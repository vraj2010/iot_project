// ============================================================
//  I2Sonic assistant "Friday"  —  ESP32-S3   (Phase 6: music)
//
//  Say the wake word near the laptop (LAPTOP_MIC 1: the laptop's mic
//  listens), hold BOOT and speak, or type a question in the serial
//  monitor. The S3 sends it to friday-local on your laptop, which hears it
//  (faster-whisper), answers with Gemini using its memory and tools
//  (PostgreSQL, web search, your music library) and speaks the answer.
//  Songs stream from the laptop and play under Friday's voice.
//  If the laptop is off, the same request goes to the Cloudflare Worker.
//
//  Edge:  record, clean up, mix and play, HTTP client, offline SAM voice
//  Fog:   laptop -> Whisper -> Gemini + memory + tools -> voice, music
//  Cloud: Worker (backup) -> Whisper -> Gemini -> voice
// ============================================================

#include <stdio.h>
#include <string.h>

#include "driver/gpio.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "audio.h"
#include "cloud.h"
#include "config.h"
#include "music.h"
#include "net.h"
#include "player.h"
#include "sam_tts.h"
#include "textin.h"

static const char *TAG = "i2sonic";

#define MAX_RECORD_SAMPLES ((size_t)SAMPLE_RATE_HZ * MAX_RECORD_SEC)
#define MIN_RECORD_SEC     0.4f

static int16_t *s_record;             // spoken question, lives in PSRAM

// One turn at a time. Whoever holds s_turn owns the mic, the speaker, the
// record buffer and the HTTPS connection (none of those are shareable).
// The main loop takes it when BOOT goes down; the assistant task gives it
// back once the answer has finished playing.
static SemaphoreHandle_t s_turn;
static QueueHandle_t     s_voice_q;   // recorded sample counts, main -> assistant
static QueueHandle_t     s_wake_q;    // wake ids from the laptop's ears, ears -> assistant

// ---------------------------------------------------------------- push-to-talk

static bool ptt_down(void)
{
    return gpio_get_level(PTT_GPIO) == 0;     // BOOT is active-low
}

// Records while the button is held. Returns the number of samples captured.
static size_t record_question(void)
{
    size_t n = 0;
    int    meter_tick = 0;

    audio_flush_mic();                        // drop whatever buffered while idle
    printf("\n[listening] hold BOOT and speak...\n");

    while (ptt_down() && n + AUDIO_BLOCK <= MAX_RECORD_SAMPLES) {
        audio_capture(s_record + n);
        n += AUDIO_BLOCK;

#if SHOW_METER
        if (++meter_tick >= 15) {             // roughly four updates a second
            meter_tick = 0;
            float db = audio_level_db();
            int fill = (int)((db + 60.0f) * 0.5f);
            if (fill < 0)  fill = 0;
            if (fill > 30) fill = 30;
            char bar[31];
            for (int i = 0; i < 30; i++) {
                bar[i] = i < fill ? '#' : '-';
            }
            bar[30] = '\0';
            printf("\r  %6.1f dBFS [%s] %4.1fs", db, bar,
                   (double)n / SAMPLE_RATE_HZ);
            fflush(stdout);
        }
#else
        (void)meter_tick;
#endif
    }
    printf("\n");
    if (ptt_down()) {
        printf("[recorded] %d s limit reached (MAX_RECORD_SEC)\n", MAX_RECORD_SEC);
    }
    return n;
}

// Runs in the main task while it holds s_turn. Returns the samples to send,
// or 0 if the recording is too short to be a question.
static size_t capture_question(void)
{
    music_talk_begin();                       // music ducks (or pauses) while you talk
    player_set_listening(true);               // ...and is silent while the mic records
    audio_chime_start();
    player_sfx_drain();                       // start recording once the chime (and the song) are quiet
    size_t n = record_question();
    player_set_listening(false);

    float seconds = (float)n / SAMPLE_RATE_HZ;
    if (seconds < MIN_RECORD_SEC) {
        printf("[too short] hold BOOT a little longer\n");
        music_talk_end();
        return 0;
    }

    float gain = audio_normalize(s_record, n);
    printf("[recorded] %.1f s, normalised %.1fx\n", (double)seconds, (double)gain);
    audio_chime_done();
    return n;
}

// ---------------------------------------------------------------- one turn

static char s_reply[1024];            // last reply text, for the offline voice
static bool s_laptop_listening;       // music is silent so the laptop's mic hears you

// The laptop has heard the whole question: the music can come back up (ducked under her reply).
static void laptop_heard_you(void)
{
    if (s_laptop_listening) {
        s_laptop_listening = false;
        player_set_listening(false);
    }
}

static void on_cloud_heard(const char *heard)
{
    laptop_heard_you();
    if (heard[0] != '\0') {
        printf("[heard] \"%s\"\n", heard);
    } else {
        printf("[heard] nothing clear\n");
    }
    fflush(stdout);
}

static void on_cloud_text(const char *reply)
{
    strncpy(s_reply, reply, sizeof(s_reply) - 1);
    s_reply[sizeof(s_reply) - 1] = '\0';
    if (reply[0] != '\0') {
        printf("friday> %s\n", reply);
        fflush(stdout);
    }
}

// Audio goes into the playback buffer, never straight to the speaker, so the
// download can run ahead and network stalls don't become gaps in the speech.
static void on_cloud_audio(const int16_t *pcm, size_t samples)
{
    player_write(pcm, samples);
}

// Speak `text` with SAM, the on-device voice: no internet, no quota.
static void say_offline(const char *text)
{
    player_start();
    float secs = sam_say(text, player_write);
    player_finish();
    player_wait_done((PLAYER_BUFFER_SEC + 5) * 1000);
    printf("[voice] offline (SAM), %.1f s\n", (double)secs);
}

static void print_failure(int http_status, const char *why, bool chime)
{
    if (http_status == 0) {
        printf("[failed] %s\n", why);
    } else {
        printf("[failed] HTTP %d: %s\n", http_status, why);
    }
    if (http_status == 401) {
        printf("         DEVICE_TOKEN in secrets.h doesn't match the server's (friday-local .env / Worker secret)\n");
    } else if (http_status == 429) {
        printf("         free-tier limit hit; wait a minute and try again\n");
    }
    if (chime) {
        audio_chime_error();
    }
}

// Ask the server and speak the answer. Typed questions go to /say; recordings go to
// /ask, which hears them first; wake_id > 0 fetches what the laptop's mic heard (/listen).
static void ask_and_speak(const char *question, const int16_t *pcm, size_t samples, int wake_id)
{
    static const cloud_say_cb_t cb = {
        .on_heard    = on_cloud_heard,
        .on_text     = on_cloud_text,
        .on_audio    = on_cloud_audio,
        .on_action   = MUSIC_ENABLED ? music_apply : NULL,
        .music_state = MUSIC_ENABLED ? music_state_header : NULL,
    };
    static char why[256];
    cloud_result_t r;

    int64_t asked = esp_timer_get_time();
    s_reply[0] = '\0';
    player_start();

    esp_err_t err = wake_id > 0 ? cloud_listen_answer(wake_id, &cb, why, sizeof(why), &r)
                  : question    ? cloud_say(question, &cb, why, sizeof(why), &r)
                                : cloud_ask(pcm, samples, &cb, why, sizeof(why), &r);
    int64_t downloaded = esp_timer_get_time();
    if (wake_id > 0 && err == ESP_ERR_NOT_FOUND) {   // the wake word, then nothing
        player_finish();
        printf("[heard] nothing after the wake word\n");
        return;
    }

    // Let whatever arrived finish playing, even if the download broke off.
    player_finish();
    player_wait_done((PLAYER_BUFFER_SEC + 5) * 1000);
    player_stats_t p = player_stats();

    // No cloud audio at all? The reply still gets spoken, by SAM on the S3.
    bool fallback = SAM_FALLBACK && err != ESP_OK && p.played_samples == 0;

    if (err != ESP_OK) {
        print_failure(r.http_status, why, !fallback);
    }
    if (fallback) {
        say_offline(s_reply[0] ? s_reply : "Sorry, I could not reach the cloud.");
        return;
    }
    if (p.played_samples > 0 || err == ESP_OK) {
        char hearing[32] = "";
        if (!question) {
            snprintf(hearing, sizeof(hearing), "Whisper %d ms + ", r.stt_ms);
        }
        printf("[time] %s | first sound %d ms | %sGemini %d ms + %s %d ms | download %s %d ms\n",
               cloud_brain_name(r.brain),
               p.first_sound_us ? (int)((p.first_sound_us - asked) / 1000) : -1,
               hearing, r.chat_ms, cloud_provider_name(r.provider), r.tts_ms,
               err == ESP_OK ? "done" : "broke at", (int)((downloaded - asked) / 1000));
        printf("[audio] %.1f s spoken of %.1f s received, buffer peak %.1f s, %d underrun%s\n",
               (double)p.played_samples / SAMPLE_RATE_HZ,
               (double)r.audio_samples / SAMPLE_RATE_HZ,
               (double)p.peak_samples / SAMPLE_RATE_HZ,
               p.underruns, p.underruns == 1 ? "" : "s");
    }
}

static void ask_text_only(const char *question)
{
    static char answer[1024];
    cloud_result_t r;

    esp_err_t err = cloud_chat(question, answer, sizeof(answer), &r);
    if (err == ESP_OK) {
        printf("friday> %s\n", answer);
        printf("[time] %s | %d ms total, %d ms on the server\n", cloud_brain_name(r.brain), r.total_ms, r.worker_ms);
        audio_chime_done();
    } else {
        print_failure(r.http_status, answer, true);
    }
}

static bool online_or_say_so(void)
{
    if (net_is_online()) {
        return true;
    }
    printf("[offline] no Wi-Fi, can't reach the cloud\n");
    if (SAM_FALLBACK) {
        say_offline("I can't reach the internet right now.");
    } else {
        audio_chime_error();
    }
    return false;
}

// music_talk_begin() ducks (or pauses) the music for the whole turn, from the
// moment BOOT goes down until the answer has played; this puts it back.
static void turn_over(void)
{
    music_talk_end();
    player_volume_save();                     // "volume 60" is remembered across reboots
}

static void voice_turn(size_t samples)
{
    if (online_or_say_so()) {
        printf("[thinking]\n");
        ask_and_speak(NULL, s_record, samples, 0);   // spoken questions always get spoken answers
    }
    turn_over();
}

static void typed_turn(const char *question)
{
    music_talk_begin();
    if (online_or_say_so()) {
        printf("[thinking]\n");
        if (SPEAK_REPLIES) {
            ask_and_speak(question, NULL, 0, 0);
        } else {
            ask_text_only(question);
        }
    }
    turn_over();
}

// The laptop heard the wake word (or BOOT asked it to listen): chime, silence the music
// so its mic hears you and not the song, and play the answer it sends.
static void laptop_turn(int wake_id)
{
    printf("\n[listening] on the laptop's mic...\n");
    music_talk_begin();
    player_set_listening(true);
    s_laptop_listening = true;
    audio_chime_start();
    if (online_or_say_so()) {
        ask_and_speak(NULL, NULL, 0, wake_id);
    }
    laptop_heard_you();                       // also when nothing was heard
    turn_over();
}

// ---------------------------------------------------------------- the laptop's ears (LAPTOP_MIC 1)
//
// Waits for the laptop to hear the wake word, then takes the turn and hands it to the
// assistant task, which plays the answer (like BOOT hands over a recording). It only
// asks while no turn is running: the laptop takes each new poll to mean "the speaker
// has finished talking", and only then listens for the wake word again.
// Its 8 KB stack is enough for the small long poll; the answer is decoded elsewhere.

static void ears_task(void *arg)
{
    printf("[friday] say the wake word near the laptop, press BOOT, or type a question\n");
    for (;;) {
        xSemaphoreTake(s_turn, portMAX_DELAY);     // wait for any turn (and its reply) to finish
        xSemaphoreGive(s_turn);
        int wake = cloud_listen_wait(LISTEN_HOLD_SEC);
        if (wake < 0) {
            vTaskDelay(pdMS_TO_TICKS(2000));       // laptop away: the finder task looks for it
            continue;
        }
        if (wake == 0 || xSemaphoreTake(s_turn, 0) != pdTRUE) {
            continue;                              // nothing yet, or a typed / BOOT turn got there first
        }
        xQueueSend(s_wake_q, &wake, portMAX_DELAY);   // the assistant gives s_turn back when it's done
    }
}

// BOOT with LAPTOP_MIC: ask the laptop to listen. False = it can't (use push-to-talk).
static bool laptop_press(void)
{
    int status = cloud_listen_press();
    if (status == 202) {
        return true;                               // ears_task gets the wake and does the rest
    }
    if (status == 409) {
        printf("\n[busy] Friday is still answering; ask again when she's done\n");
        return true;
    }
    if (status == 0) {
        printf("\n[ears] can't reach the laptop; push-to-talk on the S3's mic instead\n");
    } else {
        printf("\n[ears] the laptop isn't listening (HTTP %d); push-to-talk on the S3's mic instead\n", status);
    }
    return false;
}

// ---------------------------------------------------------------- assistant task
//
// The task that runs every turn (laptop or Worker): it watches the keyboard, the push-to-talk
// queue and the laptop's wakes together, and runs one turn at a time. (The ears task only
// waits for the wake word, on a connection of its own.)

static void prompt(void)
{
    printf("\nyou> ");
    fflush(stdout);
}

static void assistant_task(void *arg)
{
    static char question[480];        // Worker accepts up to 500 characters
    bool typed = false;               // a typed line is waiting for its turn

    if (!LAPTOP_MIC) {                // with the laptop's mic, ears_task says how to talk to her
        printf("[friday] hold BOOT and speak, or type a question and press Enter\n");
    }
    prompt();

    for (;;) {
        size_t samples;
        // While a typed line waits (BOOT got there first), block on the queue
        // so the spoken question is handled straight away.
        TickType_t wait = typed ? pdMS_TO_TICKS(50) : 0;
        if (xQueueReceive(s_voice_q, &samples, wait) == pdTRUE) {
            voice_turn(samples);              // main handed us s_turn with it
            xSemaphoreGive(s_turn);
            prompt();
            continue;
        }
        int wake;
        if (s_wake_q && xQueueReceive(s_wake_q, &wake, 0) == pdTRUE) {
            laptop_turn(wake);                // the ears task handed us s_turn with it
            xSemaphoreGive(s_turn);
            prompt();
            continue;
        }

        if (!typed) {
            typed = textin_poll(question, sizeof(question), 50);
        }
        if (typed && xSemaphoreTake(s_turn, 0) == pdTRUE) {
            typed = false;
            typed_turn(question);
            xSemaphoreGive(s_turn);
            prompt();
        }
    }
}

// ---------------------------------------------------------------- boot

void app_main(void)
{
    textin_init();

    printf("\n=== I2Sonic: Friday (Phase 6, music) ===\n");
    printf("mic     : %s\n", MIC_TYPE == MIC_MAX4466 ? "MAX4466 (analog)" : "INMP441 (I2S)");
    printf("speaker : MAX98357A  BCLK %d  LRC %d  DIN %d  SD %d\n",
           AMP_BCLK_GPIO, AMP_LRC_GPIO, AMP_DIN_GPIO, AMP_SD_GPIO);
    if (USE_LAPTOP) {
        printf("brain   : laptop (friday-local, found automatically)%s\n",
               USE_WORKER_FALLBACK ? ", Worker as backup" : "");
    }
    if (USE_WORKER_FALLBACK || !USE_LAPTOP) {
        printf("cloud   : %s\n", WORKER_URL);
    }
    if (MUSIC_ENABLED) {
        printf("music   : from the laptop's library, %s while you talk\n",
               MUSIC_ON_TALK == MUSIC_PAUSE ? "paused" : "ducked");
    }
    if (LAPTOP_MIC) {
        printf("ears    : the laptop's mic listens for the wake word; BOOT = listen now\n");
    }
    printf("\n");

    gpio_config_t ptt = {
        .pin_bit_mask = 1ULL << PTT_GPIO,
        .mode         = GPIO_MODE_INPUT,
        .pull_up_en   = GPIO_PULLUP_ENABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&ptt));

    s_record = heap_caps_malloc(MAX_RECORD_SAMPLES * sizeof(int16_t),
                                MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_record) {
        ESP_LOGW(TAG, "no PSRAM, falling back to internal RAM (shorter questions)");
        s_record = heap_caps_malloc(MAX_RECORD_SAMPLES * sizeof(int16_t), MALLOC_CAP_8BIT);
    }
    s_turn    = xSemaphoreCreateBinary();
    s_voice_q = xQueueCreate(1, sizeof(size_t));
    if (!s_record || !s_turn || !s_voice_q) {
        ESP_LOGE(TAG, "out of memory at boot");
        return;
    }
    xSemaphoreGive(s_turn);                   // binary semaphores start empty

    audio_init();
    player_init();

    if (net_wifi_start() != ESP_OK) {
        printf("[warning] Wi-Fi did not connect — check WIFI_SSID / WIFI_PASSWORD\n");
    }
    player_volume_load();                     // NVS is ready once Wi-Fi has started
    cloud_start();                            // find the laptop (or settle for the Worker)
    music_init();

    // 32 KB stack: TLS needs ~8 KB and the MP3 decoder keeps ~16 KB of
    // scratch on the stack for each frame.
    xTaskCreate(assistant_task, "friday", 32768, NULL, 5, NULL);
    if (LAPTOP_MIC) {
        s_wake_q = xQueueCreate(1, sizeof(int));
        if (s_wake_q) {
            xTaskCreate(ears_task, "ears", 8192, NULL, 5, NULL);
        }
    }

    for (;;) {
        if (ptt_down()) {
            if (LAPTOP_MIC && s_wake_q && laptop_press()) {
                while (ptt_down()) {              // no holding needed: the laptop listens until you stop
                    vTaskDelay(pdMS_TO_TICKS(20));
                }
                continue;
            }
            if (xSemaphoreTake(s_turn, 0) == pdTRUE) {
                size_t n = capture_question();
                if (n > 0) {
                    xQueueSend(s_voice_q, &n, portMAX_DELAY);   // assistant gives s_turn back
                } else {
                    xSemaphoreGive(s_turn);
                }
            } else {
                printf("\n[busy] Friday is still answering; ask again when she's done\n");
            }
            while (ptt_down()) {              // wait for the button to come up
                vTaskDelay(pdMS_TO_TICKS(20));
            }
        }
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}
