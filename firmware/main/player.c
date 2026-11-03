#include "player.h"

#include <string.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/idf_additions.h"
#include "freertos/stream_buffer.h"
#include "freertos/task.h"
#include "nvs.h"

#include "audio.h"
#include "config.h"

static const char *TAG = "player";

#define BIT_START     BIT0   // a reply begins
#define BIT_FINISHED  BIT1   // the reply's audio is all in the buffer
#define BIT_DONE      BIT2   // the reply has finished playing
#define BIT_WAKE      BIT3   // something was written: the mixer should look again
#define BIT_M_FLUSH   BIT4   // music task -> mixer: empty the music buffer
#define BIT_M_FLUSHED BIT5   // mixer -> music task: done
#define BIT_STARTED   BIT6   // mixer -> player_start(): voice buffer emptied, new reply armed

#define BPS            ((size_t)SAMPLE_RATE_HZ * 2)            // bytes per second
#define VOICE_BYTES    (BPS * PLAYER_BUFFER_SEC)
#define MUSIC_BYTES    (BPS * MUSIC_BUFFER_SEC)
#define SFX_BYTES      (BPS / 2)
#define PREBUFFER_B    (BPS * PREBUFFER_MS / 1000)
#define REBUFFER_B     (BPS * REBUFFER_MS / 1000)
#define M_PREBUFFER_B  (BPS * MUSIC_PREBUFFER_MS / 1000)
#define M_REBUFFER_B   (BPS * 300 / 1000)
#define HOLD_BLOCKS    25         // ~400 ms of silence before the amp is switched off

// Gain changes are smoothed per sample so they never click.
#define DUCK_DOWN_STEP (1.0f / (0.06f * SAMPLE_RATE_HZ))   // ~60 ms to duck
#define DUCK_UP_STEP   (1.0f / (0.50f * SAMPLE_RATE_HZ))   // ~0.5 s to come back up
#define VOL_STEP       (1.0f / (0.03f * SAMPLE_RATE_HZ))
#define HP_A           (1.0f / (1.0f + 2.0f * 3.14159265f * (float)MUSIC_HIGHPASS_HZ / SAMPLE_RATE_HZ))

typedef enum { ST_IDLE, ST_WAIT, ST_PLAY } stream_state_t;

// A buffer the mixer reads from. Writers can split a sample across two writes,
// so the reader keeps an odd byte back until its partner arrives.
typedef struct {
    StreamBufferHandle_t sb;
    bool    has_carry;
    uint8_t carry;
} source_t;

static source_t           s_voice, s_music, s_sfx;
static EventGroupHandle_t s_ev;
static player_stats_t     s_stats;

static stream_state_t     s_vstate = ST_IDLE;     // mixer-owned
static size_t             s_vneed;
static stream_state_t     s_mstate = ST_IDLE;     // mixer-owned
static size_t             s_mneed;
static volatile bool      s_m_on;                 // a song is loaded
static volatile bool      s_m_eos;                // ...and all its data is in the buffer
static volatile uint32_t  s_m_played;

static volatile bool      s_duck;
static volatile bool      s_listening;
static volatile bool      s_amp_running;          // the mixer is feeding the amp (DMA queue is full)
static volatile int       s_volume = DEFAULT_VOLUME;
static volatile float     s_master = 1.0f;        // target master gain from s_volume
static int                s_saved_volume = -1;

/* ------------------------------------------------------------------ */

static size_t avail_samples(source_t *s)
{
    return (xStreamBufferBytesAvailable(s->sb) + (s->has_carry ? 1 : 0)) / 2;
}

static size_t take(source_t *s, int16_t *out, size_t samples)
{
    uint8_t *p = (uint8_t *)out;
    size_t got = 0;
    if (s->has_carry) {
        p[0] = s->carry;
        s->has_carry = false;
        got = 1;
    }
    got += xStreamBufferReceive(s->sb, p + got, samples * 2 - got, 0);
    if (got & 1) {
        s->carry     = p[got - 1];
        s->has_carry = true;
        got--;
    }
    return got / 2;
}

static void flush(source_t *s)
{
    xStreamBufferReset(s->sb);
    s->has_carry = false;
}

// Write all of `bytes`, never stopping between the two halves of a sample.
// Returns the bytes written (even), fewer only if the buffer stayed full.
static size_t send_samples(StreamBufferHandle_t sb, const int16_t *pcm, size_t samples, TickType_t wait)
{
    size_t sent = xStreamBufferSend(sb, pcm, samples * 2, wait);
    if (sent & 1) {
        // Half a sample went in: the other half must follow, the reader is draining.
        while (xStreamBufferSend(sb, (const uint8_t *)pcm + sent, 1, pdMS_TO_TICKS(100)) == 0) {
        }
        sent++;
    }
    if (sent > 0) {
        xEventGroupSetBits(s_ev, BIT_WAKE);
    }
    return sent;
}

static inline int16_t clamp16(float v)
{
    if (v > 32767.0f)  return 32767;
    if (v < -32768.0f) return -32768;
    return (int16_t)v;
}

/* ------------------------------------------------------------------ */
/*  The mixer task: the only thing that writes to the speaker         */
/* ------------------------------------------------------------------ */

static void mixer_task(void *arg)
{
    static int16_t v[AUDIO_BLOCK], m[AUDIO_BLOCK], fx[AUDIO_BLOCK], out[AUDIO_BLOCK];
    bool  amp   = false;
    int   quiet = 0;
    float mgain = 1.0f;          // current music gain (ducking)
    float vol   = s_master;      // current master gain
    float hp_x  = 0.0f, hp_y = 0.0f;   // music high-pass state

    for (;;) {
        EventBits_t bits = xEventGroupGetBits(s_ev);
        if (bits & BIT_M_FLUSH) {
            flush(&s_music);
            s_mstate = ST_WAIT;
            s_mneed  = M_PREBUFFER_B;
            xEventGroupClearBits(s_ev, BIT_M_FLUSH);
            xEventGroupSetBits(s_ev, BIT_M_FLUSHED);
        }
        if (bits & BIT_START) {
            // Emptied here, by the only reader, so the reset can't race a read.
            flush(&s_voice);
            s_vstate = ST_WAIT;
            s_vneed  = PREBUFFER_B;
            xEventGroupClearBits(s_ev, BIT_START);
            xEventGroupSetBits(s_ev, BIT_STARTED);
        }
        bool vfin = (bits & BIT_FINISHED) != 0;
        size_t nv = 0, nm = 0, nf = 0;

        // ---- Friday's voice: prebuffer, play, refill if the network stalls
        if (s_vstate == ST_WAIT && (xStreamBufferBytesAvailable(s_voice.sb) >= s_vneed || vfin)) {
            s_vstate = ST_PLAY;
        }
        if (s_vstate == ST_PLAY) {
            size_t have = avail_samples(&s_voice);
            if (have >= AUDIO_BLOCK || (vfin && have > 0)) {
                nv = take(&s_voice, v, AUDIO_BLOCK);
                if (s_stats.first_sound_us == 0) {
                    s_stats.first_sound_us = esp_timer_get_time();
                }
                s_stats.played_samples += nv;
            } else if (vfin) {
                s_vstate = ST_IDLE;                      // the whole reply has played
                xEventGroupSetBits(s_ev, BIT_DONE);
            } else {
                s_stats.underruns++;                     // ran dry mid-reply: refill a little
                s_vstate = ST_WAIT;
                s_vneed  = REBUFFER_B;
            }
        }

        // ---- the song
        if (s_m_on) {
            if (s_mstate == ST_IDLE) {                   // a song was loaded without a flush
                s_mstate = ST_WAIT;
                s_mneed  = M_PREBUFFER_B;
            }
            if (s_mstate == ST_WAIT && (xStreamBufferBytesAvailable(s_music.sb) >= s_mneed || s_m_eos)) {
                s_mstate = ST_PLAY;
            }
            if (s_mstate == ST_PLAY) {
                size_t have = avail_samples(&s_music);
                if (have >= AUDIO_BLOCK || (s_m_eos && have > 0)) {
                    nm = take(&s_music, m, AUDIO_BLOCK);
                    s_m_played += nm;
                } else if (s_m_eos) {
                    s_m_on   = false;                    // song over
                    s_mstate = ST_IDLE;
                } else {
                    s_mstate = ST_WAIT;
                    s_mneed  = M_REBUFFER_B;
                }
            }
        }

        // ---- chimes
        size_t have_fx = avail_samples(&s_sfx);
        if (have_fx > 0) {
            nf = take(&s_sfx, fx, have_fx < AUDIO_BLOCK ? have_fx : AUDIO_BLOCK);
        }

        size_t n = nv > nm ? nv : nm;
        if (nf > n) n = nf;

        if (n > 0) {
            if (!amp) {
                audio_play_begin();
                amp = true;
                s_amp_running = true;
            }
            float target = s_listening ? 0.0f
                         : (s_vstate != ST_IDLE || s_duck) ? MUSIC_DUCK_LEVEL : 1.0f;
            float master = s_master;
            for (size_t i = 0; i < AUDIO_BLOCK; i++) {
                if (mgain > target) {
                    mgain -= DUCK_DOWN_STEP;
                    if (mgain < target) mgain = target;
                } else if (mgain < target) {
                    mgain += DUCK_UP_STEP;
                    if (mgain > target) mgain = target;
                }
                if (vol > master) {
                    vol -= VOL_STEP;
                    if (vol < master) vol = master;
                } else if (vol < master) {
                    vol += VOL_STEP;
                    if (vol > master) vol = master;
                }
                float x = 0.0f;
                if (i < nv) x += (float)v[i];
                if (i < nm) {
#if MUSIC_HIGHPASS_HZ > 0
                    // one-pole high-pass: y = a * (y_prev + x - x_prev)
                    float in = (float)m[i];
                    hp_y = HP_A * (hp_y + in - hp_x);
                    hp_x = in;
                    x += hp_y * mgain * MUSIC_GAIN;
#else
                    x += (float)m[i] * mgain * MUSIC_GAIN;
#endif
                }
                if (i < nf) x += (float)fx[i];
                out[i] = clamp16(x * vol);
            }
            audio_play_chunk(out, AUDIO_BLOCK);          // blocks: paces the loop at real time
            quiet = 0;
        } else if (amp) {
            if (++quiet < HOLD_BLOCKS) {                 // a short gap: keep the amp on
                memset(out, 0, sizeof(out));
                audio_play_chunk(out, AUDIO_BLOCK);
            } else {
                audio_play_end();
                amp = false;
                s_amp_running = false;
            }
        } else {
            xEventGroupWaitBits(s_ev, BIT_WAKE, pdTRUE, pdFALSE, pdMS_TO_TICKS(10));
        }
    }
}

static StreamBufferHandle_t make_buffer(size_t bytes)
{
    StreamBufferHandle_t sb = xStreamBufferCreateWithCaps(bytes, 1, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!sb) {
        ESP_LOGE(TAG, "could not allocate %u bytes of PSRAM for playback", (unsigned)bytes);
        abort();
    }
    return sb;
}

void player_init(void)
{
    s_ev          = xEventGroupCreate();
    s_voice.sb    = make_buffer(VOICE_BYTES);
    s_music.sb    = make_buffer(MUSIC_BYTES);
    s_sfx.sb      = make_buffer(SFX_BYTES);
    xEventGroupSetBits(s_ev, BIT_DONE);              // nothing playing yet
    player_set_volume(DEFAULT_VOLUME);
    // Own core (Wi-Fi runs on core 0) and above everything else of ours, so
    // playback never waits behind HTTP, TLS or MP3 decoding.
    if (xTaskCreatePinnedToCore(mixer_task, "mixer", 4096, NULL, 10, NULL, 1) != pdPASS) {
        ESP_LOGE(TAG, "could not start the mixer task");
        abort();
    }
    audio_set_tone_sink(player_sfx);
    ESP_LOGI(TAG, "mixer: voice %d s, music %d s buffers; speech starts after %d ms",
             PLAYER_BUFFER_SEC, MUSIC_BUFFER_SEC, PREBUFFER_MS);
}

/* ------------------------------------------------------------------ */
/*  Voice                                                              */
/* ------------------------------------------------------------------ */

void player_start(void)
{
    if (!(xEventGroupGetBits(s_ev) & BIT_DONE)) {
        // The last reply is somehow still playing: let it end first.
        xEventGroupSetBits(s_ev, BIT_FINISHED);
        xEventGroupWaitBits(s_ev, BIT_DONE, pdFALSE, pdFALSE, pdMS_TO_TICKS(3000));
    }
    s_stats = (player_stats_t){0};
    xEventGroupClearBits(s_ev, BIT_FINISHED | BIT_DONE | BIT_STARTED);
    xEventGroupSetBits(s_ev, BIT_START | BIT_WAKE);
    // The mixer empties the voice buffer and arms the reply; wait for that, so
    // nothing written after this returns can be thrown away.
    xEventGroupWaitBits(s_ev, BIT_STARTED, pdTRUE, pdFALSE, pdMS_TO_TICKS(500));
}

void player_write(const int16_t *pcm, size_t samples)
{
    while (samples > 0) {
        // Only blocks if PLAYER_BUFFER_SEC is already full; the mixer drains
        // it at 32 KB/s, so this can't wait long.
        size_t sent = send_samples(s_voice.sb, pcm, samples, pdMS_TO_TICKS(2000)) / 2;
        if (sent == 0) {
            ESP_LOGW(TAG, "buffer stuck full, dropping %u samples", (unsigned)samples);
            return;
        }
        pcm     += sent;
        samples -= sent;
    }
    size_t held = xStreamBufferBytesAvailable(s_voice.sb) / 2;
    if (held > s_stats.peak_samples) {
        s_stats.peak_samples = held;
    }
}

void player_finish(void)
{
    xEventGroupSetBits(s_ev, BIT_FINISHED | BIT_WAKE);
}

bool player_wait_done(int timeout_ms)
{
    EventBits_t b = xEventGroupWaitBits(s_ev, BIT_DONE, pdFALSE, pdFALSE, pdMS_TO_TICKS(timeout_ms));
    return (b & BIT_DONE) != 0;
}

player_stats_t player_stats(void)
{
    return s_stats;
}

/* ------------------------------------------------------------------ */
/*  Music                                                              */
/* ------------------------------------------------------------------ */

static void music_flush(void)
{
    xEventGroupClearBits(s_ev, BIT_M_FLUSHED);
    xEventGroupSetBits(s_ev, BIT_M_FLUSH | BIT_WAKE);
    xEventGroupWaitBits(s_ev, BIT_M_FLUSHED, pdTRUE, pdFALSE, pdMS_TO_TICKS(500));
}

void player_music_begin(void)
{
    s_m_on = false;
    music_flush();
    s_m_played = 0;
    s_m_eos    = false;
    s_m_on     = true;
}

size_t player_music_room(void)
{
    return xStreamBufferSpacesAvailable(s_music.sb) / 2;
}

size_t player_music_write(const int16_t *pcm, size_t samples, int timeout_ms)
{
    return send_samples(s_music.sb, pcm, samples, pdMS_TO_TICKS(timeout_ms)) / 2;
}

void player_music_end(void)
{
    s_m_eos = true;
    xEventGroupSetBits(s_ev, BIT_WAKE);
}

void player_music_stop(void)
{
    s_m_on = false;
    music_flush();
}

bool player_music_idle(void)
{
    return !s_m_on;
}

uint32_t player_music_played(void)
{
    return s_m_played;
}

/* ------------------------------------------------------------------ */
/*  Levels and chimes                                                  */
/* ------------------------------------------------------------------ */

void player_set_duck(bool on)
{
    s_duck = on;
}

void player_set_listening(bool on)
{
    s_listening = on && MUSIC_MUTE_LISTENING;
}

void player_set_volume(int percent)
{
    if (percent < 0)   percent = 0;
    if (percent > 100) percent = 100;
    s_volume = percent;
    float f  = (float)percent / 100.0f;
    s_master = f * f;                                // roughly even steps to the ear
}

int player_volume(void)
{
    return s_volume;
}

void player_volume_load(void)
{
    nvs_handle_t h;
    uint8_t v = DEFAULT_VOLUME;
    if (nvs_open("friday", NVS_READONLY, &h) == ESP_OK) {
        nvs_get_u8(h, "volume", &v);
        nvs_close(h);
    }
    s_saved_volume = v;
    player_set_volume(v);
}

void player_volume_save(void)
{
    if (s_volume == s_saved_volume) {
        return;
    }
    nvs_handle_t h;
    if (nvs_open("friday", NVS_READWRITE, &h) == ESP_OK) {
        if (nvs_set_u8(h, "volume", (uint8_t)s_volume) == ESP_OK && nvs_commit(h) == ESP_OK) {
            s_saved_volume = s_volume;
        }
        nvs_close(h);
    }
}

void player_sfx(const int16_t *pcm, size_t samples)
{
    send_samples(s_sfx.sb, pcm, samples, pdMS_TO_TICKS(200));
}

void player_sfx_drain(void)
{
    for (int i = 0; i < 60 && xStreamBufferBytesAvailable(s_sfx.sb) > 0; i++) {
        vTaskDelay(pdMS_TO_TICKS(5));
    }
    // The mixer has taken the last of it; now let it come out of the speaker.
    // While the amp is running the I2S DMA queue holds ~256 ms; from idle, very little.
    vTaskDelay(pdMS_TO_TICKS(s_amp_running ? 280 : 70));
}
