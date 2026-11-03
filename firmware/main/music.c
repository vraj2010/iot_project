#include "music.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "cJSON.h"
#include "esp_heap_caps.h"
#include "esp_http_client.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#include "cloud.h"
#include "config.h"
#include "mp3stream.h"
#include "player.h"

static const char *TAG = "music";

#define READ_BYTES  4096
#define FEED_BYTES  1024         // MP3 bytes decoded per step (~128 ms at 64 kbps)
#define FEED_ROOM   4096         // ...when the mixer has room for this many samples
#define READ_WAIT_MS 50          // socket read timeout while streaming
#define STALL_MS    15000        // no data this long = the stream is broken
// minimp3 keeps ~16 KB of scratch on the stack per frame. Internal RAM, not
// PSRAM: ESP-IDF advises against PSRAM stacks (unusable while flash is busy).
#define TASK_STACK  (24 * 1024)

typedef enum { CMD_PLAY, CMD_PAUSE, CMD_RESUME, CMD_STOP } cmd_op_t;
typedef struct {
    cmd_op_t op;
    int      track;
} cmd_t;

typedef enum { M_STOPPED, M_PLAYING, M_PAUSED } mstate_t;
typedef enum { S_END, S_BROKEN, S_ABORT, S_NONE, S_FAIL } stream_result_t;

static QueueHandle_t     s_cmds;
static volatile mstate_t s_state = M_STOPPED;
static volatile int      s_track;          // song playing (or paused, or last played)
static volatile float    s_base;           // seconds into the song where this stream started
static volatile float    s_paused_at;
static volatile bool     s_talk_paused;    // paused only because you're talking to Friday
static mp3s_t           *s_dec;
static uint8_t          *s_mp3;            // the song as downloaded (PSRAM)
static size_t            s_mp3_cap;
static char              s_title[96];
static int               s_hdr_track;      // from the response headers
static char              s_hdr_title[96];

/* ------------------------------------------------------------------ */

// A command is waiting: whatever the task is doing should stop.
static bool pending(void)
{
    return s_cmds && uxQueueMessagesWaiting(s_cmds) > 0;
}

static float position(void)
{
    switch (s_state) {
    case M_PLAYING: return s_base + (float)player_music_played() / SAMPLE_RATE_HZ;
    case M_PAUSED:  return s_paused_at;
    default:        return 0.0f;
    }
}

static void send_cmd(cmd_op_t op, int track)
{
    if (!s_cmds) {
        return;
    }
    cmd_t c = { .op = op, .track = track };
    if (xQueueSend(s_cmds, &c, 0) != pdTRUE) {
        ESP_LOGW(TAG, "too many music commands at once, one dropped");
    }
}

void music_play(int track) { send_cmd(CMD_PLAY, track); }
void music_pause(void)     { send_cmd(CMD_PAUSE, 0); }
void music_resume(void)    { send_cmd(CMD_RESUME, 0); }
void music_stop(void)      { send_cmd(CMD_STOP, 0); }

// "Kesariya%20by%20Arijit" -> "Kesariya by Arijit"
static void url_decode(const char *in, char *out, size_t len)
{
    size_t o = 0;
    for (size_t i = 0; in[i] && o + 1 < len; i++) {
        if (in[i] == '%' && in[i + 1] && in[i + 2]) {
            char hex[3] = { in[i + 1], in[i + 2], 0 };
            out[o++] = (char)strtol(hex, NULL, 16);
            i += 2;
        } else {
            out[o++] = in[i];
        }
    }
    out[o] = '\0';
}

static esp_err_t on_http_event(esp_http_client_event_t *e)
{
    if (e->event_id == HTTP_EVENT_ON_HEADER && e->header_key && e->header_value) {
        if (strcasecmp(e->header_key, "X-Track-Id") == 0) {
            s_hdr_track = atoi(e->header_value);
        } else if (strcasecmp(e->header_key, "X-Track-Title") == 0) {
            url_decode(e->header_value, s_hdr_title, sizeof(s_hdr_title));
        }
    }
    return ESP_OK;
}

// Decoded audio -> the mixer. Normally there is room (the caller checks first);
// if not, waits, but lets go at once when a command arrives.
static void on_pcm(const int16_t *pcm, size_t samples)
{
    while (samples > 0 && !pending()) {
        size_t took = player_music_write(pcm, samples, 100);
        pcm     += took;
        samples -= took;
    }
}

// One HTTP stream: /music/<id>?t=... or /music/next?after=...
static stream_result_t stream(const char *path)
{
    char base[48];
    if (!cloud_laptop_base(base, sizeof(base))) {
        printf("[music] the laptop isn't reachable, so there's no music\n");
        return S_FAIL;
    }
    char url[128];
    snprintf(url, sizeof(url), "%s%s", base, path);

    esp_http_client_config_t cfg = {
        .url           = url,
        .method        = HTTP_METHOD_GET,
        .timeout_ms    = 15000,               // the first play of a song is converted on the fly
        .buffer_size   = 2048,
        .event_handler = on_http_event,
    };
    esp_http_client_handle_t client = esp_http_client_init(&cfg);
    if (!client) {
        return S_FAIL;
    }
    esp_http_client_set_header(client, "X-Device-Token", DEVICE_TOKEN);
    esp_http_client_set_header(client, "User-Agent", "i2sonic-s3/0.5");
    s_hdr_track    = 0;
    s_hdr_title[0] = '\0';

    stream_result_t result = S_FAIL;
    if (esp_http_client_open(client, 0) != ESP_OK) {
        printf("[music] couldn't connect to the laptop\n");
        goto done;
    }
    if (esp_http_client_fetch_headers(client) < 0) {
        printf("[music] the laptop didn't answer\n");
        goto done;
    }
    int status = esp_http_client_get_status_code(client);
    if (status == 204) {
        result = S_NONE;
        goto done;
    }
    if (status != 200) {
        char msg[160];
        int n = esp_http_client_read(client, msg, sizeof(msg) - 1);
        msg[n > 0 ? n : 0] = '\0';
        printf("[music] the laptop said HTTP %d: %s\n", status, msg);
        goto done;
    }
    if (s_hdr_track > 0) {
        s_track = s_hdr_track;
    }
    if (s_hdr_title[0]) {
        strncpy(s_title, s_hdr_title, sizeof(s_title) - 1);
        s_title[sizeof(s_title) - 1] = '\0';
    }
    printf("[music] playing %s%s\n", s_title, s_base > 0.5f ? " (carrying on)" : "");

    // Download as fast as the laptop sends (into PSRAM), decode only as fast as
    // the mixer plays. Unread data never piles up in the socket, where it would
    // hold Wi-Fi receive buffers that Friday's next reply needs.
    esp_http_client_set_timeout_ms(client, READ_WAIT_MS);
    player_music_begin();
    mp3s_start(s_dec, on_pcm);
    size_t have = 0, used = 0;
    int    stalled_ms = 0;
    bool   eof = false, broken = false;
    for (;;) {
        if (pending()) {
            result = S_ABORT;
            break;
        }
        bool busy = false;
        if (!eof) {
            if (have == s_mp3_cap && used > 0) {           // longer than the buffer: slide it
                memmove(s_mp3, s_mp3 + used, have - used);
                have -= used;
                used  = 0;
            }
            if (have < s_mp3_cap) {
                size_t want = s_mp3_cap - have < READ_BYTES ? s_mp3_cap - have : READ_BYTES;
                int n = esp_http_client_read(client, (char *)s_mp3 + have, (int)want);
                if (n > 0) {
                    have += (size_t)n;
                    stalled_ms = 0;
                    busy = true;
                } else if (n == -ESP_ERR_HTTP_EAGAIN) {      // nothing arrived within READ_WAIT_MS
                    stalled_ms += READ_WAIT_MS;
                    if (stalled_ms >= STALL_MS) {
                        eof = broken = true;
                    }
                } else {
                    eof    = true;
                    broken = n < 0 || !esp_http_client_is_complete_data_received(client);
                }
            }
        }
        if (used < have && player_music_room() >= FEED_ROOM) {
            size_t k = have - used < FEED_BYTES ? have - used : FEED_BYTES;
            mp3s_push(s_dec, s_mp3 + used, k);
            used += k;
            busy = true;
        } else if (eof && used == have) {
            result = broken ? S_BROKEN : S_END;
            break;
        }
        if (!busy && (eof || have == s_mp3_cap)) {
            vTaskDelay(pdMS_TO_TICKS(20));                 // waiting for the mixer to make room
        }
    }
    if (result == S_END || result == S_BROKEN) {
        if (broken) {
            printf("[music] the stream broke off\n");
        }
        mp3s_finish(s_dec);
        player_music_end();
    }

done:
    esp_http_client_close(client);
    esp_http_client_cleanup(client);
    return result;
}

// Play `track` from `pos` seconds, then whatever the laptop queues next,
// until a command arrives or the queue runs out.
static void play_from(int track, float pos)
{
    char path[48];
    snprintf(path, sizeof(path), "/music/%d?t=%.1f", track, (double)pos);
    s_track = track;
    s_base  = pos;
    s_state = M_PLAYING;

    int retries = 0;
    for (;;) {
        stream_result_t r = stream(path);
        if (r == S_ABORT) {
            break;
        }
        if (r == S_FAIL) {
            player_music_stop();
            s_state = M_STOPPED;
            return;
        }
        if (r == S_NONE) {
            printf("[music] that's the end of the queue\n");
            s_state = M_STOPPED;
            return;
        }
        while (!player_music_idle() && !pending()) {     // let what arrived play out
            vTaskDelay(pdMS_TO_TICKS(50));
        }
        if (pending()) {
            break;
        }
        if (r == S_BROKEN && retries < 2) {               // Wi-Fi hiccup: pick the song up where it broke
            retries++;
            float at = position();
            snprintf(path, sizeof(path), "/music/%d?t=%.1f", s_track, (double)at);
            s_base = at;
            continue;
        }
        retries = 0;
        snprintf(path, sizeof(path), "/music/next?after=%d", s_track);
        s_base = 0.0f;
    }

    // A command arrived. Hold our place: it may be "pause" (or a resume of this song).
    s_paused_at = position();
    s_state     = M_PAUSED;
    player_music_stop();
}

static void music_task(void *arg)
{
    cmd_t c;
    for (;;) {
        xQueueReceive(s_cmds, &c, portMAX_DELAY);
        // If another command is already queued, pending() makes this one stop at once.
        switch (c.op) {
        case CMD_PLAY:
            play_from(c.track, 0.0f);
            break;
        case CMD_RESUME:
            if (s_state == M_PAUSED) {
                play_from(s_track, s_paused_at);
            }
            break;
        case CMD_PAUSE:
            if (s_state != M_STOPPED) {
                printf("[music] paused at %d s\n", (int)s_paused_at);
            }
            break;
        case CMD_STOP:
            if (s_state != M_STOPPED) {
                printf("[music] stopped\n");
            }
            s_state = M_STOPPED;
            player_music_stop();
            break;
        }
    }
}

void music_init(void)
{
    if (!MUSIC_ENABLED) {
        return;
    }
    s_dec     = mp3s_new();
    s_cmds    = xQueueCreate(8, sizeof(cmd_t));
    s_mp3_cap = (size_t)MUSIC_DOWNLOAD_KB * 1024;
    s_mp3     = heap_caps_malloc(s_mp3_cap, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_dec || !s_cmds || !s_mp3 ||
        xTaskCreatePinnedToCore(music_task, "music", TASK_STACK, NULL, 4, NULL, 1) != pdPASS) {
        ESP_LOGE(TAG, "not enough memory for music");
        s_cmds = NULL;
    }
}

/* ------------------------------------------------------------------ */
/*  Friday's actions and the turn around them                          */
/* ------------------------------------------------------------------ */

void music_apply(const char *json)
{
    cJSON *doc = cJSON_Parse(json);
    if (!doc) {
        return;
    }
    const cJSON *vol = cJSON_GetObjectItemCaseSensitive(doc, "volume");
    if (cJSON_IsNumber(vol)) {
        player_set_volume(vol->valueint);
        printf("[music] volume %d%%\n", player_volume());
    }
    const cJSON *what = cJSON_GetObjectItemCaseSensitive(doc, "music");
    if (cJSON_IsString(what) && s_cmds) {
        s_talk_paused = false;               // Friday changed the music: don't undo it after the turn
        const char *m = what->valuestring;
        const cJSON *track = cJSON_GetObjectItemCaseSensitive(doc, "track");
        const cJSON *title = cJSON_GetObjectItemCaseSensitive(doc, "title");
        if (strcmp(m, "play") == 0 && cJSON_IsNumber(track)) {
            if (cJSON_IsString(title)) {
                strncpy(s_title, title->valuestring, sizeof(s_title) - 1);
                s_title[sizeof(s_title) - 1] = '\0';
            }
            music_play(track->valueint);
        } else if (strcmp(m, "pause") == 0) {
            music_pause();
        } else if (strcmp(m, "resume") == 0) {
            music_resume();
        } else if (strcmp(m, "stop") == 0) {
            music_stop();
        }
    }
    cJSON_Delete(doc);
}

void music_talk_begin(void)
{
    player_set_duck(true);
    if (MUSIC_ON_TALK == MUSIC_PAUSE && s_state == M_PLAYING) {
        s_talk_paused = true;
        music_pause();
    }
}

void music_talk_end(void)
{
    player_set_duck(false);
    if (s_talk_paused) {
        s_talk_paused = false;
        music_resume();
    }
}

void music_state_header(char *buf, size_t len)
{
    mstate_t st = s_state;
    if (st == M_STOPPED) {
        snprintf(buf, len, "state=stopped;vol=%d", player_volume());
        return;
    }
    const char *name = (st == M_PLAYING || s_talk_paused) ? "playing" : "paused";
    snprintf(buf, len, "state=%s;track=%d;pos=%.1f;vol=%d", name, s_track, (double)position(), player_volume());
}
