#include "cloud.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "cJSON.h"
#include "esp_crt_bundle.h"
#include "esp_heap_caps.h"
#include "esp_http_client.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "config.h"
#include "mp3stream.h"
#include "net.h"
#include "resample.h"

static const char *TAG = "cloud";

#define BODY_MAX      4096   // JSON replies (and error bodies) fit easily
#define REPLY_MAX     1024   // longest reply text kept from a /say frame
#define HEAD_V1       20     // "I2S1" rate chat_ms tts_ms text_len
#define HEAD_V2       28     // "I2S2" format rate provider chat_ms tts_ms text_len
#define HEAD_V3       36     // "I2S3" format rate provider chat_ms tts_ms stt_ms heard_len text_len
#define HEAD_V4       40     // "I2S4" ...the same... action_len, then heard, text, action (JSON), audio
#define HEARD_MAX     512    // longest transcript kept
#define ACTION_MAX    512    // longest action JSON kept
#define PCM_CHUNK     1024   // samples handed to on_audio at a time
#define FMT_PCM       1
#define FMT_MP3       2

static esp_http_client_handle_t s_client;   // kept open so later calls skip the TLS handshake
static char s_base[80];                     // the server s_client talks to (laptop or Worker)
static int  s_used_brain;                   // which one answered the last request
static bool s_asking;                       // the request in flight is /ask
static bool s_heap_logged;

/* ------------------------------------------------------------------ */
/*  Response handling                                                  */
/*                                                                     */
/*  /chat and every error: JSON, collected into s_body.                */
/*  /say on 200: a binary frame, parsed as it streams in, so audio     */
/*  reaches the speaker before the whole download has finished.        */
/* ------------------------------------------------------------------ */

typedef enum { RX_JSON, RX_SAY } rx_mode_t;
typedef enum { SAY_HEAD, SAY_HEARD, SAY_TEXT, SAY_ACTION, SAY_AUDIO, SAY_BAD } say_stage_t;

static struct {
    rx_mode_t mode;
    bool      decided;          // first data seen: status known, binary or JSON chosen
    bool      binary;           // /say answered 200 -> parse the frame

    char      body[BODY_MAX];   // JSON path
    int       body_len;
    bool      body_overflow;

    say_stage_t stage;          // /say path
    uint8_t   head[HEAD_V4];
    int       head_len, head_need;
    uint32_t  format, rate, provider, chat_ms, tts_ms, text_len, text_got;
    uint32_t  stt_ms, heard_len, heard_got;
    char      heard[HEARD_MAX];
    uint32_t  action_len, action_got;
    char      action[ACTION_MAX];
    bool      asked;            // /ask (a recording) rather than /say (typed)
    bool      resampling;       // PCM that isn't already 16 kHz
    resampler_t rs;
    int16_t   rs_out[PCM_CHUNK * 4];
    char      text[REPLY_MAX];
    bool      have_carry;       // odd byte left over between network chunks
    uint8_t   carry;
    int16_t   pcm[PCM_CHUNK];
    size_t    samples;

    const cloud_say_cb_t *cb;
} rx;

static uint32_t le32(const uint8_t *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static void collect_json(const uint8_t *d, int n)
{
    int room = BODY_MAX - 1 - rx.body_len;
    int take = n < room ? n : room;
    if (take > 0) {
        memcpy(rx.body + rx.body_len, d, take);
        rx.body_len += take;
    }
    if (take < n) {
        rx.body_overflow = true;
    }
}

// Every 16 kHz sample reaches the caller through here.
static void deliver(const int16_t *pcm, size_t n)
{
    rx.samples += n;
    if (rx.cb && rx.cb->on_audio) {
        rx.cb->on_audio(pcm, n);
    }
}

static void deliver_pcm(const int16_t *pcm, size_t n)
{
    if (!rx.resampling) {
        deliver(pcm, n);
        return;
    }
    size_t got = rs_process(&rx.rs, pcm, n, rx.rs_out, sizeof(rx.rs_out) / sizeof(rx.rs_out[0]));
    if (got) {
        deliver(rx.rs_out, got);
    }
}

static void emit_pcm(const uint8_t *d, int n)
{
    while (n > 0) {
        size_t filled = 0;
        if (rx.have_carry) {                       // finish a sample split across chunks
            rx.pcm[filled++] = (int16_t)(rx.carry | (d[0] << 8));
            rx.have_carry = false;
            d++;
            n--;
        }
        int whole = n / 2;
        int room  = PCM_CHUNK - (int)filled;
        int take  = whole < room ? whole : room;
        memcpy(&rx.pcm[filled], d, (size_t)take * 2);   // the S3 is little-endian too
        filled += take;
        d += take * 2;
        n -= take * 2;

        if (n == 1) {                              // one byte left: keep it for next time
            rx.carry      = d[0];
            rx.have_carry = true;
            n = 0;
        }
        if (filled > 0) {
            deliver_pcm(rx.pcm, filled);
        }
    }
}

static void parse_say(const uint8_t *d, int n)
{
    while (n > 0 && rx.stage != SAY_BAD) {
        if (rx.stage == SAY_HEAD) {
            if (rx.head_need == 0) {
                rx.head_need = 4;                      // read the magic first
            }
            int take = rx.head_need - rx.head_len;
            if (take > n) take = n;
            memcpy(rx.head + rx.head_len, d, take);
            rx.head_len += take;
            d += take;
            n -= take;
            if (rx.head_len < rx.head_need) {
                return;
            }
            if (rx.head_need == 4) {                   // magic known: how long is the header?
                if (memcmp(rx.head, "I2S4", 4) == 0) {
                    rx.head_need = HEAD_V4;
                } else if (memcmp(rx.head, "I2S3", 4) == 0) {
                    rx.head_need = HEAD_V3;
                } else if (memcmp(rx.head, "I2S2", 4) == 0) {
                    rx.head_need = HEAD_V2;
                } else if (memcmp(rx.head, "I2S1", 4) == 0) {
                    rx.head_need = HEAD_V1;
                } else {
                    ESP_LOGE(TAG, "reply is not an I2S frame");
                    rx.stage = SAY_BAD;
                    return;
                }
                continue;
            }
            if (rx.head_need == HEAD_V3 || rx.head_need == HEAD_V4) {
                rx.format    = le32(rx.head + 4);
                rx.rate      = le32(rx.head + 8);
                rx.provider  = le32(rx.head + 12);
                rx.chat_ms   = le32(rx.head + 16);
                rx.tts_ms    = le32(rx.head + 20);
                rx.stt_ms    = le32(rx.head + 24);
                rx.heard_len = le32(rx.head + 28);
                rx.text_len  = le32(rx.head + 32);
                if (rx.head_need == HEAD_V4) {
                    rx.action_len = le32(rx.head + 36);
                }
            } else if (rx.head_need == HEAD_V2) {
                rx.format   = le32(rx.head + 4);
                rx.rate     = le32(rx.head + 8);
                rx.provider = le32(rx.head + 12);
                rx.chat_ms  = le32(rx.head + 16);
                rx.tts_ms   = le32(rx.head + 20);
                rx.text_len = le32(rx.head + 24);
            } else {
                rx.format   = FMT_PCM;
                rx.rate     = le32(rx.head + 4);
                rx.chat_ms  = le32(rx.head + 8);
                rx.tts_ms   = le32(rx.head + 12);
                rx.text_len = le32(rx.head + 16);
            }
            if (rx.format == FMT_MP3) {
                mp3s_begin(deliver);
            } else if (rx.format == FMT_PCM) {
                rx.resampling = rx.rate != SAMPLE_RATE_HZ && rx.rate >= 8000 && rx.rate <= 48000;
                if (rx.resampling) {
                    rs_init(&rx.rs, rx.rate);
                }
            } else {
                ESP_LOGE(TAG, "unknown audio format %u", (unsigned)rx.format);
                rx.stage = SAY_BAD;
                return;
            }
            if (rx.head_need == HEAD_V3 || rx.head_need == HEAD_V4) {
                rx.stage = SAY_HEARD;
                if (rx.heard_len == 0) {               // nothing usable was heard (or a typed question)
                    if (rx.asked && rx.cb && rx.cb->on_heard) rx.cb->on_heard("");
                    rx.stage = SAY_TEXT;
                }
            } else {
                rx.stage = SAY_TEXT;
            }
            if (rx.stage == SAY_TEXT && rx.text_len == 0) {
                if (rx.cb && rx.cb->on_text) rx.cb->on_text("");
                rx.stage = SAY_ACTION;
            }
            if (rx.stage == SAY_ACTION && rx.action_len == 0) {
                rx.stage = SAY_AUDIO;
            }
        } else if (rx.stage == SAY_HEARD) {
            uint32_t left = rx.heard_len - rx.heard_got;
            int take = (uint32_t)n < left ? n : (int)left;
            for (int i = 0; i < take; i++) {           // keep what fits, skip the rest
                if (rx.heard_got + i < HEARD_MAX - 1) {
                    rx.heard[rx.heard_got + i] = (char)d[i];
                }
            }
            rx.heard_got += take;
            d += take;
            n -= take;
            if (rx.heard_got == rx.heard_len) {
                rx.heard[rx.heard_len < HEARD_MAX - 1 ? rx.heard_len : HEARD_MAX - 1] = '\0';
                if (rx.cb && rx.cb->on_heard) rx.cb->on_heard(rx.heard);
                rx.stage = SAY_TEXT;
                if (rx.text_len == 0) {
                    if (rx.cb && rx.cb->on_text) rx.cb->on_text("");
                    rx.stage = rx.action_len ? SAY_ACTION : SAY_AUDIO;
                }
            }
        } else if (rx.stage == SAY_TEXT) {
            uint32_t left = rx.text_len - rx.text_got;
            int take = (uint32_t)n < left ? n : (int)left;
            for (int i = 0; i < take; i++) {           // keep what fits, skip the rest
                if (rx.text_got + i < REPLY_MAX - 1) {
                    rx.text[rx.text_got + i] = (char)d[i];
                }
            }
            rx.text_got += take;
            d += take;
            n -= take;
            if (rx.text_got == rx.text_len) {
                rx.text[rx.text_len < REPLY_MAX - 1 ? rx.text_len : REPLY_MAX - 1] = '\0';
                if (rx.cb && rx.cb->on_text) {
                    rx.cb->on_text(rx.text);
                }
                rx.stage = rx.action_len ? SAY_ACTION : SAY_AUDIO;
            }
        } else if (rx.stage == SAY_ACTION) {
            uint32_t left = rx.action_len - rx.action_got;
            int take = (uint32_t)n < left ? n : (int)left;
            for (int i = 0; i < take; i++) {
                if (rx.action_got + i < ACTION_MAX - 1) {
                    rx.action[rx.action_got + i] = (char)d[i];
                }
            }
            rx.action_got += take;
            d += take;
            n -= take;
            if (rx.action_got == rx.action_len) {
                rx.action[rx.action_len < ACTION_MAX - 1 ? rx.action_len : ACTION_MAX - 1] = '\0';
                if (rx.action_len > 2 && rx.cb && rx.cb->on_action) {   // "{}" = nothing to do
                    rx.cb->on_action(rx.action);
                }
                rx.stage = SAY_AUDIO;
            }
        } else {                                       // SAY_AUDIO
            if (rx.format == FMT_MP3) {
                mp3s_feed(d, (size_t)n);
            } else {
                emit_pcm(d, n);
            }
            return;
        }
    }
}

static esp_err_t on_http_event(esp_http_client_event_t *e)
{
    if (e->event_id != HTTP_EVENT_ON_DATA || e->data_len <= 0) {
        return ESP_OK;
    }
    if (!rx.decided) {
        rx.decided = true;
        rx.binary  = rx.mode == RX_SAY && esp_http_client_get_status_code(e->client) == 200;
    }
    if (rx.binary) {
        parse_say(e->data, e->data_len);
    } else {
        collect_json(e->data, e->data_len);
    }
    return ESP_OK;
}

/* ------------------------------------------------------------------ */
/*  Choosing the brain: the laptop (friday-local) when it's there,     */
/*  otherwise the Cloudflare Worker.                                   */
/* ------------------------------------------------------------------ */

static SemaphoreHandle_t s_brain_mu;
static char s_laptop_ip[16];        // "" = laptop not found (yet)
static int  s_laptop_port;
static int  s_announced;            // last brain printed, so changes are shown once

static void laptop_set(const char *ip, int port)
{
    xSemaphoreTake(s_brain_mu, portMAX_DELAY);
    bool changed = strcmp(s_laptop_ip, ip) != 0 || s_laptop_port != port;
    strncpy(s_laptop_ip, ip, sizeof(s_laptop_ip) - 1);
    s_laptop_ip[sizeof(s_laptop_ip) - 1] = '\0';
    s_laptop_port = port;
    xSemaphoreGive(s_brain_mu);
    if (changed) {
        printf("\n[brain] found the laptop at %s:%d\n", ip, port);
    }
}

static void laptop_lost(const char *why)
{
    xSemaphoreTake(s_brain_mu, portMAX_DELAY);
    bool had = s_laptop_ip[0] != '\0';
    s_laptop_ip[0] = '\0';
    xSemaphoreGive(s_brain_mu);
    if (had) {
        printf("[brain] laptop %s\n", why);
    }
}

static bool laptop_get(char *ip, size_t ip_len, int *port)
{
    xSemaphoreTake(s_brain_mu, portMAX_DELAY);
    bool known = s_laptop_ip[0] != '\0';
    if (known) {
        strncpy(ip, s_laptop_ip, ip_len - 1);
        ip[ip_len - 1] = '\0';
        *port = s_laptop_port;
    }
    xSemaphoreGive(s_brain_mu);
    return known;
}

bool cloud_laptop_base(char *base, size_t len)
{
    char ip[16];
    int  port;
    if (!USE_LAPTOP || !s_brain_mu || !laptop_get(ip, sizeof(ip), &port)) {
        return false;
    }
    snprintf(base, len, "http://%s:%d", ip, port);
    return true;
}

static void look_for_laptop(int timeout_ms)
{
    if (LAPTOP_IP_FIXED[0] != '\0') {
        laptop_set(LAPTOP_IP_FIXED, LAPTOP_HTTP_PORT);   // checked before each question anyway
        return;
    }
    char ip[16];
    int  port = 0;
    if (net_find_laptop(timeout_ms, ip, sizeof(ip), &port) == ESP_OK) {
        laptop_set(ip, port);
    }
}

// Keeps looking while the laptop is away, so it's picked up again when it wakes.
static void finder_task(void *arg)
{
    char ip[16];
    int  port;
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(LAPTOP_RETRY_SEC * 1000));
        if (net_is_online() && !laptop_get(ip, sizeof(ip), &port)) {
            look_for_laptop(1500);
        }
    }
}

void cloud_start(void)
{
    s_brain_mu = xSemaphoreCreateMutex();
    if (!USE_LAPTOP) {
        return;
    }
    if (net_is_online()) {
        look_for_laptop(2000);          // once at boot, so the first question can use it
    }
    xTaskCreate(finder_task, "laptop-finder", 4096, NULL, 3, NULL);
}

const char *cloud_brain_name(int brain)
{
    switch (brain) {
    case CLOUD_BRAIN_LAPTOP: return "laptop";
    case CLOUD_BRAIN_WORKER: return "cloud";
    default:                 return "none";
    }
}

// Pick where this question goes and fill in its base URL.
static int pick_brain(char *base, size_t len)
{
    char ip[16];
    int  port;
    int  brain = CLOUD_BRAIN_NONE;
    if (USE_LAPTOP && s_brain_mu && laptop_get(ip, sizeof(ip), &port)) {
        if (net_can_connect(ip, port, LAPTOP_PROBE_MS)) {
            snprintf(base, len, "http://%s:%d", ip, port);
            brain = CLOUD_BRAIN_LAPTOP;
        } else {
            laptop_lost("is not answering (asleep, off, or friday-local not running)");
        }
    }
    if (brain == CLOUD_BRAIN_NONE && (USE_WORKER_FALLBACK || !USE_LAPTOP)) {
        snprintf(base, len, "%s", WORKER_URL);
        brain = CLOUD_BRAIN_WORKER;
    }
    if (brain != s_announced) {
        s_announced = brain;
        if (brain == CLOUD_BRAIN_LAPTOP) {
            printf("[brain] using the laptop (memory on)\n");
        } else if (brain == CLOUD_BRAIN_WORKER) {
            printf("[brain] using the cloud Worker%s\n", USE_LAPTOP ? " (laptop not found, no memory)" : "");
        }
    }
    return brain;
}

/* ------------------------------------------------------------------ */
/*  Connection                                                         */
/* ------------------------------------------------------------------ */

static void drop_client(void)
{
    if (s_client) {
        esp_http_client_cleanup(s_client);
        s_client = NULL;
    }
}

static esp_err_t ensure_client(void)
{
    if (s_client) {
        return ESP_OK;
    }

    char url[120];
    snprintf(url, sizeof(url), "%s/chat", s_base);
    esp_http_client_config_t cfg = {
        .url               = url,
        .method            = HTTP_METHOD_POST,
        .timeout_ms        = CLOUD_TIMEOUT_MS,
        .crt_bundle_attach = esp_crt_bundle_attach,   // trusts Cloudflare's certificate (unused on plain http)
        .event_handler     = on_http_event,
        .buffer_size       = 4096,
        .buffer_size_tx    = 1024,
    };
    s_client = esp_http_client_init(&cfg);
    if (!s_client) {
        return ESP_ERR_NO_MEM;
    }

    esp_http_client_set_header(s_client, "Content-Type", "application/json");
    esp_http_client_set_header(s_client, "X-Device-Token", DEVICE_TOKEN);
    esp_http_client_set_header(s_client, "User-Agent", "i2sonic-s3/0.4");
    return ESP_OK;
}

// Switching between laptop and Worker means a different host: start a new connection.
static void use_base(const char *base)
{
    if (strcmp(s_base, base) != 0) {
        drop_client();
        strncpy(s_base, base, sizeof(s_base) - 1);
        s_base[sizeof(s_base) - 1] = '\0';
    }
}

static bool nothing_arrived(void)
{
    return rx.samples == 0 && rx.text_got == 0 && rx.heard_got == 0 && rx.action_got == 0 && rx.body_len == 0;
}

static void rx_reset(rx_mode_t mode, const cloud_say_cb_t *cb)
{
    memset(&rx, 0, sizeof(rx));
    rx.mode  = mode;
    rx.stage = SAY_HEAD;
    rx.cb    = cb;
    rx.asked = s_asking;
}

// One POST to `path` (a GET when body is NULL). Returns ESP_OK if an HTTP response (any status) came back.
static esp_err_t post_once(const char *path, const char *body, size_t len, const char *content_type,
                           rx_mode_t mode, const cloud_say_cb_t *cb, int *status)
{
    esp_err_t err = ensure_client();
    if (err != ESP_OK) {
        return err;
    }

    char url[160];
    snprintf(url, sizeof(url), "%s%s", s_base, path);
    esp_http_client_set_url(s_client, url);        // same host: the connection is kept
    if (body) {
        esp_http_client_set_method(s_client, HTTP_METHOD_POST);
        esp_http_client_set_header(s_client, "Content-Type", content_type);
        esp_http_client_set_post_field(s_client, body, (int)len);
    } else {
        esp_http_client_set_method(s_client, HTTP_METHOD_GET);
        esp_http_client_delete_header(s_client, "Content-Type");
        esp_http_client_set_post_field(s_client, NULL, 0);
    }
    if (cb && cb->music_state) {
        char state[96];
        cb->music_state(state, sizeof(state));
        esp_http_client_set_header(s_client, "X-Music-State", state);
    } else {
        esp_http_client_delete_header(s_client, "X-Music-State");
    }

    // The laptop may search the web and think for a while; the Worker is quicker or gone.
    esp_http_client_set_timeout_ms(s_client, s_used_brain == CLOUD_BRAIN_LAPTOP ? LAPTOP_TIMEOUT_MS
                                                                                  : CLOUD_TIMEOUT_MS);
    rx_reset(mode, cb);
    err = esp_http_client_perform(s_client);
    if (err != ESP_OK) {
        drop_client();     // a dead or half-closed connection: start fresh next time
        return err;
    }
    *status = esp_http_client_get_status_code(s_client);
    rx.body[rx.body_len] = '\0';
    return ESP_OK;
}

// POST with one retry on a fresh connection (the kept-open one may have timed
// out). If the laptop stops answering mid-request, ask the Worker instead.
// Never retries once anything has been shown or played.
static esp_err_t post_raw(const char *path, const char *body, size_t len, const char *content_type,
                          rx_mode_t mode, const cloud_say_cb_t *cb, int *status, int *total_ms)
{
    int64_t t0 = esp_timer_get_time();
    char base[80];
    int brain = pick_brain(base, sizeof(base));
    s_used_brain = brain;
    if (brain == CLOUD_BRAIN_NONE) {
        rx_reset(mode, cb);
        *total_ms = 0;
        return ESP_ERR_NOT_FOUND;
    }
    use_base(base);

    esp_err_t err = post_once(path, body, len, content_type, mode, cb, status);
    // A quick failure is a stale kept-open connection: retry the same server.
    // A slow one (timed out) means the server is there but busy: asking again,
    // or asking the Worker, would only make you wait twice as long.
    bool slow = (esp_timer_get_time() - t0) > 5000000;
    if (err != ESP_OK && nothing_arrived() && !slow) {
        ESP_LOGW(TAG, "request failed (%s), retrying on a new connection", esp_err_to_name(err));
        err = post_once(path, body, len, content_type, mode, cb, status);
    }
    if (err != ESP_OK && nothing_arrived() && brain == CLOUD_BRAIN_LAPTOP && slow) {
        printf("[brain] the laptop took more than %d s to answer (see its window)\n", LAPTOP_TIMEOUT_MS / 1000);
    } else if (err != ESP_OK && nothing_arrived() && brain == CLOUD_BRAIN_LAPTOP) {
        laptop_lost("stopped answering");
        if (USE_WORKER_FALLBACK) {
            printf("[brain] asking the cloud Worker instead\n");
            s_used_brain = CLOUD_BRAIN_WORKER;
            s_announced  = CLOUD_BRAIN_WORKER;
            use_base(WORKER_URL);
            err = post_once(path, body, len, content_type, mode, cb, status);
            if (err != ESP_OK && nothing_arrived()) {
                err = post_once(path, body, len, content_type, mode, cb, status);
            }
        }
    }
    *total_ms = (int)((esp_timer_get_time() - t0) / 1000);

    if (err == ESP_OK && !s_heap_logged && s_used_brain == CLOUD_BRAIN_WORKER) {
        s_heap_logged = true;
        ESP_LOGI(TAG, "TLS up. free internal RAM %u bytes, PSRAM %u bytes",
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    }
    return err;
}

// JSON {"text": "..."} POST, for /chat and /say.
static esp_err_t post(const char *path, const char *text, rx_mode_t mode,
                      const cloud_say_cb_t *cb, int *status, int *total_ms)
{
    cJSON *req = cJSON_CreateObject();
    cJSON_AddStringToObject(req, "text", text);      // escapes quotes and newlines
    char *json = cJSON_PrintUnformatted(req);
    cJSON_Delete(req);
    if (!json) {
        return ESP_ERR_NO_MEM;
    }
    esp_err_t err = post_raw(path, json, strlen(json), "application/json", mode, cb, status, total_ms);
    cJSON_free(json);
    return err;
}

/* ------------------------------------------------------------------ */

static void copy_out(char *out, size_t out_len, const char *src)
{
    if (!out || out_len == 0) {
        return;
    }
    strncpy(out, src ? src : "", out_len - 1);
    out[out_len - 1] = '\0';
}

static bool token_missing(char *out, size_t out_len)
{
    if (strstr(DEVICE_TOKEN, "PASTE-YOUR") != NULL) {
        copy_out(out, out_len, "DEVICE_TOKEN in secrets.h is still the placeholder");
        return true;
    }
    return false;
}

static void unreachable(char *out, size_t out_len, esp_err_t err)
{
    char why[112];
    if (err == ESP_ERR_NOT_FOUND && s_used_brain == CLOUD_BRAIN_NONE) {
        snprintf(why, sizeof(why), "laptop not found and USE_WORKER_FALLBACK is 0");
    } else {
        snprintf(why, sizeof(why), "could not reach the %s (%s)",
                 s_used_brain == CLOUD_BRAIN_LAPTOP ? "laptop" : "Worker", esp_err_to_name(err));
    }
    copy_out(out, out_len, why);
}

esp_err_t cloud_chat(const char *text, char *out, size_t out_len, cloud_result_t *res)
{
    cloud_result_t r = {0};
    copy_out(out, out_len, "");
    if (token_missing(out, out_len)) {
        if (res) *res = r;
        return ESP_ERR_INVALID_STATE;
    }

    int status = 0;
    esp_err_t err = post("/chat", text, RX_JSON, NULL, &status, &r.total_ms);
    r.brain = s_used_brain;
    if (err != ESP_OK) {
        unreachable(out, out_len, err);
        if (res) *res = r;
        return err;
    }
    r.http_status = status;
    if (rx.body_overflow) {
        ESP_LOGW(TAG, "reply longer than %d bytes was cut", BODY_MAX - 1);
    }

    // Worker always answers JSON: {"reply": "...", "ms": n} or {"error": "...", "ms": n}
    cJSON *doc = cJSON_Parse(rx.body);
    if (!doc) {
        char why[96];
        snprintf(why, sizeof(why), "HTTP %d with a non-JSON body", status);
        copy_out(out, out_len, why);
        if (res) *res = r;
        return ESP_FAIL;
    }

    const cJSON *ms    = cJSON_GetObjectItemCaseSensitive(doc, "ms");
    const cJSON *reply = cJSON_GetObjectItemCaseSensitive(doc, "reply");
    const cJSON *error = cJSON_GetObjectItemCaseSensitive(doc, "error");
    if (cJSON_IsNumber(ms)) {
        r.worker_ms = ms->valueint;
    }

    esp_err_t result;
    if (status == 200 && cJSON_IsString(reply)) {
        copy_out(out, out_len, reply->valuestring);
        result = ESP_OK;
    } else {
        copy_out(out, out_len, cJSON_IsString(error) ? error->valuestring : "unexpected reply");
        result = ESP_FAIL;
    }

    cJSON_Delete(doc);
    if (res) *res = r;
    return result;
}

// Shared by /say and /ask once the response is in: flush the MP3 decoder,
// fill in the result, and turn JSON error replies into a reason to print
// (passing along any "heard" / "reply" text they still carry).
static esp_err_t finish_reply(esp_err_t err, int status, const cloud_say_cb_t *cb,
                              char *err_out, size_t err_len, cloud_result_t *r)
{
    if (rx.binary && rx.stage == SAY_AUDIO && rx.format == FMT_MP3) {
        mp3s_end();                                    // decode the last frames, even after a drop
    }
    r->audio_samples = rx.samples;
    r->chat_ms       = (int)rx.chat_ms;
    r->tts_ms        = (int)rx.tts_ms;
    r->stt_ms        = (int)rx.stt_ms;
    r->provider      = (int)rx.provider;
    r->format        = (int)rx.format;

    if (err != ESP_OK) {
        if (rx.samples > 0) {
            copy_out(err_out, err_len, "connection dropped mid-reply");
        } else {
            unreachable(err_out, err_len, err);
        }
        return err;
    }
    r->http_status = status;

    if (rx.binary) {
        if (rx.stage == SAY_BAD) {
            copy_out(err_out, err_len, "the server sent an unreadable reply");
            return ESP_FAIL;
        }
        if (rx.stage != SAY_AUDIO || rx.samples == 0) {
            copy_out(err_out, err_len, "reply ended before any audio");
            return ESP_FAIL;
        }
        return ESP_OK;
    }

    // Not 200: the Worker sent JSON. It may still carry what was heard and the reply.
    cJSON *doc = cJSON_Parse(rx.body);
    const cJSON *heard = doc ? cJSON_GetObjectItemCaseSensitive(doc, "heard") : NULL;
    const cJSON *reply = doc ? cJSON_GetObjectItemCaseSensitive(doc, "reply") : NULL;
    const cJSON *error = doc ? cJSON_GetObjectItemCaseSensitive(doc, "error") : NULL;
    if (cJSON_IsString(heard) && cb && cb->on_heard) {
        cb->on_heard(heard->valuestring);
    }
    if (cJSON_IsString(reply) && cb && cb->on_text) {
        cb->on_text(reply->valuestring);
    }
    const cJSON *action = doc ? cJSON_GetObjectItemCaseSensitive(doc, "action") : NULL;
    if (cJSON_IsObject(action) && action->child && cb && cb->on_action) {
        char *json = cJSON_PrintUnformatted(action);   // the voice failed, the music still happens
        if (json) {
            cb->on_action(json);
            cJSON_free(json);
        }
    }
    if (cJSON_IsString(error)) {
        copy_out(err_out, err_len, error->valuestring);
    } else {
        char why[64];
        snprintf(why, sizeof(why), "HTTP %d", status);
        copy_out(err_out, err_len, why);
    }
    cJSON_Delete(doc);
    return ESP_FAIL;
}

esp_err_t cloud_say(const char *text, const cloud_say_cb_t *cb,
                    char *err_out, size_t err_len, cloud_result_t *res)
{
    cloud_result_t r = {0};
    copy_out(err_out, err_len, "");
    if (token_missing(err_out, err_len)) {
        if (res) *res = r;
        return ESP_ERR_INVALID_STATE;
    }

    int status = 0;
    s_asking = false;
    esp_err_t err = post("/say", text, RX_SAY, cb, &status, &r.total_ms);
    r.brain = s_used_brain;
    err = finish_reply(err, status, cb, err_out, err_len, &r);
    if (res) *res = r;
    return err;
}

esp_err_t cloud_ask(const int16_t *pcm, size_t samples, const cloud_say_cb_t *cb,
                    char *err_out, size_t err_len, cloud_result_t *res)
{
    cloud_result_t r = {0};
    copy_out(err_out, err_len, "");
    if (token_missing(err_out, err_len)) {
        if (res) *res = r;
        return ESP_ERR_INVALID_STATE;
    }

    // The recording goes up as-is: raw 16 kHz s16le mono (the S3 is little-endian).
    int status = 0;
    s_asking = true;
    esp_err_t err = post_raw("/ask", (const char *)pcm, samples * sizeof(int16_t),
                             "application/octet-stream", RX_SAY, cb, &status, &r.total_ms);
    r.brain = s_used_brain;
    err = finish_reply(err, status, cb, err_out, err_len, &r);
    if (res) *res = r;
    return err;
}

/* ------------------------------------------------------------------ */
/*  The laptop's mic: GET /listen                                      */
/*                                                                     */
/*  The idle poll has its own connection and buffer: it waits up to    */
/*  LISTEN_HOLD_SEC while a typed question may use s_client and rx.    */
/*  The answer (/listen?wake=n) comes through s_client and rx, like    */
/*  /ask, by whoever owns the turn.                                    */
/* ------------------------------------------------------------------ */

#define LISTEN_BODY_MAX 256

static esp_http_client_handle_t s_listen;   // the idle poll's connection, kept open between polls
static char s_listen_base[40];
static char s_listen_body[LISTEN_BODY_MAX];
static int  s_listen_len;
static int  s_listen_said;                  // last odd status printed, so it's said once

static esp_err_t on_listen_event(esp_http_client_event_t *e)
{
    if (e->event_id == HTTP_EVENT_ON_DATA && e->data_len > 0) {
        int room = LISTEN_BODY_MAX - 1 - s_listen_len;
        int take = e->data_len < room ? e->data_len : room;
        if (take > 0) {
            memcpy(s_listen_body + s_listen_len, e->data, take);
            s_listen_len += take;
        }
    }
    return ESP_OK;
}

static void listen_drop(void)
{
    if (s_listen) {
        esp_http_client_cleanup(s_listen);
        s_listen = NULL;
    }
}

int cloud_listen_wait(int hold_s)
{
    char ip[16];
    int  port;
    if (!USE_LAPTOP || !s_brain_mu || !laptop_get(ip, sizeof(ip), &port)) {
        return -1;
    }
    char base[40];
    snprintf(base, sizeof(base), "http://%s:%d", ip, port);
    if (s_listen && strcmp(base, s_listen_base) != 0) {
        listen_drop();                              // the laptop moved: new connection
    }
    if (!s_listen) {
        char url[64];
        snprintf(url, sizeof(url), "%s/listen", base);
        esp_http_client_config_t cfg = {
            .url            = url,
            .method         = HTTP_METHOD_GET,
            .timeout_ms     = (hold_s + 10) * 1000,   // the laptop answers within hold_s
            .event_handler  = on_listen_event,
            .buffer_size    = 1024,
            .buffer_size_tx = 512,
        };
        s_listen = esp_http_client_init(&cfg);
        if (!s_listen) {
            return -1;
        }
        esp_http_client_set_header(s_listen, "X-Device-Token", DEVICE_TOKEN);
        esp_http_client_set_header(s_listen, "User-Agent", "i2sonic-s3/0.5");
        strncpy(s_listen_base, base, sizeof(s_listen_base) - 1);
        s_listen_base[sizeof(s_listen_base) - 1] = '\0';
    }

    s_listen_len = 0;
    int64_t t0 = esp_timer_get_time();
    esp_err_t err = esp_http_client_perform(s_listen);
    if (err != ESP_OK) {
        listen_drop();
        // A quick failure is usually a stale kept-open connection; a laptop that has gone away doesn't
        // answer the probe either, and is looked for again.
        if (esp_timer_get_time() - t0 < 5000000 && !net_can_connect(ip, port, LAPTOP_PROBE_MS)) {
            laptop_lost("is not answering (asleep, off, or friday-local not running)");
        }
        return -1;
    }
    int status = esp_http_client_get_status_code(s_listen);
    s_listen_body[s_listen_len] = '\0';
    if (status == 204) {
        s_listen_said = 0;
        return 0;
    }
    if (status == 202) {
        s_listen_said = 0;
        cJSON *doc = cJSON_Parse(s_listen_body);
        const cJSON *ev = doc ? cJSON_GetObjectItemCaseSensitive(doc, "event") : NULL;
        const cJSON *id = doc ? cJSON_GetObjectItemCaseSensitive(doc, "id") : NULL;
        int wake = cJSON_IsString(ev) && strcmp(ev->valuestring, "wake") == 0 && cJSON_IsNumber(id) ? id->valueint : 0;
        cJSON_Delete(doc);
        return wake;
    }
    if (status != s_listen_said) {
        s_listen_said = status;
        printf("[ears] the laptop answered /listen with HTTP %d%s\n", status,
               status == 401 ? ": DEVICE_TOKEN in secrets.h doesn't match friday-local's .env" :
               status == 503 ? ": its mic is off (EARS in friday-local's .env, or see its window)" :
               status == 404 ? ": friday-local is older than this firmware" : "");
    }
    return -1;
}

esp_err_t cloud_listen_answer(int wake_id, const cloud_say_cb_t *cb,
                              char *err_out, size_t err_len, cloud_result_t *res)
{
    cloud_result_t r = {0};
    copy_out(err_out, err_len, "");
    char ip[16];
    int  port;
    if (!USE_LAPTOP || !s_brain_mu || !laptop_get(ip, sizeof(ip), &port)) {
        copy_out(err_out, err_len, "the laptop isn't connected");
        if (res) *res = r;
        return ESP_ERR_INVALID_STATE;
    }
    char base[80], path[32];
    snprintf(base, sizeof(base), "http://%s:%d", ip, port);
    snprintf(path, sizeof(path), "/listen?wake=%d", wake_id);
    use_base(base);
    s_used_brain = CLOUD_BRAIN_LAPTOP;
    s_asking     = true;                            // the reply carries what was heard, like /ask

    int64_t t0 = esp_timer_get_time();
    int status = 0;
    esp_err_t err = post_once(path, NULL, 0, NULL, RX_SAY, cb, &status);
    if (err != ESP_OK && nothing_arrived() && esp_timer_get_time() - t0 < 5000000) {
        err = post_once(path, NULL, 0, NULL, RX_SAY, cb, &status);   // a stale kept-open connection
    }
    r.total_ms = (int)((esp_timer_get_time() - t0) / 1000);
    r.brain    = CLOUD_BRAIN_LAPTOP;
    if (err == ESP_OK && status == 202) {           // {"event": "cancel"}: nothing was said
        if (res) *res = r;
        return ESP_ERR_NOT_FOUND;
    }
    err = finish_reply(err, status, cb, err_out, err_len, &r);
    if (res) *res = r;
    return err;
}

int cloud_listen_press(void)
{
    char ip[16];
    int  port;
    if (!USE_LAPTOP || !s_brain_mu || !laptop_get(ip, sizeof(ip), &port)) {
        return 0;
    }
    char url[64];
    snprintf(url, sizeof(url), "http://%s:%d/listen/wake", ip, port);
    esp_http_client_config_t cfg = {
        .url        = url,
        .method     = HTTP_METHOD_POST,
        .timeout_ms = 3000,
    };
    esp_http_client_handle_t c = esp_http_client_init(&cfg);   // its own: the other two may be busy
    if (!c) {
        return 0;
    }
    esp_http_client_set_header(c, "X-Device-Token", DEVICE_TOKEN);
    esp_http_client_set_post_field(c, "", 0);
    int status = esp_http_client_perform(c) == ESP_OK ? esp_http_client_get_status_code(c) : 0;
    esp_http_client_cleanup(c);
    return status;
}

const char *cloud_provider_name(int provider)
{
    switch (provider) {
    case 1:  return "MeloTTS";
    case 2:  return "Aura";
    case 3:  return "Gemini";
    case 4:  return "Piper";
    case 5:  return "Windows voice";
    case 6:  return "Edge voice";
    default: return "voice";
    }
}
