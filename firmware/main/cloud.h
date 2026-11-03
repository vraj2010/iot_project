// Talks to Friday's server: friday-local on the laptop (plain HTTP on the home
// Wi-Fi) when it can be found, otherwise the Cloudflare Worker over HTTPS.
// Both speak the same protocol, so the rest of the firmware doesn't care which.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

enum { CLOUD_BRAIN_NONE = 0, CLOUD_BRAIN_LAPTOP = 1, CLOUD_BRAIN_WORKER = 2 };

typedef struct {
    int    http_status;    // 200 on success; 401, 429, 502... from the server; 0 = never connected
    int    worker_ms;      // /chat: time the Worker spent (mostly Gemini)
    int    chat_ms;        // /say: Gemini thinking time
    int    tts_ms;         // /say: Gemini voice time
    int    total_ms;       // full round trip measured on the S3, including TLS and playback
    size_t audio_samples;  // /say: samples handed to on_audio (16 kHz)
    int    provider;       // /say: 1 MeloTTS, 2 Aura, 3 Gemini, 4 Piper, 5 Windows voice, 6 Edge voice
    int    format;         // /say: 1 PCM, 2 MP3
    int    stt_ms;         // /ask: time Whisper spent hearing the question
    int    brain;          // CLOUD_BRAIN_LAPTOP or CLOUD_BRAIN_WORKER: who answered
} cloud_result_t;

// Call once after Wi-Fi is up: looks for the laptop (up to 2 s) and keeps
// looking in the background while it's away.
void cloud_start(void);

const char *cloud_provider_name(int provider);
const char *cloud_brain_name(int brain);

// "http://192.168.1.5:8765" when the laptop is known (for music). False if it isn't.
bool cloud_laptop_base(char *base, size_t len);

// ---------------------------------------------------------------- text only

// Send `text` to POST /chat. On success `out` holds Gemini's reply and the
// function returns ESP_OK. On failure `out` holds a short reason to print.
// Reuses one HTTPS connection between calls, so only the first is slow.
esp_err_t cloud_chat(const char *text, char *out, size_t out_len, cloud_result_t *res);

// ---------------------------------------------------------------- text + voice

typedef struct {
    // /ask only: what Whisper heard. "" means nothing usable (silence or noise).
    void (*on_heard)(const char *heard);
    // Called once with Gemini's reply, before any audio. Also called when the
    // voice failed but the text arrived, so the reply is never lost.
    void (*on_text)(const char *reply);
    // Called repeatedly with 16 kHz mono PCM as it downloads. Play it straight away.
    void (*on_audio)(const int16_t *pcm, size_t samples);
    // Optional, for music. on_action gets what the reply asks the speaker to do
    // (JSON, e.g. {"music":"play","track":12}), before any audio. music_state
    // fills the X-Music-State header that tells the laptop what's playing; with
    // it set, the laptop offers Friday its music tools.
    void (*on_action)(const char *json);
    void (*music_state)(char *buf, size_t len);
} cloud_say_cb_t;

// Send `text` to POST /say: Gemini answers and the answer is spoken.
// Returns ESP_OK when text and audio both arrived. On failure `err` holds a
// short reason to print (on_text may still have been called).
esp_err_t cloud_say(const char *text, const cloud_say_cb_t *cb,
                    char *err, size_t err_len, cloud_result_t *res);

// ---------------------------------------------------------------- voice in, voice out

// Upload a recorded question (16 kHz mono PCM) to POST /ask: Whisper hears it,
// Gemini answers, the answer is spoken. on_heard reports the transcript first,
// then on_text and on_audio work exactly as for cloud_say().
esp_err_t cloud_ask(const int16_t *pcm, size_t samples, const cloud_say_cb_t *cb,
                    char *err, size_t err_len, cloud_result_t *res);

// ---------------------------------------------------------------- the laptop's mic (LAPTOP_MIC 1)
//
// The laptop listens for the wake word on its own mic; the speaker asks it, with GET /listen, what it heard.

// Wait up to hold_s for the wake word, on a connection of its own (safe beside a turn in another task).
// Returns its wake id (> 0), 0 if nothing happened (ask again), -1 if the laptop can't be asked (wait a bit).
int cloud_listen_wait(int hold_s);

// After a wake: the laptop records until you stop talking, then answers exactly like cloud_ask (on_heard,
// on_text, on_audio, on_action). ESP_ERR_NOT_FOUND = nothing was said. Call it only while you own the turn,
// like cloud_say and cloud_ask: the three share one connection.
esp_err_t cloud_listen_answer(int wake_id, const cloud_say_cb_t *cb,
                              char *err, size_t err_len, cloud_result_t *res);

// BOOT pressed: ask the laptop to listen now, as if it had heard the wake word.
// Returns the HTTP status: 202 listening, 409 busy answering, 0 laptop not reachable.
int cloud_listen_press(void);
