#include "sam_tts.h"

#include <ctype.h>
#include <stdbool.h>
#include <string.h>

#include "esp_log.h"
#include "reciter.h"
#include "sam.h"

#include "config.h"
#include "resample.h"

static const char *TAG = "sam";

#define SAM_RATE     22050    // SAM renders 8-bit unsigned audio at 22.05 kHz
#define PHRASE_MAX   60       // characters per phrase: spelled-out letters expand ~3x into SAM's 256-byte buffer
#define WORD_MAX     16       // longer words are split; SAM hangs on very long letter runs
#define GAP_MS       120      // pause between phrases
#define SAM_BUF_BYTES (22050 * 10)   // SAM's output buffer (sam.c Init)
#define BLOCK        256

// Keep letters, digits and the punctuation SAM uses for pacing; everything
// else becomes a space. Uppercase, because the reciter's rules expect it.
static void clean(const char *in, char *out, size_t cap)
{
    size_t n = 0;
    size_t word = 0;
    bool space = true;
    for (; *in && n + 2 < cap; in++) {
        unsigned char c = (unsigned char)*in;
        if (isalnum(c) || c == '.' || c == ',' || c == '?' || c == '!' || c == '\'') {
            if (word == WORD_MAX) {          // break up absurdly long words
                out[n++] = ' ';
                word = 0;
            }
            out[n++] = (char)toupper(c);
            word++;
            space = false;
        } else if (!space) {
            out[n++] = ' ';
            word = 0;
            space = true;
        }
    }
    while (n > 0 && out[n - 1] == ' ') {
        n--;
    }
    out[n] = '\0';
}

// Length of the next phrase: up to PHRASE_MAX, ending after punctuation or at a space.
static size_t next_phrase(const char *s)
{
    size_t len = strlen(s);
    if (len <= PHRASE_MAX) {
        return len;
    }
    size_t cut = 0;
    for (size_t i = 0; i < PHRASE_MAX; i++) {
        if (s[i] == '.' || s[i] == ',' || s[i] == '?' || s[i] == '!') {
            cut = i + 1;
        } else if (s[i] == ' ' && cut == 0) {
            cut = i;
        }
    }
    if (cut == 0) {
        for (size_t i = PHRASE_MAX; i > 0; i--) {
            if (s[i] == ' ') { cut = i; break; }
        }
    }
    return cut ? cut : PHRASE_MAX;
}

static void emit(resampler_t *rs, const int16_t *pcm, size_t n, sam_out_t out)
{
    static int16_t res[BLOCK * SAMPLE_RATE_HZ / 16000 + 8];
    size_t got = rs_process(rs, pcm, n, res, sizeof(res) / sizeof(res[0]));
    if (got > 0 && out) {
        out(res, got);
    }
}

float sam_say(const char *text, sam_out_t out)
{
    static char cleaned[512];
    static char input[256];
    static int16_t block[BLOCK];
    resampler_t rs;
    size_t total = 0;

    clean(text, cleaned, sizeof(cleaned));
    rs_init(&rs, SAM_RATE);

    const char *p = cleaned;
    while (*p) {
        while (*p == ' ') p++;
        if (!*p) break;

        size_t len = next_phrase(p);
        memset(input, 0, sizeof(input));
        memcpy(input, p, len);
        p += len;

        strncat(input, "[", sizeof(input) - strlen(input) - 1);    // reciter end marker
        if (!TextToPhonemes((unsigned char *)input)) {
            ESP_LOGW(TAG, "couldn't pronounce a phrase, skipping it");
            continue;
        }
        SetInput(input);
        if (!SAMMain()) {
            ESP_LOGW(TAG, "render failed, skipping a phrase");
            continue;
        }

        const unsigned char *buf = (const unsigned char *)GetBuffer();
        size_t samples = (size_t)GetBufferLength() / 50;          // SAM's length is in 1/50 samples
        if (samples > SAM_BUF_BYTES - 5) {
            samples = SAM_BUF_BYTES - 5;                           // phrase longer than SAM's 10 s buffer
        }
        for (size_t i = 0; i < samples; i += BLOCK) {
            size_t n = samples - i < BLOCK ? samples - i : BLOCK;
            for (size_t k = 0; k < n; k++) {
                block[k] = (int16_t)(((int)buf[i + k] - 128) * 200);  // 8-bit -> 16-bit, a little headroom
            }
            emit(&rs, block, n, out);
        }
        total += samples;

        memset(block, 0, sizeof(block));                           // short pause between phrases
        for (size_t gap = SAM_RATE * GAP_MS / 1000; gap > 0; ) {
            size_t n = gap < BLOCK ? gap : BLOCK;
            emit(&rs, block, n, out);
            gap -= n;
        }
    }
    return (float)total / SAM_RATE;
}
