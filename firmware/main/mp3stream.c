#include "mp3stream.h"

#include <stdbool.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "minimp3.h"

#include "config.h"
#include "resample.h"

static const char *TAG = "mp3";

#define IN_BUF      16384     // undecoded MP3 bytes held (in PSRAM)
#define DECODE_AT   4096      // decode once this much is buffered: >2 max-size frames
#define OUT_CAP     (MINIMP3_MAX_SAMPLES_PER_FRAME * 16000 / 8000 + 8)

struct mp3s {
    mp3dec_t     dec;
    uint8_t     *in;
    size_t       len;
    mp3s_out_t   out;
    resampler_t  rs;
    int          rate;
    int          frames;
    int          skipped;
    int16_t      pcm[MINIMP3_MAX_SAMPLES_PER_FRAME];
    int16_t      mono[MINIMP3_MAX_SAMPLES_PER_FRAME / 2];
    int16_t      res[OUT_CAP];
};

static void *psram_or_internal(size_t n)
{
    void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    return p ? p : heap_caps_malloc(n, MALLOC_CAP_8BIT);
}

mp3s_t *mp3s_new(void)
{
    mp3s_t *d = psram_or_internal(sizeof(mp3s_t));
    if (!d) {
        return NULL;
    }
    memset(d, 0, sizeof(*d));
    d->in = psram_or_internal(IN_BUF);
    if (!d->in) {
        heap_caps_free(d);
        return NULL;
    }
    return d;
}

void mp3s_start(mp3s_t *d, mp3s_out_t out)
{
    mp3dec_init(&d->dec);
    d->len     = 0;
    d->out     = out;
    d->rate    = 0;
    d->frames  = 0;
    d->skipped = 0;
}

// Decode as many whole frames as the buffer holds. With `final` set, also
// decode the tail, which may be shorter than DECODE_AT.
static void decode(mp3s_t *d, bool final)
{
    size_t off = 0;

    while (d->len - off > 0 && (final || d->len - off >= DECODE_AT)) {
        mp3dec_frame_info_t info;
        int samples = mp3dec_decode_frame(&d->dec, d->in + off, (int)(d->len - off), d->pcm, &info);

        if (info.frame_bytes == 0) {
            break;                                  // need more data (or the end is junk)
        }
        off += info.frame_bytes;
        if (samples == 0) {
            d->skipped++;                           // ID3 tag or a frame it couldn't decode
            continue;
        }

        if (d->rate != info.hz) {                   // first frame (or a rate change)
            d->rate = info.hz;
            rs_init(&d->rs, (uint32_t)info.hz);
            ESP_LOGI(TAG, "%d Hz, %d ch, %d kbps -> %d Hz", info.hz, info.channels,
                     info.bitrate_kbps, SAMPLE_RATE_HZ);
        }

        const int16_t *mono = d->pcm;
        if (info.channels == 2) {
            for (int i = 0; i < samples; i++) {
                d->mono[i] = (int16_t)(((int)d->pcm[2 * i] + (int)d->pcm[2 * i + 1]) / 2);
            }
            mono = d->mono;
        }

        if (d->rate == SAMPLE_RATE_HZ) {
            if (d->out) d->out(mono, (size_t)samples);
        } else {
            size_t n = rs_process(&d->rs, mono, (size_t)samples, d->res, OUT_CAP);
            if (n > 0 && d->out) d->out(d->res, n);
        }
        d->frames++;
    }

    if (off > 0) {                                  // keep the unconsumed tail
        memmove(d->in, d->in + off, d->len - off);
        d->len -= off;
    }
    if (final) {
        d->len = 0;
    }
}

void mp3s_push(mp3s_t *d, const uint8_t *data, size_t len)
{
    if (!d || !d->in) {
        return;
    }
    while (len > 0) {
        size_t room = IN_BUF - d->len;
        size_t take = len < room ? len : room;
        memcpy(d->in + d->len, data, take);
        d->len += take;
        data   += take;
        len    -= take;
        decode(d, false);
        if (d->len == IN_BUF) {
            // A full buffer that still won't decode is not MP3; drop it rather than stall.
            ESP_LOGW(TAG, "undecodable data, skipping %d bytes", IN_BUF);
            d->len = 0;
        }
    }
}

void mp3s_finish(mp3s_t *d)
{
    if (d && d->in) {
        decode(d, true);
    }
    if (d && d->skipped > 2) {
        ESP_LOGW(TAG, "%d frames skipped", d->skipped);
    }
}

int mp3s_rate_of(const mp3s_t *d)   { return d ? d->rate : 0; }
int mp3s_frames_of(const mp3s_t *d) { return d ? d->frames : 0; }

/* ------------------------------------------------------------------ */
/*  The shared default decoder (Friday's voice), same API as before    */
/* ------------------------------------------------------------------ */

static mp3s_t *s_default;

void mp3s_begin(mp3s_out_t out)
{
    if (!s_default) {
        s_default = mp3s_new();
        if (!s_default) {
            ESP_LOGE(TAG, "out of memory for the MP3 decoder");
            return;
        }
    }
    mp3s_start(s_default, out);
}

void mp3s_feed(const uint8_t *data, size_t len) { mp3s_push(s_default, data, len); }
void mp3s_end(void)                             { mp3s_finish(s_default); }
int  mp3s_source_rate(void)                     { return mp3s_rate_of(s_default); }
int  mp3s_frames(void)                          { return mp3s_frames_of(s_default); }
