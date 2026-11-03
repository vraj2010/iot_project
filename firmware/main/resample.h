// Streaming sample-rate converter to the speaker's 16 kHz.
// Feeds any input rate (MeloTTS MP3 at 44.1/24 kHz, SAM at 22.05 kHz) in
// chunks of any size; keeps its state between calls so chunk edges are seamless.
#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct {
    float   step;        // input samples per output sample
    float   t;           // position between `prev` and the next input sample
    float   prev;        // last (filtered) input sample
    float   lp;          // anti-alias low-pass state
    float   lp_a;        // low-pass coefficient (0 = off)
    uint32_t in_rate;
} resampler_t;

void rs_init(resampler_t *rs, uint32_t in_rate);

// Convert `n` input samples. Writes up to `out_cap` samples to `out` and
// returns how many were written. Size `out` for n * 16000 / in_rate + 2.
size_t rs_process(resampler_t *rs, const int16_t *in, size_t n, int16_t *out, size_t out_cap);
