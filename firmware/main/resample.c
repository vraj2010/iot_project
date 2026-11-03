#include "resample.h"

#include <math.h>

#include "config.h"

void rs_init(resampler_t *rs, uint32_t in_rate)
{
    rs->in_rate = in_rate;
    rs->step    = (float)in_rate / (float)SAMPLE_RATE_HZ;
    rs->t       = 0.0f;
    rs->prev    = 0.0f;
    rs->lp      = 0.0f;
    // Downsampling: one-pole low-pass near 7 kHz so treble above the new
    // 8 kHz limit doesn't fold back as hiss. Speech loses almost nothing.
    if (in_rate > SAMPLE_RATE_HZ) {
        rs->lp_a = 1.0f - expf(-2.0f * (float)M_PI * 7000.0f / (float)in_rate);
    } else {
        rs->lp_a = 0.0f;
    }
}

size_t rs_process(resampler_t *rs, const int16_t *in, size_t n, int16_t *out, size_t out_cap)
{
    size_t produced = 0;

    for (size_t i = 0; i < n; i++) {
        float x = (float)in[i];
        if (rs->lp_a > 0.0f) {
            rs->lp += rs->lp_a * (x - rs->lp);
            x = rs->lp;
        }
        // Emit every output sample that falls between prev and x.
        while (rs->t <= 1.0f) {
            if (produced >= out_cap) {
                return produced;
            }
            float y = rs->prev + (x - rs->prev) * rs->t;
            if (y > 32767.0f)  y = 32767.0f;
            if (y < -32768.0f) y = -32768.0f;
            out[produced++] = (int16_t)lrintf(y);
            rs->t += rs->step;
        }
        rs->t   -= 1.0f;
        rs->prev = x;
    }
    return produced;
}
