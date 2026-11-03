// Offline fallback voice: SAM (Software Automatic Mouth). Robotic, but needs no
// internet and no quota. Speaks any English text at 16 kHz through `out`.
#pragma once

#include <stddef.h>
#include <stdint.h>

typedef void (*sam_out_t)(const int16_t *pcm, size_t samples);

// Splits long text into short phrases (SAM handles ~80 characters at a time),
// renders each and hands the audio to `out`. Returns seconds of audio produced.
float sam_say(const char *text, sam_out_t out);
