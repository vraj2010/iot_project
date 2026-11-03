// Microphone capture, speaker playback and the cleanup chain in between.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define AUDIO_BLOCK 256          // samples per block = 16 ms at 16 kHz

// Start both I2S ports (and the ADC, for the MAX4466). The amp starts muted.
void audio_init(void);

// Fill `out` with AUDIO_BLOCK cleaned-up samples. Blocks until the mic has them,
// so calling this in a loop paces itself at real time.
void audio_capture(int16_t *out);

// Throw away whatever the mic buffered while we were busy.
void audio_flush_mic(void);

// Un-mute the amp, play, mute again. Call audio_play_begin() once, then
// audio_play_chunk() as often as you like, then audio_play_end().
void audio_play_begin(void);
void audio_play_chunk(const int16_t *pcm, size_t samples);
void audio_play_end(void);

// Short tones for feedback. Before the mixer starts they play directly (and
// handle begin/end themselves); once a sink is set they're handed to it, so
// they mix with whatever else is playing.
void audio_tone(float freq_hz, int ms, float amplitude);
void audio_set_tone_sink(void (*sink)(const int16_t *pcm, size_t samples));
void audio_chime_start(void);    // "I'm listening"
void audio_chime_done(void);     // "sent"
void audio_chime_error(void);    // "something went wrong"

// Scale a recorded buffer so its peak sits at NORMALIZE_PEAK. Returns the gain
// applied, so the caller can log it.
float audio_normalize(int16_t *pcm, size_t samples);

// Level of the most recent captured block, in dBFS (-90 = silence).
float audio_level_db(void);

// How many ADC samples were repaired in the last capture (MAX4466 only).
int audio_glitch_count(void);
