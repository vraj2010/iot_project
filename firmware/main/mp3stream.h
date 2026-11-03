// Streaming MP3 decoder: feed bytes as they download, get 16 kHz mono PCM out.
// Uses minimp3 (mono or stereo, any MPEG-1/2 Layer III rate), downmixes and
// resamples to the speaker rate.
//
// Two ways to use it:
//   * mp3s_begin / mp3s_feed / mp3s_end: the shared default decoder, used for
//     Friday's voice (one reply at a time, from the assistant task).
//   * mp3s_new + mp3s_start / mp3s_push / mp3s_finish: a separate decoder, used
//     by the music task, so a song and a spoken reply can decode at once.
// Each decoder must be used from one task only.
//
// Stack: minimp3 keeps ~16 KB of scratch on the stack per frame, so a task that
// feeds a decoder needs a big stack (the chat task has 32 KB, music 24 KB).
#pragma once

#include <stddef.h>
#include <stdint.h>

typedef void (*mp3s_out_t)(const int16_t *pcm, size_t samples);
typedef struct mp3s mp3s_t;

// ---- separate decoders
mp3s_t *mp3s_new(void);                                  // buffers in PSRAM; NULL if out of memory
void    mp3s_start(mp3s_t *d, mp3s_out_t out);
void    mp3s_push(mp3s_t *d, const uint8_t *data, size_t len);
void    mp3s_finish(mp3s_t *d);                          // decode whatever is left
int     mp3s_rate_of(const mp3s_t *d);                   // e.g. 44100; 0 until the first frame
int     mp3s_frames_of(const mp3s_t *d);

// ---- the shared default decoder (unchanged API)
void mp3s_begin(mp3s_out_t out);
void mp3s_feed(const uint8_t *data, size_t len);
void mp3s_end(void);
int  mp3s_source_rate(void);
int  mp3s_frames(void);
