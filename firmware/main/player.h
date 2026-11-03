// The speaker's mixer: Friday's voice, music and chimes play at the same time.
//
// One task owns the speaker and mixes three buffers into it at 16 kHz:
//   voice  - reply audio. Starts once PREBUFFER_MS is stored, so short network
//            stalls are absorbed instead of heard as gaps (same as before).
//   music  - the current song, fed by the music task. Ducked (quieter) while
//            Friday speaks or while you talk to her.
//   chimes - short tones, mixed on top.
// The amp is switched off when nothing has played for a moment.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    int64_t first_sound_us;   // esp_timer time the reply started playing, 0 = never
    int     underruns;        // times the reply ran dry mid-way
    size_t  peak_samples;     // most reply audio held at once
    size_t  played_samples;
} player_stats_t;

// Allocate the buffers (in PSRAM) and start the mixer task. Call once.
void player_init(void);

// ---- Friday's voice (one reply at a time)

// Begin a new reply. Call before the first player_write().
void player_start(void);

// Queue 16 kHz mono samples. Returns quickly; blocks only if the buffer is full.
void player_write(const int16_t *pcm, size_t samples);

// No more audio for this reply. Whatever is buffered still plays.
void player_finish(void);

// Wait until the reply has finished playing (or timeout_ms passes).
bool player_wait_done(int timeout_ms);

player_stats_t player_stats(void);

// ---- Music (used by the music task only)

void     player_music_begin(void);                    // new song: empty the music buffer
size_t   player_music_room(void);                     // samples the music buffer can take now
size_t   player_music_write(const int16_t *pcm, size_t samples, int timeout_ms);   // samples taken
void     player_music_end(void);                      // the song's data is complete; play out the rest
void     player_music_stop(void);                     // silence the song now, drop what's buffered
bool     player_music_idle(void);                     // nothing of the song left to play
uint32_t player_music_played(void);                   // samples of this song played so far

// ---- Levels

void player_set_duck(bool on);                        // music quieter (you're talking to Friday)
void player_set_listening(bool on);                   // music silent while the mic records
void player_set_volume(int percent);                  // master volume 0-100, all sounds
int  player_volume(void);
void player_volume_load(void);                        // from flash (call after Wi-Fi init: NVS is ready)
void player_volume_save(void);                        // to flash, only if it changed

// Chimes and other short sounds, mixed on top (audio_tone() uses this).
void player_sfx(const int16_t *pcm, size_t samples);
// Wait until queued chimes have actually come out of the speaker (so the mic
// doesn't record the "listening" chime).
void player_sfx_drain(void);
