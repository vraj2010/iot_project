// Music from the laptop's library (friday-local). The laptop decides what to
// play; this task streams it: GET /music/<id> from the laptop, decode the MP3,
// hand the audio to the mixer, and ask for /music/next when a song ends.
//
// Friday's replies carry the commands (play track 12, pause, volume 60), and
// every question tells the laptop what's playing (X-Music-State), so she can
// answer "what's this song?" or "turn it down".
#pragma once

#include <stdbool.h>
#include <stddef.h>

// Start the music task. Call once, after cloud_start().
void music_init(void);

// Commands. Safe from any task; they return straight away.
void music_play(int track);
void music_pause(void);
void music_resume(void);
void music_stop(void);

// An action from Friday's reply, e.g. {"music":"play","track":12,"title":"..."}
// or {"music":"pause"} or {"volume":60}. Unknown keys are ignored.
void music_apply(const char *json);

// Around each turn: duck (or pause, see MUSIC_ON_TALK in config.h) while you
// talk to Friday and she answers, then back to normal.
void music_talk_begin(void);
void music_talk_end(void);

// The X-Music-State header value, e.g. "state=playing;track=12;pos=45.2;vol=80".
void music_state_header(char *buf, size_t len);
