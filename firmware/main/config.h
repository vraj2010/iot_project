// ============================================================
//  I2Sonic Nexa  —  all settings live here
//  Edit, then:  idf.py build  &&  idf.py -p COMx flash monitor
// ============================================================
#pragma once

// ---------- Private settings: Wi-Fi, laptop IP, Worker URL, device token ----------
// These live in secrets.h, which git ignores. Copy secrets.example.h to secrets.h and fill it in.
#if __has_include("secrets.h")
#include "secrets.h"
#else
#error "Missing firmware/main/secrets.h: copy secrets.example.h to secrets.h and fill it in"
#endif

// ---------- Brain: your laptop first, the Cloudflare Worker as backup ----------
// The laptop runs friday-local (Whisper + Gemini + memory + voice). The board
// finds it by UDP broadcast, so there is no IP to hardcode. When the laptop is
// off or asleep, questions go to the Worker instead (no memory there).
#define USE_LAPTOP           1      // 1 = look for friday-local on the laptop
#define LAPTOP_HTTP_PORT     8765   // HTTP_PORT in friday-local's .env
#define DISCOVERY_PORT       5006   // UDP_PORT in friday-local's .env
#define LAPTOP_RETRY_SEC     15     // how often to look again while the laptop is away
#define LAPTOP_PROBE_MS      700    // "is it still there?" check before each question
#define USE_WORKER_FALLBACK  1      // 1 = use the Worker when the laptop can't be reached

// ---------- The laptop's mic (wake word) ----------
// 1 = the laptop listens on its own mic for the wake word ("hey jarvis", or your own "hey friday") and
//     sends the answer here; BOOT then means "listen now" (no holding), and falls back to push-to-talk on
//     the S3's mic when the laptop can't be reached. Needs friday-local with EARS=1 in its .env.
// 0 = push-to-talk on the S3's own mic only, as before.
#define LAPTOP_MIC           1
#define LISTEN_HOLD_SEC      20     // LISTEN_HOLD_S in friday-local's .env

// ---------- Microphone ----------
#define MIC_MAX4466   1          // analog module -> ADC
#define MIC_INMP441   2          // digital I2S mic
#define MIC_TYPE      MIC_MAX4466

#define MAX4466_OUT_GPIO    7    // must be an ADC1 pin: GPIO 1-10

#define INMP441_SCK_GPIO    5
#define INMP441_WS_GPIO     4
#define INMP441_SD_GPIO     6
#define INMP441_RIGHT_SLOT  0    // flip to 1 if the mic reads silent

// ---------- MAX98357A amp (verified pins) ----------
#define AMP_BCLK_GPIO      15
#define AMP_LRC_GPIO       16
#define AMP_DIN_GPIO       17
#define AMP_SD_GPIO        18    // driven low between replies, so no idle hiss

// ---------- Button ----------
#define PTT_GPIO            0    // BOOT button: hold to talk (with LAPTOP_MIC: press to make the laptop listen)

// ---------- Audio ----------
#define SAMPLE_RATE_HZ   16000   // what Whisper and ESP-SR both expect
#define MAX_RECORD_SEC      10   // longest question
#define MAX_REPLY_SEC       60   // longest spoken answer
#define VOLUME            1.0f   // 0.0 - 1.0 hardware ceiling (amp on 3V3 is quieter)
#define DEFAULT_VOLUME      100  // 0-100 at first boot. "Friday, volume 60" changes it; it's remembered

// ---------- Cleanup for the noisy MAX4466 ----------
#define HPF_CUTOFF_HZ     100.0f // kills mains hum and rumble
#define NOISE_GATE_DBFS   -32.0f // below this is treated as silence
#define GATE_FLOOR        0.06f  // how far gated audio is ducked (not to zero)
#define NORMALIZE_PEAK    0.80f  // peak-normalize before upload, helps Whisper
#define NORMALIZE_MAX_X    8.0f  // but never amplify more than this

// ---------- Debug ----------
#define SHOW_METER          1    // print a mic level bar while recording

// ---------- Cloud (your Cloudflare Worker, the backup brain) ----------
// WORKER_URL and DEVICE_TOKEN are in secrets.h.
#define CLOUD_TIMEOUT_MS  20000                            // give up on a Worker request after this
#define LAPTOP_TIMEOUT_MS 45000                            // ...and on the laptop (web searches can take a while)
#define SPEAK_REPLIES     1                                // 1 = /say (voice), 0 = /chat (text only)
#define SAM_FALLBACK      1                                // 1 = speak with the offline SAM voice when the cloud voice fails

// ---------- Playback buffer (smooths out network hiccups) ----------
#define PLAYER_BUFFER_SEC  30     // longest reply held in PSRAM (30 s = 960 KB)
#define PREBUFFER_MS       500    // audio stored before speaking starts
#define REBUFFER_MS        300    // refill this much if the network stalls mid-reply

// ---------- Music (songs from your laptop's library; needs friday-local) ----------
// "Play Kesariya", "pause", "next song", "louder"... Songs stream from the laptop
// as 16 kHz mono MP3 and play under Friday's voice.
#define MUSIC_ENABLED       1
#define MUSIC_DUCK          1
#define MUSIC_PAUSE         2
#define MUSIC_ON_TALK       MUSIC_DUCK  // while you hold BOOT and Friday answers: DUCK = music
                                        // plays quietly underneath, PAUSE = stops and carries on after
#define MUSIC_DUCK_LEVEL    0.15f       // music level while Friday answers (0.15 = about -16 dB)
#define MUSIC_MUTE_LISTENING 1          // 1 = silent while you hold BOOT, so Whisper hears you, not the song
#define MUSIC_HIGHPASS_HZ   120         // cut deep bass the small speaker can't play anyway: it
                                        // draws big current spikes that can brown out the board (0 = off)
#define MUSIC_GAIN          0.60f       // songs vs Friday's voice (songs are mastered loud)
#define MUSIC_BUFFER_SEC    4           // song audio held in PSRAM (4 s = 128 KB)
#define MUSIC_PREBUFFER_MS  600         // stored before a song starts
#define MUSIC_DOWNLOAD_KB   3072        // the whole song is downloaded into PSRAM as it plays
                                        // (3 MB = 6 min of the laptop's 64 kbps), so the Wi-Fi
                                        // link stays free for Friday's replies

