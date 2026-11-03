# firmware — Phase 6 (music from your laptop)

**Hold BOOT and speak** → the S3 sends the recording to **friday-local on your laptop**
→ faster-whisper hears it → Gemini answers as Friday, using her **memory** (PostgreSQL on
the laptop), **web search** and now **your music library** → the voice speaks it → the S3 plays it.

**New in Phase 6:** "Play Kesariya", "play some Arijit Singh", "pause", "next song",
"louder", "volume 40", "stop". Songs stream from the laptop and play **under** Friday's voice:
while you hold BOOT and while she answers, the music drops to a low level (ducks), then comes
back up. Prefer it to stop instead? Set `MUSIC_ON_TALK MUSIC_PAUSE` in `config.h`.

The S3 finds the laptop by itself (UDP broadcast). If the laptop is off, questions go to the
**Cloudflare Worker** as before (no memory, no music). If no voice is available at all, the
S3 speaks the reply itself with **SAM**.

## The laptop's mic as Friday's ears (`LAPTOP_MIC 1`)

While the S3's own mic is the noisy MAX4466, the **laptop listens** instead: say **"Hey Jarvis"** near it and
ask. The S3 waits for it with `GET /listen` (a long poll, on a connection of its own), and on a wake it
**chimes**, silences the music so the laptop hears you and not the song, and plays the answer like any other
reply. BOOT becomes **"listen now"** (press, don't hold); if the laptop can't be reached, BOOT falls back to
push-to-talk on the S3's mic. Typing still works. Needs friday-local with `EARS=1` (see its README).

| Setting (`config.h`) | Does |
|---|---|
| `LAPTOP_MIC` | `1` = the laptop's mic + wake word, `0` = push-to-talk on the S3 only (as before) |
| `LISTEN_HOLD_SEC` | How long the laptop holds each poll: same as `LISTEN_HOLD_S` in its `.env` |

The S3 asks for the wake word only while no turn is running, and the laptop takes each new poll to mean "the
speaker has finished talking": that is how Friday never wakes herself up with her own voice.

## What changed from Phase 5

| File | Change |
|---|---|
| `main/player.c/.h` | Rewritten as a **mixer**: one task owns the speaker and mixes Friday's voice, the song and chimes. Ducks the song while she speaks or you talk. Master volume (0–100), remembered in flash |
| `main/music.c/.h` | **New.** Music task: streams `GET /music/<id>` from the laptop, decodes it, hands it to the mixer, asks for `/music/next` when a song ends. Pause / resume / stop / play commands. 24 KB stack in internal RAM |
| `main/mp3stream.c/.h` | The decoder can now have several instances (voice and music decode at the same time). The old functions still work |
| `main/cloud.c/.h` | Understands the new `I2S4` reply, which carries an action for the speaker (`{"music":"play","track":12}`, `{"volume":60}`). Sends `X-Music-State` (what's playing) with each question. Knows the Edge voice by name |
| `main/audio.c/.h` | Chimes go through the mixer once it's running, so they play on top of music |
| `main/main.c` | Ducks (or pauses) the music for each turn; saves the volume after a turn; starts the music task |
| `main/config.h` | New `DEFAULT_VOLUME` and the `MUSIC_*` block |
| `main/CMakeLists.txt` | Adds `music.c` |

Wi-Fi, laptop IP, Worker URL and token now live in `main/secrets.h` (copy `main/secrets.example.h`); git ignores it, so none of it gets committed.

## Run it

1. Start the laptop side first: `friday-local` (see its README). It prints
   `[friday] listening on port 8765`.
2. Laptop and S3 must be on the **same Wi-Fi** (your phone hotspot is fine).
3. Same `DEVICE_TOKEN` in `main/secrets.h` and in friday-local's `.env`.

Then, in an ESP-IDF terminal (`& C:\esp\v5.5.5\esp-idf\export.ps1` first),
with the cable on the **UART/COM** port:

```
cd firmware
idf.py build
idf.py -p COMx flash monitor
```

## Pass checklist

At boot you should see:

```
brain   : laptop (friday-local, found automatically), Worker as backup
[brain] found the laptop at 192.168.x.x:8765
```

A spoken question prints (numbers are only examples):

```
[brain] using the laptop (memory on)
[heard] "Remember that the spare key is in the kitchen drawer"
friday> Got it, boss. The spare key is in the kitchen drawer.
[time] laptop | first sound 2400 ms | Whisper 600 ms + Gemini 900 ms + Piper 300 ms | download done 2600 ms
```

| # | Test | Expected |
|---|---|---|
| 1 | Hold BOOT: "Remember that the spare key is in the kitchen drawer" | `[time] laptop`, she confirms; the laptop prints `[tool] save_memory` |
| 2 | Restart the S3, then ask "Where is the spare key?" | She knows: memory lives on the laptop, not the board |
| 3 | "What's the gold price today?" | Laptop prints `[tool] web_search` (needs billing on the Gemini key, see friday-local README) |
| 4 | Stop friday-local (Ctrl+C), ask anything | `[brain] laptop is not answering...`, `[brain] using the cloud Worker`, and `[time] cloud` |
| 5 | Start friday-local again, wait 15 s | `[brain] found the laptop at ...`; next question says `[time] laptop` |
| 6 | Hold BOOT and say nothing for 2 s | `[heard] nothing clear`, she says she didn't catch that |
| 6a | `LAPTOP_MIC 1`: near the laptop, "Hey Jarvis, what time is it?" | Chime, `[listening] on the laptop's mic...`, she tells the time (the laptop prints `local time`, no Gemini) |
| 6b | "Hey Jarvis" ... say nothing | `[heard] nothing after the wake word` after about 5 s, no reply |
| 6c | Play a song, then "Hey Jarvis, pause" | The music goes silent while you speak, then pauses: `local pause (phrase)` on the laptop |
| 7 | Type a question at `you>` | Answered by the laptop the same way |
| 8 | Mobile data off on the hotspot (laptop still on Wi-Fi) | Hearing, memory and voice still work locally; only Gemini fails, and SAM speaks the error |
| 9 | "Play" + a song you have | `[action] {"music":"play",...}`, then `[music] playing <title>`; the song starts quietly under her reply, then comes up |
| 10 | Hold BOOT while it plays: "what's this song?" | Music ducks while you talk and while she answers; she names it |
| 11 | "Pause", then "resume" | `[music] paused at N s`; resume carries on from there |
| 12 | "Next song" / let a song finish | A different song starts (`[music] playing ...`) |
| 13 | "Volume 40" | `[music] volume 40%`; still 40 after a reset |
| 14 | "Stop the music" | `[music] stopped` |

## If something fails

| You see | Meaning |
|---|---|
| Never `[brain] found the laptop` | Windows Firewall is blocking Python: allow it on **private** networks (or run the two `netsh` lines in friday-local's README). Or the S3 and laptop are on different networks (a 5 GHz guest SSID counts). Last resort: set `LAPTOP_IP_FIXED` |
| Found, but every question says `not answering` | friday-local is running on a different port: `LAPTOP_HTTP_PORT` must match `HTTP_PORT` in `.env` (discovery normally tells the S3 the right port) |
| `HTTP 401` | `DEVICE_TOKEN` differs between `secrets.h` and friday-local's `.env` (or the Worker secret) |
| `HTTP 502: gemini ...` | Gemini problem (key, model name, quota); the laptop's window shows the full message |
| Text prints, then `HTTP 502: voice: ...` and SAM speaks | Piper or the Windows voice failed on the laptop; its window says why |
| `[heard]` is wrong or garbled | Speak closer to the mic, or set `WHISPER_MODEL=small` in `.env`. Play `last_question.wav` on the laptop |
| `[heard] nothing clear` every time | Mic reads silence: watch the meter bar while you talk. For the INMP441 try `INMP441_RIGHT_SLOT 1` |
| `HTTP 400: audio too short` | Hold BOOT a little longer |
| Still choppy, `underruns` > 0 | Network slower than real time. Raise `PREBUFFER_MS` to 1500 in `config.h` |
| Choppy but `0 underruns` | Not the network: power or wiring to the amp. Try a 100–470 µF capacitor across amp VIN/GND |
| `connection dropped mid-reply` | Wi-Fi dropped during speech; it does not retry, so it won't repeat itself |
| Want the Worker only again | `USE_LAPTOP 0` in `config.h` |
| "I can't play music right now" | She's on the Worker (laptop not found), or friday-local printed `[music] off`. Check `[brain]` lines |
| `[music] the laptop said HTTP 503 ... install ffmpeg` | That song is M4A (or a video rip named .mp3): install ffmpeg on the laptop |
| `[music] the laptop said HTTP 404` | The song's file moved or was deleted: run `python music.py scan` |
| Music too loud / quiet next to her voice | `MUSIC_GAIN` in `config.h` (0.6 default) |
| Music too loud while you talk | Lower `MUSIC_DUCK_LEVEL`, or `MUSIC_ON_TALK MUSIC_PAUSE` |
| `E BOD: Brownout detector was triggered` or a `TG1WDT` reset when a song starts | The amp pulls more current than the supply gives (loud bass at high volume). Power the board from a 5 V / 2 A adapter or a powered hub rather than a laptop port, and put a 470–1000 µF capacitor across the amp's VIN/GND. `MUSIC_HIGHPASS_HZ` (120) already cuts the deepest bass; lowering `MUSIC_GAIN` or the volume also helps |
| `[ears] the laptop answered /listen with HTTP 503` | friday-local's mic is off: `EARS=1` in its `.env`, and look for `[ears] off: ...` in its window |
| `[ears] the laptop answered /listen with HTTP 404` | friday-local is older than this firmware: update it |
| "Hey Jarvis" does nothing | The laptop's window shows `[ears] heard "hey jarvis"` when it hears you; if it doesn't, see the ears section of its README |
| `[brain] the laptop took more than 45 s` | Something on the laptop was very slow (usually a web search): its window shows which step |
| Music stutters | `MUSIC_PREBUFFER_MS` up to 1500; check the laptop isn't busy converting (first play of a song converts it, later plays are instant) |

## Files

| File | What it does |
|---|---|
| `main/main.c` | The main loop records while BOOT is held (with `LAPTOP_MIC`: asks the laptop to listen); one **assistant task** (32 KB stack) runs every turn, typed, push-to-talk or from the laptop's mic, one at a time; the small **ears task** (8 KB) only waits for the wake word |
| `main/cloud.c/.h` | HTTP(S) client for the laptop or the Worker (same protocol). `cloud_ask()` uploads a recording, `cloud_say()` a typed question, `cloud_listen_answer()` fetches what the laptop's mic heard; all parse the reply frame as it arrives and hand audio to the speaker. `cloud_listen_wait()` is the wake-word long poll, `cloud_listen_press()` the BOOT button |
| `main/net.c/.h` | Wi-Fi, laptop discovery, the quick reachability check |
| `main/textin.c/.h` | Reads typed lines from the serial monitor without blocking push-to-talk |
| `main/player.c/.h` | The mixer: voice, music and chimes into the speaker at 16 kHz. Voice prebuffers and refills on stalls; music ducks under it; master volume |
| `main/music.c/.h` | Music task: streams songs from the laptop, pause / resume / next |
| `main/mp3stream.c/.h` | Streaming MP3 decoder (minimp3), one instance for the voice, one for music |
| `main/resample.c/.h` | Converts 44.1 / 24 / 22.05 kHz audio to the speaker's 16 kHz |
| `main/sam_tts.c/.h` | Offline voice (SAM) |
| `main/audio.c/.h` | Mic capture and cleanup (high-pass, noise gate, normalise), chimes |
| `components/minimp3/`, `components/sam/` | Unchanged |
