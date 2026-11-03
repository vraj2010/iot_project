# Friday

A DIY smart speaker with an Iron Man–style assistant. Ask her something and she answers out
loud. Your laptop does the hearing (faster-whisper), the thinking (Gemini), the remembering
(PostgreSQL) and the voice (an Irish neural voice); the board just listens and plays. If the
laptop is off, a Cloudflare Worker answers instead (no memory or music).

```
ESP32-S3 ── Wi-Fi ──▶ server/   laptop: Whisper · Gemini · memory · voice · music
    │
    └── laptop off ──▶ worker/   Cloudflare Worker: Whisper · Gemini · voice
```

There are two hardware options, both driven by the same laptop brain:

| Build | Board | Mic | How you talk to her |
|---|---|---|---|
| ESP32-S3 (original) | ESP32-S3 board (`firmware/`) | Onboard mic, or the laptop's mic | Hold BOOT, or say "Hey Jarvis" near the laptop |
| Bluetooth speaker | Plain ESP32 DevKit + MAX98357A amp (`friday_speaker/`) | The laptop's mic only | Press the board's BOOT button, or Enter in `speaker.py` |

## How it works

Every question goes through the same four stages, whichever board asked it:

1. **Wake** — BOOT is pressed, or the laptop's always-on wake word ("Hey Jarvis") fires.
   Nothing is recorded or sent anywhere before this point.
2. **Hear** — the recording is turned into text by faster-whisper, running locally on the
   laptop. A handful of exact commands ("pause", "louder", "what's playing") are matched
   right here and answered immediately, with no Gemini round trip.
3. **Think** — anything else goes to **Gemini**, along with her memory (profile, facts,
   recent conversation, from PostgreSQL) and whatever tools the question needs: web search,
   weather, music control, or Cortex (your uploaded documents). If the laptop is off, the
   **Cloudflare Worker** answers instead, with no memory or music.
4. **Speak** — the reply becomes audio (an Irish neural voice online, Piper or the Windows
   voice offline) and streams to whichever board asked, mixed under any music that's playing.

```
 wake word / BOOT
        │
        ▼
 faster-whisper (hearing) ──local command──▶ answered instantly (pause, volume, time, ...)
        │
        │ everything else
        ▼
 Gemini ── memory (Postgres) ── web search / weather ── music control ── Cortex (your docs)
        │
        ▼
 voice (Edge / Piper / SAPI) ──▶ board plays it back, music ducked underneath
```

She also learns by herself in the background (`AUTO_MEMORY`), without slowing a reply down.

| Folder | What it is |
|---|---|
| [`firmware/`](firmware) | ESP32-S3 firmware (ESP-IDF 5.5): mic, speaker mixer, Wi-Fi, laptop discovery, music, offline SAM voice |
| [`friday_speaker/`](friday_speaker) | Arduino sketch for the plain-ESP32 Bluetooth speaker board — a simpler alternative to `firmware/` |
| [`server/`](server) | The laptop brain in Python: hearing, Gemini + tools, memory, voice, music library |
| [`worker/`](worker) | Cloudflare Worker used as the backup brain |
| [`db/`](db) | PostgreSQL + Adminer in Docker: Friday's memory |
| [`cortex-web/`](cortex-web) | Cortex upload side, hosted on Vercel: upload page + ingest (parse, chunk, embed into Supabase pgvector). See [`docs/cortex.md`](docs/cortex.md) |
| [`Cortex/`](Cortex) | The original Cortex RAG project; its parser was reused in `cortex-web/`. Friday no longer calls it |
| [`docs/`](docs) | [Architecture overview](docs/architecture.html) and [why replies take the time they do](docs/latency.md) |

## Running it

**1. Database** (Friday's memory):

```
cd db
copy .env.example .env  &&  notepad .env     REM set a password
docker compose up -d
```

**2. Server** (the brain: hearing, Gemini, memory, voice, music):

```
cd server
py -m venv .venv  &&  .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env  &&  notepad .env     REM GEMINI_API_KEY, DEVICE_TOKEN, DATABASE_URL
python server.py
```

Leave this running — you'll see `[friday] listening on port 8765`. Firewall rules, the wake
word, and the music library are covered in [`server/README.md`](server/README.md).

**3. Pick one board:**

- **ESP32-S3** — copy `firmware/main/secrets.example.h` to `secrets.h`, fill it in (same
  `DEVICE_TOKEN` as the server's `.env`), then, in an ESP-IDF terminal:

  ```
  cd firmware
  idf.py build
  idf.py -p COMx flash monitor
  ```

  Hold BOOT and speak. Details in [`firmware/README.md`](firmware/README.md).

- **Bluetooth speaker** (`friday_speaker/`, no S3 needed) — flash
  `friday_speaker/friday_speaker.ino`, pair **Friday** under Windows Bluetooth & devices,
  set it as the sound output, then run a second process alongside the server:

  ```
  cd server
  python speaker.py
  ```

  Press the board's BOOT button (or Enter in the `speaker.py` window) and speak. More in the
  "laptop as the speaker" section of [`server/README.md`](server/README.md).

**4. Worker** (optional — the backup brain for when the laptop is off): see
[`worker/README.md`](worker/README.md).

## Secrets

Nothing private is committed. `server/.env`, `db/.env`, `worker/.dev.vars` and
`firmware/main/secrets.h` are git-ignored, and each has an `.example` template beside it.
`DEVICE_TOKEN` must be the same in the firmware (or `friday_speaker` setup), the server and
the Worker.
