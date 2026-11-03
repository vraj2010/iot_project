# friday-local: Friday's brain on your laptop

> In the Friday repo this is the `server/` folder.

The laptop hears (faster-whisper), remembers (PostgreSQL), speaks (an Irish neural
voice online, or Piper / the Windows voice offline) and asks **Gemini** to think. It speaks the same protocol as the
Cloudflare Worker, so the S3 uses whichever one is available: the laptop when it's
on, the Worker when it's not.

```
S3 --(Wi-Fi, HTTP)--> friday-local --> faster-whisper   (hearing, on the laptop)
                                   --> PostgreSQL       (memory, on the laptop)
                                   --> Gemini API       (thinking, web search)
                                   --> Piper / Windows  (voice, on the laptop)
```

## What Friday can do now

| Say | What happens |
|---|---|
| "Hey Jarvis" (near the laptop), then your question | The laptop's mic hears the wake word, records until you stop and the speaker answers ([the laptop's ears](#the-laptops-ears-wake-word)) |
| "Remember that the spare key is in the kitchen drawer" | Saved to memory (`save_memory`) |
| "Where's the spare key?" | Answered from memory: related memories are put in her prompt automatically |
| "What did I tell you about my car?" | Searches memories and past conversations (`recall_memory`) |
| "Forget my favourite colour" | Removes that memory (`forget_memory`) |
| "My name is Deep" / "I live in Bengaluru" | Stored in your profile, which is in every prompt (`update_profile`) |
| "What's the gold price today?" / "Any news on..." | Live web search (`web_search`): Google and DuckDuckGo are asked at once, the first good answer wins |
| "How's the weather in Gandhinagar?" / "Will it rain tomorrow?" | Open-Meteo forecast (`get_weather`), free and usually under a second. No city = your profile's city |
| "Play Kesariya" / "Play some Arijit Singh" / "Play the Brahmastra songs" | Plays from your own songs (`play_music`), then keeps going with others |
| "Pause" / "Resume" / "Next song" / "Go back" / "Stop" / "Louder" / "Volume 40" | Answered on the laptop in milliseconds, no Gemini ([commands without Gemini](#commands-without-gemini)); other wordings go to Gemini's `music_control` |
| "What's playing?" / "What time is it?" / "What's the date?" | Also answered on the laptop |
| "Play Tum Hi Ho next" | Queues it after the current song |
| "I added some songs" | Indexes new files in the music folder (`check_new_music`) |
| "What do my biology notes say about mitochondria?" / "How do I reset the router, it's in the manual" | Answers from the documents you uploaded to Cortex and says which one, and which page, it came from |
| Normal chat | She remembers the last 8 exchanges; after 10 minutes of silence a new conversation starts and the old one is summarised |

She also **learns by herself**: after each reply, a background job asks Gemini whether you
mentioned anything worth keeping ("I've started going to the gym at 6") and saves it.
That never slows a reply down. Set `AUTO_MEMORY=0` to keep only what you ask her to remember.

## 1. PostgreSQL (already running in Docker)

Your database runs in Docker from the repo's `db/` folder (`pgvector/pgvector:pg17`, user `friday`,
database `friday`, port 5432; the password is `FRIDAY_DB_PASSWORD` in `db\.env`). Make sure it's up before starting friday-local:

```
cd db
docker compose up -d
docker compose ps
```

`friday-pg` should say **running** (if not: `docker compose start`, or just open Docker Desktop).

You don't need to load any schema: friday-local creates its own tables (`profile`, `facts`,
`episodes`, `turns`, `items`, `music_tracks`, `settings`) the first time it starts.
**Don't also load the old Supabase schema into this database**: it uses the same table names
with different columns, and the two would clash.

## 2. Set up friday-local (once)

Create a virtual environment (or reuse one that already has faster-whisper):

```
cd server
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
notepad .env
```

In `.env`, fill in the first three lines:

| Setting | Value |
|---|---|
| `GEMINI_API_KEY` | Same key as your Worker |
| `DEVICE_TOKEN` | Copy from `DEVICE_TOKEN` in the firmware's `main\secrets.h` |
| `DATABASE_URL` | Replace `YOUR_DB_PASSWORD` with `FRIDAY_DB_PASSWORD` from `db\.env` |

**Allow it through Windows Firewall.** Open a terminal **as Administrator** and run:

```
netsh advfirewall firewall add rule name="FRIDAY http" dir=in action=allow protocol=TCP localport=8765 profile=private
netsh advfirewall firewall add rule name="FRIDAY discovery" dir=in action=allow protocol=UDP localport=5006 profile=private
```

These rules only apply on networks Windows treats as **Private**. A phone hotspot often
joins as Public: Settings → Network & internet → Wi-Fi → (your hotspot) → **Private network**.

## 3. Check it without the speaker

```
python server.py --test "Remember that my bike is red"
python server.py --test "What colour is my bike?"
```

The first prints `[tool] save_memory(...)`, the second answers "red". Both write
`last_reply.wav`; play it to hear the voice.

## 4. Run it

```
python server.py
```

You'll see `[memory] PostgreSQL ready`, `[tts] using ...`, `[stt] ready` and
`[friday] listening on port 8765`. Leave it running and power the S3: it prints
`[brain] found the laptop`. Every question then shows up here with timings:

```
[tool] save_memory({"fact": "The spare key is in the kitchen drawer."}) -> saved memory 3: ...
[turn] heard "Remember the spare key is in the kitchen drawer" | Whisper 640 ms | Gemini 1100 ms (tools: save_memory) | voice 280 ms
friday> Got it, boss. The spare key is in the kitchen drawer.
[memory] learned: saved memory 4: The user goes to the gym at 6 am.
```

To test from the laptop as if you were the S3: `python client.py --say "hello"` (or
`--ask question.wav`). It finds the server the same way the S3 does.

## The laptop's ears (wake word)

Until the speaker has a good microphone, the **laptop's own mic** listens. Say **"Hey Jarvis"**, then your
question in the same breath or after the chime; when you stop talking, the speaker answers. Only that phrase
wakes her: other speech, music and noise score close to zero.

```
laptop mic ─80 ms─▶ wake word ──"hey jarvis"──▶ record until you stop ──▶ Whisper ─▶ local command or Gemini ─▶ speaker
              openWakeWord, always on            Silero speech detector
```

Nothing is recorded or sent anywhere before the wake word: each 80 ms of sound is scored and thrown away.
The speaker never gets pushed to: it asks with `GET /listen` and the laptop holds that request until the wake
word is heard (firmware `LAPTOP_MIC 1`). On a wake the speaker chimes and silences the music, so the mic hears
you and not the song.

**She doesn't wake herself up.** While she's answering (and until the speaker has finished playing the reply)
the wake word is switched off, then stays off another `WAKE_COOLDOWN_S` for the echo. You can't interrupt her
by voice yet; BOOT works.

**Setup:** `pip install -r requirements.txt` (adds `sounddevice` and `openwakeword`). The first start downloads
the wake word model from openWakeWord's GitHub (about 5 MB). You'll see
`[ears] listening for "hey jarvis" on Microphone (Realtek...)`.

**Check it without the speaker:**

```
python -m ears.mic                  which microphones there are, and a 5 s level meter (talk!)
python client.py --listen           be the speaker: say "hey jarvis, what's the time" and it prints the answer
```

| Setting | Default | Does |
|---|---|---|
| `EARS` | `1` | `0` = no laptop mic (BOOT push-to-talk on the S3 only) |
| `MIC_DEVICE` | blank | The Windows default mic, or a number / part of a name from `python -m ears.mic` |
| `WAKE_MODEL` | `hey_jarvis` | A built-in phrase (`hey_jarvis`, `alexa`, `hey_mycroft`) or your own model's path, e.g. `models/hey_friday.onnx` |
| `WAKE_THRESHOLD` | `0.6` | 0..1. Raise it if she wakes up when nobody called her, lower it if she misses you |
| `WAKE_DEBUG` | `0` | `1` prints each near-miss (`[ears] near miss: 0.43`), which is how you pick a threshold for your room |
| `WAKE_COOLDOWN_S` | `0.8` | Deaf this long after the speaker has finished a reply |
| `EARS_START_S` / `EARS_SILENCE_MS` / `EARS_MAX_S` | `5` / `800` / `12` | After the wake word: how long you have to start, how much silence ends the question, the longest question |

**Your own "Hey Friday".** "Hey Jarvis" is built in so everything works today. A custom phrase is a small
model trained on synthetic speech (an hour or two, e.g. with [livekit-wakeword](https://pypi.org/project/livekit-wakeword/),
which exports openWakeWord-compatible ONNX). Put the `.onnx` in `server/models/` and set
`WAKE_MODEL=models/hey_friday.onnx`; nothing else changes. Use two words ("hey friday"), not "friday" alone:
a single word false-triggers in normal conversation ("it's Friday").

When the INMP441 arrives, the same job moves onto the S3 (ESP-SR WakeNet); the rest of this pipeline stays.

## The laptop as the speaker (Bluetooth board, button only)

The S3 is gone: the ESP32 DevKit is now a plain **Bluetooth speaker** (`friday_speaker/friday_speaker.ino`).
It has no mic, so the laptop's mic listens, and the wake word is **off** (`WAKE_ENABLED=0` is the default): Friday
listens only after you press **BOOT** on the board. `speaker.py` plays the voice and the music on the laptop's
sound output, which Windows sends to the board.

```
python server.py        # terminal 1: the brain (unchanged)
python speaker.py       # terminal 2: the button, her voice and your music
```

1. Flash `friday_speaker/friday_speaker.ino`, then pair **Friday** in Windows (Bluetooth & devices).
2. Set Windows' sound output to Friday, or pin it: `python speaker.py --out Friday` (`--list-devices` lists them).
3. Plug the board's USB cable in (BOOT is read over its serial port), close the Arduino Serial Monitor, and run
   `pip install -r requirements.txt` once (adds `pyserial`).
4. Press BOOT, wait for the chime, talk. Pressing it while she talks cuts her off and listens again. **Enter** in
   the `speaker.py` window does the same, so you can test without the board.

What `speaker.py` does for you: the chime and a quiet song while you talk; her voice as it arrives (sentence by
sentence); songs from your library (play, pause, resume, stop, "volume 60", then the next song by itself) ducked
under her voice; the volume commands change its own software volume.

| `.env` | Default | |
|---|---|---|
| `WAKE_ENABLED` | `0` | `1` brings the wake word (`WAKE_MODEL`, `WAKE_THRESHOLD`) back next to the button |
| `PTT_PORT` | `auto` | the board's COM port (`COM5`), `auto` to find it, `off` for the Enter key only |
| `SPEAKER_DEVICE` | blank | part of the sound output's name, or its number; blank = Windows' default |
| `SPEAKER_VOLUME` | `80` | start volume, 0 to 100 |

`python ptt.py` checks the button on its own: it names the ports it sees and prints every press.

## Commands without Gemini

Plain commands are answered **on the laptop** (`intents.py`): no tokens, no Gemini round trip, the reply starts
as soon as Whisper is done. Short replies like "Paused." and "Volume 40 percent." are already cached as audio,
so they play instantly.

| Command | Examples |
|---|---|
| Pause / resume / stop | "pause", "pause the music", "resume", "carry on", "stop" |
| Skip / back / restart | "next song", "skip this one", "go back", "play it again", "from the top" |
| Volume | "louder", "a bit quieter", "volume 40", "set the volume to forty five", "full volume" |
| Play | "play Kesariya", "play something by Arijit", "play Tum Hi Ho next", "play some music" |
| What's on | "what's playing?", "who sings this?" |
| Time and date | "what time is it?", "what's the date today?" |

Exact phrases are matched first. Other wordings ("hold the music a sec", "make it a bit softer") go through
a small sentence model (all-MiniLM-L6-v2 on onnxruntime, ~90 MB, downloaded once into the Hugging Face cache;
nothing new to install). It only takes a sentence it's sure of: no negations ("don't stop"), no questions
except "what's playing" and the time, at most `INTENT_MAX_WORDS` words, and a clear lead over the next-best
command. A song the library can't find goes to Gemini, which can work out what you meant.

```
python intents.py "turn it down a bit"      what Friday would do, and the model's score for each command
```

These turns aren't saved to her conversation history or memory (there's nothing to remember in "pause"), and
the console shows them as `local pause (phrase) 0 ms, no Gemini`.

| Setting | Default | Does |
|---|---|---|
| `LOCAL_COMMANDS` | `1` | `0` = everything goes to Gemini, as before |
| `INTENT_MODEL` | `1` | `0` = exact phrases only (no download) |
| `INTENT_MIN_SIMILARITY` | `0.8` | How sure the model must be. Raise it if a sentence is taken for a command it isn't |
| `INTENT_MARGIN` | `0.05` | How far the best command must be ahead of the next one |
| `INTENT_MAX_WORDS` | `8` | Longer sentences always go to Gemini |

## Looking at and editing memory

Open **Adminer** at <http://localhost:8080> (System PostgreSQL, Server `db`, Username `friday`,
your password, Database `friday`). Click a table → **Select data** to browse, **edit** on a row to change
it, **New item** to add one. Changes apply on the next question, no restart.

| Table | What's in it |
|---|---|
| `profile` | One row; `content` is JSON, e.g. `{"name": "Deep", "city": "Bengaluru"}`. Always in her prompt |
| `facts` | Long-term memories. Add rows by hand too: just fill `content` and she'll index it. Set `is_active` false to hide one |
| `episodes` | One per conversation, with a summary |
| `turns` | Every question and answer, with which tools ran |
| `items` | Free-form storage (notes, lists...), for later features |
| `music_tracks` | Your music library: official titles, artists, albums and artwork from Apple, play counts. Fix a wrong match by editing `title` / `artist` |
| `settings` | `extra_instructions`: anything you want added to her personality, e.g. `"Call me Deep. Answer in Hinglish if I speak Hinglish."` |

## Settings worth knowing (`.env`)

| Setting | Does |
|---|---|
| `GEMINI_MODEL` | Default `gemini-3.5-flash-lite`, the same model as the Worker |
| `WEB_SEARCH` / `SEARCH_PROVIDER` | Live web search. `auto` (default) tries Google Search grounding first; on a key without billing Google refuses, and Friday uses **DuckDuckGo** instead (free, no key) for the next hour. `duckduckgo` skips Google entirely. `SEARCH_REGION=in-en` gives Indian results |
| `AUTO_MEMORY` | `1` = learn facts by herself after each reply (one small extra Gemini call, in the background) |
| `HISTORY_TURNS` / `SESSION_GAP_MIN` | How much recent conversation she keeps word for word, and how long a silence starts a new one |
| `WHISPER_MODEL` | `base` is fast; `small` hears a noisy mic better but is ~3x slower on CPU |
| `WHISPER_DEVICE` | `cpu` works everywhere. `cuda` + `WHISPER_COMPUTE=int8_float16` uses your Quadro (needs CUDA 12 cuBLAS and cuDNN 9 installed); if CUDA fails it falls back to the CPU by itself |
| `TTS_ENGINE` / `EDGE_VOICE` | `edge` (default) speaks with Microsoft's Irish female neural voice `en-IE-EmilyNeural` over the internet, through the free Edge read-aloud service (unofficial: Microsoft could change or block it). If it fails, the reply is spoken by the local voice instead. `EDGE_RATE=+10%` speeds her up. `local` = offline voices only |
| `SAPI_VOICE` | Windows voice for the local backup, default `Zira` (female). `python tts.py` lists what you have |
| `PIPER_EXE` / `PIPER_MODEL` | Set both for a much nicer voice than the Windows one. Get a Piper Windows build and an `.onnx` voice such as `en_US-amy-medium` (plus its `.json`) |

## Music

Friday plays songs from your own folder (`MUSIC_DIR`, now `D:\songs`). The laptop keeps the
library and the queue; the speaker streams one song at a time and mixes it under her voice
(quieter while you talk to her, back up after).

**Once:** install ffmpeg (it converts songs for the speaker and makes M4A files playable),
then index the folder:

```
winget install Gyan.FFmpeg
python music.py scan
```

Open a **new** terminal after installing ffmpeg so it's on the PATH. The scan looks each new
song up on Apple's iTunes Search API (free, ~20 lookups a minute), then converts every song
once into `music_cache\` (16 kHz mono, the speaker's format; about 2 MB per song). Later
scans only touch new or changed files, and a renamed file keeps its metadata.

**New songs:** drop them in the folder. Friday checks every `MUSIC_SCAN_MIN` minutes (30 by
default), or say "I added some songs". `python music.py list --review` shows matches worth a look.

| Setting | Does |
|---|---|
| `MUSIC_DIR` | The folder with your songs |
| `MUSIC_SCAN_MIN` | How often to look for new songs (0 = only when you ask) |
| `MUSIC_AUTOPLAY` | `1` = after what you asked for, keep playing other songs (artist first, then a shuffle) |
| `MUSIC_LOUDNORM` | `1` = even out loud and quiet songs |
| `FFMPEG` | Full path to `ffmpeg.exe` if it isn't on the PATH |

Check it without the speaker: `python server.py --test "play kesariya" --music` prints the
`[action]` the speaker would get, and `python client.py --song 12` downloads song 12 exactly as
the speaker receives it (`song.mp3`).

Music needs the laptop: when the S3 falls back to the Cloudflare Worker, Friday says she
can't play music right now.

## Your documents (Cortex)

Friday can answer from your own PDFs, Word and text files, and says where the answer came from
("your biology notes, page three, say..."). The design and the numbers behind it are in
[`docs/cortex.md`](../docs/cortex.md).

```
UPLOAD (any browser)                                    ASK (this laptop)
Cortex upload page (Vercel) ── file ──▶ Supabase        question ─▶ friday-local
   parse ▸ chunk ▸ Gemini embeddings ──▶ cortex.chunks       │  about your documents? (a local check, microseconds)
                                          ▲                   ├─ yes: embed the question + cortex.search() ─▶ passages
                                          └─ one SQL call ────┤       in the prompt ─▶ Gemini answers in ONE round
                                                              └─ no: search in the background, offer search_documents
```

**Uploading** happens on the hosted Cortex page (password protected), not here. A document can be asked about
once the page shows it as *ready*; Friday re-reads the list every `CORTEX_LIST_S` seconds (60).

**Connecting Friday** (once): in Supabase open *Connect*, copy the **Session pooler** string (port 5432, not the
transaction pooler on 6543), put the database password in it and set it in `.env`:

```
CORTEX_DATABASE_URL=postgresql://postgres.<project-ref>:<password>@aws-0-ap-south-1.pooler.supabase.com:5432/postgres
```

TLS (`sslmode=require`) is added for you. At start-up friday-local connects in the background (it never delays
start-up) and prints `[cortex] 3 document(s) ready` and `[cortex] ready in ... ms`. `/health` shows
`"documents": 3`, or `"off"` / `"connecting"` / `"unreachable"`. Leave the setting empty and Cortex is off: no
connection, no extra prompt text, Friday works exactly as before.

**How Friday decides.** Each question first goes through a quick local check: phrases like *my notes*, *the
manual*, *the pdf*, *according to my notes*, *what does the manual say*, or a distinctive word from one of your
documents' names or subjects used as a reference ("the router", "in biology" for `Router_Manual.txt` and
`biology-notes.pdf`). "How do routers work" or "what does the fox say" don't count: those go the tool way.

- *Yes* (the **inject** path): the question is embedded and searched while her memory builds the prompt, and the
  best passages go straight into Gemini's instructions, so she answers in one round. If nothing is similar enough,
  Gemini is told so and she says she couldn't find it in your documents.
- *No* (the **tool** path): she gets a `search_documents` tool and the list of your documents, and the same search
  starts in the background as the question arrives (**prefetch**). If Gemini decides to look, the tool hands back
  the prefetched passages at once instead of searching again, as long as she's looking for the same thing. A
  follow-up ("tell me more about that one") or a named document is searched with her own wording.
- If the database or Gemini's embedding fails, or takes longer than `CORTEX_TIMEOUT` (2.5 s), she answers without
  the documents and says she couldn't reach them. A document problem never breaks a normal answer. A database call
  that runs out of time is cancelled and its connection replaced, and if the database stops answering altogether
  Cortex is marked unreachable within a few seconds, so later questions don't wait for it.

**Citations.** Gemini marks each fact with the number of the passage it came from (`[1]`) and names the source once
in words. The markers are never spoken (they're removed before the voice, also when a marker arrives split across
pieces of a streamed reply). Which passages count as cited: the ones she marked; if she marked none, on the
inject path none (she didn't use them: the question only looked like a document question), on the tool path all the
passages the search returned. Where they show:

- the console, under the reply (`[cortex] inject: [1] biology-notes.pdf p.3 (0.82)`);
- `/chat` and `/ask?debug=1` JSON, as `citations`;
- the `X-Citations` header of a `/say` or `/ask` reply that **isn't** streamed. A streamed reply (the default, what
  the speaker gets) sends its headers before the answer exists, so it has no citations: for those, the console and
  the upload page are the place to look, and she says the source out loud;
- the upload page's *Recent answers*: each answer that cited something (and had no error) is saved to
  `cortex.answers` by a small background worker of its own, with links to the cited pages. `X-No-Log: 1` skips it,
  like the history.

**Check it without the speaker:**

```
python cortex.py                                          what Cortex has
python cortex.py "what is the function of mitochondria"   what a question finds, its similarities, and how long it took
python server.py --test "what do my biology notes say about mitochondria"
```

| Setting | Default | Does |
|---|---|---|
| `CORTEX_DATABASE_URL` | empty | Supabase session pooler string. Empty = Cortex off |
| `CORTEX_TOP_K` | `4` | Passages handed to Gemini per question |
| `CORTEX_MIN_SIMILARITY` | `0.5` | Passages less similar than this (0 to 1) are left out. Lower it if she says she can't find things that are there; raise it if she quotes passages that don't fit. `python cortex.py "question"` lists every candidate's similarity, including those below the threshold |
| `CORTEX_TIMEOUT` | `2.5` | Seconds a question waits for its passages at most |
| `CORTEX_INJECT` | `1` | `0` = never put passages in the prompt up front; Gemini always decides with the tool (a second round) |
| `CORTEX_PREFETCH` | `1` | `0` = search only when Gemini asks; questions then go to Google's embedding API only when they're about your documents |
| `CORTEX_LIST_S` | `60` | How often the list of documents is re-read |
| `CORTEX_MAX_CHARS` | `1200` | Characters per passage |
| `CORTEX_LOG_ANSWERS` | `1` | Save document answers to `cortex.answers` |

**Speed.** A question the check catches costs about 0.2-0.4 s more than an ordinary one (embedding the question at
Google, then one database call to Mumbai, both on connections that are kept open), instead of a whole second Gemini
round. An ordinary question costs nothing extra in time: the check takes microseconds and the prefetch runs in the
background.

**The prefetch and your embedding quota.** While you have documents, almost every question of three or more words
("what's the weather like", "tell me a joke about cats") is sent once to Gemini's embedding API in the background, in
case Gemini decides to search. That uses the same Gemini embedding quota as your uploads (the free tier has a daily
limit). `CORTEX_PREFETCH=0` turns it off: then a question goes to the embedding API only when it is about your
documents (the check said so, or Gemini called the tool), and a tool search costs about 0.3 s more.
The `[latency]` line and `latency.log` say how long each question waited for the documents (`rag_mode`, `rag_ms`,
`embed_ms`, `search_ms`).

**Privacy.** Your documents live in your Supabase project (Mumbai), not on this laptop. Friday reads them with the
database password in `.env`; the browser never sees a Supabase key. The few passages that match a question go to
Gemini with that question, like your memories do. With `CORTEX_PREFETCH=1` (the default) and documents uploaded,
almost every question of three or more words also goes to Gemini's embedding API; with `CORTEX_PREFETCH=0` only
questions about your documents do. Text inside a document is treated as reference, never as instructions, and it
can't break out of the passage markers it is given to Gemini in.

**If it doesn't work:**

| Problem | Fix |
|---|---|
| `[cortex] can't reach postgresql://...` at start-up | Wrong password, or the transaction pooler (6543) instead of the session pooler (5432), or the Supabase project is paused (free projects pause after a week unused: open the dashboard to wake it). Friday keeps trying every 10 s |
| `"documents": 0` | Nothing is *ready* yet: check the upload page; documents still processing aren't searched |
| She says she can't find something that's there | Lower `CORTEX_MIN_SIMILARITY` (try 0.4), or ask with "my notes" / the document's name so the check catches it |
| `[cortex] no passages within 2.5 s` | The network to Google or Mumbai is slow; raise `CORTEX_TIMEOUT` a little |
| `[cortex] search-3 didn't answer in time: cancelled, closed and replaced` | A database call hung (a network that drops traffic). Friday has already replaced the connection; if it keeps happening, `/health` will show `"unreachable"` until the line recovers |
| `[cortex] embedding failed: gemini rate limit hit` | Too many questions for the free embedding quota: set `CORTEX_PREFETCH=0` so only document questions are embedded |

## Speed

Every question prints where its time went, and the same numbers go into `latency.log`
(tab-separated, opens in Excel):

```
[latency] 2140 ms to first audio = hearing 610 + memory/prompt 12 + Gemini 980 + tools 0 + voice 520
```

That's measured on the laptop, from the moment the recording arrives until the first audio leaves.
The S3's `[time]` line adds the upload and the speaker's buffer on top.

| Setting | Effect |
|---|---|
| `GEMINI_REASONING=low` | Gemini thinks briefly before answering (Gemini 3 can't skip thinking entirely). `medium` or `high` for harder questions, at a cost in speed. If the model refuses the setting, Friday notices and uses Google's default |
| `MEMORY_ALL_FACTS_MAX=40` | While she knows fewer than 40 things about you, they all go in her prompt and no memory search runs (that search cost a trip to Google on every question) |
| `TTS_CACHE_MAX=500` | Short lines she says often ("Paused.", "Volume 50 percent.") are kept as audio in `voice_cache\`, so they play at once instead of waiting 1–3 s for Microsoft. The common ones are made in the background when the server starts |
| `FAST_MUSIC_REPLIES=1` | "Pause", "next song", "play X" are answered by the laptop ("Here's Kesariya.") instead of a second Gemini call |
| `WHISPER_DEVICE=cuda` + `WHISPER_COMPUTE=int8_float16` | Hearing on the Quadro GPU instead of the CPU, usually 3–4x faster. Needs the CUDA 12 cuBLAS and cuDNN 9 libraries; if they're missing, the server says so and stays on the CPU |

The database isn't a bottleneck: all of its queries together take about 10–20 ms per question,
and memories and the music library are kept in RAM.

## Making replies faster

Friday no longer waits for Gemini to finish the whole reply before she starts talking. The answer is
cut into sentences as it arrives, the first one is spoken while the rest is still being written, and the
voice makes the next ones in the background so there's no gap. This works for spoken questions (`/ask`)
and typed ones (`/say`). [`docs/latency.md`](../docs/latency.md) has the measurements behind it and the
ideas that were left out on purpose.

| Setting | Default | Does |
|---|---|---|
| `STREAM_REPLIES` | `1` | Start speaking while Gemini is still writing. `0` = the old way, the whole reply first |
| `STREAM_HOLD_MS` | `1500` | A reply is started when Gemini has finished the round that writes it, or this many ms after the first sentence of that round, whichever comes first. Once it has started, a music command can't be added to it |
| `TOOL_FILLER` | `1` | Says "One sec, boss." while a slow tool (web search, a weather lookup that isn't cached) runs |
| `TTS_PARALLEL` | `3` | How many sentences of one reply the Edge voice makes at the same time. `1` = one after another |
| `STREAM_MAX_FAILS` | `3` | After this many turns in a row where streaming broke before sending anything, it switches itself off until the server restarts. Those turns were answered the old way |
| `GEMINI_HEDGE_S` | `3` | No first word from Gemini after this many seconds: ask again on a fresh connection and use whichever answers first. `0` = never |
| `GEMINI_KEEPALIVE_S` | `40` | A tiny request to Google when the line has been quiet this long, so its connection doesn't go stale. `0` = off |
| `WEATHER_CACHE_S`, `WEATHER_PREFETCH`, `WEATHER_TIMEOUT` | `600`, `1`, `4` | A weather lookup is reused for 10 minutes, your profile's city is refreshed in the background, and Open-Meteo gets 4 s before Friday gives up on it |
| `STT_WARMUP` | `1` | Runs Whisper once on a second of noise when the server starts, so the first question isn't the slow one |

`STREAM_REPLIES=0` brings back the old behaviour exactly. The server also falls back to it by itself for a
question where streaming fails before it has sent anything (the console says
`[stream] broke before sending anything (...)`, with the reason). `python server.py --test` always uses the old way.

### Measuring

With the server running, in another terminal on the same laptop:

```
python tools/bench_latency.py                    8 questions x 3 rounds, each asked with streaming off and on, in turns
python tools/bench_latency.py --runs 5 --gap 60  steadier, with a minute between questions so the connections go cold
python tools/bench_latency.py --tools            also questions that need the weather or a web search
python tools/bench_latency.py --wav q.wav        also a recording (that includes Whisper); --csv times.csv keeps every timing
python tools/diag_network.py                     what a new connection to Google, Open-Meteo and the Edge voice costs
python tools/diag_network.py --idle              and how long an idle connection to Google survives (about 8 minutes)
```

The benchmark prints the time to the first sound and to the end of each reply as median, slowest 10 % and
worst. Look at the slowest 10 %: that's what you notice. `diag_network.py` says in words what its numbers
mean, and suggests a `GEMINI_KEEPALIVE_S` if idle connections die on your network. Neither changes any setting.

Two request headers help when testing by hand (the speaker never sends them). `X-Stream: 0` or `1` decides
streaming for that one request, even after the automatic switch-off. `X-No-Log: 1` answers normally but keeps
the exchange out of the history and out of what she learns, which is how the benchmark avoids filling her
memory with its questions.

### Reading `latency.log`

One line per question, tab-separated, with a header row. (A log from an older version, with other columns,
is renamed to `latency.<date>.log` and a new one is started.)

| Column | Meaning |
|---|---|
| `first_audio_ms` | From the question reaching the laptop to the first sound leaving it. If she said "One sec, boss." first, that is the filler: `answer_ms` is when the answer began |
| `stt_ms`, `prompt_ms`, `gemini_ms`, `tool_ms`, `tts_ms` | Hearing, building the prompt, Gemini (`a+b` = several rounds), tools, and the voice (for a streamed reply: how long the first sentence took to become audio) |
| `quick` | 1 = a music command answered by the laptop without a second Gemini call |
| `stream` | 1 = the reply was streamed. Then, for a plain reply (no filler, no tool call), `first_audio_ms` is about `stt_ms` + `first_sentence_ms` + `tts_ms`; the console line shows any wait for the hold as "held back". `gemini_ms` is all of Gemini's writing, which carries on after that |
| `first_sentence_ms`, `ttft_ms` | Streamed only: Gemini's time to the first full sentence, and to its first word |
| `hedged` | Streamed only: 1 = the first word took so long that Friday asked again |
| `filler` | Streamed only: 1 = a filler line was spoken while a tool ran |
| `answer_ms` | Streamed only: when the answer itself started to sound: the first audio of the last round that was heard, so not the filler and not something said before a tool call |
| `rag_mode` | How the documents were used: `inject` (passages in the prompt, waited for before Gemini), `prefetch` (the tool answered from the background search), `tool` (the tool searched when asked); empty = not used |
| `rag_ms`, `embed_ms`, `search_ms` | How long the question waited for the documents (for `inject` before Gemini, otherwise inside `tool_ms`), and of that search, the question's embedding (`0` = cached) and the database call |

## Privacy

With `EARS=1` the laptop's mic is always on, but only to score each 80 ms for the wake word, on the laptop;
nothing is kept or sent until the wake word is heard. Recordings, transcripts and memories stay on this laptop; your uploaded documents live in your own Supabase
project (see [Your documents](#your-documents-cortex)). What leaves the laptop: the text of each question (plus the
profile, relevant memories and any matching document passages) goes to Gemini, each memory is sent once to Gemini's
embedding API to index it, and while you have documents uploaded almost every question of three or more words is
sent once to the embedding API too (the Cortex prefetch; with `CORTEX_PREFETCH=0` only questions about your
documents). Passwords, PINs and card numbers are refused.

## If something fails

| Problem | Fix |
|---|---|
| `[ears] off: openwakeword isn't installed` (or `sounddevice`) | `pip install -r requirements.txt` in the server's venv |
| `[ears] off: no microphone found` / `the mic has gone quiet` | Windows Settings → Privacy → Microphone: allow desktop apps. Check with `python -m ears.mic` |
| `[ears] heard "hey jarvis" ..., but the speaker isn't connected` | The S3's firmware needs `LAPTOP_MIC 1`, or test with `python client.py --listen` |
| She wakes up when nobody called her | Raise `WAKE_THRESHOLD` (0.7); keep the laptop away from the TV |
| She doesn't hear "hey jarvis" | `WAKE_DEBUG=1` shows how close you got; lower `WAKE_THRESHOLD` a little, or speak towards the laptop |
| She cuts you off mid-sentence | Raise `EARS_SILENCE_MS` (1200) |
| A command went to Gemini | `python intents.py "what you said"` shows why. Add the wording to `EXAMPLES` in `intents.py`, or lower `INTENT_MIN_SIMILARITY` a little |
| `[local] command model off (...)` | No internet for the first download, or Hugging Face blocked: exact phrases still work |
| `can't connect to PostgreSQL ... password authentication failed` | Wrong password in `DATABASE_URL`: use the one Adminer accepts |
| `can't connect to PostgreSQL ... Connection refused` | Docker Desktop isn't running, or the container is stopped: `cd db` then `docker compose start` |
| `GEMINI_API_KEY is not set` | `.env` missing or in the wrong folder (it must sit next to `server.py`) |
| `gemini 400/404 ... model` | Model name wrong for your key; try `GEMINI_MODEL=gemini-3.5-flash` |
| `[search] Google grounding refused ...` | Normal on a free key: searches go to DuckDuckGo. Set `SEARCH_PROVIDER=duckduckgo` to skip the Google attempt |
| `[tool] web_search(...) -> {"error": ...duckduckgo...}` | `pip install ddgs`, or DuckDuckGo is rate-limiting you: wait a minute |
| `time zone Asia/Kolkata not found` | `pip install tzdata` |
| The S3 never finds the laptop | Firewall rules above, network set to Private, same Wi-Fi. Check with `python client.py --find` from another PC |
| First answer slow | Whisper loads and warms up on start (`STT_WARMUP`), and the first Gemini call opens its connection: wait for `[stt] warmed up` and `[gemini] connection ready` |
| `[stream] broke before sending anything (...)` | Streaming failed before any sound went out, so that question was answered the old way. The text in brackets is the reason. After `STREAM_MAX_FAILS` in a row it stays off until the restart; `STREAM_REPLIES=0` turns it off for good |
| `[gemini] nothing after 3 s; asking again on a fresh connection` | Normal now and then: Gemini was slow to start, so Friday asked a second time and used whichever answered first. If it's every question, run `python tools/diag_network.py` |
| `[tts] stopped mid-reply: ...` | The voice failed after she had started speaking, so the reply ends where it stopped. The reason follows the colon (usually the internet) |
| `[music] 0 songs in the library` | Run `python music.py scan` |
| `ffmpeg not found` at start-up | `winget install Gyan.FFmpeg`, open a new terminal; or set `FFMPEG=C:\path\to\ffmpeg.exe` in `.env`. Without it MP3s still play, M4As don't |
| She plays the wrong song | Check the match with `python music.py list`; fix `title` / `artist` in Adminer |
