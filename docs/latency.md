# Why a reply takes the time it takes

Notes from tracing a normal chat reply, from the moment the laptop has the question to the first sound
at the speaker. They were written while building streamed replies (branch `perf/streaming-replies`), so
they say what was measured, what was changed, what was left alone on purpose, and how to find out
whether it helped on your network.

## What the old log showed

`latency.log` from the old folder: 11 typed questions in one evening. That is too few to prove anything,
but enough to say where to look.

| | Typical | Range |
|---|---|---|
| First audio, measured on the laptop | 4.6 s (median) | 2.4 s to 33.3 s |
| One Gemini call | 1.4 to 2.0 s | 1.0 s to 16.4 s |
| The voice's first chunk | 2.7 s (median) | 1.0 s to 3.7 s |
| Building the prompt (memory, history) | under 25 ms | |

Three things stand out.

1. **A plain chat reply is about 1.6 s of Gemini plus 2.7 s of voice.** The voice does not get faster
   when the reply is short: seven of the eleven first chunks took between 2.4 s and 2.9 s whatever was
   said. A cost that doesn't depend on the text is a start-up cost (reaching Microsoft's service and
   opening a connection), not the work of making the sound.
2. **Four of the eleven replies took 9 to 33 s.** Two were tool rounds, with 12.6 s and 3.1 s spent in
   tools; the likeliest cause of the longer one is a weather lookup that ran into its time-out and was then
   answered by a web search. The other two were Gemini calls of 11 to 16 s each, three in a row, the first
   one ten minutes after the previous question. That fits a dead connection that was only given up on after
   its full time-out, but it also fits a bad few minutes on the network or at Google. The log cannot tell
   which, so the changes below cover the first without assuming it, and `tools/diag_network.py --idle`
   finds out whether idle connections die on your network.
3. **Memory and the database are not the problem** (6 to 23 ms per question).

## What was changed

| Change | Aimed at | What to expect | Turn it off |
|---|---|---|---|
| Speak sentence by sentence while Gemini is still writing | Waiting for the whole reply before any sound | Gains grow with the length of the answer. A one-sentence reply gains almost nothing, because Gemini has finished by the time its first sentence is complete | `STREAM_REPLIES=0` |
| Make up to 3 sentences at once, play them in order | Gaps between sentences | No pause between sentences when the voice is slower than speech | `TTS_PARALLEL=1` |
| Ask Gemini again on a fresh connection if there is no first word after 3 s | The 11 to 16 s stalls | A stall costs about 3 s plus a normal call, not the whole time-out | `GEMINI_HEDGE_S=0` |
| TCP keep-alive, a tiny request to Google when the line has been quiet for 40 s, and the connection kept (not closed) after a streamed answer | Connections that die while idle, and a new secure connection for every question | The first question after a quiet spell is as fast as the others | `GEMINI_KEEPALIVE_S=0` (only the ping) |
| Remember each weather lookup for 10 minutes, refresh your home city in the background, give Open-Meteo 4 s | The 12.6 s weather round | "What's the weather" answered from memory in milliseconds | `WEATHER_CACHE_S=0`, `WEATHER_PREFETCH=0` |
| Say "One sec, boss." while a slow tool runs | Silence during a web search | Doesn't make it faster, makes the wait feel shorter | `TOOL_FILLER=0` |
| Run Whisper once at start-up | The slow first question after a restart | The model is awake before the first real recording | `STT_WARMUP=0` |
| Send small writes to the speaker at once (`TCP_NODELAY`) | The first bytes of a sentence waiting in the socket for the next packet or an acknowledgement | Tens of milliseconds at most. Python's `http.server` doesn't do this by itself; xiaozhi's asyncio server does | none |
| Replace a voice that has said nothing for `EDGE_TIMEOUT` (15 s) with the next one, before the reply starts | A request to Microsoft's service that hangs instead of failing, using up the 45 s the speaker waits for the laptop | The sentence is made by the local voice instead of the speaker giving up | `EDGE_TIMEOUT` |

The voice's 2.7 s start-up is the biggest single piece of a short reply, and **none of these changes
touch it**, except that for longer replies it now overlaps with Gemini's writing instead of following it.

### How streaming stays safe

The speaker reads a reply as one frame: a header (the text, the timings and the *action*, for example
"play track 12") followed by the audio. The header cannot change once it has been sent, and whether the
reply carries an action is only known when Gemini has finished the round that writes it. So:

- A reply is started when Gemini finishes the round without asking for a tool, or `STREAM_HOLD_MS` (1.5 s)
  after the first sentence of that round, whichever comes first. After that point a music command can no
  longer be added: the music tool answers that the reply has already started, and Gemini says so instead of
  promising something the speaker won't do.
- While a slow tool runs (web search, an uncached weather lookup) the filler line is spoken first, unless
  Microsoft's voice looks unreachable: the filler comes from the voice cache, which would make that voice
  the voice of the whole reply (MP3 and PCM can't be mixed in one reply). The server only learns that the
  voice is down from a request that fails, so the first tool question after an outage begins can still end
  after the filler; the next one is spoken by the local voice.
- If streaming breaks before a single byte has gone out, the turn is answered the old way, unless a tool had
  already run (then the error is returned rather than running the tool a second time). Only turns that the
  old way then answers count towards `STREAM_MAX_FAILS`; after that many in a row streaming switches itself
  off until the server is restarted. A failure the old way shares (Gemini down, say) is not blamed on
  streaming.
- If Gemini fails after the reply has started, she finishes with a short apology ("Sorry, something went
  wrong, boss.") rather than stopping mid-sentence. If the voice itself fails, the reply ends where it
  stopped (`[tts] stopped mid-reply` in the console) and Gemini is told to stop writing.
- If the speaker hangs up mid-reply, Gemini and the voice are told to stop.
- `X-Stream: 0` or `1` on a request overrides all of this for that request (even after the automatic
  switch-off). The benchmark uses it to ask the same question both ways.

## Left alone, on purpose

These might help, but they can't be judged from here: the machine this was built on can't reach Microsoft's
voice service or Gemini, and the Edge voice is an unofficial interface. The first two depend on one
comparison: how long `diag_network.py` says the voice takes to start a short sentence ("Good morning,
boss.") on your laptop, against the 2.7 s the server's own log showed.

| Idea | Build it if |
|---|---|
| Keep one event loop and one worker for the voice, instead of a new loop and a new thread for every sentence | `diag_network.py` shows the voice starting much faster than the server's log does (say under 1.2 s against 2.7 s). The time is then being lost inside the server, where it competes with Whisper and Gemini, and this is the cheap fix |
| Keep the Edge connection open between sentences | `diag_network.py` shows about the same 2 to 3 s as the server's log. The cost is then outside the server. The `edge-tts` package opens a new connection for every sentence and cannot reuse one, so this means speaking Microsoft's unofficial protocol ourselves: worth it only if that start-up is the thing you most want gone |
| Ask the voice twice when the first request is slow (like the Gemini retry) | `diag_network.py --repeat 10` shows an occasional very slow start (look at "worst") rather than a steady one |
| Speak the first sentence with the local Piper voice and the rest with Edge | You'd accept a different voice for the first sentence in exchange for a first sound well under 2 s |
| Lower `STREAM_HOLD_MS` | The benchmark shows long answers waiting for the hold, and you rarely say "play" after a long question |

## Compared with xiaozhi

xiaozhi-esp32 and its server are open source and the best-known ESP32 voice assistant, so their code is the
obvious thing to compare with. Reading it:

| | xiaozhi (default setup) | Friday |
|---|---|---|
| Connection | One WebSocket per session, Opus frames both ways | One HTTP request per question, the reply streamed back as one body |
| Microphone | Streams live while you talk; the server finds the end of speech | The recording is uploaded after the button is released |
| Answer to speech | LLM streamed, cut at punctuation, spoken segment by segment | The same, on this branch |
| Voice | EdgeTTS, one whole segment at a time. Only the "double-stream" cloud voices keep a connection open | Edge, up to 3 sentences at once, the local voice as fallback |
| Pacing | First 5 frames (0.3 s) at once, then one every 60 ms | The speaker buffers 0.5 s, then plays |
| Server sockets | asyncio, `TCP_NODELAY` by default | Set explicitly now |

With its default pieces xiaozhi is not faster than this branch. Its advantages are live microphone audio and,
when configured, streaming voice services. What those would cost here (the gains are estimates):

| Idea | Gain | Cost |
|---|---|---|
| Stream the microphone while the button is held | 0.2 to 0.5 s (the upload after release) | Firmware and server change |
| A voice that starts faster than Edge's 2.7 s | The largest single piece of a short reply | See the table above; run `diag_network.py` first |
| Take the music action out of the audio header, so a reply never waits for `STREAM_HOLD_MS` | Up to 1.5 s on tool and music answers | Firmware change |
| `WHISPER_DEVICE=cuda` | Only if `stt_ms` is high in `latency.log` | An NVIDIA GPU with CUDA 12 |

## Finding out if it helped

On the laptop that runs the server, with the server running:

```
python tools/diag_network.py --idle     # about 8 minutes: what a new connection costs, and how long idle ones survive
python tools/bench_latency.py --runs 5  # the same questions with streaming off and on, taken in turns
```

The benchmark reports the time to the first sound and to the end of each reply as median, slowest 10 % and
worst. Look at the worst and the slowest 10 %, not only the median: the stalls are what you notice.
Its questions are sent with `X-No-Log: 1`, so they don't end up in the history or in her memory.

## What has not been checked

All of this is built and tested against stand-ins for Gemini, the voice and Whisper, including real HTTP
between a fake speaker and the server (182 tests, `python -m unittest discover -s tests` in `server/`).
It has not been run against the real services or with the speaker, so every "what to expect" above is an
expectation, not a measurement. If anything sounds wrong on the first real run, set `STREAM_REPLIES=0`
and she behaves exactly as she did before.
