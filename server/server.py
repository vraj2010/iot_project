"""FRIDAY local server: the laptop does the hearing, the voice and the memory;
Gemini does the thinking.

It speaks exactly the same HTTP protocol as the Cloudflare Worker (/ask, /say,
/chat, /health), so the S3 can use either one. The S3 finds this laptop by UDP
broadcast and falls back to the Worker when the laptop is off. Music is laptop
only: GET /music/<id> and /music/next stream songs from your library.

The laptop can also be Friday's ears (ears/): it listens for the wake word on its
own mic, and the S3 collects what was said with GET /listen. Simple commands
("pause", "volume 40", "play Kesariya") are answered here without Gemini (intents.py).

    python server.py                       run it
    python server.py --test "hi there"     check Gemini, memory and the voice, no hardware
    python server.py --test-wav q.wav      the whole pipeline on a recording
    python server.py --test "play kesariya" --music   as if the speaker can play music
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import numpy as np

import config
import frames
import music
from audio_utils import bytes_to_pcm, load_wav, normalize, resample, save_wav, split_sentences
from replystream import StreamingReply

HERE = Path(__file__).resolve().parent
NOT_HEARD_REPLY = "Sorry, I didn't catch that. Could you say it again?"
# Said while a slow tool runs (web search, weather), and when a streamed reply breaks off half way
FILLERS = ["One sec, boss.", "Let me check.", "Checking that now.", "Just a moment.", "On it, boss."]
ERROR_LINE = "Sorry, something went wrong, boss."
QUICK_LINES = (["Paused.", "Back on.", "Stopped.", "It's already playing.", "Nothing was playing.", "From the top.",
                NOT_HEARD_REPLY, ERROR_LINE] + FILLERS + [f"Volume {n} percent." for n in range(0, 101, 5)])
MAX_BODY = config.MAX_AUDIO_SECONDS * config.SAMPLE_RATE * 2 + 64 * 1024
CHUNK = 16 * 1024


class TurnError(Exception):
    """A turn that failed; becomes a JSON error reply the S3 can print."""

    def __init__(self, message: str, status: int = 502, **extra) -> None:
        super().__init__(message)
        self.status = status
        self.extra = extra


# ====================================================================
#  The assistant: hearing -> thinking -> voice
# ====================================================================

class Assistant:
    def __init__(self, with_stt: bool = True) -> None:
        from brain import Brain
        from memory import Background, Memory
        from tts import TTS

        self.turn_lock = threading.Lock()      # one turn at a time: there is one speaker
        self.stream_fails = 0                  # streamed turns in a row that broke before sending anything and
                                               # were then answered the normal way
        self.memory = Memory()
        self.brain = Brain(self.memory)
        self.background = Background()
        self.tts = TTS()
        self.tts.warm_cache(QUICK_LINES)        # "Paused.", "Volume 50 percent."... ready before they're needed
        self.jukebox = None
        try:
            from jukebox import Jukebox

            self.jukebox = Jukebox(self.memory)
        except Exception as exc:              # music is optional: never stop Friday starting
            print(f"[music] off: {type(exc).__name__}: {exc}")
        self.router = None
        if config.LOCAL_COMMANDS:
            from intents import Router

            self.router = Router(load_model=config.INTENT_MODEL)   # the model loads in the background
        self.stt = None
        if with_stt:
            from stt import STT

            self.stt = STT()
            if config.STT_WARMUP:
                self.stt.warmup()

    # ------------------------------------------------------------ pieces

    def hear(self, pcm: np.ndarray) -> tuple[str, str, int]:
        """Returns (cleaned transcript, raw transcript, ms)."""
        from stt import clean_transcript

        t0 = time.time()
        raw = self.stt.transcribe(normalize(pcm))
        import langs

        langs.set_turn(self.stt.language)            # the reply is in the language that was spoken
        return clean_transcript(raw), raw, int((time.time() - t0) * 1000)

    def think(self, heard: str, music_state: str | None = None) -> dict:
        """music_state: the speaker's X-Music-State header (None = it can't play music)."""
        import gemini

        jb = self.jukebox if self.jukebox and self.jukebox.observe(music_state) else None
        try:
            return self.brain.answer(heard, jb)
        except gemini.GeminiError as exc:
            raise TurnError(str(exc), exc.status, heard=heard) from None

    def local(self, heard: str, music_state: str | None = None) -> dict | None:
        """A command answered on the laptop without Gemini ("pause", "volume 40"), as a turn result; None = ask
        Gemini. Never raises: a broken local command must not stop the question from being answered."""
        router = getattr(self, "router", None)
        if router is None or not heard:
            return None
        t0 = time.time()
        try:
            jb = self.jukebox if self.jukebox and self.jukebox.observe(music_state) else None
            hit = router.route(heard, jb)
        except Exception as exc:
            print(f"[local] {type(exc).__name__}: {exc}; asking Gemini instead")
            return None
        if hit is None:
            return None
        ms = int((time.time() - t0) * 1000)
        return {"reply": hit.reply, "ms": ms, "actions": hit.action, "citations": [], "local": hit,
                "tools": [{"tool": f"local:{hit.intent}", "args": {}, "ms": ms, "result": json.dumps(hit.result)[:300]}],
                "timing": {"prompt_ms": 0, "gemini_ms": [], "tool_ms": ms, "fast": True, "local": True}}

    def stream_enabled(self) -> bool:
        return config.STREAM_REPLIES and self.stream_fails < config.STREAM_MAX_FAILS

    def think_stream(self, heard: str, music_state: str | None, sink) -> dict:
        """think(), but each sentence goes to `sink` while Gemini is still writing the rest."""
        import gemini

        jb = self.jukebox if self.jukebox and self.jukebox.observe(music_state) else None
        try:
            return self.brain.answer_stream(heard, jb, sink)
        except gemini.GeminiError as exc:
            raise TurnError(str(exc), exc.status, heard=heard) from None

    def voice_start(self, reply: str) -> tuple:
        """(backend, first audio bytes, rest iterator, ms); raises TTSError."""
        return self.tts.start(reply)

    def after_turn(self, heard: str, result: dict) -> None:
        """Log the exchange now (fast), learn from it later (slow, background)."""
        cx = result.get("cortex")
        if heard and cx:                                        # a document answer: for "Recent answers"
            import cortex

            if cortex.should_log(result):                       # cited something, nothing failed
                latency = dict(cx.get("latency") or {})
                latency["first_audio_ms"] = (result.get("timing") or {}).get("first_audio_ms")
                cortex.queue_answer(heard, result["reply"], result.get("citations") or [], cx.get("mode"), latency)
        if not heard or not result.get("episode_id"):
            return
        try:
            self.memory.log_turn(result["episode_id"], heard, result["reply"], result.get("tools", []),
                                 result.get("ms", 0))
        except Exception as exc:
            print(f"[memory] couldn't log the turn: {exc}")
        explicit = any(t["tool"] in ("save_memory", "forget_memory", "update_profile")
                       for t in result.get("tools", []))
        if not explicit:
            self.background.submit(self.memory.learn_from_turn, heard, result["reply"])
        if result.get("new_session") and result.get("previous_id"):
            self.background.submit(self.memory.summarise_episode, result["previous_id"])


# ====================================================================
#  HTTP
# ====================================================================

def read_audio(body: bytes) -> np.ndarray:
    """Raw 16 kHz s16le mono from the S3, or a 16-bit WAV file (laptop tests)."""
    if body[:4] == b"RIFF" and body[8:12] == b"WAVE":
        import io
        import wave

        with wave.open(io.BytesIO(body), "rb") as w:
            if w.getsampwidth() != 2:
                raise TurnError("WAV must be 16-bit", 400)
            ch, rate = w.getnchannels(), w.getframerate()
            pcm = bytes_to_pcm(w.readframes(w.getnframes()))
        if ch > 1:
            pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16)
        pcm = resample(pcm, rate, config.SAMPLE_RATE)
    else:
        pcm = bytes_to_pcm(body)
    seconds = pcm.size / config.SAMPLE_RATE
    if seconds < config.MIN_AUDIO_SECONDS:
        raise TurnError("audio too short", 400)
    if seconds > config.MAX_AUDIO_SECONDS:
        raise TurnError(f"audio longer than {config.MAX_AUDIO_SECONDS} s", 400)
    return pcm


class _HttpTransport:
    """How a StreamingReply writes its response: a chunked 200, started when the first audio is ready."""

    def __init__(self, handler: "Handler") -> None:
        self.h = handler

    def begin(self) -> None:
        h = self.h
        h.send_response(200)
        h.send_header("Content-Type", "application/octet-stream")
        h.send_header("Transfer-Encoding", "chunked")
        h.end_headers()

    def write(self, data: bytes) -> None:
        self.h._chunk(data)

    def end(self) -> None:
        self.h.wfile.write(b"0\r\n\r\n")
        self.h.wfile.flush()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"          # keep-alive: the S3 reuses one connection
    server_version = "nexa-local/1.0"
    disable_nagle_algorithm = True         # TCP_NODELAY: a small write (the header, the end of a sentence) goes out
                                           # now, not after the S3 has got round to acknowledging the one before
    assistant: Assistant = None            # set in main()
    ears = None                            # ears.Ears when the laptop's mic is on (set in main())

    def log_message(self, fmt, *args) -> None:   # quiet; turns print their own summary
        pass

    # ------------------------------------------------------------ replies

    def _json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _chunk(self, data: bytes) -> None:
        for i in range(0, len(data), CHUNK):
            part = data[i:i + CHUNK]
            self.wfile.write(f"{len(part):X}\r\n".encode() + part + b"\r\n")
        self.wfile.flush()

    def _stream(self, header: bytes, first: bytes, rest, backend, citations=None) -> bytes:
        """Send the frame header with the first audio straight away, then the
        rest of the reply (same voice) as soon as each piece is ready."""
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Transfer-Encoding", "chunked")
        if citations:
            self.send_header("X-Citations", _citations_header(citations))
        self.end_headers()
        sent = [first]
        self._chunk(header + first)
        self.first_audio_ms = int((time.time() - getattr(self, "t_start", time.time())) * 1000)
        while True:
            try:
                data = next(rest)
            except StopIteration:
                break
            except Exception as exc:
                print(f"[tts] stopped mid-reply: {exc}")
                break
            self._chunk(data)
            sent.append(data)
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
        return b"".join(sent)

    # ------------------------------------------------------------ routes

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path.startswith("/music/"):
            try:
                return self._music(url)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return                         # the speaker skipped or paused: normal
        if url.path == "/listen":
            if not self._token_ok():
                return self._json({"error": "bad or missing X-Device-Token"}, 401)
            try:
                return self._listen(url)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return                         # the speaker went away while waiting: it polls again
        if url.path != "/health":
            return self._json({"error": "not found"}, 404)
        a = self.assistant
        jb = a.jukebox
        router = getattr(a, "router", None)
        self._json({"ok": True, "server": "nexa-local", "model": config.GEMINI_MODEL,
                    "voice": a.tts.name, "stt": config.WHISPER_MODEL if a.stt else None,
                    "facts": a.memory.count_facts(), "web_search": config.WEB_SEARCH, "stream": a.stream_enabled(),
                    "documents": _documents_health(),
                    "music": {"songs": jb.count(), "ffmpeg": bool(jb.ffmpeg)} if jb else None,
                    "ears": self.ears.status() if self.ears else "off",
                    "local_commands": ("phrases + model" if router.embedder else "phrases") if router else "off"})

    # ------------------------------------------------------------ the laptop's ears

    def _token_ok(self) -> bool:
        return bool(config.DEVICE_TOKEN) and self.headers.get("X-Device-Token", "") == config.DEVICE_TOKEN

    def _busy(self):
        """Around every turn: the laptop's ears ignore the wake word while Friday may be talking."""
        return self.ears.busy() if self.ears else nullcontext()

    def _listen(self, url) -> None:
        """GET /listen: the speaker waiting for the laptop's ears.

        /listen          idle poll: held up to LISTEN_HOLD_S. 202 {"event": "wake", "id": n} as soon as the wake
                         word is heard (or BOOT asked for it), 204 when nothing happened (poll again).
        /listen?wake=n   the speaker has chimed and turned the music down: held until you stop talking, then
                         answered exactly like /ask (200 + reply frame). 202 {"event": "cancel"} if nothing was said.
        """
        ears = self.ears
        if ears is None:
            return self._json({"error": "the laptop's mic is off (EARS=0 in .env, or it failed at start-up: "
                                        "see the server window)"}, 503)
        wake = (parse_qs(url.query).get("wake") or [""])[0]
        if not wake:
            wake_id = ears.wait_for_wake(config.LISTEN_HOLD_S)
            if wake_id is None:
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            return self._json({"event": "wake", "id": wake_id, "source": ears.source}, 202)
        try:
            wake_id = int(wake)
        except ValueError:
            return self._json({"error": "wake must be a number"}, 400)
        pcm = ears.wait_for_question(wake_id, config.EARS_START_S + config.EARS_MAX_S + 3)
        if pcm is None:
            return self._json({"event": "cancel"}, 202)
        self.t_start = time.time()             # the question is complete: the speaker is now waiting
        self.music_state = self.headers.get("X-Music-State")
        with ears.busy(), self.assistant.turn_lock:
            self._turn(lambda: self._answer_audio(pcm, debug=False))

    # ------------------------------------------------------------ music

    def _music(self, url) -> None:
        """GET /music/<id>?t=<seconds> or /music/next?after=<id>: one song, 16 kHz mono MP3."""
        if not config.DEVICE_TOKEN or self.headers.get("X-Device-Token", "") != config.DEVICE_TOKEN:
            return self._json({"error": "bad or missing X-Device-Token"}, 401)
        jb = self.assistant.jukebox
        if not jb:
            return self._json({"error": "music is off on the laptop (see the [music] line at start-up)"}, 503)
        q = parse_qs(url.query)
        name = url.path[len("/music/"):].strip("/")
        try:
            if name == "next":
                after = int((q.get("after") or ["0"])[0] or 0) or None
                track_id, start = jb.next_after(after), 0.0
                if track_id is None:
                    self.send_response(204)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    print("[music] queue finished")
                    return
            else:
                track_id = int(name)
                start = max(0.0, float((q.get("t") or ["0"])[0] or 0))
        except ValueError:
            return self._json({"error": "bad track"}, 400)

        t = jb.track(track_id)
        src = jb.source_file(t) if t else None
        if not src:
            return self._json({"error": f"song {track_id} is not in the library (or its file is gone)"}, 404)
        if start < 1:
            jb.played(track_id)
        print(f"[music] {'playing' if start < 1 else f'resuming at {int(start)} s:'} {jb.label(t)}")

        head = {"X-Track-Id": str(track_id), "X-Track-Title": quote(jb.label(t)[:120]),
                "X-Track-Duration": str(t.get("duration_s") or 0)}
        if jb.ffmpeg:
            cache = jb.cache_file(t)
            if cache.exists():
                size = cache.stat().st_size
                offset = min(size, int(start * _bitrate_bytes()))
                return self._send_file(cache, offset, size, head)
            return self._send_ffmpeg(src, start, cache, head)
        span = music.mp3_span(src)
        if not span:
            return self._json({"error": f"{src.name} isn't an MP3; install ffmpeg to play it"}, 503)
        first, end = span
        if start > 0 and t.get("duration_s"):
            first = min(end, first + int((end - first) * start / t["duration_s"]))
        self._send_file(src, first, end, head)

    def _audio_headers(self, head: dict, length: int | None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "audio/mpeg")
        for k, v in head.items():
            self.send_header(k, v)
        if length is None:
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header("Content-Length", str(length))
        self.end_headers()

    def _send_file(self, path: Path, first: int, end: int, head: dict) -> None:
        """Bytes [first, end) of a file. The speaker reads at its own pace."""
        self._audio_headers(head, end - first)
        with open(path, "rb") as f:
            f.seek(first)
            left = end - first
            while left > 0:
                data = f.read(min(CHUNK, left))
                if not data:
                    break
                self.wfile.write(data)
                left -= len(data)
        self.wfile.flush()

    def _send_ffmpeg(self, src: Path, start: float, cache: Path, head: dict) -> None:
        """Convert while sending (first play of a song). From the start, the result is
        also saved to the cache, so the next play skips ffmpeg."""
        proc = subprocess.Popen(music.ffmpeg_cmd(src, start), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        tee, tmp = None, None
        try:
            first = proc.stdout.read1(CHUNK) if hasattr(proc.stdout, "read1") else proc.stdout.read(CHUNK)
            if not first:
                err = proc.stderr.read().decode(errors="replace").strip()[:200]
                proc.wait()
                print(f"[music] ffmpeg couldn't convert {src.name}: {err}")
                return self._json({"error": f"ffmpeg failed: {err}"}, 500)
            if start < 1:
                cache.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache.with_suffix(f".{os.getpid()}.{threading.get_ident()}.part")
                tee = open(tmp, "wb")
            self._audio_headers(head, None)
            data = first
            while data:
                if tee:
                    tee.write(data)
                self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                data = proc.stdout.read1(CHUNK) if hasattr(proc.stdout, "read1") else proc.stdout.read(CHUNK)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            ok = proc.wait() == 0
            if tee:
                tee.close()
                tee = None
                if ok:
                    try:
                        os.replace(tmp, cache)
                        tmp = None
                    except OSError:            # Windows: another copy is being written or read
                        pass
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if tee:
                tee.close()
            if tmp:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

    def do_POST(self) -> None:
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self.close_connection = True
            return self._json({"error": "request too large"}, 413)
        body = self.rfile.read(length) if length else b""
        self.t_start = time.time()             # the question is in: the speaker is now waiting

        if url.path not in ("/ask", "/say", "/chat", "/listen/wake"):
            return self._json({"error": "not found"}, 404)
        if not self._token_ok():
            return self._json({"error": "bad or missing X-Device-Token"}, 401)
        if url.path == "/listen/wake":         # BOOT on the speaker, with the laptop as its ears
            if self.ears is None:
                return self._json({"error": "the laptop's mic is off"}, 503)
            if self.ears.wake_now():
                return self._json({"ok": True}, 202)
            return self._json({"error": "busy: Friday is answering"}, 409)

        self.music_state = self.headers.get("X-Music-State")   # sent by speakers that can play music
        with self._busy(), self.assistant.turn_lock:
            self._turn(lambda: self._post_turn(url, body))

    def _post_turn(self, url, body: bytes) -> None:
        if url.path == "/ask":
            return self._ask(body, parse_qs(url.query).get("debug") == ["1"])
        try:
            text = str(json.loads(body or b"{}").get("text", "")).strip()
        except ValueError:
            raise TurnError('body must be JSON like {"text": "..."}', 400) from None
        if not text:
            raise TurnError("text is empty", 400)
        if len(text) > config.MAX_INPUT_CHARS:
            raise TurnError(f"text longer than {config.MAX_INPUT_CHARS} characters", 400)
        (self._chat if url.path == "/chat" else self._say)(text)

    def _turn(self, run) -> None:
        """Run one turn, turning its failures into replies the speaker can print."""
        try:
            run()
        except TurnError as exc:
            print(f"[turn] failed: {exc}")
            self._json({"error": str(exc), **exc.extra}, exc.status)
        except (BrokenPipeError, ConnectionResetError):
            print("[turn] the speaker hung up mid-reply")
        except Exception as exc:
            print(f"[turn] crashed: {type(exc).__name__}: {exc}")
            try:
                self._json({"error": f"server error: {exc}"}, 500)
            except OSError:
                pass

    def _first_voice(self, reply: str, **extra) -> tuple:
        try:
            return self.assistant.voice_start(reply)
        except Exception as exc:
            raise TurnError(f"voice: {exc}", 502, reply=reply, **extra) from None

    def _frame(self, heard: str | None, text: str, chat_ms: int, actions: dict | None, backend, tts_ms: int,
               stt_ms: int) -> bytes:
        """I2S4 (with the music action) for speakers that can play music, else the old frames."""
        if self.music_state is not None:
            return frames.reply_header(heard, text, backend.provider, chat_ms, tts_ms, stt_ms, backend.fmt, actions)
        if heard is None:
            return frames.say_header(text, backend.provider, chat_ms, tts_ms, backend.fmt)
        return frames.ask_header(heard, text, backend.provider, chat_ms, tts_ms, stt_ms, backend.fmt)

    def _header(self, heard: str | None, result: dict, backend, tts_ms: int, stt_ms: int) -> bytes:
        return self._frame(heard, result["reply"], result["ms"], result.get("actions"), backend, tts_ms, stt_ms)

    # ------------------------------------------------------------ a reply that starts while Gemini is still writing

    def _wants_stream(self) -> bool:
        """An X-Stream: 0 or 1 header on the request decides (that's how the two ways are compared);
        otherwise STREAM_REPLIES, unless streaming has broken too often in a row and switched itself off."""
        flag = (self.headers.get("X-Stream") or "").strip().lower()
        if flag:
            return flag not in ("0", "false", "no", "off")
        return self.assistant.stream_enabled()

    def _turn_streamed(self, heard: str, stt_ms: int, typed: bool) -> bool:
        """Answer `heard` with the audio starting while Gemini is still writing.

        True: the turn is dealt with (answered, or failed with an error the caller has been given).
        False: the streamed path broke before it had sent anything; answer the normal way instead."""
        from brain import TurnCancelled

        a = self.assistant
        shown = None if typed else heard
        reply = StreamingReply(
            a.tts.session(), _HttpTransport(self),
            lambda voice, text, chat_ms, tts_ms, actions: self._frame(shown, text, chat_ms, actions, voice, tts_ms,
                                                                      stt_ms),
            a.brain.actions, offset_ms=int((time.time() - self.t_start) * 1000),
            hold_s=config.STREAM_HOLD_MS / 1000.0, fillers=FILLERS if config.TOOL_FILLER else (),
            apology=ERROR_LINE, filler_ok=a.tts.filler_ok)
        reply.start()
        try:
            try:
                result = a.think_stream(heard, self.music_state, reply)
            except TurnCancelled:
                print("[turn] the speaker hung up mid-reply" if reply.dead
                      else "[turn] the voice stopped mid-reply, so Gemini was stopped too")
                return True
            except TurnError as exc:
                reply.finish(error=exc)
                reply.wait()
                if reply.dead:
                    print("[turn] the speaker hung up mid-reply")
                    return True
                if not reply.started:
                    raise                      # nothing reached the speaker: the caller answers with the error
                print(f"[turn] failed after the reply had started: {exc}")
                return True
            except Exception as exc:
                reply.finish(error=exc)
                reply.wait()
                if reply.dead or reply.started:
                    print(f"[turn] the streamed reply broke after it had started: {type(exc).__name__}: {exc}")
                    return True
                if reply.tools_ran:
                    raise                      # the tools have run: asking the question again would repeat them
                print(f"[stream] broke before sending anything ({type(exc).__name__}: {exc}); "
                      f"answering the normal way")
                return False
            reply.finish()
            reply.wait()
        finally:
            reply.close()

        if not reply.started and not reply.dead:       # nothing could be said, so the speaker gets the text
            raise TurnError(f"voice: {reply.voice_error or 'nothing was said'}", 502, reply=result["reply"],
                            action=result.get("actions") or {}, **({} if typed else {"heard": heard}))
        a.stream_fails = 0
        if reply.dead:
            print("[turn] the speaker hung up mid-reply")
        else:
            timing = result.get("timing") or {}
            timing["first_audio_ms"] = reply.first_audio_ms
            _latency("typed" if typed else "spoken", heard, stt_ms, result, reply.tts_ms or 0,
                     reply.first_audio_ms or 0,
                     {"first_sentence_ms": timing.get("first_sentence_ms"), "ttft_ms": timing.get("ttft_ms"),
                      "hedged": timing.get("hedged"), "filler": reply.filler, "answer_ms": reply.answer_ms})
            print(f"[turn] {'typed' if typed else 'heard'} \"{heard}\" | "
                  f"{'' if typed else f'Whisper {stt_ms} ms | '}Gemini {result['ms']} ms {_tools(result)}| "
                  f"{reply.voice.name} {reply.tts_ms} ms (streamed)\nnexa> {result['reply']}{_action(result)}"
                  f"{_cites(result)}")
            _save_reply(bytes(reply.audio), reply.voice.fmt)
        self._after(heard, result)
        return True

    def _stream_failed(self) -> None:
        """A streamed turn broke before it sent anything, and the normal way has now answered the same question:
        that points at streaming. Enough of these in a row and it switches itself off."""
        a = self.assistant
        a.stream_fails += 1
        if not a.stream_enabled():
            print("[stream] this keeps happening; streaming is off until the server is restarted")

    def _after(self, heard: str, result: dict) -> None:
        """Log the exchange and learn from it, unless the request says it's a test (X-No-Log: 1):
        the benchmark asks dozens of questions and shouldn't fill the history and the memory with them."""
        if (self.headers.get("X-No-Log") or "").strip() == "1":
            return
        self.assistant.after_turn(heard, result)

    def _local_turn(self, heard: str, stt_ms: int, typed: bool, result: dict) -> None:
        """Speak a command the laptop answered by itself: no Gemini, no memory (it's only "pause")."""
        backend, first, rest, tts_ms = self._first_voice(result["reply"], action=result.get("actions") or {},
                                                         **({} if typed else {"heard": heard}))
        header = self._header(None if typed else heard, result, backend, tts_ms, stt_ms)
        audio = self._stream(header, first, rest, backend)
        result["timing"]["first_audio_ms"] = self.first_audio_ms
        _latency("local", heard, stt_ms, result, tts_ms, self.first_audio_ms)
        hit = result["local"]
        print(f"[turn] {'typed' if typed else 'heard'} \"{heard}\" | {'' if typed else f'Whisper {stt_ms} ms | '}"
              f"local {hit.intent} ({hit.how}) {result['ms']} ms, no Gemini | {backend.name} {tts_ms} ms\n"
              f"nexa> {result['reply']}{_action(result)}")
        _save_reply(audio, backend.fmt)

    def _chat(self, text: str) -> None:
        local = self.assistant.local(text, self.music_state)
        if local:
            self._json({"reply": local["reply"], "ms": local["ms"], "tools": [t["tool"] for t in local["tools"]],
                        "action": local["actions"], "citations": []})
            print(f"[turn] typed \"{text}\" -> {local['reply']} (local {local['local'].intent}, no Gemini)")
            return
        result = self.assistant.think(text, self.music_state)
        self._json({"reply": result["reply"], "ms": result["ms"], "tools": [t["tool"] for t in result["tools"]],
                    "action": result.get("actions") or {}, "citations": result.get("citations") or []})
        print(f"[turn] typed \"{text}\" -> {result['reply']}{_cites(result)}")
        self._after(text, result)

    def _say(self, text: str) -> None:
        local = self.assistant.local(text, self.music_state)
        if local:
            return self._local_turn(text, 0, True, local)
        fell_back = False
        if self._wants_stream():
            if self._turn_streamed(text, 0, typed=True):
                return
            fell_back = True
        result = self.assistant.think(text, self.music_state)
        if fell_back:
            self._stream_failed()
        backend, first, rest, tts_ms = self._first_voice(result["reply"], action=result.get("actions") or {})
        header = self._header(None, result, backend, tts_ms, 0)
        audio = self._stream(header, first, rest, backend, result.get("citations"))
        (result.get("timing") or {})["first_audio_ms"] = self.first_audio_ms
        _latency("typed", text, 0, result, tts_ms, self.first_audio_ms)
        print(f"[turn] typed \"{text}\" | Gemini {result['ms']} ms {_tools(result)}| {backend.name} {tts_ms} ms\n"
              f"nexa> {result['reply']}{_action(result)}{_cites(result)}")
        _save_reply(audio, backend.fmt)
        self._after(text, result)

    def _ask(self, body: bytes, debug: bool) -> None:
        self._answer_audio(read_audio(body), debug)

    def _answer_audio(self, pcm: np.ndarray, debug: bool) -> None:
        """A spoken question, recorded by the speaker (/ask) or by the laptop's own mic (/listen)."""
        a = self.assistant
        _save("last_question.wav", pcm)
        heard, raw, stt_ms = a.hear(pcm)

        local = None if debug else a.local(heard, self.music_state)
        if local:
            return self._local_turn(heard, stt_ms, False, local)

        fell_back = False
        if heard and not debug and self._wants_stream():
            if self._turn_streamed(heard, stt_ms, typed=False):
                return
            fell_back = True

        result = {"reply": NOT_HEARD_REPLY, "ms": 0, "tools": [], "actions": {}}
        if heard:
            result = a.think(heard, self.music_state)
            if fell_back:
                self._stream_failed()

        if debug:
            self._json({"heard": heard, "raw_transcript": raw, "reply": result["reply"], "stt_ms": stt_ms,
                        "chat_ms": result["ms"], "tools": result["tools"], "citations": result.get("citations") or [],
                        "seconds": round(pcm.size / config.SAMPLE_RATE, 2)})
        else:
            backend, first, rest, tts_ms = self._first_voice(result["reply"], heard=heard,
                                                             action=result.get("actions") or {})
            header = self._header(heard, result, backend, tts_ms, stt_ms)
            audio = self._stream(header, first, rest, backend, result.get("citations"))
            if result.get("timing") is not None:
                result["timing"]["first_audio_ms"] = self.first_audio_ms
            _latency("spoken", heard, stt_ms, result, tts_ms, self.first_audio_ms)
            _save_reply(audio, backend.fmt)
            print(f"[turn] heard \"{heard or '(nothing clear)'}\" | Whisper {stt_ms} ms | Gemini {result['ms']} ms "
                  f"{_tools(result)}| {backend.name} {tts_ms} ms\nnexa> {result['reply']}{_action(result)}"
                  f"{_cites(result)}")
        self._after(heard, result)


LATENCY_COLUMNS = ("time", "kind", "first_audio_ms", "stt_ms", "prompt_ms", "gemini_ms", "tool_ms", "tts_ms", "quick",
                   "stream", "first_sentence_ms", "ttft_ms", "hedged", "filler", "answer_ms",
                   "rag_mode", "rag_ms", "embed_ms", "search_ms", "said")


def _latency(kind: str, said: str, stt_ms: int, result: dict, tts_ms: int, first_audio_ms: int,
             stream: dict | None = None) -> None:
    """Where the time went, on the console and as one line in latency.log.
    `stream` is given for a reply that started while Gemini was still writing."""
    t = result.get("timing") or {}
    rounds = t.get("gemini_ms") or []
    gemini = "+".join(map(str, rounds)) or "0"
    s = stream or {}
    if t.get("local"):
        line = (f"[latency] {first_audio_ms} ms to first audio = hearing {stt_ms} + local command "
                f"{t.get('tool_ms', 0)} (no Gemini) + voice {tts_ms}")
    elif stream is None:
        line = (f"[latency] {first_audio_ms} ms to first audio = hearing {stt_ms} + memory/prompt "
                f"{t.get('prompt_ms', 0)} + Gemini {gemini}"
                f"{' (quick reply)' if t.get('fast') else ''} + tools {t.get('tool_ms', 0)} + voice {tts_ms}")
    elif s.get("filler"):
        # The first audio was the filler line, said once Gemini's first round (the one that asked for the tool)
        # was over; the first sentence of the answer came after the tool.
        line = (f"[latency] {first_audio_ms} ms to first audio (streamed, the filler {s['filler']!r}) = hearing "
                f"{stt_ms} + memory/prompt {t.get('prompt_ms', 0)} + Gemini's first round "
                f"{rounds[0] if rounds else 0} + voice {tts_ms}; the answer itself from {s.get('answer_ms')} ms "
                f"(Gemini {s.get('first_sentence_ms')} to its first sentence, first word {s.get('ttft_ms')}"
                f"{', retried' if s.get('hedged') else ''}); Gemini {gemini} and tools {t.get('tool_ms', 0)} in all")
    else:
        held = first_audio_ms - stt_ms - (s.get("first_sentence_ms") or 0) - tts_ms
        line = (f"[latency] {first_audio_ms} ms to first audio (streamed) = hearing {stt_ms} + Gemini "
                f"{s.get('first_sentence_ms')} to the first sentence (memory/prompt {t.get('prompt_ms', 0)}, "
                f"first word {s.get('ttft_ms')}{', retried' if s.get('hedged') else ''}) + voice {tts_ms}")
        if held > 150:                      # the reply waited to see whether a tool call followed (or for a voice)
            line += f" + held back {held}"
        line += (f"; the answer itself from {s.get('answer_ms')} ms; "
                 f"Gemini {gemini} and tools {t.get('tool_ms', 0)} in all")
    if t.get("rag_mode"):
        where = "before Gemini, not in memory/prompt" if t["rag_mode"] == "inject" else "inside tools"
        line += (f"; documents ({t['rag_mode']}) waited {t.get('rag_ms', 0)} ms {where} "
                 f"[embed {_ms(t.get('embed_ms'))}, search {'-' if t.get('search_ms') is None else t['search_ms']} ms]")
    print(line)
    if not config.LATENCY_LOG:
        return
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "first_audio_ms": first_audio_ms,
           "stt_ms": stt_ms, "prompt_ms": t.get("prompt_ms", 0), "gemini_ms": "+".join(map(str, rounds)),
           "tool_ms": t.get("tool_ms", 0), "tts_ms": tts_ms, "quick": int(bool(t.get("fast"))),
           "stream": int(stream is not None), "first_sentence_ms": s.get("first_sentence_ms"),
           "ttft_ms": s.get("ttft_ms"), "hedged": int(bool(s.get("hedged"))) if stream is not None else "",
           "filler": int(bool(s.get("filler"))) if stream is not None else "", "answer_ms": s.get("answer_ms"),
           "rag_mode": t.get("rag_mode") or "", "rag_ms": t.get("rag_ms") if t.get("rag_mode") else None,
           "embed_ms": t.get("embed_ms"), "search_ms": t.get("search_ms"),
           "said": (said or "")[:80].replace("\t", " ")}
    try:
        path = HERE / "latency.log"
        header = "\t".join(LATENCY_COLUMNS)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                if f.readline().rstrip("\n") != header:      # an older log with other columns: keep it aside
                    f.close()
                    path.replace(HERE / f"latency.{time.strftime('%Y%m%d-%H%M%S')}.log")
        new = not path.exists()
        with open(path, "a", encoding="utf-8") as f:
            if new:
                f.write(header + "\n")
            f.write("\t".join("" if row[c] is None else str(row[c]) for c in LATENCY_COLUMNS) + "\n")
    except OSError:
        pass


def _action(result: dict) -> str:
    act = result.get("actions")
    return f"\n[action] {json.dumps(act, ensure_ascii=False)}" if act else ""


def _ms(value) -> str:
    """An embedding time: 0 means it came from the cache."""
    return "cached" if value == 0 else ("-" if value is None else f"{value} ms")


def _cites(result: dict) -> str:
    """The console line under a document answer: which passages it used."""
    cx = result.get("cortex")
    if not cx:
        return ""
    if cx.get("error"):
        return f"\n[cortex] {cx.get('mode')}: no documents ({cx['error']})"
    cites = result.get("citations") or []
    if not cites:
        if cx.get("passages"):
            return f"\n[cortex] {cx.get('mode')}: {cx['passages']} passage(s) given, none cited (not saved)"
        return f"\n[cortex] {cx.get('mode')}: nothing in the documents matched"
    return f"\n[cortex] {cx.get('mode')}: " + "; ".join(
        f"[{c['n']}] {c['filename']}" + (f" p.{c['page']}" if c.get("page") else "") + f" ({c['similarity']:.2f})"
        for c in cites)


def _citations_header(citations: list) -> str:
    """X-Citations on a /say or /ask reply that wasn't streamed: compact ASCII JSON (a header can't be long).
    A streamed reply's headers leave before the answer exists: its citations show in the console and on the
    hosted page (cortex.answers) only."""
    short = [{"n": c.get("n"), "document_id": c.get("document_id"), "filename": c.get("filename"),
              "page": c.get("page"), "chunk_id": c.get("chunk_id")} for c in citations]
    while short:
        text = json.dumps(short, ensure_ascii=True, separators=(",", ":"))
        if len(text) <= 1500:
            return text
        short = short[:-1]
    return "[]"


def _bitrate_bytes() -> float:
    """Bytes per second of the converted songs (MUSIC_BITRATE like "64k")."""
    b = config.MUSIC_BITRATE.lower().rstrip("bps")
    try:
        return float(b[:-1]) * 1000 / 8 if b.endswith("k") else float(b) / 8
    except ValueError:
        return 8000.0


def _tools(result: dict) -> str:
    names = [t["tool"] for t in result.get("tools", [])]
    return f"(tools: {', '.join(names)}) " if names else ""


def _save_reply(audio: bytes, fmt: str) -> None:
    """last_reply.wav (local voice) or last_reply.mp3 (Edge voice)."""
    if not config.SAVE_AUDIO or not audio:
        return
    try:
        if fmt == "mp3":
            (HERE / "last_reply.mp3").write_bytes(audio)
        else:
            save_wav(HERE / "last_reply.wav", np.frombuffer(audio, dtype="<i2"), config.SAMPLE_RATE)
    except OSError:
        pass


def _save(name: str, pcm: np.ndarray) -> None:
    if config.SAVE_AUDIO and pcm.size:
        try:
            save_wav(HERE / name, pcm, config.SAMPLE_RATE)
        except OSError:
            pass


def _exclusive(sock: socket.socket) -> None:
    """On Windows, SO_REUSEADDR lets a second program bind a port that is already in use, silently,
    and the speaker then talks to whichever copy of Friday Windows picks (usually the old one).
    SO_EXCLUSIVEADDRUSE makes the second bind fail instead."""
    if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = sys.platform != "win32"   # see _exclusive(): on Windows reuse means port sharing

    def server_bind(self) -> None:
        _exclusive(self.socket)
        super().server_bind()

    def handle_error(self, request, client_address) -> None:
        # The speaker hangs up whenever a song is skipped or paused: not worth a traceback.
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


# ====================================================================
#  Discovery: the S3 broadcasts "I2SONIC?", we answer "I2SONIC!<http port>"
# ====================================================================

def discovery_loop() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if sys.platform == "win32":
        _exclusive(sock)                       # a second Friday must not answer the speaker too
    else:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", config.UDP_PORT))
    except OSError as exc:
        print(f"[discovery] UDP port {config.UDP_PORT} is taken ({exc}): is another Friday server running? "
              f"The speaker won't find this one.")
        return
    reply = f"I2SONIC!{config.HTTP_PORT}".encode()
    while True:
        try:
            data, addr = sock.recvfrom(64)
        except OSError:
            continue
        if data.startswith(b"I2SONIC?"):
            sock.sendto(reply, addr)
            print(f"[discovery] speaker at {addr[0]} found us")


def lan_ips() -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))                # no packet is sent; just picks the LAN interface
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


# ====================================================================
#  Entry point
# ====================================================================

def run_test(a: Assistant, text: str | None, wav: str | None, music_state: str | None = None) -> None:
    if wav:
        pcm, rate = load_wav(wav)
        heard, raw, stt_ms = a.hear(resample(pcm, rate, config.SAMPLE_RATE))
        print(f"[stt] {stt_ms} ms: \"{raw}\"")
        text = heard or None
        if not text:
            print("[stt] nothing clear was heard")
            return
    result = a.think(text, music_state)
    print(f"[gemini] {result['ms']} ms {_tools(result)}\nnexa> {result['reply']}{_action(result)}{_cites(result)}")
    t0 = time.time()
    backend, first, rest, _ms = a.voice_start(result["reply"])
    audio = first + b"".join(rest)
    print(f"[tts] {int((time.time() - t0) * 1000)} ms with {backend.name}")
    name = "last_reply.mp3" if backend.fmt == "mp3" else "last_reply.wav"
    if backend.fmt == "mp3":
        (HERE / name).write_bytes(audio)
    else:
        save_wav(HERE / name, np.frombuffer(audio, dtype="<i2"), config.SAMPLE_RATE)
    print(f"[tts] wrote {HERE / name}; play it to hear Nexa")
    a.after_turn(text, result)
    a.background.idle(60)


def start_ears():
    """The laptop's mic + wake word, or None (EARS=0, or something is missing: Friday works without it)."""
    if not config.EARS:
        return None
    try:
        from ears import Ears

        ears = Ears()
        ears.start()
    except Exception as exc:
        print(f"[ears] off: {exc}")
        return None
    if ears.phrase:
        print(f"[ears] listening for \"{ears.phrase}\" on {ears.mic_name} (threshold {config.WAKE_THRESHOLD})")
    else:
        print(f"[ears] button only (WAKE_ENABLED=0): records from {ears.mic_name} after BOOT / Enter in speaker.py")
    return ears


def _documents_health():
    """For /health: "off", "connecting", "unreachable", or how many documents are ready (RAM only)."""
    import cortex

    return cortex.status()


def main() -> None:
    ap = argparse.ArgumentParser(description="NEXA local server")
    ap.add_argument("--test", metavar="TEXT", help="ask one question and write last_reply.wav")
    ap.add_argument("--test-wav", metavar="FILE", help="run a recording through the whole pipeline")
    ap.add_argument("--music", nargs="?", const="state=stopped", metavar="STATE",
                    help='with --test: pretend the speaker can play music, e.g. "state=playing;track=12;pos=30;vol=70"')
    args = ap.parse_args()

    bad = config.problems()
    if bad:
        print("Can't start:\n  " + "\n  ".join(bad) + "\nCopy .env.example to .env and fill it in.")
        sys.exit(1)

    import cortex

    cortex.start()                             # connects and warms up in the background: never delays start-up
    try:
        assistant = Assistant(with_stt=not args.test)
    except Exception as exc:
        print(f"Can't start: {exc}")
        sys.exit(1)

    if args.test or args.test_wav:
        cortex.wait_ready(8)                   # a one-off question: give the list of documents a moment
        try:
            run_test(assistant, args.test, args.test_wav, args.music)
        except TurnError as exc:
            print(f"[failed] {exc}")
        return

    Handler.assistant = assistant
    Handler.ears = start_ears()
    import gemini

    gemini.keepalive_start()
    import weather

    weather.start_prefetch(lambda: str(assistant.memory.profile().get("city") or ""))
    threading.Thread(target=discovery_loop, name="discovery", daemon=True).start()
    try:
        httpd = Server(("0.0.0.0", config.HTTP_PORT), Handler)
    except OSError as exc:
        print(f"Can't start: port {config.HTTP_PORT} is already in use ({exc}).\n"
              f"Another Nexa server is probably still running (an old terminal window). Close it, or find it with\n"
              f"  netstat -ano | findstr :{config.HTTP_PORT}\n"
              f"and stop that process:  taskkill /PID <number> /F")
        sys.exit(1)
    print(f"\n[nexa] listening on port {config.HTTP_PORT} (discovery on UDP {config.UDP_PORT})")
    for ip in lan_ips():
        print(f"         http://{ip}:{config.HTTP_PORT}/health")
    print("         Ctrl+C to stop\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[nexa] stopping")


if __name__ == "__main__":
    main()
