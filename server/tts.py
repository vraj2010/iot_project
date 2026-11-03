"""Text to speech.

Online (optional, tried first when TTS_ENGINE=edge):
  * Edge    - Microsoft's neural voices through the free Edge read-aloud service
              (edge-tts). Default en-IE-EmilyNeural, an Irish female voice.
              Returns MP3, which the S3 already knows how to play.
Local, used when Edge is off or fails:
  * Piper   - better voice, fully offline. Used when PIPER_EXE and PIPER_MODEL
              are set in .env.
  * SAPI    - the voice already built into Windows. Nothing to download.
Both return int16 mono audio at config.SAMPLE_RATE, ready for the speaker.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

import config
import langs
from audio_utils import bytes_to_pcm, load_wav, resample
from speech import Stop


class TTSError(RuntimeError):
    pass


# Provider ids sent to the S3 in the reply frame (cloud.c prints the name).
PROVIDER_PIPER = 4
PROVIDER_SAPI = 5
PROVIDER_EDGE = 6

_com_ready = threading.local()


def _com_init() -> None:
    """SAPI is a COM object: each thread that uses it must initialise COM once."""
    if getattr(_com_ready, "done", False):
        return
    try:
        import comtypes  # installed with pyttsx3 on Windows

        comtypes.CoInitialize()
    except Exception:
        pass
    _com_ready.done = True


class _PiperVoice:
    name = "piper"
    provider = PROVIDER_PIPER

    def __init__(self) -> None:
        self.exe = Path(config.PIPER_EXE)
        self.model = Path(config.PIPER_MODEL)
        if not self.exe.exists():
            raise TTSError(f"PIPER_EXE not found: {self.exe}")
        if not self.model.exists():
            raise TTSError(f"PIPER_MODEL not found: {self.model}")

        # Piper writes at the voice's own rate; its config file tells us which.
        self.rate = 22050
        meta = Path(str(self.model) + ".json")
        if meta.exists():
            try:
                self.rate = int(json.loads(meta.read_text())["audio"]["sample_rate"])
            except Exception:
                pass

    def synth(self, text: str) -> np.ndarray:
        result = subprocess.run(
            [str(self.exe), "--model", str(self.model), "--output_raw"],
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        if result.returncode != 0 or not result.stdout:
            raise TTSError(result.stderr.decode(errors="ignore")[:300] or "piper produced nothing")
        return resample(bytes_to_pcm(result.stdout), self.rate, config.SAMPLE_RATE)


class _SapiVoice:
    name = "windows sapi"
    provider = PROVIDER_SAPI

    def __init__(self) -> None:
        try:
            import pyttsx3  # noqa: F401
        except ImportError as exc:
            raise TTSError("pyttsx3 is not installed (pip install pyttsx3)") from exc

    def synth(self, text: str) -> np.ndarray:
        import pyttsx3

        _com_init()

        # A fresh engine per phrase: pyttsx3 is unreliable when reused across
        # runAndWait() calls, and a phrase is short enough that it costs little.
        path = Path(tempfile.gettempdir()) / f"i2sonic_tts_{os.getpid()}_{threading.get_ident()}.wav"
        engine = pyttsx3.init()
        engine.setProperty("rate", config.SAPI_RATE)
        if config.SAPI_VOICE:                       # e.g. "Zira" or "Hazel" (part of the name is enough)
            for v in engine.getProperty("voices"):
                if config.SAPI_VOICE.lower() in (v.name or "").lower():
                    engine.setProperty("voice", v.id)
                    break
        engine.save_to_file(text, str(path))
        engine.runAndWait()
        engine.stop()

        if not path.exists() or path.stat().st_size < 64:
            raise TTSError("SAPI produced no audio")
        pcm, rate = load_wav(path)
        try:
            path.unlink()
        except OSError:
            pass
        return resample(pcm, rate, config.SAMPLE_RATE)


def for_neural_voice(text: str) -> str:
    """Neural voices pause hard at ", boss." (MeloTTS doesn't). Drop that comma
    in what the voice reads; the text shown on screen is unchanged."""
    import re

    return re.sub(r",\s*(boss\b)", r" \1", text, flags=re.I) if config.EDGE_NO_BOSS_COMMA else text


class _EdgeVoice:
    """Microsoft neural voice via edge-tts. Needs internet; returns MP3 bytes."""

    provider = PROVIDER_EDGE
    fmt = "mp3"
    _up = True                         # did the last live request to Microsoft work? (none made yet: yes)

    def __init__(self) -> None:
        try:
            import edge_tts  # noqa: F401
        except ImportError as exc:
            raise TTSError("edge-tts is not installed (pip install edge-tts)") from exc
        self.name = f"edge {config.EDGE_VOICE}"

    def looks_up(self) -> bool:
        """Does Microsoft's service look reachable? Only live requests tell: a line from the voice cache doesn't."""
        return self._up

    def stalled(self) -> None:
        """A request went unanswered for too long and was given up on: treat the service as down until one works."""
        self._up = False

    def audio(self, text: str) -> bytes:
        import asyncio

        import edge_tts

        async def run() -> bytes:
            com = edge_tts.Communicate(for_neural_voice(text), langs.voice_for(text), rate=config.EDGE_RATE,
                                       pitch=config.EDGE_PITCH)
            buf = bytearray()
            async for chunk in com.stream():
                if chunk.get("type") == "audio":
                    buf += chunk["data"]
            return bytes(buf)

        try:
            data = asyncio.run(asyncio.wait_for(run(), timeout=config.EDGE_TIMEOUT))
        except asyncio.TimeoutError:
            self._up = False
            raise TTSError("edge voice timed out") from None
        except TTSError:
            self._up = False
            raise
        except Exception as exc:  # network down, service changed...
            self._up = False
            raise TTSError(f"edge voice failed: {exc}") from None
        if len(data) < 200:
            self._up = False
            raise TTSError("edge voice returned no audio")
        self._up = True
        return data

    def speak(self, text: str, put) -> None:
        """Say `text` for a SpeechSession: a line said before comes from the voice cache, anything else
        from Microsoft, handed to put() as it arrives. Runs in the calling thread."""
        cached = _cache_get(text)
        if cached:
            put(cached)
            return
        keep = [] if _cache_path(text) else None

        def forward(data: bytes) -> None:
            if keep is not None:
                keep.append(data)
            put(data)

        self._synth_into(text, forward)
        if keep:
            _cache_store(text, b"".join(keep))      # short lines are kept for next time

    def _synth_into(self, text: str, put) -> None:
        import asyncio

        import edge_tts

        async def run() -> None:
            com = edge_tts.Communicate(for_neural_voice(text), langs.voice_for(text), rate=config.EDGE_RATE,
                                       pitch=config.EDGE_PITCH)
            got = 0
            async for chunk in com.stream():
                if chunk.get("type") == "audio" and chunk.get("data"):
                    got += len(chunk["data"])
                    put(chunk["data"])
            if got < 200:
                raise TTSError("edge voice returned no audio")

        try:
            asyncio.run(asyncio.wait_for(run(), timeout=config.EDGE_TIMEOUT + 30))
        except Stop:
            raise                              # the sentence wasn't wanted any more: says nothing about the service
        except asyncio.TimeoutError:
            self._up = False
            raise TTSError("edge voice timed out") from None
        except TTSError:
            self._up = False
            raise
        except Exception as exc:  # network down, service changed...
            self._up = False
            raise TTSError(f"edge voice failed: {exc}") from None
        self._up = True

    def _start(self, text: str):
        """Begin synthesising `text` in the background; returns the queue its MP3 chunks arrive on."""
        import asyncio
        import queue

        import edge_tts

        q: queue.Queue = queue.Queue()

        def worker() -> None:
            async def run() -> None:
                com = edge_tts.Communicate(for_neural_voice(text), langs.voice_for(text), rate=config.EDGE_RATE,
                                           pitch=config.EDGE_PITCH)
                async for chunk in com.stream():
                    if chunk.get("type") == "audio" and chunk.get("data"):
                        q.put(chunk["data"])

            try:
                asyncio.run(asyncio.wait_for(run(), timeout=config.EDGE_TIMEOUT + 30))
            except Exception as exc:  # network down, service changed, timeout...
                q.put(exc)
            finally:
                q.put(_DONE)

        threading.Thread(target=worker, name="edge-voice", daemon=True).start()
        return q

    @staticmethod
    def _drain(q, required: bool):
        got_audio = False
        while True:
            try:
                item = q.get(timeout=config.EDGE_TIMEOUT)
            except Exception:
                raise TTSError("edge voice stalled") from None
            if item is _DONE:
                if required and not got_audio:
                    raise TTSError("edge voice returned no audio")
                return
            if isinstance(item, Exception):
                raise TTSError(f"edge voice failed: {item}")
            got_audio = True
            yield item

    def stream(self, text: str):
        """Yield MP3 chunks for the reply as Microsoft sends them.

        Microsoft starts sending later the longer the text is, so a reply of
        several sentences is sent as two requests at once: the first sentence
        (which comes back quickly and starts playing) and the rest (which is
        being made meanwhile and follows straight after)."""
        first, rest = split_first_sentence(text)
        q1 = self._start(first)
        q2 = self._start(rest) if rest else None
        yield from self._drain(q1, required=True)
        if q2 is not None:
            try:
                yield from self._drain(q2, required=False)
            except TTSError as exc:          # the first sentence already played: just stop there
                print(f"[tts] rest of the reply failed: {exc}")


_DONE = object()
_LOCAL_LOCK = threading.Lock()     # the laptop's own voices make one sentence at a time


def split_first_sentence(text: str, min_chars: int = 12) -> tuple[str, str]:
    """("First sentence.", "The rest of it.") or (text, "") when it's one sentence."""
    import re

    # a sentence ends at . ! ? or the Hindi danda, and the next one starts with a capital, digit, quote or Indian letter
    for m in re.finditer(r"[.!?।](?=\s+[A-Z0-9\"'ऀ-ൿ])", text):
        if m.end() >= min_chars:
            first, rest = text[:m.end()].strip(), text[m.end():].strip()
            return (first, rest) if len(rest) >= 2 else (text, "")
    return text, ""


class _Local:
    """Wraps Piper or SAPI so every backend hands out bytes the same way."""

    fmt = "pcm"

    def __init__(self, voice) -> None:
        self._voice = voice
        self.name = voice.name
        self.provider = voice.provider

    def pcm(self, text: str) -> np.ndarray:
        return self._voice.synth(text)

    def audio(self, text: str) -> bytes:
        return self.pcm(text).astype("<i2").tobytes()

    def speak(self, text: str, put) -> None:
        with _LOCAL_LOCK:
            put(self.audio(text))


# ---------------------------------------------------------------- voice cache
# Short lines Friday says often ("Paused.", "Volume 50 percent.") are kept as MP3
# files, so they play instantly instead of waiting 1-3 s for Microsoft.

_CACHE_CHARS = 80


def _cache_path(text: str) -> Path | None:
    text = (text or "").strip()
    if config.TTS_CACHE_MAX <= 0 or not text or len(text) > _CACHE_CHARS:
        return None
    import hashlib

    key = hashlib.sha1(f"{langs.voice_for(text)}|{config.EDGE_RATE}|{config.EDGE_PITCH}|{text}".encode()).hexdigest()[:16]
    return Path(config.TTS_CACHE_DIR) / f"{key}.mp3"


def _cache_get(text: str) -> bytes | None:
    path = _cache_path(text)
    if path and path.exists():
        try:
            data = path.read_bytes()
            os.utime(path)                          # recently used: keep it when pruning
            return data if len(data) > 200 else None
        except OSError:
            return None
    return None


def _cache_store(text: str, data: bytes) -> None:
    path = _cache_path(text)
    if not path or len(data) < 200:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.part")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        files = sorted(path.parent.glob("*.mp3"), key=lambda f: f.stat().st_mtime)
        for old in files[:-config.TTS_CACHE_MAX]:
            old.unlink(missing_ok=True)
    except OSError:
        pass


def _cache_put_after(text: str, first: bytes, rest):
    """Pass the stream through; once it has all arrived, keep it if it's a short line."""
    if not _cache_path(text):
        yield from rest
        return
    parts = [first]
    for chunk in rest:
        parts.append(chunk)
        yield chunk
    _cache_store(text, b"".join(parts))


class TTS:
    def __init__(self) -> None:
        voice = None
        problems: list[str] = []

        if config.PIPER_EXE and config.PIPER_MODEL:
            try:
                voice = _PiperVoice()
            except TTSError as exc:
                problems.append(f"piper: {exc}")

        if voice is None:
            try:
                voice = _SapiVoice()
            except TTSError as exc:
                problems.append(f"sapi: {exc}")

        self.local = _Local(voice) if voice else None
        self.edge = None
        if config.TTS_ENGINE == "edge":
            try:
                self.edge = _EdgeVoice()
            except TTSError as exc:
                problems.append(f"edge: {exc}")

        if self.local is None and self.edge is None:
            raise TTSError("No working text-to-speech backend.\n  " + "\n  ".join(problems))
        for p in problems:
            print(f"[tts] {p}")
        print(f"[tts] using {self.name}")

    def warm_cache(self, lines: list[str]) -> None:
        """Make the Edge voice for common short replies ahead of time (background thread)."""
        if not self.edge or config.TTS_CACHE_MAX <= 0:
            return

        def run() -> None:
            made = 0
            for line in lines:
                if _cache_get(line):
                    continue
                try:
                    _cache_store(line, self.edge.audio(line))
                    made += 1
                except TTSError:
                    return                          # offline: try again next start
            if made:
                print(f"[tts] prepared {made} quick replies")

        threading.Thread(target=run, name="tts-warm", daemon=True).start()

    def chain(self) -> list:
        """Backends in the order they're tried for each reply."""
        return [b for b in (self.edge, self.local) if b is not None]

    @property
    def name(self) -> str:
        return " -> ".join(b.name for b in self.chain())

    def start(self, reply: str) -> tuple:
        """Start speaking a whole reply with the first voice that works.
        Returns (backend, first audio bytes, iterator over the rest, ms to first audio).
        Edge streams it (first sentence and the rest in parallel); local voices go sentence by sentence."""
        import time

        from audio_utils import split_sentences

        errors = []
        for backend in self.chain():
            t0 = time.time()
            try:
                if isinstance(backend, _EdgeVoice):
                    cached = _cache_get(reply)
                    if cached:                       # said this exact line before: no trip to Microsoft
                        return backend, cached, iter(()), int((time.time() - t0) * 1000)
                    gen = backend.stream(reply.strip())
                    first = next(gen)
                    rest = _cache_put_after(reply, first, gen)
                else:
                    sentences = split_sentences(reply)
                    first = backend.audio(sentences[0])
                    rest = (backend.audio(s) for s in sentences[1:])
                return backend, first, rest, int((time.time() - t0) * 1000)
            except Exception as exc:
                errors.append(f"{backend.name}: {exc}")
                print(f"[tts] {backend.name} failed ({exc}); trying the next voice")
        raise TTSError("; ".join(errors))

    def filler_ok(self) -> bool:
        """May a reply begin with a line from the voice cache? Such a line comes from Edge, and then Edge has to
        say the rest of the reply too (MP3 and PCM can't be mixed): if it looks unreachable, better to start
        with whichever voice does work."""
        chain = self.chain()
        check = getattr(chain[0], "looks_up", None) if chain else None
        return True if check is None else bool(check())

    def session(self, cond=None):
        """A SpeechSession: add() sentences as they are written and read their audio back in order."""
        from speech import SpeechSession

        return SpeechSession(self.chain(), parallel=config.TTS_PARALLEL, cond=cond)

    def first(self, text: str) -> tuple:
        """Voice the first sentence with the first backend that works.
        Returns (backend, audio bytes, ms). The rest of the reply uses the same backend."""
        import time

        errors = []
        for backend in self.chain():
            t0 = time.time()
            try:
                return backend, backend.audio(text.strip()), int((time.time() - t0) * 1000)
            except Exception as exc:
                errors.append(f"{backend.name}: {exc}")
                print(f"[tts] {backend.name} failed ({exc}); trying the next voice")
        raise TTSError("; ".join(errors))

    def synth(self, text: str) -> np.ndarray:
        """Local voice as 16 kHz samples (Piper or Windows)."""
        text = text.strip()
        if not text or self.local is None:
            return np.zeros(0, dtype=np.int16)
        return self.local.pcm(text)

if __name__ == "__main__":
    # python tts.py  -> lists the Windows voices installed on this laptop
    import pyttsx3

    _com_init()
    for v in pyttsx3.init().getProperty("voices"):
        print(f"{v.name}    ({v.id})")
