"""The laptop as Nexa's speaker, with the ESP32 as a plain Bluetooth speaker.

    python server.py        in one terminal (the brain)
    python speaker.py       in another (this: the button, the voice and the music)

Set Windows' sound output to the "Nexa" Bluetooth speaker first (or name it: --out Nexa). Then press the
board's BOOT button (or Enter in this window), say your question, and the answer plays through the board.

What it does, the same things the S3 firmware did:
  * BOOT / Enter    server /listen/wake: the laptop's mic records until you stop talking
  * a short chime, and the music goes quiet while you talk
  * the reply is played as it arrives (server streams it sentence by sentence); music is ducked under it
  * music from your library: play / pause / resume / stop / volume, and the next song when one ends
  * pressing the button while she talks cuts her off and listens again

    python speaker.py --list-devices      the sound outputs Windows offers (to pick one with --out)
"""

from __future__ import annotations

import argparse
import http.client
import json
import struct
import sys
import threading
import time
from collections import deque
from urllib.parse import unquote

import numpy as np

import client
import config
import frames

OUT_RATE = 44100


# ---------------------------------------------------------------- audio decoding (PyAV, which faster-whisper installs)

def _av():
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV isn't installed: pip install av") from exc
    return av


class _Converter:
    """Anything PyAV decodes -> 16-bit mono at the output's rate, a chunk at a time."""

    def __init__(self, rate: int) -> None:
        self.av = _av()
        self.resampler = self.av.AudioResampler(format="s16", layout="mono", rate=rate)

    def _convert(self, frame) -> list[np.ndarray]:
        frame.pts = None
        out = self.resampler.resample(frame)
        if out is None:
            return []
        if not isinstance(out, list):
            out = [out]
        return [f.to_ndarray().reshape(-1).astype(np.int16) for f in out]

    def _drain(self) -> list[np.ndarray]:
        out = self.resampler.resample(None)
        if out is None:
            return []
        if not isinstance(out, list):
            out = [out]
        return [f.to_ndarray().reshape(-1).astype(np.int16) for f in out]


class Mp3Decoder(_Converter):
    """MP3 bytes in any chunking (one song, or several sentences' MP3s back to back) -> PCM."""

    def __init__(self, rate: int = OUT_RATE) -> None:
        super().__init__(rate)
        self.codec = self.av.CodecContext.create("mp3", "r")

    def _decode(self, packet) -> list:
        try:
            return self.codec.decode(packet)
        except self.av.error.InvalidDataError:
            return []                    # the Xing/Info header frame at the start of a file: not audio

    def feed(self, data: bytes) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for packet in self.codec.parse(data):
            for frame in self._decode(packet):
                out += self._convert(frame)
        return out

    def flush(self) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for frame in self._decode(None):
            out += self._convert(frame)
        return out + self._drain()


class PcmDecoder(_Converter):
    """16-bit mono PCM at `src_rate` (the server's 16 kHz voice) -> PCM at the output's rate."""

    def __init__(self, src_rate: int = 16000, rate: int = OUT_RATE) -> None:
        super().__init__(rate)
        self.src_rate = src_rate
        self.left = b""

    def feed(self, data: bytes) -> list[np.ndarray]:
        data = self.left + data
        n = len(data) // 2 * 2
        self.left = data[n:]
        if not n:
            return []
        arr = np.frombuffer(data[:n], dtype="<i2").copy()
        frame = self.av.AudioFrame.from_ndarray(arr.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self.src_rate
        return self._convert(frame)

    def flush(self) -> list[np.ndarray]:
        return self._drain()


# ---------------------------------------------------------------- the output

def _sd():
    try:
        import sounddevice
    except ImportError as exc:
        raise RuntimeError("sounddevice isn't installed: pip install -r requirements.txt") from exc
    return sounddevice


def find_output(spec: str):
    """None for Windows' default output, a device number, or the first output whose name contains `spec`."""
    spec = (spec or "").strip()
    if not spec:
        return None
    if spec.isdigit():
        return int(spec)
    for i, d in enumerate(_sd().query_devices()):
        if d["max_output_channels"] > 0 and spec.lower() in d["name"].lower():
            return i
    raise RuntimeError(f"no sound output matching {spec!r} (python speaker.py --list-devices shows them; "
                       f"is the Bluetooth speaker connected in Windows?)")


def chime(rate: int = OUT_RATE) -> np.ndarray:
    """Two quick rising notes: "I'm listening"."""
    parts = []
    for hz in (660, 880):
        t = np.arange(int(rate * 0.07)) / rate
        env = np.minimum(1.0, np.minimum(t / 0.008, (0.07 - t) / 0.03))
        parts.append(np.sin(2 * np.pi * hz * t) * env)
    return (np.concatenate(parts) * 0.30 * 32767).astype(np.int16)


class Mixer:
    """Nexa's voice and the music, mixed into one output stream (the S3's mixer, on the laptop).

    Music is quiet while you talk (listening), ducked while she answers (replying or voice queued), full otherwise.
    """

    MUSIC_GAIN = 0.60            # songs vs her voice (songs are mastered loud): the firmware's MUSIC_GAIN
    DUCK = 0.15                  # music level while she answers: the firmware's MUSIC_DUCK_LEVEL

    def __init__(self, device: str = "", volume: int = 80, rate: int = OUT_RATE) -> None:
        self.spec, self.volume, self.rate = device, volume, rate
        self.lock = threading.Lock()
        self._voice: deque[np.ndarray] = deque()
        self._voice_n = 0
        self._music: deque[np.ndarray] = deque()
        self._music_n = 0
        self.music_paused = False
        self.listening = False
        self.replying = False
        self.music_played = 0                # samples of this song that have gone to the output
        self._gain = 1.0
        self.stream = None
        self.latency = 0.1
        self.device_name = ""

    # --- the output stream

    def open(self) -> None:
        sd = _sd()
        device = find_output(self.spec)
        info = sd.query_devices(device, "output")
        self.stream = sd.OutputStream(samplerate=self.rate, channels=2, dtype="int16", device=device,
                                      callback=self._callback)
        self.stream.start()
        self.latency = float(self.stream.latency or 0.1)
        self.device_name = info["name"]

    def ensure_open(self) -> bool:
        """Open the output, or open it again after Bluetooth dropped and came back. False = still not there."""
        if self.stream is not None and self.stream.active:
            return True
        try:
            if self.stream is not None:
                try:
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
            self.open()
            return True
        except Exception as exc:
            if getattr(self, "_warned", "") != str(exc):
                print(f"[speaker] can't use the sound output: {exc}")
                self._warned = str(exc)
            return False

    def close(self) -> None:
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception:
                pass

    # --- queues

    @staticmethod
    def _pull(q: deque, n: int) -> np.ndarray:
        out, need = [], n
        while need and q:
            a = q[0]
            if a.size <= need:
                out.append(a)
                q.popleft()
                need -= a.size
            else:
                out.append(a[:need])
                q[0] = a[need:]
                need = 0
        return np.concatenate(out) if out else np.empty(0, np.int16)

    def push_voice(self, pcm: np.ndarray) -> None:
        with self.lock:
            self._voice.append(pcm)
            self._voice_n += pcm.size

    def push_music(self, pcm: np.ndarray) -> None:
        with self.lock:
            self._music.append(pcm)
            self._music_n += pcm.size

    def music_room(self) -> bool:
        return self._music_n < 2 * self.rate          # about 2 s buffered is plenty

    def clear_voice(self) -> None:
        with self.lock:
            self._voice.clear()
            self._voice_n = 0

    def new_song(self) -> None:
        with self.lock:
            self._music.clear()
            self._music_n = 0
            self.music_played = 0

    def wait_voice_done(self, abort: threading.Event | None = None, timeout: float = 120.0) -> None:
        """Until what she's saying has come out of the speaker (the next idle poll turns the wake word back on)."""
        end = time.time() + timeout
        while self._voice_n > 0 and time.time() < end and not (abort and abort.is_set()):
            time.sleep(0.05)
        time.sleep(self.latency + 0.1)

    # --- the mix (runs on the audio thread)

    def _callback(self, outdata, frames_n, time_info, status) -> None:
        with self.lock:
            voice = self._pull(self._voice, frames_n)
            self._voice_n -= voice.size
            speaking = self.replying or self._voice_n > 0 or voice.size > 0
            if self.music_paused or self.listening:
                want = 0.0
            else:
                want = self.DUCK if speaking else 1.0
            if want > 0 or self._gain > 0.001:       # ramping down still plays the last few ms
                music = self._pull(self._music, frames_n)
                self._music_n -= music.size
                self.music_played += music.size
            else:
                music = np.empty(0, np.int16)
            g0, self._gain = self._gain, want
            volume = self.volume
        mix = np.zeros(frames_n, np.float32)
        if voice.size:
            mix[:voice.size] += voice
        if music.size:
            ramp = np.linspace(g0, want, frames_n, endpoint=False, dtype=np.float32)[:music.size]
            mix[:music.size] += music * ramp * self.MUSIC_GAIN
        mix *= volume / 100.0
        np.clip(mix, -32768, 32767, out=mix)
        out = mix.astype(np.int16)
        outdata[:, 0] = out
        outdata[:, 1] = out


# ---------------------------------------------------------------- talking to the server

def open_stream(host: str, port: int, path: str, headers: dict, timeout: float = 90.0):
    """A GET whose body can be read as it arrives (read1), unlike client.request which waits for all of it."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    conn.request("GET", path, headers=headers)
    return conn, conn.getresponse()


class ByteStream:
    """Exact reads (for the reply's header) and then the rest in whatever pieces the network delivers."""

    def __init__(self, resp) -> None:
        self.resp = resp
        self.buf = bytearray()

    def read(self, n: int) -> bytes:
        while len(self.buf) < n:
            piece = self.resp.read1(8192)
            if not piece:
                raise ValueError("the reply ended before its header did")
            self.buf += piece
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out

    def chunks(self):
        if self.buf:
            yield bytes(self.buf)
            self.buf.clear()
        while True:
            piece = self.resp.read1(8192)
            if not piece:
                return
            yield piece


_HEADERS = {b"I2S4": ("<9I", 40), b"I2S3": ("<8I", 36), b"I2S2": ("<6I", 28)}


def read_reply_header(bs: ByteStream) -> dict:
    """The reply frame's header (heard, text, action...), leaving the audio in the stream. See frames.py."""
    magic = bs.read(4)
    if magic not in _HEADERS:
        raise ValueError(f"not an I2S frame: {magic!r}")
    fmt, total = _HEADERS[magic]
    fixed = bs.read(total - 4)
    v = struct.unpack(fmt, fixed)
    if magic == b"I2S4":
        extra = v[6] + v[7] + v[8]           # heard + text + action
    elif magic == b"I2S3":
        extra = v[6] + v[7]                  # heard + text
    else:
        extra = v[5]                         # text
    return frames.parse(magic + fixed + bs.read(extra))


class MusicPlayer:
    """Streams songs from the server's /music into the mixer, and plays the next one when a song ends."""

    def __init__(self, mixer: Mixer, server: list, decoder=Mp3Decoder, log=print) -> None:
        self.mixer, self.server, self.decoder, self.log = mixer, server, decoder, log
        self.lock = threading.Lock()
        self.token = 0                   # every play / stop bumps it: a running stream sees that and quits
        self.track = 0
        self.state = "stopped"           # stopped | playing | paused

    def play(self, track_id: int) -> None:
        with self.lock:
            self.token += 1
            token = self.token
            self.track, self.state = track_id, "playing"
            self.mixer.music_paused = False
        self.mixer.new_song()
        threading.Thread(target=self._run, args=(token, track_id), name="music", daemon=True).start()

    def pause(self) -> None:
        if self.state == "playing":
            self.state = "paused"
            self.mixer.music_paused = True

    def resume(self) -> None:
        if self.state == "paused":
            self.state = "playing"
            self.mixer.music_paused = False

    def stop(self) -> None:
        with self.lock:
            self.token += 1
            self.track, self.state = 0, "stopped"
            self.mixer.music_paused = False
        self.mixer.new_song()

    def header(self) -> str:
        """X-Music-State: what the server's jukebox needs to know (see jukebox.observe)."""
        parts = [f"state={self.state}"]
        if self.state != "stopped" and self.track:
            parts += [f"track={self.track}", f"pos={self.mixer.music_played / self.mixer.rate:.1f}"]
        parts.append(f"vol={self.mixer.volume}")
        return ";".join(parts)

    def _run(self, token: int, track_id: int) -> None:
        path = f"/music/{track_id}"
        headers = {"X-Device-Token": config.DEVICE_TOKEN}
        while token == self.token:
            conn = None
            try:
                conn, resp = open_stream(self.server[0], self.server[1], path, headers, timeout=30)
                if resp.status == 204:
                    self.log("[music] that was the last song")
                    break
                if resp.status != 200:
                    self.log(f"[music] HTTP {resp.status}: {resp.read(300).decode(errors='replace')}")
                    break
                track_id = int(resp.getheader("X-Track-Id") or track_id)
                with self.lock:
                    if token != self.token:
                        return
                    self.track = track_id
                self.mixer.new_song()
                self.log(f"[music] playing {unquote(resp.getheader('X-Track-Title') or '#' + str(track_id))}")
                self._stream(token, resp)
            except Exception as exc:
                self.log(f"[music] stopped: {type(exc).__name__}: {exc}")
                break
            finally:
                if conn is not None:
                    conn.close()
            if token != self.token:
                return
            while self.mixer._music_n > 0 and token == self.token:       # let the song's last seconds play out
                time.sleep(0.1)
            path = f"/music/next?after={track_id}"
        with self.lock:
            if token == self.token:
                self.track, self.state = 0, "stopped"

    def _stream(self, token: int, resp) -> None:
        dec = self.decoder(self.mixer.rate)
        while token == self.token:
            piece = resp.read1(8192)
            if not piece:
                break
            for pcm in dec.feed(piece):
                while token == self.token and not self.mixer.music_room():
                    time.sleep(0.03)             # paused or you're talking: the queue isn't draining
                if token != self.token:
                    return
                self.mixer.push_music(pcm)
        if token == self.token:
            for pcm in dec.flush():
                self.mixer.push_music(pcm)


# ---------------------------------------------------------------- the speaker itself

class Speaker:
    def __init__(self, server: list, mixer: Mixer, music: MusicPlayer, mp3_decoder=Mp3Decoder,
                 pcm_decoder=PcmDecoder, log=print) -> None:
        self.server, self.mixer, self.music, self.log = server, mixer, music, log
        self.mp3_decoder, self.pcm_decoder = mp3_decoder, pcm_decoder
        self.phase = "idle"                  # idle | listening (recording you) | replying (reading and playing)
        self.cancel = threading.Event()      # set by the button: drop the reply that's being read
        self.token_headers = {"X-Device-Token": config.DEVICE_TOKEN}

    # --- the button

    def press(self) -> None:
        """BOOT or Enter: stop talking, and listen."""
        if self.phase == "listening":
            return                           # already recording you
        if self.phase == "replying":
            self.cancel.set()
            self.mixer.clear_voice()
        end = time.time() + 10
        while time.time() < end:
            try:
                status, data, _ = client.request(self.server[0], self.server[1], "/listen/wake", b"",
                                                 "application/octet-stream")
            except OSError as exc:
                self.log(f"[speaker] can't reach the server: {exc}")
                return
            if status == 202:
                return
            if status == 409:                # her turn on the server hasn't finished winding down yet
                if self.phase == "listening":
                    return                   # an earlier press already got us listening: don't queue another
                time.sleep(0.25)
                continue
            self.log(f"[speaker] HTTP {status}: {data.decode(errors='replace')[:200]}")
            return
        self.log("[speaker] she was still answering: press again")

    # --- one question and answer

    def listen_once(self) -> None:
        """One idle poll. A wake (button) -> record -> reply. Returns when it's time to poll again."""
        self.phase = "idle"
        status, data, _ = client.request(self.server[0], self.server[1], "/listen", None)
        if status == 204:
            return
        if status == 503:
            self.log("[speaker] the server's mic is off (EARS=0, or it failed to start: see the server window)")
            time.sleep(5)
            return
        if status != 202:
            self.log(f"[speaker] HTTP {status}: {data.decode(errors='replace')[:200]}")
            time.sleep(2)
            return
        wake = json.loads(data)["id"]
        self.phase = "listening"
        self.cancel.clear()
        self.mixer.ensure_open()
        self.mixer.listening = True              # the song goes quiet, so Whisper hears you, not it
        self.mixer.push_voice(chime())
        self.log("[speaker] listening...")
        conn = None
        try:
            headers = dict(self.token_headers, **{"X-Music-State": self.music.header()})
            conn, resp = open_stream(self.server[0], self.server[1], f"/listen?wake={wake}", headers)
            self.mixer.listening = False
            if resp.status == 202:
                self.log("[speaker] didn't hear anything")
                return
            if resp.status != 200:
                self.log(f"[speaker] HTTP {resp.status}: {resp.read(300).decode(errors='replace')}")
                return
            self.phase = "replying"
            self.play_reply(resp)
        finally:
            self.mixer.listening = False
            self.mixer.replying = False
            if conn is not None:
                conn.close()

    def play_reply(self, resp) -> None:
        bs = ByteStream(resp)
        f = read_reply_header(bs)
        self.log(f"[heard] {f['heard'] or '(nothing clear)'}")
        self.log(f"nexa> {f['text']}")
        self.apply(f.get("action"))
        if not self.mixer.ensure_open():
            self.log("[speaker] no sound output, so that was text only (is the Bluetooth speaker connected?)")
            return
        self.mixer.replying = True
        if f["format"] == frames.FORMAT_MP3:
            dec = self.mp3_decoder(self.mixer.rate)
        else:
            dec = self.pcm_decoder(f["rate"] or 16000, self.mixer.rate)
        for piece in bs.chunks():
            if self.cancel.is_set():
                self.mixer.clear_voice()
                return
            for pcm in dec.feed(piece):
                self.mixer.push_voice(pcm)
        for pcm in dec.flush():
            self.mixer.push_voice(pcm)
        self.mixer.wait_voice_done(self.cancel)

    def apply(self, action: dict | None) -> None:
        """What the reply asks the speaker to do: {"music": "play", "track": 12}, {"volume": 60}..."""
        if not action:
            return
        if "volume" in action:
            try:
                self.mixer.volume = max(0, min(100, int(action["volume"])))
            except (TypeError, ValueError):
                pass
        what = action.get("music")
        if what == "play" and action.get("track"):
            self.music.play(int(action["track"]))
        elif what == "pause":
            self.music.pause()
        elif what == "resume":
            self.music.resume()
        elif what == "stop":
            self.music.stop()

    def run(self) -> None:
        while True:
            try:
                self.listen_once()
            except KeyboardInterrupt:
                raise
            except (OSError, http.client.HTTPException) as exc:
                self.log(f"[speaker] lost the server ({type(exc).__name__}: {exc}); looking for it again")
                self.phase = "idle"
                self.mixer.listening = self.mixer.replying = False
                found = None
                while not found:
                    found = client.find_server()
                    if not found:
                        time.sleep(2)
                self.server[0], self.server[1] = found
            except Exception as exc:         # a bad reply must not kill the speaker
                self.log(f"[speaker] {type(exc).__name__}: {exc}")
                self.phase = "idle"
                time.sleep(1)


# ---------------------------------------------------------------- command line

def list_devices() -> None:
    sd = _sd()
    default = sd.default.device[1]
    print("sound outputs (use --out with part of the name, or the number):")
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0:
            print(f"  {i:>3}{'*' if i == default else ' '} {d['name']}")
    print("* = Windows' default output")


def check_packages() -> None:
    """Say which Python this is and what it lacks, with the exact command: the usual trouble is installing the
    requirements into one Python and running speaker.py with another (a second terminal without the venv)."""
    import importlib.util

    missing = {m: pip for m, pip in (("sounddevice", "sounddevice"), ("av", "av"), ("serial", "pyserial"))
               if importlib.util.find_spec(m) is None}
    if not missing:
        return
    print(f"[speaker] this Python ({sys.executable}) is missing: {', '.join(missing.values())}")
    print(f"          install them into THIS Python:  \"{sys.executable}\" -m pip install {' '.join(missing.values())}")
    print("          (server.py may be running in a different Python or venv: activate the same one here)")
    if "sounddevice" in missing or "av" in missing:
        sys.exit(1)                                  # no sound without these; pyserial only costs the BOOT button


def main() -> None:
    try:
        sys.stdout.reconfigure(errors="replace")     # song titles in any script, on a Windows console
    except Exception:
        pass
    check_packages()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host")
    ap.add_argument("--port", type=int, default=config.HTTP_PORT)
    ap.add_argument("--out", default=config.SPEAKER_DEVICE, help="part of the sound output's name, or its number")
    ap.add_argument("--volume", type=int, default=config.SPEAKER_VOLUME)
    ap.add_argument("--ptt", default=config.PTT_PORT, help='the board\'s COM port, "auto", or "off" (Enter only)')
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        return list_devices()

    host, port = args.host, args.port
    if not host:
        print("[speaker] looking for the server (python server.py)...")
        found = None
        while not found:
            found = client.find_server()
            if not found:
                time.sleep(1)
        host, port = found
    server = [host, port]
    print(f"[speaker] server at {host}:{port}")

    mixer = Mixer(args.out, max(0, min(100, args.volume)))
    if mixer.ensure_open():
        print(f"[speaker] sound output: {mixer.device_name}")
    else:
        print("[speaker] (will keep trying to open the sound output)")
    music = MusicPlayer(mixer, server)
    speaker = Speaker(server, mixer, music)

    if args.ptt.lower() != "off":
        try:
            import ptt

            ptt.SerialButton(args.ptt, on_press=speaker.press).start()
        except RuntimeError as exc:
            print(f"[speaker] no BOOT button: {exc}. Enter still works")
    if sys.stdin and sys.stdin.isatty():
        def keys() -> None:
            for _ in sys.stdin:
                speaker.press()

        threading.Thread(target=keys, name="keys", daemon=True).start()
    print("[speaker] press the board's BOOT button (or Enter here) and talk. Ctrl+C to stop")
    try:
        speaker.run()
    except KeyboardInterrupt:
        print()
    finally:
        mixer.close()


if __name__ == "__main__":
    main()
