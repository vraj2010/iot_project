"""The laptop's microphone as a stream of 80 ms frames at 16 kHz mono, which is what both the wake word and
Whisper want.

    python -m ears.mic          list the microphones, then show the default one's level for 5 seconds
    python -m ears.mic 2        the same for microphone 2 (or part of its name, e.g. "Realtek")
"""

from __future__ import annotations

import queue
import sys

import numpy as np

from audio_utils import resample

RATE = 16000
FRAME = 1280                     # 80 ms: one openWakeWord step


class MicError(RuntimeError):
    pass


def _sd():
    try:
        import sounddevice
    except ImportError as exc:
        raise MicError("sounddevice isn't installed: pip install -r requirements.txt") from exc
    return sounddevice


def find_device(spec: str):
    """None for the default mic, a device number, or the first input whose name contains `spec`."""
    spec = (spec or "").strip()
    if not spec:
        return None
    if spec.isdigit():
        return int(spec)
    for i, d in enumerate(_sd().query_devices()):
        if d["max_input_channels"] > 0 and spec.lower() in d["name"].lower():
            return i
    raise MicError(f"no microphone matching {spec!r} (python -m ears.mic lists them)")


class Mic:
    def __init__(self, device: str = "", gain_db: float = 0.0) -> None:
        """gain_db: made louder by this much after recording (for a quiet laptop mic). Raising the mic's own
        level in Windows is better when it goes high enough: it adds less hiss."""
        sd = _sd()
        self.gain = 10 ** (gain_db / 20)
        self.device = find_device(device)
        try:
            info = sd.query_devices(self.device, "input")
        except Exception as exc:
            raise MicError(f"no microphone found ({exc})") from None
        self.name = info["name"]
        # Ask for 16 kHz; Windows usually converts for us. If this mic refuses, record at its own rate and
        # convert here instead.
        self.native = RATE
        try:
            sd.check_input_settings(device=self.device, samplerate=RATE, channels=1, dtype="int16")
        except Exception:
            self.native = int(info["default_samplerate"])
        self.dropped = 0                         # frames lost because nobody read them in time
        self._q: queue.Queue = queue.Queue(maxsize=200)    # 16 s of audio
        self._carry = np.zeros(0, np.int16)
        self._stream = sd.InputStream(device=self.device, samplerate=self.native, channels=1, dtype="int16",
                                      blocksize=int(round(FRAME * self.native / RATE)), callback=self._on_audio)

    def _on_audio(self, indata, frames, time_info, status) -> None:
        """PortAudio's thread: copy the block and get out of the way."""
        try:
            self._q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            self.dropped += 1

    def start(self) -> None:
        self._stream.start()

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """The next frame (FRAME int16 samples), or None if the mic sent nothing for `timeout` seconds."""
        while self._carry.size < FRAME:
            try:
                block = self._q.get(timeout=timeout)
            except queue.Empty:
                return None
            if self.native != RATE:
                block = resample(block, self.native, RATE)
            self._carry = np.concatenate([self._carry, block])
        frame, self._carry = self._carry[:FRAME], self._carry[FRAME:]
        if self.gain != 1.0:
            frame = np.clip(frame.astype(np.float32) * self.gain, -32768, 32767).astype(np.int16)
        return frame


def _main() -> None:
    import time

    sd = _sd()
    print("Microphones (use the number or part of the name as MIC_DEVICE in .env):")
    try:
        default_in = sd.default.device[0]            # (input, output); -1 = none set: ask PortAudio
        if default_in is None or default_in < 0:
            default_in = sd.query_devices(kind="input")["index"]
    except Exception:
        default_in = None
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  {i:3d}  {d['name']}{'   <- default' if i == default_in else ''}")
    import config

    mic = Mic(sys.argv[1] if len(sys.argv) > 1 else config.MIC_DEVICE, config.MIC_GAIN_DB)
    print(f"\nLevel of '{mic.name}' ({mic.native} Hz) for 5 s. Talk:")
    mic.start()
    end = time.time() + 5
    try:
        while time.time() < end:
            frame = mic.read()
            if frame is None:
                print("  (nothing from the mic)")
                continue
            db = 20 * np.log10(max(1.0, float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))) / 32768.0)
            bar = "#" * max(0, min(40, int((db + 70) * 0.6)))
            print(f"\r  {db:6.1f} dBFS {bar:<40}", end="", flush=True)
    finally:
        mic.close()
    print()


if __name__ == "__main__":
    _main()
