"""The wake word engine for Nexa.

Continuously detects "hey nexa" or "nexa" using an asynchronous speech spotter (VAD + Whisper),
or a custom ONNX model from server/models/ (e.g. models/hey_nexa.onnx).

    python -m ears.wake       live meter: mic level, speech, and the wake word's score as you say it
"""

from __future__ import annotations

import queue
import re
import threading
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]           # the server folder: custom model paths are relative to it


class WakeWord:
    def __init__(self, model: str = "hey_nexa") -> None:
        self.phrase = "hey nexa"
        self._onnx_model = None

        path = Path(model)
        onnx_file = path if path.is_absolute() else HERE / path
        if path.suffix in (".onnx", ".tflite") and onnx_file.is_file():
            try:
                from openwakeword.model import Model
                from openwakeword.utils import download_models

                download_models([onnx_file.stem])
                self._onnx_model = Model(wakeword_models=[str(onnx_file)], inference_framework="onnx")
                self._onnx_key = next(iter(self._onnx_model.models))
                self.phrase = onnx_file.stem.replace("_", " ").split(" v0")[0]
            except Exception as exc:
                print(f"[wake] couldn't load ONNX model {onnx_file}: {exc}; using native spotter")
                self._onnx_model = None

        if self._onnx_model is None:
            from faster_whisper import WhisperModel

            from .vad import SpeechDetector

            self.phrase = "hey nexa"
            self.vad = SpeechDetector()
            self._whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
            self._pattern = re.compile(r"\b(?:hey\s+)?n[eai]x[au]s?\b", re.IGNORECASE)
            self._triggered = False
            self._buf: list[np.ndarray] = []
            self._speech_count = 0
            self._silence_count = 0
            self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=3)
            self._worker_thread = threading.Thread(target=self._worker, name="wake-worker", daemon=True)
            self._worker_thread.start()

    def _worker(self) -> None:
        while True:
            audio = self._queue.get()
            if audio is None:
                break
            try:
                segments, _ = self._whisper.transcribe(
                    audio,
                    language="en",
                    beam_size=1,
                    vad_filter=False,
                    initial_prompt="Hey Nexa. Nexa.",
                )
                text = " ".join(s.text for s in segments).strip()
                if self._pattern.search(text):
                    print(f"\n[ears] wake word detected: \"{text}\"")
                    self._triggered = True
            except Exception:
                pass
            finally:
                self._queue.task_done()

    def score(self, frame: np.ndarray) -> float:
        if self._onnx_model is not None:
            return float(self._onnx_model.predict(frame)[self._onnx_key])

        if self._triggered:
            self._triggered = False
            self.reset()
            return 1.0

        prob = self.vad.prob(frame)
        if prob >= 0.35:
            self._speech_count += 1
            self._silence_count = 0
            self._buf.append(frame)
        elif self._speech_count > 0:
            self._silence_count += 1
            self._buf.append(frame)

        # Submit speech chunk for background transcription (approx 350ms to 1.5s of speech)
        if self._speech_count >= 4 and (self._silence_count >= 2 or len(self._buf) >= 18):
            pcm = np.concatenate(self._buf)
            audio = pcm.astype(np.float32) / 32768.0
            if not self._queue.full():
                self._queue.put_nowait(audio)
            self._buf = []
            self._speech_count = 0
            self._silence_count = 0

        if self._silence_count >= 5:
            self.reset()

        return 0.0

    def reset(self) -> None:
        """Reset the buffer and state."""
        if self._onnx_model is not None:
            self._onnx_model.reset()
        else:
            self._triggered = False
            self._buf.clear()
            self._speech_count = 0
            self._silence_count = 0
            if hasattr(self, "vad"):
                self.vad.reset()


def _main() -> None:
    """Say the wake word a few times and see what it scores, to check the mic and choose WAKE_THRESHOLD."""
    import sys
    import time

    import config

    from .mic import Mic
    from .vad import SpeechDetector

    wake = WakeWord(sys.argv[1] if len(sys.argv) > 1 else config.WAKE_MODEL)
    vad = SpeechDetector()
    mic = Mic(config.MIC_DEVICE, config.MIC_GAIN_DB)
    threshold = config.WAKE_THRESHOLD
    gain = f", MIC_GAIN_DB {config.MIC_GAIN_DB:g}" if config.MIC_GAIN_DB else ""
    _log(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')}  \"{wake.phrase}\"  mic {mic.name}{gain}  threshold {threshold}")
    print(f'Mic: {mic.name} ({mic.native} Hz{gain})\nSay "{wake.phrase}" a few times, normally, from where you\'ll sit. '
          f"It wakes at {threshold} (WAKE_THRESHOLD). Ctrl+C to stop.\n")
    mic.start()
    peak, loudest, speech, n = 0.0, -90.0, 0.0, 0
    voice: list[float] = []                        # the level of every frame the speech detector called speech
    try:
        while True:
            frame = mic.read()
            if frame is None:
                print("\r  nothing from the mic: is it muted, or blocked in Windows privacy settings?")
                continue
            score, prob = wake.score(frame), vad.prob(frame)
            rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
            loudest = max(loudest, 20 * np.log10(max(rms, 1.0) / 32768))
            speech = max(speech, prob)
            if prob >= 0.5:
                voice.append(20 * np.log10(max(rms, 1.0) / 32768))
            if score >= 0.1:
                peak = max(peak, score)
            elif peak:
                verdict = "-> she wakes" if peak >= threshold else f"(below {threshold})"
                print(f"\r  \"{wake.phrase}\" scored {peak:.2f} {verdict}" + " " * 30)
                _log(f"scored {peak:.2f}")
                peak = 0.0
            n += 1
            if n % 5 == 0:                         # about 2.5 updates a second
                bar = "#" * max(0, min(30, int((loudest + 70) * 0.5)))
                print(f"\r  level {loudest:6.1f} dBFS {bar:<30} speech {speech:.2f}", end="", flush=True)
                loudest, speech = -90.0, 0.0
    except KeyboardInterrupt:
        print()
    finally:
        mic.close()
    _advice(voice)


def _log(line: str) -> None:
    try:
        with open(HERE / "wake_test.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


TARGET_DB = -26.0                                  # a comfortable speech level for the wake word and Whisper


def _advice(voice: list[float]) -> None:
    """How loud your voice came in, and the MIC_GAIN_DB that would bring it to a good level."""
    import config

    if len(voice) < 5:
        print("\nNo speech heard: check the mic (python -m ears.mic) and that it isn't muted.")
        return
    level = float(np.percentile(voice, 90))        # your louder syllables, not the pauses
    _log(f"voice level {level:.0f} dBFS")
    print(f"\nYour voice came in at about {level:.0f} dBFS (a good level is around {TARGET_DB:.0f}).")
    if level >= TARGET_DB - 6:
        print("That's loud enough.")
        return
    total = config.MIC_GAIN_DB + (TARGET_DB - level)
    print("Too quiet. Best: turn the mic up in Windows (Settings > System > Sound > Input > volume 100, and\n"
          "Microphone Boost under More sound settings > Recording > Microphone > Properties > Levels), then run this again.")
    print(f"If it's already at the top, put this in server\\.env:  MIC_GAIN_DB={min(40, round(total))}")


if __name__ == "__main__":
    _main()
