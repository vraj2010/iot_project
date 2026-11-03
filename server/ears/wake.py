"""The wake word. openWakeWord scores every 80 ms frame from 0 to 1 for ONE phrase: "hey jarvis" (built in)
or your own model, e.g. "hey friday". Other speech, music and noise stay near 0; crossing WAKE_THRESHOLD
means the phrase was said.

The model files (about 5 MB) are downloaded from openWakeWord's GitHub releases the first time.

    python -m ears.wake       live meter: mic level, speech, and the wake word's score as you say it
                              (also written to wake_test.log, for comparing runs)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]           # the server folder: custom model paths are relative to it


class WakeWord:
    def __init__(self, model: str = "hey_jarvis") -> None:
        try:
            from openwakeword.model import Model
            from openwakeword.utils import download_models
        except ImportError as exc:
            raise RuntimeError("openwakeword isn't installed: pip install -r requirements.txt") from exc
        path = Path(model)
        if path.suffix in (".onnx", ".tflite"):
            path = path if path.is_absolute() else HERE / path
            if not path.exists():
                raise RuntimeError(f"wake word model not found: {path}")
            download_models([path.stem])                 # the shared feature models (an official name is fetched too)
            spec, self.phrase = str(path), path.stem
        else:
            download_models([model])                     # the shared feature models + this phrase, first run only
            spec, self.phrase = model, model
        self.model = Model(wakeword_models=[spec], inference_framework="onnx")
        self.key = next(iter(self.model.models))
        self.phrase = self.phrase.replace("_", " ").split(" v0")[0]

    def score(self, frame: np.ndarray) -> float:
        return float(self.model.predict(frame)[self.key])

    def reset(self) -> None:
        """Forget the last second of audio, so one "hey jarvis" can't trigger twice."""
        self.model.reset()


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
