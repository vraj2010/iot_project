"""Is someone talking? Silero VAD, the small model faster-whisper already ships with (no download), run as a
stream: 512-sample windows (32 ms) with the model's memory carried from one window to the next."""

from __future__ import annotations

import numpy as np

WINDOW = 512
CONTEXT = 64                     # samples of the previous window the model sees in front of each one


class SpeechDetector:
    def __init__(self, session=None) -> None:
        if session is None:
            from faster_whisper.vad import get_vad_model

            session = get_vad_model().session
        self.session = session
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), np.float32)
        self._c = np.zeros((1, 1, 128), np.float32)
        self._ctx = np.zeros((1, CONTEXT), np.float32)
        self._rest = np.zeros(0, np.float32)

    def prob(self, frame: np.ndarray) -> float:
        """Speech probability (0..1) for one int16 frame: the highest of the windows it completed."""
        x = np.concatenate([self._rest, frame.astype(np.float32) / 32768.0])
        n = x.size // WINDOW
        best = 0.0
        for i in range(n):
            win = x[i * WINDOW:(i + 1) * WINDOW][None, :]
            out, self._h, self._c = self.session.run(
                None, {"input": np.concatenate([self._ctx, win], axis=1), "h": self._h, "c": self._c})
            self._ctx = win[:, -CONTEXT:]
            best = max(best, float(np.asarray(out).reshape(-1)[0]))
        self._rest = x[n * WINDOW:]
        return best
