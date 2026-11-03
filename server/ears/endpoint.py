"""When does the question start, and when is it over? Fed one speech probability per frame after the wake word.

    WAITING ──speech──▶ SPEAKING ──silence──▶ DONE
       │                    └──too long──▶ TOO_LONG (keep what was said)
       └──nothing for start_ms──▶ NOTHING
"""

from __future__ import annotations

WAITING, SPEAKING, DONE, TOO_LONG, NOTHING = "waiting", "speaking", "done", "too long", "nothing"


class Endpointer:
    def __init__(self, frame_ms: int = 80, threshold: float = 0.5, start_ms: int = 5000, silence_ms: int = 800,
                 max_ms: int = 12000, min_speech_ms: int = 240) -> None:
        self.frame_ms = frame_ms
        self.on = threshold                        # at or above: speech
        self.off = max(0.05, threshold - 0.15)     # below: silence (in between keeps what's going on)
        self.start_ms, self.silence_ms, self.max_ms = start_ms, silence_ms, max_ms
        self.min_speech_ms = min_speech_ms         # shorter blips (a click, the tail of the wake word) don't count
        self.elapsed = 0
        self.speech_ms = 0
        self.quiet_ms = 0
        self.started = False

    def feed(self, prob: float) -> str:
        self.elapsed += self.frame_ms
        if prob >= self.on:
            self.speech_ms += self.frame_ms
            self.quiet_ms = 0
            if self.speech_ms >= self.min_speech_ms:
                self.started = True
        elif prob < self.off:
            self.quiet_ms += self.frame_ms
        if self.elapsed >= self.max_ms:
            return TOO_LONG if self.started else NOTHING
        if not self.started:
            return NOTHING if self.elapsed >= self.start_ms else WAITING
        return DONE if self.quiet_ms >= self.silence_ms else SPEAKING
