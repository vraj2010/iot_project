"""The always-on loop, and the hand-off to the speaker.

The mic thread scores every frame for the wake word. When it fires, it records until you stop talking
(speech detector + Endpointer) and keeps the recording as "the question". The speaker never gets pushed to:
it asks, with two kinds of GET /listen (server.py):

    idle poll   wait_for_wake()       "tell me when you hear the wake word"  ->  a wake id, or nothing (204)
    awake poll  wait_for_question()   "give me what was said after it"       ->  the recording, or None

Friday mustn't wake herself up. Whenever a turn is running (busy()) and until the speaker's next idle poll,
the wake word is ignored: the speaker only polls again once its reply has finished playing, so that poll is
the moment she has stopped talking. WAKE_COOLDOWN_S then covers the echo of her last word.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import numpy as np

import config

from . import endpoint
from .mic import FRAME, RATE

IDLE, LISTENING = "idle", "listening"
NEAR_MISS = 0.3                  # WAKE_DEBUG prints wake scores from here up that didn't cross the threshold
HANDOFF_S = 2.5                  # the speaker has this long to pick up a wake word before it's dropped
SPEAKER_GONE_S = 3.0             # no idle poll for this long (and none waiting) = no speaker to talk to


class NoWake:
    """Stands in for the wake word when WAKE_ENABLED=0: it never fires, so only the button starts a question."""

    phrase = ""

    def score(self, frame: np.ndarray) -> float:
        return 0.0

    def reset(self) -> None:
        pass


class Ears:
    def __init__(self, mic=None, wake=None, vad=None, clock=time.monotonic) -> None:
        self.mic, self.wake, self.vad = mic, wake, vad      # made in start() unless given (the tests give fakes)
        self.clock = clock
        self.cv = threading.Condition()
        self.state = IDLE
        self.source = ""                 # "wake" or "button" (BOOT pressed on the speaker)
        self.wake_id = 0                 # one per wake word heard
        self.delivered = 0               # the last wake id an idle poll picked up
        self.wake_at = 0.0
        self.question: tuple[int, np.ndarray | None] | None = None   # (wake id, recording or None = nothing said)
        self.turns = 0                   # turns running now, of any kind
        self.muted = False               # she is (or may still be) talking: deaf until the next idle poll
        self.cooldown_until = 0.0
        self.polling = 0                 # idle polls waiting now
        self.last_poll = float("-inf")
        self.release = 0                 # bumped by busy(): idle polls waiting then are answered at once
        self._buf: list[np.ndarray] = []
        self._ep: endpoint.Endpointer | None = None
        self._reset_wake = False
        self._peak = 0.0                 # highest wake score of the current near-miss (WAKE_DEBUG)
        self._warned = float("-inf")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.phrase = getattr(wake, "phrase", "")
        self.mic_name = getattr(mic, "name", "")

    # ------------------------------------------------------------ start / stop

    def start(self) -> None:
        """Load the models and open the mic. Raises if any of it fails (the server then runs without ears)."""
        if self.wake is None:
            if config.WAKE_ENABLED:
                from .wake import WakeWord

                self.wake = WakeWord(config.WAKE_MODEL)
            else:
                self.wake = NoWake()             # button only: no openWakeWord model to load
        if self.vad is None:
            from .vad import SpeechDetector

            self.vad = SpeechDetector()
        if self.mic is None:
            from .mic import Mic

            self.mic = Mic(config.MIC_DEVICE, config.MIC_GAIN_DB)
        self.phrase = getattr(self.wake, "phrase", "")
        self.mic_name = getattr(self.mic, "name", "")
        self.mic.start()
        self._thread = threading.Thread(target=self._run, name="ears", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self.cv:
            self.cv.notify_all()
        if self.mic is not None:
            self.mic.close()

    def _run(self) -> None:
        silent = False
        while not self._stop.is_set():
            frame = self.mic.read(timeout=1.0)
            if frame is None:
                if not silent:
                    print("[ears] the mic has gone quiet (unplugged, or muted in Windows?)")
                silent = True
                continue
            if silent:
                print("[ears] the mic is back")
                silent = False
            try:
                self.feed(frame)
            except Exception as exc:             # never let one bad frame kill the ears
                print(f"[ears] {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------ the mic thread

    def feed(self, frame: np.ndarray) -> None:
        """One 80 ms frame. Called by the mic thread (and directly by the tests)."""
        if self._reset_wake:
            self._reset_wake = False
            self.wake.reset()
        score = self.wake.score(frame)           # always fed, so its last second of audio stays current
        with self.cv:
            now = self.clock()
            if self.state == LISTENING:
                self._record(frame, now)
            elif score >= config.WAKE_THRESHOLD:
                self._peak = 0.0
                self._heard_wake(score, now)
            elif config.WAKE_DEBUG:
                if score >= NEAR_MISS:
                    self._peak = max(self._peak, score)
                elif self._peak:
                    print(f"[ears] near miss: {self._peak:.2f} (WAKE_THRESHOLD is {config.WAKE_THRESHOLD})")
                    self._peak = 0.0

    def _heard_wake(self, score: float, now: float) -> None:
        if self.turns or self.muted or now < self.cooldown_until:
            return                               # she's talking: that may be her own voice
        if not self._speaker_here(now):
            if now - self._warned > 30:
                self._warned = now
                print(f"[ears] heard \"{self.phrase}\" ({score:.2f}), but the speaker isn't connected "
                      f"(its firmware needs LAPTOP_MIC 1)")
            return
        print(f"[ears] heard \"{self.phrase}\" ({score:.2f}), listening...")
        self._begin(now, "wake")

    def _begin(self, now: float, source: str) -> None:
        self.wake_id += 1
        self.state = LISTENING
        self.source = source
        self.wake_at = now
        self.question = None
        self._buf = []
        self._ep = None                          # made on the first frame, by the mic thread (it owns the VAD)
        self._reset_wake = True
        self.cv.notify_all()

    def _record(self, frame: np.ndarray, now: float) -> None:
        if self.delivered != self.wake_id and now - self.wake_at > HANDOFF_S:
            print("[ears] the speaker didn't answer the wake word; stopped listening")
            self._end(None)
            return
        if self._ep is None:
            self.vad.reset()
            self._ep = endpoint.Endpointer(frame_ms=FRAME * 1000 // RATE, threshold=config.EARS_VAD_THRESHOLD,
                                           start_ms=int(config.EARS_START_S * 1000),
                                           silence_ms=config.EARS_SILENCE_MS, max_ms=int(config.EARS_MAX_S * 1000))
        self._buf.append(frame)
        verdict = self._ep.feed(self.vad.prob(frame))
        if verdict in (endpoint.DONE, endpoint.TOO_LONG):
            pcm = np.concatenate(self._buf)
            print(f"[ears] got it: {pcm.size / RATE:.1f} s{' (EARS_MAX_S reached)' if verdict == endpoint.TOO_LONG else ''}")
            self._end(pcm)
        elif verdict == endpoint.NOTHING:
            print("[ears] nothing said after the wake word")
            self._end(None)

    def _end(self, pcm: np.ndarray | None) -> None:
        self.question = (self.wake_id, pcm)
        self.state = IDLE
        if pcm is not None:
            self.muted = True                    # a reply is on its way: deaf until the speaker polls again
        self._buf = []
        self._ep = None
        self._reset_wake = True
        self.cv.notify_all()

    def _speaker_here(self, now: float) -> bool:
        return self.polling > 0 or now - self.last_poll < SPEAKER_GONE_S

    # ------------------------------------------------------------ the speaker's side (HTTP threads)

    def wait_for_wake(self, timeout: float) -> int | None:
        """An idle poll. Returns a wake id as soon as the wake word is heard, or None after `timeout`."""
        with self.cv:
            now = self.clock()
            self.polling += 1
            self.last_poll = now
            try:
                # The speaker is idle again, so whatever she was saying has finished playing.
                if self.state == LISTENING and self.delivered == self.wake_id:
                    print("[ears] the speaker moved on; stopped listening")
                    self._end(None)
                if self.muted and not self.turns:
                    self.muted = False
                    self.cooldown_until = now + config.WAKE_COOLDOWN_S
                release, end = self.release, now + timeout
                while True:
                    if self.state == LISTENING and self.delivered != self.wake_id:
                        self.delivered = self.wake_id
                        return self.wake_id
                    left = end - self.clock()
                    if left <= 0 or self.release != release or self._stop.is_set():
                        return None
                    self.cv.wait(left)
            finally:
                self.polling -= 1
                self.last_poll = self.clock()

    def wait_for_question(self, wake_id: int, timeout: float) -> np.ndarray | None:
        """An awake poll: the recording made after wake `wake_id`, once you've stopped talking.
        None = nothing was said, it was cancelled, or `wake_id` is out of date."""
        with self.cv:
            end = self.clock() + timeout
            while True:
                q = self.question
                if q is not None and q[0] == wake_id:
                    self.question = None
                    return q[1]
                if wake_id != self.wake_id or self.state != LISTENING or self._stop.is_set():
                    return None
                left = end - self.clock()
                if left <= 0:
                    print("[ears] gave up waiting for the end of the question")
                    self._end(None)
                    self.question = None
                    return None
                self.cv.wait(left)

    def wake_now(self) -> bool:
        """BOOT pressed on the speaker: listen as if the wake word had been heard. False while a turn runs."""
        with self.cv:
            if self.turns or self.state != IDLE:
                return False
            print("[ears] listening (BOOT pressed)...")
            self._begin(self.clock(), "button")
            return True

    @contextmanager
    def busy(self):
        """Around every turn (/ask, /say, /chat, /listen): she may talk, so the wake word is off. Idle polls
        waiting when it starts or ends are answered at once: the speaker polls again once it has finished
        playing the reply, and that poll turns the wake word back on."""
        with self.cv:
            if self.state == LISTENING:          # a typed or push-to-talk question beat the wake word
                self._end(None)
            self.turns += 1
            self.muted = True
            self.release += 1
            self.cv.notify_all()
        try:
            yield
        finally:
            with self.cv:
                self.turns -= 1
                self.release += 1                # a poll that arrived during the turn mustn't keep her deaf
                self.cv.notify_all()

    def status(self) -> dict:
        """For /health."""
        with self.cv:
            now = self.clock()
            return {"wake_word": self.phrase, "mic": self.mic_name, "state": self.state,
                    "speaker": self._speaker_here(now), "deaf": bool(self.turns or self.muted)}
