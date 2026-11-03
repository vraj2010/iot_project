"""A pretend voice, so the speech code can be tested without Microsoft or Windows."""

import threading
import time

from speech import Stop


def wait_until(check, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if check():
            return True
        time.sleep(0.005)
    return False


class FakeVoice:
    """Makes b"<name:text>" in `pieces` parts. With hold=True a sentence waits for gate(text) to be set."""

    provider = 6

    def __init__(self, name="edge", fmt="mp3", pieces=1, hold=False):
        self.name = name
        self.fmt = fmt
        self.pieces = pieces
        self.hold = hold
        self.fail = {}             # text -> exception raised before any audio
        self.fail_after = {}       # text -> exception raised after the first piece
        self.silent = set()        # texts that return without making any audio
        self.mid_gate = {}         # text -> Event the voice waits for after its first piece
        self._gates = {}
        self.calls = []            # texts, in the order they were started
        self.done = []             # texts, in the order they were finished
        self.stopped = []          # texts whose put() said Stop
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()

    def gate(self, text):
        with self._lock:
            return self._gates.setdefault(text, threading.Event())

    def audio(self, text):
        return f"<{self.name}:{text}>".encode()

    def speak(self, text, put):
        with self._lock:
            self.calls.append(text)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.hold:
                self.gate(text).wait(5)
            if text in self.fail:
                raise self.fail[text]
            if text in self.silent:
                return
            data = self.audio(text)
            size = max(1, len(data) // self.pieces)
            for i in range(self.pieces):
                part = data[i * size:] if i == self.pieces - 1 else data[i * size:(i + 1) * size]
                try:
                    put(part)
                except Stop:
                    with self._lock:
                        self.stopped.append(text)
                    raise
                if i == 0 and text in self.mid_gate:
                    self.mid_gate[text].wait(5)
                if i == 0 and text in self.fail_after:
                    raise self.fail_after[text]
            with self._lock:
                self.done.append(text)
        finally:
            with self._lock:
                self.active -= 1
