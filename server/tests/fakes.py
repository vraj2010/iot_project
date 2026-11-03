"""Stand-ins for the network, so the tests never call Google or Microsoft."""

import json
import threading
import time

import requests


def delta(text=None, **extra):
    """One streamed chunk in the OpenAI format Gemini uses."""
    d = {}
    if text is not None:
        d["content"] = text
    d.update(extra)
    return {"choices": [{"index": 0, "delta": d}]}


def sse(*events) -> bytes:
    """Server-sent events, ending the way Gemini ends a stream."""
    return b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events) + b"data: [DONE]\n\n"


class FakeResponse:
    """What requests hands back for a streamed call. hang=True never sends anything, like a dead connection."""

    def __init__(self, chunks=(), status=200, ctype="text/event-stream", hang=False, text="", json_data=None, gap=0.0):
        self.chunks = list(chunks)
        self.status_code = status
        self.headers = {"Content-Type": ctype}
        self.hang = hang
        self.text = text
        self.gap = gap
        self._json = json_data
        self.closed = threading.Event()

    def iter_content(self, chunk_size=None):
        if self.hang:
            self.closed.wait(20)
            raise requests.ConnectionError("connection closed")
        for chunk in self.chunks:
            if self.gap:
                time.sleep(self.gap)
            if self.closed.is_set():
                return
            yield chunk

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def close(self):
        self.closed.set()


class FakeSession:
    """Plays back prepared responses (or raises prepared exceptions), one per call."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def _next(self, kind, url, kw):
        self.calls.append({"kind": kind, "url": url, **kw})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def post(self, url, json=None, headers=None, timeout=None, stream=False):
        return self._next("post", url, {"json": json, "stream": stream})

    def get(self, url, headers=None, timeout=None, params=None):
        return self._next("get", url, {"params": params})

    def close(self):
        self.closed = True
