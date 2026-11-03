"""A pretend memory, jukebox and Gemini, so Brain can be exercised without any of them."""

import copy


class FakeMemory:
    def __init__(self, city="Rajkot"):
        self.city = city
        self.saved = []

    def begin_turn(self):
        return {"episode_id": 7, "new_session": False, "previous_id": None, "history": [], "last_summary": ""}

    def profile(self):
        return {"city": self.city} if self.city else {}

    def relevant_facts(self, text):
        return []

    def setting(self, key, default=None):
        return default

    def count_facts(self):
        return 0

    def save_fact(self, content, category=None, source="explicit"):
        self.saved.append(content)
        return "saved"


class FakeJukebox:
    ffmpeg = None

    def __init__(self):
        self.played = []

    def count(self):
        return 3

    def observe(self, header):
        return header is not None

    def describe(self):
        return "Music: nothing is playing."

    def play(self, query, kind, artist, when, shuffle):
        self.played.append(query)
        return {"playing": "Kesariya"}, {"music": "play", "track": 12, "title": "Kesariya"}

    def control(self, action, level):
        return {"done": "paused"}, {"music": "pause"}


class Recorder:
    """A sink that writes down what it is told."""

    def __init__(self):
        self.events = []
        self.cancel = False
        self.locked = False

    def sentence(self, text):
        self.events.append(("sentence", text))

    def round_end(self, final, tools=(), slow=False):
        self.events.append(("round_end", final, list(tools), slow))

    def cancelled(self):
        return self.cancel

    def music_ok(self):
        return not self.locked

    def sentences(self):
        return [e[1] for e in self.events if e[0] == "sentence"]


def script(*rounds):
    """A stand-in for gemini.chat_stream: round N plays rounds[N], a list of ("text", ...) / ("end", msg).
    Returns (function, calls) where calls records the messages and tools each round was sent."""
    calls = []

    def fake(messages, tools=None, model=None, max_tokens=1024, reasoning=None, hedge_s=None, info=None):
        calls.append({"messages": copy.deepcopy(messages), "tools": tools, "closed": False})
        record = calls[-1]
        events = rounds[len(calls) - 1]
        if info is not None:
            info.update(ttft_ms=120, hedged=False)

        def gen():
            try:
                for event in events:
                    yield event
            finally:
                record["closed"] = True
        return gen()

    return fake, calls


def always(*events):
    """Like script(), but every round plays the same events (for runs that ask many questions)."""
    calls = []

    def fake(messages, tools=None, model=None, max_tokens=1024, reasoning=None, hedge_s=None, info=None):
        calls.append({"messages": copy.deepcopy(messages), "tools": tools})
        if info is not None:
            info.update(ttft_ms=120, hedged=False)

        def gen():
            yield from events
        return gen()

    return fake, calls


def tool_call(name, args="{}", call_id="call_1", **extra):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}, **extra}
