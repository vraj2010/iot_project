"""Commands Nexa answers by herself on the laptop: no Gemini, no tokens, a few milliseconds.

    heard ─▶ exact phrases ─▶ similar wordings (MiniLM) ─▶ otherwise Gemini, as before
             "pause"           "hold the music a sec"
             "volume 40"       "make it a bit quieter"
             "play kesariya"   "who's singing this"

Only clear-cut commands are taken: anything with a negation, a number the phrases didn't catch, more than
INTENT_MAX_WORDS words, or a song the library can't find goes to Gemini, which understands it properly.

    python intents.py "turn it down a bit"      what Nexa would do, and the model's score for each command
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

import config

# ------------------------------------------------------------------ cleaning up what Whisper heard

_WAKE = re.compile(r"^(?:(?:hey|hi|ok|okay)\s+)?(?:nexa|Nexa)\b\s*")
_LEAD = re.compile(r"^(?:(?:can|could|would|will) you\s+|please\s+|just\s+|now\s+|ok(?:ay)?\s+|so\s+|and\s+|um+\s+|uh+\s+)+")
_TAIL = re.compile(r"(?:\s+(?:please|for me|now|thanks|thank you|nexa|Nexa))+$")
_NEGATION = re.compile(r"\b(?:dont|do not|never|not|no|cant|cannot|wont|stop asking)\b")
_QUESTION = re.compile(r"^(?:what|whats|who|whos|why|how|when|where|which|is|are|do|does|did|can|could|would|should|will)\b")

_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                       "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i > 1}


def normalize(text: str) -> str:
    """'Hey Nexa, could you pause the music, please?' -> 'pause the music'."""
    t = (text or "").lower().replace("’", "'").replace("'", "")
    t = re.sub(r"[^\w\s%]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _WAKE.sub("", t)
    t = _LEAD.sub("", t)
    return _TAIL.sub("", t).strip()


def number(words: str) -> int | None:
    """'40', '40%', 'forty', 'forty five', 'a hundred' -> int; None if it isn't a number."""
    w = words.strip().rstrip("%").replace("percent", "").replace("-", " ").strip()
    if w.isdigit():
        return int(w)
    parts = [p for p in w.split() if p not in ("a", "and")]
    if parts == ["hundred"] or parts == ["one", "hundred"]:
        return 100
    total, tens, unit = 0, False, False          # at most one tens word, then at most one unit word
    for p in parts:
        if p in _TENS and not tens and not unit:
            total, tens = total + _TENS[p], True
        elif p in _UNITS and not unit and (not tens or _UNITS[p] < 10):
            total, unit = total + _UNITS[p], True
        else:
            return None
    return total if parts else None


# ------------------------------------------------------------------ the exact phrases

_IT = r"(?: (?:the )?(?:music|song|track|it|this|this song|playback|songs))?"
_W = (r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen"
      r"|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|a)")
_NUM = rf"(?:\d{{1,3}}|{_W}(?: {_W}){{0,2}})(?: ?%| percent)?"
_NUMBER = re.compile(rf"\b(\d{{1,3}}|{_W}(?: {_W}){{0,2}})\b")

RULES: list[tuple[str, re.Pattern]] = [(name, re.compile(rx)) for name, rx in [
    ("pause", rf"(?:pause|hold){_IT}|(?:put|keep) (?:the )?(?:music|song) on hold"),
    ("resume", rf"(?:resume|unpause|continue|carry on|keep playing|start playing){_IT}|(?:play|start) (?:the )?music again"),
    ("stop", rf"stop{_IT}|stop playing|stop everything|turn (?:the )?music off|turn off (?:the )?music"),
    ("next", rf"(?:next|skip){_IT}(?: one)?|next one|(?:play|go to) (?:the )?next(?: song| track| one)?|skip to the next(?: song| track| one)?"),
    ("previous", r"(?:previous|last)(?: song| track| one)?|go back(?: a song| one song)?|back|(?:play|go to) (?:the )?(?:previous|last)(?: song| track| one)?"),
    ("restart", r"restart(?: (?:the|this) (?:song|track)| it)?|(?:start|play) (?:it|this|this song|the song) (?:again|over|from the (?:top|start|beginning))"
                r"|from the (?:top|start|beginning)|replay(?: it| this| this song)?|once more"),
    ("volume_up", rf"(?:a (?:bit|little) |little |much )?louder|volume up|turn (?:it|this|the volume|the music|that) up|turn up(?: the)?(?: volume| music)?"
                  rf"|(?:increase|raise) (?:the )?volume|more volume|pump it up"),
    ("volume_down", rf"(?:a (?:bit|little) |little |much )?(?:quieter|softer|lower)|volume down|turn (?:it|this|the volume|the music|that) down"
                    rf"|turn down(?: the)?(?: volume| music)?|(?:decrease|reduce|lower) (?:the )?volume|less volume|too loud|not so loud"),
    ("max_volume", r"(?:full|max|maximum|top) volume|volume (?:full|max|maximum)|(?:set )?(?:the )?volume to (?:full|max|maximum)"),
    ("set_volume", rf"(?:(?:set|put|change|make) )?(?:the )?volume (?:to |at )?{_NUM}|{_NUM} volume"
                   rf"|(?:set|put|turn) (?:it|the music) (?:to|at) {_NUM}"),
    ("what_playing", r"(?:whats|what is) (?:this|playing|this song|the song|that song|this track)(?: called)?|(?:which|what) song is (?:this|that|playing)"
                     r"|who (?:sings|sang|is singing) (?:this|that)(?: song)?|whos (?:singing|this)|(?:the )?name of (?:this|the) song|what am i listening to"),
    ("time", r"(?:whats|what is) the time(?: now)?|what time is it(?: now)?|time|tell me the time|current time"),
    ("date", r"(?:whats|what is) (?:the date|todays date|the day)(?: today)?|what day is (?:it|today)|todays date|whats today"),
    ("shuffle", r"play|play (?:some |a |any |me )?(?:music|song|songs|something|anything)|play my (?:music|songs|library)|shuffle(?: my| the)?(?: music| songs| library)?"),
]]

_PLAY = re.compile(r"(?:play|put on|queue|add) (?P<q>.+?)")
_ADD = re.compile(r"add .+ (?:to (?:the |my )?queue|next)")   # a bare "add ..." is someone else's business
_WHEN = [(re.compile(r"\s+(?:next|up next|after this(?: one| song)?)$"), "next"),
         (re.compile(r"\s+(?:later|at the end|to the queue|to my queue|in the queue)$"), "later")]
_BY = re.compile(r"^(?P<q>.+?)\s+(?:by|from)\s+(?P<a>.+)$")
_ANY_SONG = {"something", "anything", "some music", "music", "songs", "some songs", "a song", "any song", "me something"}

# ------------------------------------------------------------------ examples for the model (no-number commands)

EXAMPLES: dict[str, list[str]] = {
    "pause": ["pause the music", "pause the song", "hold the music for a second", "put the song on hold"],
    "resume": ["resume the music", "continue playing the song", "unpause the music", "start the music again"],
    "stop": ["stop the music", "stop playing music", "turn the music off", "enough music for now"],
    "next": ["next song", "skip this song", "play the next track", "skip to the next one"],
    "previous": ["previous song", "go back to the last song", "play the song before this one"],
    "restart": ["play this song from the beginning", "restart the song", "start this song over"],
    "volume_up": ["turn the volume up", "make it louder", "a little louder please", "increase the volume a bit"],
    "volume_down": ["turn the volume down", "make it quieter", "the music is too loud", "lower the volume a bit"],
    "what_playing": ["what song is this", "who sings this song", "what is playing right now", "what is the name of this song"],
    "time": ["what time is it", "tell me the current time", "what is the time right now"],
}
QUESTIONS_OK = {"what_playing", "time", "date"}       # the only commands that may be phrased as a question
MUSIC = {"pause", "resume", "stop", "next", "previous", "restart", "volume_up", "volume_down", "set_volume",
         "max_volume", "what_playing", "shuffle", "play"}

# Friendly lines for the jukebox's refusals, instead of a Gemini round to word them
_REFUSALS = {"nothing is playing": "Nothing's playing right now.",
             "nothing to resume; ask for a song": "Nothing's paused. Ask me for a song.",
             "nothing else in the queue": "That's the end of the queue."}


@dataclass
class Local:
    """A command answered on the laptop."""
    intent: str
    reply: str
    action: dict = field(default_factory=dict)
    how: str = "phrase"            # "phrase", or "model 0.86"
    result: dict = field(default_factory=dict)


# ------------------------------------------------------------------ the sentence model

class Embedder:
    """all-MiniLM-L6-v2 on onnxruntime: a sentence becomes 384 numbers, similar meanings point the same way.
    Built only from packages faster-whisper already installed (onnxruntime, tokenizers, huggingface_hub)."""

    def __init__(self, repo: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        model = hf_hub_download(repo, "onnx/model.onnx")       # ~90 MB, once, into the Hugging Face cache
        self.tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        self.tok.enable_truncation(64)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(model, sess_options=opts, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}

    def __call__(self, texts: list[str]) -> np.ndarray:
        enc = self.tok.encode_batch(list(texts))
        ids = np.array([e.ids for e in enc], np.int64)
        mask = np.array([e.attention_mask for e in enc], np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self.session.run(None, feed)[0]                 # (sentences, tokens, 384)
        m = mask[..., None].astype(np.float32)
        vec = (hidden * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)   # the mean of the real tokens
        return vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)


# ------------------------------------------------------------------ the router

class Router:
    def __init__(self, embedder=None, load_model: bool = False, tz=None) -> None:
        """embedder: anything that turns a list of sentences into unit vectors. load_model=True loads MiniLM
        in the background (it never delays start-up); until it's ready only the exact phrases work."""
        self.tz = tz
        self.embedder = None
        self._names: list[str] = []
        self._matrix: np.ndarray | None = None
        if embedder is not None:
            self._use(embedder)
        elif load_model:
            threading.Thread(target=self._load, name="intent-model", daemon=True).start()

    def _load(self) -> None:
        t0 = time.time()
        try:
            self._use(Embedder(config.INTENT_MODEL_REPO))
            print(f"[local] command model ready ({int((time.time() - t0) * 1000)} ms)")
        except Exception as exc:
            print(f"[local] command model off ({type(exc).__name__}: {exc}); exact phrases still work")

    def _use(self, embedder) -> None:
        names = [n for n, phrases in EXAMPLES.items() for _ in phrases]
        matrix = np.asarray(embedder([p for phrases in EXAMPLES.values() for p in phrases]), np.float32)
        self._names, self._matrix, self.embedder = names, matrix, embedder

    # ------------------------------------------------------------ deciding

    def route(self, heard: str, jukebox=None) -> Local | None:
        """A Local answer, or None = ask Gemini. `jukebox` is None when the speaker can't play music."""
        text = normalize(heard)
        if not text:
            return None
        for name, rule in RULES:
            m = rule.fullmatch(text)
            if m:
                return self._do(name, text, jukebox, "phrase")
        m = _PLAY.fullmatch(text)
        if m and (not text.startswith("add ") or _ADD.fullmatch(text)):
            return self._play(m["q"], text, jukebox)
        name, score = self.guess(text)
        if name:
            return self._do(name, text, jukebox, f"model {score:.2f}")
        return None

    def scores(self, text: str) -> list[tuple[str, float]]:
        """The model's best similarity for each command, highest first ([] while the model isn't ready)."""
        if self.embedder is None or self._matrix is None:
            return []
        sims = self._matrix @ np.asarray(self.embedder([text]), np.float32)[0]
        best: dict[str, float] = {}
        for n, s in zip(self._names, sims):
            best[n] = max(best.get(n, -1.0), float(s))
        return sorted(best.items(), key=lambda kv: -kv[1])

    def guess(self, text: str) -> tuple[str | None, float]:
        """The command a free wording means, if the model is sure enough. text is already normalized."""
        if (self.embedder is None or len(text.split()) > config.INTENT_MAX_WORDS or _NEGATION.search(text)
                or re.search(r"\d", text) or text.startswith(("play ", "put on ", "queue ", "add "))):
            return None, 0.0
        ranked = self.scores(text)
        if not ranked:
            return None, 0.0
        (name, top), second = ranked[0], ranked[1][1] if len(ranked) > 1 else 0.0
        if top < config.INTENT_MIN_SIMILARITY or top - second < config.INTENT_MARGIN:
            return None, top
        if _QUESTION.match(text) and name not in QUESTIONS_OK:
            return None, top
        return name, top

    # ------------------------------------------------------------ doing

    def _do(self, name: str, text: str, jb, how: str) -> Local | None:
        if name in MUSIC and jb is None:
            return None                      # the speaker can't play music: Gemini says so properly
        if name == "time":
            now = self._now()
            return Local(name, f"It's {now.strftime('%I:%M %p').lstrip('0')}.", how=how)
        if name == "date":
            now = self._now()
            return Local(name, f"It's {now.strftime('%A')}, {now.day} {now.strftime('%B')}.", how=how)
        if name == "what_playing":
            t = jb.current()
            if not t:
                return Local(name, "Nothing's playing right now.", how=how)
            paused = jb.device.get("state") == "paused"
            return Local(name, f"{'That' if paused else 'This'} is {jb.label(t)}{', paused' if paused else ''}.", how=how)
        if name == "shuffle":
            if jb.device.get("state") == "paused":
                return self._music(name, *jb.control("resume"), how)
            return self._music(name, *jb.play(""), how)
        if name == "set_volume":
            m = _NUMBER.search(text)
            level = number(m.group(1)) if m else None
            if level is None or not 0 <= level <= 100:
                return None
            return self._music(name, *jb.control("set_volume", level), how)
        if name == "max_volume":
            return self._music(name, *jb.control("set_volume", 100), how)
        return self._music(name, *jb.control(name), how)

    def _play(self, query: str, text: str, jb) -> Local | None:
        if jb is None:
            return None
        when = "later" if text.startswith(("queue ", "add ")) else "now"
        for rx, w in _WHEN:
            if rx.search(query):
                query, when = rx.sub("", query), w
        artist = ""
        m = _BY.match(query)
        if m:
            query, artist = m["q"], m["a"]
        if query in _ANY_SONG:
            query = ""
            if not artist:
                return self._music("shuffle", *jb.play(""), "phrase")
        result, action = jb.play(query, "any", artist, when)
        if "error" in result:
            return None                      # not in the library (or misheard): Gemini can work out what was meant
        return self._music("play", result, action, "phrase")

    @staticmethod
    def _music(name: str, result: dict, action: dict | None, how: str) -> Local | None:
        from jukebox import spoken

        line = spoken(result) or _REFUSALS.get(str(result.get("error", "")), "")
        if not line:
            return None
        return Local(name, line, action or {}, how, result)

    def _now(self) -> datetime:
        if self.tz is None:
            try:
                from zoneinfo import ZoneInfo

                self.tz = ZoneInfo(config.TIMEZONE)
            except Exception:
                return datetime.now()
        return datetime.now(self.tz)


def _main() -> None:
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        return
    heard = " ".join(sys.argv[1:])
    text = normalize(heard)
    print(f"heard:      {heard!r}\nnormalized: {text!r}")
    rule = next((n for n, rx in RULES if rx.fullmatch(text)), None)
    if rule:
        print(f"phrase:     {rule}  (answered locally)")
        return
    if _PLAY.fullmatch(text):
        print("phrase:     play <song>  (answered locally if the library has it, otherwise Gemini)")
        return
    router = Router()
    print("loading the model...")
    router._load()
    name, score = router.guess(text)
    for n, s in router.scores(text)[:5]:
        print(f"  {s:.3f}  {n}")
    if name:
        print(f"-> {name} (answered locally)")
    else:
        print(f"-> Gemini (needs {config.INTENT_MIN_SIMILARITY} and a lead of {config.INTENT_MARGIN}; "
              f"no negations, numbers or more than {config.INTENT_MAX_WORDS} words)")


if __name__ == "__main__":
    _main()
