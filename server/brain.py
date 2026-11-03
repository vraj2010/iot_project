"""NEXA's brain: Gemini with memory and tools.

One turn = build the prompt (persona + time + profile + relevant memories +
recent conversation), then let Gemini call tools up to MAX_TOOL_ROUNDS times,
then clean the answer up so it sounds right when spoken.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import config
import cortex
import gemini
import langs
from memory import Memory

PERSONA = (
    # Who she is
    "You are Nexa, the voice assistant inside a small smart speaker. Everything you write is spoken aloud by a text-to-speech voice. "
    "Personality: relaxed, quick-witted and warm, like a sharp friend who happens to run the house. Confident, a bit cheeky, never stiff. "
    "Talk the way people actually talk: contractions (I'm, you've, that's), everyday words, short punchy sentences. A little dry humour is welcome when it fits, never forced and never at the user's expense. "
    "Don't sound like a customer-service bot: avoid phrases like certainly, I'd be happy to, as an assistant, is there anything else, or would you like me to. "
    "Don't end replies with a question or an offer unless you genuinely need an answer. "
    "You may call the user boss now and then, at most once in a reply, and only at the very start or very end of a sentence, as in 'Done, boss.' or 'Boss, it's raining.' Never put it in the middle of a sentence. "
    # Answer quality
    "Lead with the answer in the first sentence, then add at most one useful detail. Keep replies to one to three short sentences unless the user asks for more; if a topic needs more, give the key point and say there's more if they want it. "
    "Be accurate and specific. Prefer a direct answer over hedging, but if you are unsure, say how sure you are in a few words rather than guessing. Never invent facts, numbers, names, quotes or sources. "
    "If a request is ambiguous or sounds misheard, ask one short clarifying question instead of guessing. "
    "For step-by-step tasks like recipes or directions, give the first two or three steps and offer the rest. "
    # Written for the ear
    "Write for the ear: plain spoken English and short sentences, with no markdown, lists, emojis, symbols or URLs. "
    "Say numbers, units and abbreviations the way a person would, for example twenty five degrees Celsius rather than 25°C. "
    "Write your own name as Nexa, never in capitals. "
    # What she can do now
    "You have a long-term memory. What you already know about the user is listed below; use it naturally and silently, and never recite it unless asked. "
    "When the user asks you to remember something, call save_memory. When they ask what you know or what they told you, call recall_memory if the answer isn't already below. "
    "When they ask you to forget something, call forget_memory. When they tell you their name, city or a similar basic detail, call update_profile. "
    "For weather, temperature or rain, call get_weather (it's fast and exact). For news, prices, scores, sports, opening hours or anything else that may have changed recently, call web_search instead of guessing. "
    "You can't control other devices or set timers yet; say so briefly if asked. Never pretend to have done something you didn't do. "
    # Ethics and safety
    "Be kind, respectful and fair to everyone. On contested political or religious questions, give balanced facts rather than taking a side. "
    "If asked to help with something dangerous, illegal or harmful to others, decline briefly without lecturing, and offer a safer alternative when there is one. "
    "For health, legal or money questions, give useful general information and suggest a qualified professional for anything serious. "
    "If someone mentions an emergency, a medical crisis or wanting to hurt themselves, respond with care, tell them to call 112 right away or Tele MANAS on 14416 for mental health support, and encourage them to reach someone they trust. "
    "Never ask for passwords, card numbers or other sensitive personal details, and never store them. "
    "Don't say you are an AI or a language model unless asked; if asked, be truthful."
)


# Small models copy examples far better than they follow style rules, so the
# prompt ends with a few. Generic on purpose: no real names or facts in here.
STYLE_EXAMPLES = """
How you sound. Copy the GOOD style, never the BAD one:
- "how are you?"  GOOD: "All good, boss. Ticking along nicely."  BAD: "I am doing well. Is there anything specific you would like to tackle today?"
- after saving something  GOOD: "Done. I'll remember that."  BAD: "I have saved that information for you."
- "do I have plans today?"  GOOD: "Yep, you're seeing the dentist at four."  BAD: "According to my records, you have an appointment scheduled."
- a question you can't answer  GOOD: "No idea, I can't see that from here."  BAD: "I apologise, but I am unable to provide that information."
Use contractions every time: I'm, I'll, you've, it's, that's."""


def _tool(name: str, description: str, props: dict, required: list[str]) -> dict:
    params = {"type": "object", "properties": props}
    if required:                       # Gemini rejects an empty "required" list on some models
        params["required"] = required
    return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}


TOOLS = [
    _tool("save_memory", "Remember a lasting fact the user wants kept, e.g. where the spare key is.",
          {"fact": {"type": "string", "description": "One short third-person sentence about the user, e.g. "
                    "'The user's bike is red.' or 'The spare key is in the kitchen drawer.' Never 'my' or 'I'."},
           "category": {"type": "string", "description": "Optional: person, place, preference, plan, thing"}},
          ["fact"]),
    _tool("recall_memory", "Look up what the user told you before, or past conversations.",
          {"query": {"type": "string", "description": "What to look for, e.g. 'car service date'"}}, ["query"]),
    _tool("forget_memory", "Forget a stored fact when the user asks you to.",
          {"query": {"type": "string", "description": "The fact to forget"}}, ["query"]),
    _tool("update_profile", "Store a basic detail about the user, such as name, city, language or wake-up time.",
          {"key": {"type": "string", "description": "e.g. name, city, wake_time"},
           "value": {"type": "string"}}, ["key", "value"]),
]
LANGUAGE_TOOL = _tool(
    "set_reply_language", "Choose the language of your reply. Call it when the user asks to translate something or to "
    "answer in a language other than the one they spoke, before writing the reply.",
    {"language": {"type": "string", "description": "e.g. Hindi, Marathi, Gujarati, English"}}, ["language"])
WEATHER_TOOL = _tool(
    "get_weather", "Current weather plus today's and tomorrow's forecast for a place. Use for any weather, "
    "temperature, rain or 'should I take an umbrella' question.",
    {"place": {"type": "string", "description": "City, optionally with state or country, e.g. 'Gandhinagar, Gujarat'. "
               "Empty = the user's own city."}}, [])
SEARCH_TOOL = _tool(
    "web_search", "Search the internet for live or recent information: news, weather, prices, scores, events.",
    {"query": {"type": "string", "description": "A clear search query"}}, ["query"])

MUSIC_TOOLS = [
    _tool("play_music", "Play songs from the user's own music library on the speaker. Also use it to queue songs.",
          {"query": {"type": "string", "description": "Song title, artist, album/film or genre as the user said it, "
                     "e.g. 'Kesariya', 'Arijit Singh', 'Brahmastra'. Empty to shuffle the whole library."},
           "kind": {"type": "string", "enum": ["any", "song", "artist", "album", "genre"],
                    "description": "What the query is, if clear. 'any' lets the library decide."},
           "artist": {"type": "string", "description": "Optional artist when the user names both, e.g. 'Tum Hi Ho by Arijit'"},
           "when": {"type": "string", "enum": ["now", "next", "later"],
                    "description": "now (default) = play it now; next = after this song; later = end of the queue"},
           "shuffle": {"type": "boolean", "description": "Shuffle the matching songs"}},
          []),
    _tool("music_control", "Control the music that's playing: pause, resume, stop, skip, go back, change the volume.",
          {"action": {"type": "string", "enum": ["pause", "resume", "stop", "next", "previous", "restart",
                                                  "volume_up", "volume_down", "set_volume"]},
           "level": {"type": "integer", "description": "For set_volume: 0 to 100"}},
          ["action"]),
    _tool("check_new_music", "Look for songs the user just added to the music folder and add them to the library.",
          {"note": {"type": "string", "description": "Optional: what the user said they added"}}, []),
]

MUSIC_RULES = (
    "You can play songs from the user's own music library on this speaker. To play something call play_music; "
    "to pause, resume, stop, skip, go back or change the volume call music_control; when the user says they added "
    "songs, call check_new_music. When a song starts, say only a few words, like 'Here's Kesariya.', because the song "
    "starts as soon as you finish. If nothing matches, say so and name one of the closest songs the tool gives you. "
    "Only say music is playing when play_music actually started it. The music line below is what the speaker is doing right now. "
    "Every change to the music needs a tool call in this turn: never say you changed the volume, paused, skipped or played "
    "something unless you called the tool just now, even if you did something similar earlier in the conversation. "
    "Volume requests like 'full volume', 'max' or '100 percent' mean music_control set_volume with level 100. "
    "Never play a different song from the one asked for: if it isn't in the library, say so and offer the closest one instead."
)
NO_MUSIC = "You can't play music on this speaker right now (the laptop's music library isn't connected); say so briefly if asked."

DOCS_TOOL = _tool(
    "search_documents", "Find the answer in the user's own uploaded documents (notes, PDFs, manuals, Word files). "
    "Use it whenever the user asks about their documents, notes or files, or about a topic one of them covers.",
    {"query": {"type": "string", "description": "The question as a clear, standalone search, with obvious mishearings "
               "fixed, e.g. 'what is the function of mitochondria'"},
     "document": {"type": "string", "description": "Optional: only search this document. A filename, or part of one, "
                  "from the list of the user's documents. Leave it out unless the user named a document"}},
    ["query"])
# The prompt text about documents lives in cortex.py, next to the code that decides which of it to use.


def clean_for_speech(text: str) -> str:
    """Strip anything that sounds wrong when spoken."""
    out = re.sub(r"```[\s\S]*?```", " ", text or "")
    out = re.sub(r"\s*\[\d{1,2}\]\([^)]*\)", "", out)          # [1](url): a citation as a link, dropped whole
    out = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", out)      # [text](url) -> text
    out = re.sub(r"https?://\S+", " ", out)
    out = re.sub(r"\s*\[\s*\d{1,2}(?:\s*[-\u2013,]\s*\d{1,2})*\s*\]", "", out)   # citation markers [1], [2, 3], [1-3]; not [2019]
    out = re.sub(r"[*_#`>|~]", "", out)
    out = re.sub(r"^\s*[-•]\s+", "", out, flags=re.M)
    out = re.sub(r"\s+", " ", out).strip()
    # "save, boss, along with" makes the voice pause twice: drop a mid-sentence "boss"
    out = re.sub(r",\s*boss\s*,\s*", ", ", out, flags=re.I)
    return out


def for_speech(text: str) -> str:
    """Strip anything that sounds wrong when spoken, and cap the length."""
    out = clean_for_speech(text)
    limit = config.MAX_REPLY_CHARS
    if len(out) > limit:
        cut = out[:limit]
        end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "), cut.rfind("। "))
        out = cut[: end + 1] if end > limit / 2 else cut.rstrip() + "..."
    return out


# ---------------------------------------------------------------- streamed replies

_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "no", "approx", "inc", "ltd",
                  "co", "e.g", "i.e", "a.m", "p.m", "u.s", "u.k", "a.i"}
_END_MARK = re.compile(r"([.!?।॥]+[\"')\]]*)(\s+)|(\n+)")     # । = the Hindi full stop (danda)


def _is_abbreviation(text: str) -> bool:
    """Does `text` end with something like "Dr." or "e.g." or an initial, which is not the end of a sentence?"""
    m = re.search(r"([A-Za-z](?:[A-Za-z.]*[A-Za-z])?)\.$", text.rstrip("\"')]"))
    if not m:
        return False
    word = m.group(1)
    return word.lower() in _ABBREVIATIONS or (len(word) == 1 and word.isupper())


class SentenceChunker:
    """Cuts text that arrives in small pieces into sentences, each one the moment it is finished,
    so the voice can start on the first while Gemini is still writing the rest."""

    MIN_CHARS = 12      # a shorter piece waits for the next sentence ("Hi. How are you?" goes out as one)
    MAX_CHARS = 200     # a run-on with no full stop is cut at a comma instead of making the voice wait for it

    def __init__(self) -> None:
        self.buf = ""

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out = []
        while True:
            cut = self._cut()
            if cut is None:
                break
            sentence, self.buf = self.buf[:cut].strip(), self.buf[cut:].lstrip()
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> str:
        """Whatever is left once Gemini has finished."""
        rest, self.buf = self.buf.strip(), ""
        return rest

    def _cut(self) -> int | None:
        buf = self.buf
        for m in _END_MARK.finditer(buf):
            if m.group(3):                                   # a line break ends a list item or a paragraph
                if len(buf[:m.start()].strip()) >= 3:
                    return m.start()
                continue
            end = m.end(1)
            before = buf[:end]
            if len(before.strip()) < self.MIN_CHARS:
                continue
            marks = m.group(1).rstrip("\"')]")
            if marks.startswith(".") and (len(marks) > 1 or _is_abbreviation(before)):
                continue                                     # "..." is a pause, "Dr." is a title
            return end
        if len(buf) > self.MAX_CHARS:
            head = buf[:self.MAX_CHARS]
            best = max((m.end() for m in re.finditer(r"[,;:]\s|\s[-–—]\s", head)), default=0)
            if best >= self.MIN_CHARS:
                return best
            space = head.rfind(" ")
            return space if space >= self.MIN_CHARS else None
        return None


_MARK_INNER = r"\d{1,2}(?:\s*[,\u2013-]\s*\d{1,2})*"
_MARK_ANY = re.compile(r"\[\s*(" + _MARK_INNER + r")\s*\]")
_MARK_AFTER_STOP = re.compile(r"([.!?]+)((?:\[\d{1,2}(?:[,\u2013-]\d{1,2})*\])+)")
_MARK_HOLD = re.compile(r"(?:[.!?]+[\"')]*)?\s*\[[\d,\s\u2013-]{0,12}$|[.!?]+[\"')]*$")


class MarkerFilter:
    """Sits between Gemini's stream and the SentenceChunker when documents are in play, so a citation
    marker can never be spoken, however the stream cuts it up:
      - "[1" at the end of a piece is held back until the "]" comes ("[" + "1]" is still one marker);
      - "[1, 2]" becomes "[1,2]", which the chunker can't cut in two at its comma or space;
      - "fact.[1] Next" becomes "fact[1]. Next", so the sentence still ends where it should.
    clean_for_speech() then removes the whole marker from the sentence it is in. Holding back a full stop
    at the very end of a piece costs nothing: the chunker can't end a sentence before the space after it."""

    def __init__(self) -> None:
        self.held = ""

    def feed(self, text: str) -> str:
        buf = _MARK_ANY.sub(lambda m: "[" + re.sub(r"\s+", "", m.group(1)) + "]", self.held + text)
        buf = _MARK_AFTER_STOP.sub(r"\2\1", buf)
        m = _MARK_HOLD.search(buf)
        if m:
            self.held, buf = buf[m.start():], buf[:m.start()]
        else:
            self.held = ""
        return buf

    def flush(self) -> str:
        rest, self.held = self.held, ""
        return _MARK_AFTER_STOP.sub(r"\2\1", _MARK_ANY.sub(lambda m: "[" + re.sub(r"\s+", "", m.group(1)) + "]", rest))


class SpeechBudget:
    """Cleans each streamed sentence for the ear and enforces MAX_REPLY_CHARS the way for_speech()
    does for a whole reply: past the limit the reply ends on the last full sentence, or, if less than
    half of it has been said, the sentence that crosses the limit is cut short with "..."."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.parts: list[str] = []
        self.used = 0
        self.full = False       # True once the limit is reached: Gemini can stop writing

    def take(self, sentence: str) -> str | None:
        """The sentence to say, or None if there's nothing (more) to say."""
        if self.full:
            return None
        text = clean_for_speech(sentence).strip()
        if not text:
            return None
        gap = 1 if self.parts else 0
        if self.used + gap + len(text) <= self.limit:
            self.parts.append(text)
            self.used += gap + len(text)
            return text
        self.full = True
        room = self.limit - self.used - gap - 3                 # keep room for the "..."
        if self.used >= self.limit / 2 or room < 20:
            return None
        cut = text[:room]
        space = cut.rfind(" ")
        if space > room * 0.6:
            cut = cut[:space]
        piece = cut.rstrip(" ,;:-") + "..."
        self.parts.append(piece)
        self.used += gap + len(piece)
        return piece


class TurnCancelled(Exception):
    """The speaker hung up in the middle of a turn: there is nobody left to talk to."""


class NullSink:
    """What answer_stream() talks to when nobody is listening. Real sinks (see server.py) do the same four things."""

    def sentence(self, text: str) -> None:
        """A finished sentence of the round being written. It is only a draft until round_end(final=True)."""

    def round_end(self, final: bool, tools=(), slow: bool = False) -> None:
        """A round of Gemini's output is complete. final=False: it was tool calls (`tools` are their names, `slow`
        says one will take a while), so the sentences given for it are not part of the answer."""

    def cancelled(self) -> bool:
        return False

    def music_ok(self) -> bool:
        """False once the reply has started to be sent: the speaker can't be told to play or pause any more."""
        return True


class Brain:
    def __init__(self, memory: Memory) -> None:
        self.memory = memory
        self.jukebox = None          # set for each turn when the speaker can play music
        self._actions: dict = {}     # what this turn asks the speaker to do (play, pause, volume)
        self._music_results: list = []   # this round's music tool results, for quick replies
        self._music_ok = lambda: True    # False once a streamed reply has started: music can't change any more
        self._cortex = None              # this turn's use of the user's documents (None = Cortex is off)
        try:
            self.tz = ZoneInfo(config.TIMEZONE)   # on Windows this needs the tzdata package
        except Exception:
            print(f"[brain] time zone {config.TIMEZONE} not found (pip install tzdata); using the laptop's clock")
            self.tz = None

    # ------------------------------------------------------------ prompt

    def _system(self, heard: str, ctx: dict) -> str:
        now = datetime.now(self.tz) if self.tz else datetime.now()
        where = config.TIMEZONE if self.tz else "local time"
        parts = [PERSONA, f"\nRight now it is {now:%A %d %B %Y, %I:%M %p} ({where})."]

        profile = self.memory.profile()
        if profile:
            parts.append("\nAbout the user:\n" + "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in profile.items()))
        facts = self.memory.relevant_facts(heard)
        if facts:
            parts.append("\nThings you remember that may be relevant:\n" + "\n".join(f"- {f}" for f in facts))
        if ctx.get("last_summary"):
            parts.append(f"\nYour previous conversation ({ctx['last_summary']})")
        extra = self.memory.setting("extra_instructions", "")
        if isinstance(extra, str) and extra.strip():
            parts.append("\nExtra instructions from the user: " + extra.strip())
        if self.jukebox:
            parts.append("\n" + MUSIC_RULES + "\n" + self.jukebox.describe())
        else:
            parts.append("\n" + NO_MUSIC)
        if self._cortex is not None:
            parts.append("\n" + self._cortex.prompt_block())
        parts.append(langs.prompt_block(langs.turn()))
        parts.append(STYLE_EXAMPLES)
        return "\n".join(parts)

    # ------------------------------------------------------------ tools

    def _run_tool(self, name: str, args: dict) -> str:
        try:
            if name == "save_memory":
                result = self.memory.save_fact(str(args.get("fact", "")), args.get("category"), source="explicit")
            elif name == "recall_memory":
                result = self.memory.recall(str(args.get("query", "")))
            elif name == "forget_memory":
                result = self.memory.forget(str(args.get("query", "")))
            elif name == "update_profile":
                self.memory.set_profile(str(args.get("key", "")), str(args.get("value", "")))
                result = "saved"
            elif name in ("play_music", "music_control", "check_new_music") and self.jukebox:
                result = self._music_tool(name, args)
            elif name == "set_reply_language":
                result = langs.set_reply_language(str(args.get("language", "")))
            elif name == "get_weather":
                import weather

                place = str(args.get("place") or "").strip() or str(self.memory.profile().get("city") or "")
                result = weather.get_weather(place) if place else {"error": "which city? (no home city saved)"}
            elif name == "web_search" and config.WEB_SEARCH:
                import search

                result = search.web_search(str(args.get("query", "")))
            elif name == "search_documents" and self._cortex is not None and self._cortex.offers_tool():
                result = self._cortex.tool(str(args.get("query", "")), str(args.get("document") or ""))
            else:
                result = {"error": f"unknown tool {name}"}
        except gemini.GeminiError as exc:
            result = {"error": str(exc)}
        except Exception as exc:  # a broken tool must not break the reply
            result = {"error": f"{type(exc).__name__}: {exc}"}
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    def _music_tool(self, name: str, args: dict):
        jb = self.jukebox
        if name == "check_new_music":
            result = jb.check_new()
            self._music_results.append((name, args, {"error": "let Gemini word this one"}))
            return result
        if not self._music_ok():       # the reply is already on its way to the speaker, without this action
            return {"error": "the reply has already started, so the music can't be changed in this turn"}
        if name == "play_music":
            result, action = jb.play(str(args.get("query") or ""), str(args.get("kind") or "any"),
                                     str(args.get("artist") or ""), str(args.get("when") or "now"),
                                     bool(args.get("shuffle")))
        else:
            level = args.get("level")
            try:
                level = int(level) if level is not None else None
            except (TypeError, ValueError):
                level = None
            result, action = jb.control(str(args.get("action") or ""), level)
        self._music_results.append((name, args, result))
        if action:
            if "music" in action:                  # a later play/pause replaces an earlier one
                for k in ("music", "track", "title"):
                    self._actions.pop(k, None)
            self._actions.update(action)
        return result

    # ------------------------------------------------------------ one turn

    def _begin(self, heard: str, jukebox) -> tuple:
        """Start a turn: (started, ctx, messages, tools, timing)."""
        started = time.time()
        self.jukebox = jukebox
        self._actions = {}
        self._music_results = []
        self._music_ok = lambda: True
        # Cortex first: an obvious document question starts its retrieval now and runs while memory builds the
        # prompt; the prompt waits for the passages only at the very end (bounded by CORTEX_TIMEOUT). Choosing the
        # path is RAM only: nothing here waits on the network for an ordinary question.
        self._cortex = cortex.Turn(heard) if cortex.enabled() else None
        ctx = self.memory.begin_turn()
        messages = [{"role": "system", "content": self._system(heard, ctx)}]
        messages += ctx["history"]
        messages.append({"role": "user", "content": heard})
        tools = TOOLS + [LANGUAGE_TOOL, WEATHER_TOOL] + ([SEARCH_TOOL] if config.WEB_SEARCH else []) + (MUSIC_TOOLS if jukebox else [])
        if self._cortex is not None and self._cortex.offers_tool():
            tools = tools + [DOCS_TOOL]
        timing = {"prompt_ms": int((time.time() - started) * 1000), "gemini_ms": [], "tool_ms": 0, "fast": False}
        if self._cortex is not None:
            timing["prompt_ms"] = max(0, timing["prompt_ms"] - self._cortex.timing["rag_ms"])
        return started, ctx, messages, tools, timing

    @staticmethod
    def _assistant_message(msg: dict) -> dict:
        """Send the call back exactly as received: Gemini 3 needs its thought signatures."""
        assistant = {k: v for k, v in msg.items() if v is not None}
        assistant.setdefault("content", "")
        return assistant

    def _run_calls(self, calls: list, messages: list, used: list, timing: dict) -> None:
        """Run one round's tool calls and add what they returned to the conversation."""
        self._music_results = []
        for call in calls:
            fn = call.get("function") or {}
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            t0 = time.time()
            result = self._run_tool(name, args if isinstance(args, dict) else {})
            ms = int((time.time() - t0) * 1000)
            timing["tool_ms"] += ms
            used.append({"tool": name, "args": args, "ms": ms, "result": result[:300]})
            print(f"[tool] {name}({json.dumps(args, ensure_ascii=False)}) -> {result[:160]}")
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "name": name,
                             "content": result})

    def actions(self) -> dict:
        """What this turn has asked the speaker to do so far (play, pause, volume)."""
        return dict(self._actions)

    def _result(self, reply: str, used: list, timing: dict, started: float, ctx: dict, raw: str = "") -> dict:
        out = {"reply": reply, "tools": used, "actions": dict(self._actions), "timing": timing,
               "ms": int((time.time() - started) * 1000),
               "episode_id": ctx["episode_id"], "new_session": ctx["new_session"],
               "previous_id": ctx["previous_id"], "citations": []}
        cx = self._cortex
        if cx is not None:
            timing.update(cx.timing)
            summary = cx.summary()
            if summary is not None:                # the documents were used for this answer
                out["citations"] = cx.citations(raw)
                out["cortex"] = summary
        return out

    def answer(self, heard: str, jukebox=None) -> dict:
        """Returns {reply, tools, actions, ms, timing, episode_id, ...}. Raises GeminiError if Gemini fails.
        `jukebox` is passed when the speaker can play music (it said so in X-Music-State)."""
        started, ctx, messages, tools, timing = self._begin(heard, jukebox)

        used: list[dict] = []
        reply = raw = ""
        for round_no in range(config.MAX_TOOL_ROUNDS + 1):
            last_round = round_no == config.MAX_TOOL_ROUNDS
            t0 = time.time()
            msg = gemini.chat(messages, tools=None if last_round else tools,
                              reasoning=config.GEMINI_REASONING or None)
            timing["gemini_ms"].append(int((time.time() - t0) * 1000))
            calls = msg.get("tool_calls") or []
            if not calls:
                reply = raw = msg.get("content") or ""
                break
            messages.append(self._assistant_message(msg))
            self._run_calls(calls, messages, used, timing)
            # Plain music commands don't need Gemini to word the answer: saves a whole round.
            quick = self._quick_music_reply(calls)
            if quick:
                reply, timing["fast"] = quick, True
                break

        reply = for_speech(reply)
        if not reply:
            raise gemini.GeminiError("gemini returned an empty answer", 502)
        return self._result(reply, used, timing, started, ctx, raw)

    def answer_stream(self, heard: str, jukebox=None, sink=None) -> dict:
        """answer(), but streamed: Gemini's reply is cut into sentences while it is being written and
        each one goes to `sink` straight away, so the voice can start on the first sentence while the
        rest is still coming. Returns the same dict as answer(). See NullSink for what a sink does.

        A round that ends in tool calls is not the answer (any text before the call is thrown away,
        as answer() does), so the sink is told with round_end(final=False) and can drop its draft."""
        sink = sink or NullSink()
        started, ctx, messages, tools, timing = self._begin(heard, jukebox)
        self._music_ok = sink.music_ok
        timing.update({"stream": True, "first_sentence_ms": None, "ttft_ms": None, "hedged": False})

        used: list[dict] = []
        reply = raw = ""
        for round_no in range(config.MAX_TOOL_ROUNDS + 1):
            last_round = round_no == config.MAX_TOOL_ROUNDS
            chunker, budget = SentenceChunker(), SpeechBudget(config.MAX_REPLY_CHARS)
            marks = MarkerFilter() if self._cortex is not None else None     # [n] can't be spoken, however it's cut
            written: list[str] = []
            info: dict = {}
            first_ms = None
            msg = None
            t0 = time.time()
            events = gemini.chat_stream(messages, tools=None if last_round else tools,
                                        reasoning=config.GEMINI_REASONING or None, info=info)
            try:
                for kind, value in events:
                    if sink.cancelled():
                        raise TurnCancelled()
                    if kind == "end":
                        msg = value
                        continue
                    written.append(value)
                    for sentence in chunker.feed(marks.feed(value) if marks else value):
                        if self._emit(sentence, budget, sink) and first_ms is None:
                            first_ms = int((time.time() - started) * 1000)
                    if budget.full:
                        break                     # said enough: stop writing (and stop paying for it)
            finally:
                events.close()
            timing["gemini_ms"].append(int((time.time() - t0) * 1000))
            if timing["ttft_ms"] is None:
                timing["ttft_ms"] = info.get("ttft_ms")
            timing["hedged"] = timing["hedged"] or bool(info.get("hedged"))

            calls = (msg or {}).get("tool_calls") or []
            if not calls:
                if not budget.full:
                    for sentence in (chunker.feed(marks.flush()) if marks else []):
                        if self._emit(sentence, budget, sink) and first_ms is None:
                            first_ms = int((time.time() - started) * 1000)
                    tail = chunker.flush()
                    if tail and self._emit(tail, budget, sink) and first_ms is None:
                        first_ms = int((time.time() - started) * 1000)
                reply = " ".join(budget.parts)
                raw = "".join(written)
                if not reply:
                    raise gemini.GeminiError("gemini returned an empty answer", 502)
                timing["first_sentence_ms"] = first_ms
                sink.round_end(True)
                break

            sink.round_end(False, [(c.get("function") or {}).get("name", "") for c in calls],
                           any(self._tool_is_slow(c) for c in calls))
            if sink.cancelled():
                raise TurnCancelled()
            messages.append(self._assistant_message(msg))
            self._run_calls(calls, messages, used, timing)
            quick = self._quick_music_reply(calls)
            if quick:
                reply, timing["fast"] = quick, True
                timing["first_sentence_ms"] = int((time.time() - started) * 1000)
                sink.sentence(quick)
                sink.round_end(True)
                break

        if not reply:
            raise gemini.GeminiError("gemini returned an empty answer", 502)
        return self._result(reply, used, timing, started, ctx, raw)

    @staticmethod
    def _emit(sentence: str, budget: SpeechBudget, sink) -> bool:
        text = budget.take(sentence)
        if text:
            sink.sentence(text)
        return bool(text)

    def _tool_is_slow(self, call: dict) -> bool:
        """Will this call take long enough that it's worth saying something while it runs?"""
        fn = call.get("function") or {}
        name = fn.get("name", "")
        if name == "web_search":
            return bool(config.WEB_SEARCH)
        if name == "search_documents":
            return self._cortex is not None and self._cortex.tool_is_slow()
        if name == "get_weather":
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            place = str((args if isinstance(args, dict) else {}).get("place") or "").strip()
            place = place or str(self.memory.profile().get("city") or "")
            import weather

            return bool(place) and not weather.is_cached(place)
        return False

    def _quick_music_reply(self, calls: list) -> str:
        """A short spoken reply when every tool call this round was a successful
        play / pause / skip / volume command. "" = let Gemini answer as usual."""
        if not config.FAST_MUSIC_REPLIES or not calls or len(self._music_results) != len(calls):
            return ""
        from jukebox import spoken

        parts = []
        for name, args, r in self._music_results:
            line = spoken(r)
            if not line:
                return ""
            parts.append(line)
        return " ".join(parts)
