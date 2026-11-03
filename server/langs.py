"""Languages Friday can hear and speak: which Whisper code, which Edge voice, which script.

The voice for a sentence is chosen from the script it is written in (Devanagari, Gujarati...), so
a reply in Hindi is always read by the Hindi voice, whatever Gemini did or didn't announce. Hindi
and Marathi share a script: the language of the current turn (heard, or set by the
set_reply_language tool) breaks the tie.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass

import config


@dataclass(frozen=True)
class Lang:
    code: str          # Whisper's code, too
    name: str
    voice: str         # Edge voice ("" = config.EDGE_VOICE)
    script: str        # Gemini is told to write in this
    low: int           # first and last code point of the script ("" voice = Latin)
    high: int


LANGS: dict[str, Lang] = {l.code: l for l in (
    Lang("en", "English", "", "Latin", 0x0000, 0x024F),
    Lang("hi", "Hindi", "hi-IN-SwaraNeural", "Devanagari", 0x0900, 0x097F),
    Lang("mr", "Marathi", "mr-IN-AarohiNeural", "Devanagari", 0x0900, 0x097F),
    Lang("gu", "Gujarati", "gu-IN-DhwaniNeural", "Gujarati", 0x0A80, 0x0AFF),
    Lang("bn", "Bengali", "bn-IN-TanishaaNeural", "Bengali", 0x0980, 0x09FF),
    Lang("ta", "Tamil", "ta-IN-PallaviNeural", "Tamil", 0x0B80, 0x0BFF),
    Lang("te", "Telugu", "te-IN-ShrutiNeural", "Telugu", 0x0C00, 0x0C7F),
    Lang("kn", "Kannada", "kn-IN-SapnaNeural", "Kannada", 0x0C80, 0x0CFF),
    Lang("ml", "Malayalam", "ml-IN-SobhanaNeural", "Malayalam", 0x0D00, 0x0D7F),
    Lang("ur", "Urdu", "ur-IN-GulNeural", "Urdu (Arabic script)", 0x0600, 0x06FF),
)}

_SCRIPT_OWNER = {}                     # first language of each script (Hindi for Devanagari)
for _l in LANGS.values():
    if _l.code != "en":
        _SCRIPT_OWNER.setdefault((_l.low, _l.high), _l.code)

_lock = threading.Lock()
_turn = "en"                           # the language of the question being answered


def enabled() -> list[str]:
    """Codes from LANGUAGES that Friday knows (English always included)."""
    codes = [c.strip().lower() for c in config.LANGUAGES.split(",")]
    out = [c for c in codes if c in LANGS]
    return out if "en" in out else ["en"] + out


def set_turn(code: str | None) -> None:
    global _turn
    with _lock:
        _turn = code if code in LANGS else "en"


def turn() -> str:
    with _lock:
        return _turn


def name_of(code: str) -> str:
    return LANGS[code].name if code in LANGS else code


def lang_of_text(text: str) -> str:
    """The language `text` is written in: the script with most of its letters, and for a script two
    languages share (Devanagari), the turn's language if it is one of them."""
    counts: dict[tuple[int, int], int] = {}
    letters = 0
    for ch in text:
        if not ch.isalpha():
            continue
        letters += 1
        o = ord(ch)
        for (low, high) in _SCRIPT_OWNER:
            if low <= o <= high:
                counts[(low, high)] = counts.get((low, high), 0) + 1
                break
    if not letters or not counts:
        return "en"
    span, n = max(counts.items(), key=lambda kv: kv[1])
    if n / letters < 0.3:                  # a few foreign words in an English sentence
        return "en"
    code = _SCRIPT_OWNER[span]
    current = turn()
    if current in LANGS and (LANGS[current].low, LANGS[current].high) == span:
        return current
    return code


def voice_for(text: str) -> str:
    """The Edge voice to read `text` with."""
    return LANGS[lang_of_text(text)].voice or config.EDGE_VOICE


def prompt_block(heard_code: str) -> str:
    """What the language of this turn means for the reply, for the system prompt."""
    others = ", ".join(LANGS[c].name for c in enabled() if c != "en")
    lines = ["\nLanguages: you can speak English and " + others + ". Your voice reads whatever you write, so "
             "write each language in its own script (Hindi and Marathi in Devanagari, Gujarati in Gujarati script), "
             "never in Roman letters."]
    if heard_code != "en":
        n, script = LANGS[heard_code].name, LANGS[heard_code].script
        lines.append(f"The user just spoke {n} (it may be mixed with English words, e.g. Hinglish). Reply entirely in "
                     f"{n}, in {script} script, in simple everyday spoken {n}; keep your usual warm, short style. "
                     "The 'plain spoken English' rule above means plain, spoken wording, here in " + n + ".")
    else:
        lines.append("Reply in English unless the user asks for another language.")
    lines.append("If the user asks you to translate something, or to answer in a particular language, call "
                 "set_reply_language first with that language, then write only the translation or answer in that "
                 "language, with nothing in English around it.")
    return "\n".join(lines)


def set_reply_language(language: str) -> dict:
    """The set_reply_language tool: pick the language of this turn's reply by name or code."""
    want = (language or "").strip().lower()
    for lang in LANGS.values():
        if want in (lang.code, lang.name.lower()):
            set_turn(lang.code)
            return {"ok": True, "language": lang.name, "script": lang.script,
                    "note": f"Now write the reply in {lang.name} ({lang.script} script)."}
    return {"error": f"unsupported language '{language}'; supported: "
                     + ", ".join(l.name for l in LANGS.values())}
