"""Settings, read from the environment or a .env file next to this script."""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Hindi, Marathi and Gujarati text in the logs would crash a print() on a Windows console that isn't UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _load_dotenv() -> None:
    """Minimal .env reader, so there is no extra dependency to install."""
    path = HERE / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _get_int(name: str, default: int) -> int:
    try:
        return int(_get(name, str(default)))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(_get(name, str(default)))
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    return _get(name, "1" if default else "0").lower() not in ("0", "false", "no", "off", "")


# --- Security -----------------------------------------------------
# Same value as DEVICE_TOKEN in the firmware's secrets.h (and the Worker's secret).
DEVICE_TOKEN = _get("DEVICE_TOKEN")

# --- Gemini (the brain) ---------------------------------------------
GEMINI_API_KEY = _get("GEMINI_API_KEY")
GEMINI_MODEL = _get("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_SEARCH_MODEL = _get("GEMINI_SEARCH_MODEL", "") or GEMINI_MODEL
GEMINI_MEMORY_MODEL = _get("GEMINI_MEMORY_MODEL", "") or GEMINI_MODEL
GEMINI_EMBED_MODEL = _get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIMS = _get_int("EMBED_DIMS", 768)
GEMINI_BASE = _get("GEMINI_BASE", "https://generativelanguage.googleapis.com/v1beta")
GEMINI_TIMEOUT = _get_float("GEMINI_TIMEOUT", 20.0)
MAX_TOOL_ROUNDS = _get_int("MAX_TOOL_ROUNDS", 3)
# How hard Gemini thinks before answering: low | medium | high (blank = Google's default,
# which is slower). Gemini 3 can't switch thinking off; "low" is the fastest that's reliable.
GEMINI_REASONING = _get("GEMINI_REASONING", "low").lower()
MAX_REPLY_CHARS = _get_int("MAX_REPLY_CHARS", 400)
WEB_SEARCH = _get_bool("WEB_SEARCH", True)
SEARCH_PROVIDER = _get("SEARCH_PROVIDER", "auto").lower()   # auto (Google, then DuckDuckGo) | google | duckduckgo
SEARCH_REGION = _get("SEARCH_REGION", "in-en")               # DuckDuckGo region: in-en India, uk-en, us-en...
SEARCH_TIMEOUT = _get_int("SEARCH_TIMEOUT", 8)
SEARCH_GOOGLE_GRACE = _get_float("SEARCH_GOOGLE_GRACE", 1.5)  # after DuckDuckGo answers, wait this long for Google

# --- Memory (PostgreSQL) --------------------------------------------
# Plain PostgreSQL is enough: similarity search runs in Python, no pgvector needed.
DATABASE_URL = _get("DATABASE_URL", "postgresql://nexa:nexa@localhost:5433/nexa")
USER_KEY = _get("USER_KEY", "household")
MEMORY_TOP_K = _get_int("MEMORY_TOP_K", 5)
MEMORY_MIN_SIMILARITY = _get_float("MEMORY_MIN_SIMILARITY", 0.55)
# Up to this many memories, all of them go in Nexa's prompt and no search runs
# (the search sends every question to Google's embedding service: ~0.2-0.4 s).
MEMORY_ALL_FACTS_MAX = _get_int("MEMORY_ALL_FACTS_MAX", 40)
AUTO_MEMORY = _get_bool("AUTO_MEMORY", True)        # learn facts after each reply
HISTORY_TURNS = _get_int("HISTORY_TURNS", 8)          # recent exchanges kept word for word
SESSION_GAP_MIN = _get_int("SESSION_GAP_MIN", 10)     # silence that starts a new conversation
TIMEZONE = _get("TIMEZONE", "Asia/Kolkata")

# --- Your documents (Cortex) ---------------------------------------
# Documents are uploaded on the hosted Cortex page and stored in Supabase; Nexa reads them
# straight from the database. Empty = Cortex is off and Nexa works exactly as before.
# Use the SESSION pooler string (Supabase > Connect > Session pooler, port 5432).
CORTEX_DATABASE_URL = _get("CORTEX_DATABASE_URL", "")
CORTEX_TOP_K = _get_int("CORTEX_TOP_K", 4)                       # passages handed to Gemini per question
CORTEX_MIN_SIMILARITY = _get_float("CORTEX_MIN_SIMILARITY", 0.5) # passages less similar than this are dropped
CORTEX_TIMEOUT = _get_float("CORTEX_TIMEOUT", 2.5)               # seconds a turn waits for passages, at most
CORTEX_INJECT = _get_bool("CORTEX_INJECT", True)     # obvious document questions: passages in the prompt, one round
CORTEX_PREFETCH = _get_bool("CORTEX_PREFETCH", True) # other questions: search in the background in case Gemini asks
CORTEX_LIST_S = _get_int("CORTEX_LIST_S", 60)                    # re-read the list of documents this often
CORTEX_MAX_CHARS = _get_int("CORTEX_MAX_CHARS", 1200)            # per passage
CORTEX_LOG_ANSWERS = _get_bool("CORTEX_LOG_ANSWERS", True)       # keep document answers in cortex.answers

# --- Speech recognition (faster-whisper) ------------------------------
WHISPER_MODEL = _get("WHISPER_MODEL", "base")
# en = always English. auto = detect which of LANGUAGES you spoke (Hindi, Marathi...) and answer in it.
WHISPER_LANGUAGE = _get("WHISPER_LANGUAGE", "en") or None
LANGUAGES = _get("LANGUAGES", "en,hi,mr,gu")           # languages Nexa understands and speaks (see langs.py)
WHISPER_DEVICE = _get("WHISPER_DEVICE", "cpu")        # cpu | cuda (falls back to cpu if CUDA fails)
WHISPER_COMPUTE = _get("WHISPER_COMPUTE", "int8")     # int8 on cpu, int8_float16 on cuda
WHISPER_HINT = _get("WHISPER_HINT", "A person talking to a voice assistant named Nexa.")

# --- The laptop's ears: its own mic, a wake word, end-of-speech ---------
# While the speaker's own mic is a noisy MAX4466, the laptop listens instead: it waits for the wake word,
# records until you stop talking and sends the answer to the speaker (which asks for it with GET /listen).
EARS = _get_bool("EARS", True)                       # 0 = no laptop mic (push-to-talk on the S3 only)
MIC_DEVICE = _get("MIC_DEVICE", "")                  # part of the mic's name, or its number; blank = Windows default
MIC_GAIN_DB = _get_float("MIC_GAIN_DB", 0.0)         # extra loudness for a quiet mic; python -m ears.wake suggests one
# 0 = no wake word at all: Nexa only listens after the button (BOOT on the board, see PTT_PORT, or Enter in
# speaker.py). 1 = also wake on WAKE_MODEL below.
WAKE_ENABLED = _get_bool("WAKE_ENABLED", True)
# Wake phrase (default: hey_nexa, or path to a custom model: models/hey_nexa.onnx).
WAKE_MODEL = _get("WAKE_MODEL", "hey_nexa")
WAKE_THRESHOLD = _get_float("WAKE_THRESHOLD", 0.6)   # 0..1: higher = fewer false wakes, more misses
WAKE_COOLDOWN_S = _get_float("WAKE_COOLDOWN_S", 0.8) # deaf this long after she stops talking (her echo)
WAKE_DEBUG = _get_bool("WAKE_DEBUG", False)          # print near-misses (scores from 0.3 up) to tune the threshold
EARS_START_S = _get_float("EARS_START_S", 5.0)       # after the wake word, give up if you haven't started talking
EARS_SILENCE_MS = _get_int("EARS_SILENCE_MS", 800)   # this much silence = you've finished
EARS_MAX_S = _get_float("EARS_MAX_S", 12.0)          # longest question
EARS_VAD_THRESHOLD = _get_float("EARS_VAD_THRESHOLD", 0.5)  # 0..1: how sure the speech detector must be
LISTEN_HOLD_S = _get_int("LISTEN_HOLD_S", 20)        # an idle GET /listen is held this long, then answered 204

# --- The laptop as the speaker (speaker.py): the ESP32 is a plain Bluetooth speaker ----------------------------
# speaker.py plays Nexa's voice and your music on the laptop's audio output (set Windows to the "Nexa"
# Bluetooth speaker, or name it in SPEAKER_DEVICE) and reads the board's BOOT button from its USB serial port.
SPEAKER_DEVICE = _get("SPEAKER_DEVICE", "")          # part of the output device's name, or its number; blank = Windows default
SPEAKER_VOLUME = _get_int("SPEAKER_VOLUME", 80)      # 0..100, software volume (voice commands change it)
PTT_PORT = _get("PTT_PORT", "auto")                  # the board's COM port, e.g. COM5; "auto" finds it; "off" = Enter key only

# --- Local commands (no Gemini) ----------------------------------------
# "pause", "next song", "volume 40", "play Kesariya", "what's playing", "what time is it" are answered on the
# laptop in milliseconds: no tokens, no Gemini round trip. Anything else goes to Gemini as before.
LOCAL_COMMANDS = _get_bool("LOCAL_COMMANDS", True)
# A small sentence-embedding model (all-MiniLM-L6-v2, ~90 MB, downloaded once) also catches other wordings:
# "turn it down a bit", "skip this one". 0 = exact phrases only.
INTENT_MODEL = _get_bool("INTENT_MODEL", True)
INTENT_MODEL_REPO = _get("INTENT_MODEL_REPO", "sentence-transformers/all-MiniLM-L6-v2")
INTENT_MIN_SIMILARITY = _get_float("INTENT_MIN_SIMILARITY", 0.8)  # 0..1; python intents.py "..." shows the scores
INTENT_MARGIN = _get_float("INTENT_MARGIN", 0.05)    # the best command must beat the next best by this much
INTENT_MAX_WORDS = _get_int("INTENT_MAX_WORDS", 8)   # longer sentences always go to Gemini

# --- Text to speech ---------------------------------------------------
# edge  = Microsoft neural voice online (EDGE_VOICE), local voice as backup
# local = Piper or the Windows voice only (fully offline)
TTS_ENGINE = _get("TTS_ENGINE", "edge").lower()
EDGE_VOICE = _get("EDGE_VOICE", "en-IE-EmilyNeural")   # Irish female
EDGE_RATE = _get("EDGE_RATE", "+10%")                   # +10% = a touch faster (chosen by ear); -10% slower
EDGE_PITCH = _get("EDGE_PITCH", "+0Hz")
EDGE_TIMEOUT = _get_float("EDGE_TIMEOUT", 15.0)
EDGE_NO_BOSS_COMMA = _get_bool("EDGE_NO_BOSS_COMMA", True)   # "Got it, boss." is read as "Got it boss."
TTS_CACHE_MAX = _get_int("TTS_CACHE_MAX", 500)        # short replies kept as audio files (0 = off)
TTS_CACHE_DIR = _get("TTS_CACHE_DIR", str(HERE / "voice_cache"))
TTS_PARALLEL = _get_int("TTS_PARALLEL", 3)           # sentences of one reply the Edge voice makes at the same time
# Local voice: leave both blank to use the Windows voice. Set both to use Piper.
PIPER_EXE = _get("PIPER_EXE")
PIPER_MODEL = _get("PIPER_MODEL")
SAPI_VOICE = _get("SAPI_VOICE", "Zira")   # Windows voice when Piper isn't set; part of its name
SAPI_RATE = _get_int("SAPI_RATE", 175)    # words per minute

# --- Music -------------------------------------------------------------
MUSIC_DIR = _get("MUSIC_DIR", r"D:\songs")
ITUNES_BASE = _get("ITUNES_BASE", "https://itunes.apple.com")
ITUNES_COUNTRY = _get("ITUNES_COUNTRY", "IN")          # India store first (Bollywood), then US
ITUNES_GAP_S = _get_float("ITUNES_GAP_S", 3.2)         # Apple allows roughly 20 lookups a minute
ITUNES_BACKOFF_S = _get_float("ITUNES_BACKOFF_S", 60.0)  # wait after Apple says "too many requests"
MATCH_ACCEPT = _get_float("MATCH_ACCEPT", 0.8)         # at or above: trusted match
MATCH_REVIEW = _get_float("MATCH_REVIEW", 0.55)        # between REVIEW and ACCEPT: stored, needs_review
MUSIC_SCAN_MIN = _get_int("MUSIC_SCAN_MIN", 30)        # look for new songs this often (0 = only when asked)
MUSIC_AUTOPLAY = _get_bool("MUSIC_AUTOPLAY", True)     # keep playing other songs when the request runs out
FAST_MUSIC_REPLIES = _get_bool("FAST_MUSIC_REPLIES", True)  # "Here's Kesariya." without a second Gemini call
# The speaker plays 16 kHz mono, so songs are converted on the laptop with ffmpeg
# (a proper resampler; the S3's own one is made for speech). Converted copies are
# cached, so each song is converted once. Without ffmpeg, MP3s are sent as they are
# and M4A files can't play.
FFMPEG = _get("FFMPEG", "ffmpeg")                       # full path if it isn't on PATH
MUSIC_RATE = _get_int("MUSIC_RATE", 16000)             # the speaker's rate
MUSIC_BITRATE = _get("MUSIC_BITRATE", "64k")
MUSIC_LOUDNORM = _get_bool("MUSIC_LOUDNORM", True)     # even out loud and quiet songs
MUSIC_CACHE_DIR = _get("MUSIC_CACHE_DIR", str(HERE / "music_cache"))

# --- Network ----------------------------------------------------------
HTTP_PORT = _get_int("HTTP_PORT", 8765)
UDP_PORT = _get_int("UDP_PORT", 5006)                 # the firmware's DISCOVERY_PORT

# --- Audio ------------------------------------------------------------
SAMPLE_RATE = 16000
MIN_AUDIO_SECONDS = 0.3
MAX_AUDIO_SECONDS = 30
MAX_INPUT_CHARS = 500

# --- Speed ------------------------------------------------------------
# What each one does is explained under "Making replies faster" in the README.
# No first word from Gemini after this many seconds: ask again on a fresh connection and use
# whichever answers first (0 = never). This is what turns a 15 s stall into a 4 s one.
GEMINI_HEDGE_S = _get_float("GEMINI_HEDGE_S", 3.0)
# Reuse a weather lookup for this many seconds (0 = always ask), and keep your home city's
# weather fresh in the background so "what's the weather" is instant
WEATHER_CACHE_S = _get_int("WEATHER_CACHE_S", 600)
WEATHER_PREFETCH = _get_bool("WEATHER_PREFETCH", True)
WEATHER_TIMEOUT = _get_float("WEATHER_TIMEOUT", 4.0)        # seconds to wait for Open-Meteo
# Ping Google when the line has been quiet this long, so the connection never goes stale (0 = off)
GEMINI_KEEPALIVE_S = _get_int("GEMINI_KEEPALIVE_S", 40)
# Run Whisper once at start-up so the first question isn't the slow one
STT_WARMUP = _get_bool("STT_WARMUP", True)
# Start speaking while Gemini is still writing (0 = write the whole reply first, as before). A request
# can override this with an X-Stream: 0 or 1 header, which is how the benchmark compares the two.
STREAM_REPLIES = _get_bool("STREAM_REPLIES", True)
# A reply isn't started until Gemini has finished the round that writes it (so a music command can't
# arrive too late for it), unless this many milliseconds have passed since the first sentence of that round
STREAM_HOLD_MS = _get_int("STREAM_HOLD_MS", 1500)
# Say "One sec, boss." while a slow tool (web search, weather that isn't cached) runs
TOOL_FILLER = _get_bool("TOOL_FILLER", True)
# Streaming switches itself off (until the server is restarted) after this many turns in a row where
# it broke before sending anything; those turns were answered the normal way
STREAM_MAX_FAILS = _get_int("STREAM_MAX_FAILS", 3)

# --- Debug ------------------------------------------------------------
SAVE_AUDIO = _get_bool("SAVE_AUDIO", True)
LATENCY_LOG = _get_bool("LATENCY_LOG", True)          # one line per question in latency.log


def problems() -> list[str]:
    """Settings that stop the server from starting."""
    out = []
    if not GEMINI_API_KEY:
        out.append("GEMINI_API_KEY is not set in .env")
    