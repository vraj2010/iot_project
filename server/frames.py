"""Reply frames the S3 already understands (same bytes as the Cloudflare Worker sends).

All integers are little-endian uint32.

  /say  "I2S2" | format | sample_rate | provider | chat_ms | tts_ms | text_len
               | text | audio...
  /ask  "I2S3" | format | sample_rate | provider | chat_ms | tts_ms | stt_ms
               | heard_len | text_len | heard | text | audio...
  both  "I2S4" | format | sample_rate | provider | chat_ms | tts_ms | stt_ms
               | heard_len | text_len | action_len | heard | text | action | audio...
        Sent to speakers that can play music (they send an X-Music-State header).
        action is JSON for the speaker, {} when there's nothing to do, e.g.
        {"music": "play", "track": 12, "title": "Kesariya by Arijit Singh"},
        {"music": "pause"} (also resume, stop), {"volume": 60}.

format 1 = PCM s16le mono 16 kHz, 2 = MP3 (several MP3s back to back, one per
sentence; the S3's decoder plays them as one stream). Audio has no length field: it is simply the rest of
the response, which is why it can be streamed sentence by sentence.
"""

from __future__ import annotations

import json
import struct

FORMAT_PCM = 1
FORMAT_MP3 = 2


def _fmt(fmt: str) -> tuple[int, int]:
    """(format code, sample rate field) for "pcm" (16 kHz) or "mp3" (rate is in the MP3 itself)."""
    return (FORMAT_MP3, 0) if fmt == "mp3" else (FORMAT_PCM, 16000)


def say_header(text: str, provider: int, chat_ms: int, tts_ms: int, fmt: str = "pcm") -> bytes:
    code, rate = _fmt(fmt)
    t = text.encode("utf-8")
    return b"I2S2" + struct.pack("<6I", code, rate, provider, chat_ms, tts_ms, len(t)) + t


def ask_header(heard: str, text: str, provider: int, chat_ms: int, tts_ms: int, stt_ms: int,
               fmt: str = "pcm") -> bytes:
    code, rate = _fmt(fmt)
    h = heard.encode("utf-8")
    t = text.encode("utf-8")
    return (b"I2S3" + struct.pack("<8I", code, rate, provider, chat_ms, tts_ms, stt_ms, len(h), len(t))
            + h + t)


def reply_header(heard: str | None, text: str, provider: int, chat_ms: int, tts_ms: int, stt_ms: int,
                 fmt: str, action: dict | None) -> bytes:
    code, rate = _fmt(fmt)
    h = (heard or "").encode("utf-8")
    t = text.encode("utf-8")
    a = json.dumps(action or {}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return (b"I2S4" + struct.pack("<9I", code, rate, provider, chat_ms, tts_ms, stt_ms, len(h), len(t), len(a))
            + h + t + a)


def parse(data: bytes) -> dict:
    """Decode a whole frame (used by client.py and the tests)."""
    magic = data[:4]
    action = None
    if magic == b"I2S4":
        fmt, rate, provider, chat_ms, tts_ms, stt_ms, hl, tl, al = struct.unpack_from("<9I", data, 4)
        pos = 40
        heard = data[pos:pos + hl].decode("utf-8")
        pos += hl
        text = data[pos:pos + tl].decode("utf-8")
        pos += tl
        action = json.loads(data[pos:pos + al] or b"{}")
        return {"format": fmt, "rate": rate, "provider": provider, "chat_ms": chat_ms, "tts_ms": tts_ms,
                "stt_ms": stt_ms, "heard": heard, "text": text, "action": action, "audio": data[pos + al:]}
    if magic == b"I2S3":
        fmt, rate, provider, chat_ms, tts_ms, stt_ms, hl, tl = struct.unpack_from("<8I", data, 4)
        pos = 36
        heard = data[pos:pos + hl].decode("utf-8")
        pos += hl
    elif magic == b"I2S2":
        fmt, rate, provider, chat_ms, tts_ms, tl = struct.unpack_from("<6I", data, 4)
        stt_ms, heard, pos = 0, None, 28
    else:
        raise ValueError(f"not an I2S frame: {magic!r}")
    text = data[pos:pos + tl].decode("utf-8")
    pos += tl
    return {"format": fmt, "rate": rate, "provider": provider, "chat_ms": chat_ms, "tts_ms": tts_ms,
            "stt_ms": stt_ms, "heard": heard, "text": text, "action": action, "audio": data[pos:]}
