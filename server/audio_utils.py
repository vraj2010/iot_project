"""Small helpers for moving 16-bit mono audio around."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def bytes_to_pcm(raw: bytes) -> np.ndarray:
    """Raw little-endian 16-bit bytes -> int16 array."""
    if len(raw) % 2:
        raw = raw[:-1]
    return np.frombuffer(raw, dtype="<i2")


def pcm_to_float(pcm: np.ndarray) -> np.ndarray:
    """int16 -> float32 in the -1..1 range, which is what Whisper wants."""
    return pcm.astype(np.float32) / 32768.0


def resample(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resample. Plenty for speech on a small speaker, and no extra deps."""
    if src_rate == dst_rate or pcm.size == 0:
        return pcm.astype(np.int16)
    duration = pcm.size / src_rate
    dst_len = max(1, int(round(duration * dst_rate)))
    src_idx = np.linspace(0, pcm.size - 1, dst_len, dtype=np.float64)
    out = np.interp(src_idx, np.arange(pcm.size), pcm.astype(np.float64))
    return np.clip(out, -32768, 32767).astype(np.int16)


def normalize(pcm: np.ndarray, peak: float = 0.85, max_gain: float = 12.0) -> np.ndarray:
    """Scale so the loudest sample sits at `peak`, without over-amplifying silence."""
    if pcm.size == 0:
        return pcm
    current = int(np.max(np.abs(pcm)))
    if current < 32:
        return pcm
    gain = min((peak * 32767.0) / current, max_gain)
    return np.clip(pcm.astype(np.float32) * gain, -32768, 32767).astype(np.int16)


def save_wav(path: str | Path, pcm: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.astype("<i2").tobytes())


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a mono or stereo 16-bit WAV; stereo is mixed down to mono."""
    with wave.open(str(path), "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: expected 16-bit audio, found {width * 8}-bit")
    pcm = bytes_to_pcm(frames)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return pcm, rate


def split_sentences(text: str, max_chars: int = 180) -> list[str]:
    """Break a reply into speakable chunks so the speaker can start sooner."""
    parts: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        current = candidate
        ends_sentence = word.endswith((".", "!", "?", ":", ";", "।"))
        if ends_sentence or len(current) >= max_chars:
            parts.append(current)
            current = ""
    if current:
        parts.append(current)
    return parts or [text]
