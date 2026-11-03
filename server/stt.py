"""Speech to text with faster-whisper, running on the laptop."""

from __future__ import annotations

import re
import time

import numpy as np

import config
from audio_utils import pcm_to_float

# Whisper sometimes "hears" these in silence or noise (subtitle credits from its
# training data). Nobody says them to a speaker, so a transcript that is only one
# of them counts as nothing heard. Same list as the Worker.
HALLUCINATIONS = {
    "", "you", "um", "uh", "hmm", "mm", "the end",
    "thanks for watching", "thank you for watching", "thanks for watching bye",
    "please subscribe", "subscribe", "like and subscribe",
    "subtitles by the amara org community", "transcribed by", "subtitles by",
}


def _echoes_hint(key: str) -> bool:
    """Whisper, given silence or noise, often just says its own prompt back ("A voice assistant named Friday."),
    and that would be sent to Gemini as if it were a question. Three or more words, nearly all of them from
    WHISPER_HINT = an echo. A real question has words of its own ("what", "is", "play"...)."""
    words = key.split()
    hint_words = re.sub(r"[^a-z0-9]+", " ", (config.WHISPER_HINT or "").lower()).split()
    if len(words) < 3 or not hint_words:
        return False
    if sum(w in set(hint_words) for w in words) / len(words) < 0.8:
        return False
    # ...and it must repeat a run of 4 words in the hint's own order (a real question that happens to use
    # hint words, like "is Friday a voice assistant", doesn't)
    n = min(4, len(hint_words))
    runs = {tuple(hint_words[i:i + n]) for i in range(len(hint_words) - n + 1)}
    return any(tuple(words[i:i + n]) in runs for i in range(len(words) - n + 1))


def clean_transcript(text: str) -> str:
    """Trim, and treat empty, known-hallucination or echoed-prompt transcripts as nothing heard."""
    t = re.sub(r"\s+", " ", text or "").strip()
    key = re.sub(r"[^a-z0-9؀-ۿऀ-ൿ]+", " ", t.lower()).strip()   # Latin and Indian scripts
    if key in HALLUCINATIONS or len(key) < 2 or _echoes_hint(key):
        return ""
    return t[: config.MAX_INPUT_CHARS]


class STT:
    def __init__(self) -> None:
        from faster_whisper import WhisperModel  # imported here so --help stays fast

        device, compute = config.WHISPER_DEVICE, config.WHISPER_COMPUTE
        try:
            print(f"[stt] loading Whisper '{config.WHISPER_MODEL}' ({device}/{compute})...")
            self._model = WhisperModel(config.WHISPER_MODEL, device=device, compute_type=compute)
        except Exception as exc:  # usually missing CUDA/cuDNN DLLs on Windows
            if device == "cpu":
                raise
            print(f"[stt] {device} failed ({exc}); using the CPU instead")
            self._model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
        print("[stt] ready")

    def warmup(self) -> None:
        """Run a second of hiss through Whisper, with and without the voice-activity filter, so the first real
        question isn't the one that pays for waking the model up (the first call after start-up is slower).
        Whatever it "hears" is thrown away, and a failure here never stops the server starting."""
        t0 = time.time()
        hiss = (np.random.default_rng(0).standard_normal(config.SAMPLE_RATE) * 800).astype(np.int16)
        try:
            for vad in (False, True):
                segments, _info = self._model.transcribe(pcm_to_float(hiss),
                                                         language=None if config.WHISPER_LANGUAGE == "auto" else config.WHISPER_LANGUAGE,
                                                         beam_size=1, vad_filter=vad,
                                                         condition_on_previous_text=False)
                for _ in segments:                 # the segments are made lazily: this is what runs the model
                    pass
        except Exception as exc:
            print(f"[stt] warm-up failed ({type(exc).__name__}: {exc}); the first question may be slower")
            return
        print(f"[stt] warmed up in {int((time.time() - t0) * 1000)} ms")

    language = "en"                            # the language of the last question (see transcribe)

    def _run(self, audio, language: str | None):
        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=1,
            vad_filter=True,                   # drops the silence a noisy mic leaves behind
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,  # stops one bad guess poisoning the next
            # the English hint would pull a Hindi sentence towards English
            initial_prompt=(config.WHISPER_HINT or None) if language == "en" else None,
        )
        return " ".join(s.text.strip() for s in segments).strip(), info

    def transcribe(self, pcm: np.ndarray) -> str:
        """pcm: int16 mono at config.SAMPLE_RATE. Returns the raw text it heard; self.language is the language
        it was in. With WHISPER_LANGUAGE=auto, Whisper picks the best of config.LANGUAGES (a language outside
        the list, e.g. Welsh heard in noise, is never chosen)."""
        audio = pcm_to_float(pcm)
        if config.WHISPER_LANGUAGE != "auto":
            self.language = config.WHISPER_LANGUAGE or "en"
            return self._run(audio, config.WHISPER_LANGUAGE)[0]
        import langs

        allowed = langs.enabled()
        text, info = self._run(audio, None)
        lang = info.language
        if lang not in allowed:
            probs = dict(getattr(info, "all_language_probs", None) or [])
            lang = max(allowed, key=lambda c: probs.get(c, 0.0))
            text, info = self._run(audio, lang)          # heard again as the language it most likely was
        self.language = lang
        return text
