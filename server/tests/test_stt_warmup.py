import unittest
from unittest import mock

import _path  # noqa: F401

import config
import numpy as np
import stt


class FakeModel:
    def __init__(self, fail=False):
        self.calls = []
        self.consumed = []
        self.fail = fail

    def transcribe(self, audio, **kw):
        if self.fail:
            raise RuntimeError("model broke")
        self.calls.append((audio, kw))
        n = len(self.calls)

        def segments():
            self.consumed.append(n)               # only counted if somebody reads the generator
            yield mock.Mock(text=" hmm ")

        return segments(), object()


def make(model):
    s = stt.STT.__new__(stt.STT)                  # without loading Whisper
    s._model = model
    return s


class Warmup(unittest.TestCase):
    def test_it_runs_the_model_with_and_without_the_voice_filter(self):
        model = FakeModel()
        make(model).warmup()
        self.assertEqual([kw["vad_filter"] for _a, kw in model.calls], [False, True])
        self.assertEqual(model.consumed, [1, 2])                  # the lazy segments were really read
        audio = model.calls[0][0]
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(len(audio), config.SAMPLE_RATE)
        self.assertTrue(0 < float(np.abs(audio).max()) < 0.2)      # quiet hiss, not silence and not loud

    def test_it_uses_the_same_settings_as_a_real_question(self):
        model = FakeModel()
        make(model).warmup()
        for _audio, kw in model.calls:
            self.assertEqual(kw["beam_size"], 1)
            self.assertEqual(kw["language"], None if config.WHISPER_LANGUAGE == "auto" else config.WHISPER_LANGUAGE)
            self.assertFalse(kw["condition_on_previous_text"])

    def test_a_failure_never_stops_the_server_starting(self):
        make(FakeModel(fail=True)).warmup()                        # prints a warning, raises nothing


if __name__ == "__main__":
    unittest.main()
