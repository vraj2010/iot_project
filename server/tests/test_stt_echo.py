"""Whisper saying its own prompt back (when it hears silence) is "nothing heard", not a question for Gemini."""

import unittest
from unittest import mock

import _path  # noqa: F401

import config  # noqa: E402
from stt import clean_transcript  # noqa: E402

HINT = "A person talking to a voice assistant named Friday."


class EchoTest(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(config, "WHISPER_HINT", HINT)
        p.start()
        self.addCleanup(p.stop)

    def test_the_prompt_coming_back_is_nothing_heard(self):
        for echo in ("The voice assistant named Friday.", "A voice assistant named Friday.",
                     "A person talking to a voice assistant named Friday.", "voice assistant named friday"):
            self.assertEqual(clean_transcript(echo), "", echo)

    def test_real_questions_survive(self):
        for q in ("Friday", "Friday, what time is it?", "What is a voice assistant?", "Play something by Arijit",
                  "Is Friday a voice assistant", "Hello.", "Volume 60", "what's the weather in Gandhinagar"):
            self.assertEqual(clean_transcript(q), q, q)

    def test_no_hint_no_filter(self):
        with mock.patch.object(config, "WHISPER_HINT", ""):
            self.assertEqual(clean_transcript("A voice assistant named Friday."), "A voice assistant named Friday.")


if __name__ == "__main__":
    unittest.main()
