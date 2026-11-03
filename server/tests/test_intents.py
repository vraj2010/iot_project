"""Local commands: what Friday answers by herself, and what she leaves to Gemini."""

import datetime
import re
import unittest
from unittest import mock

import _path  # noqa: F401

import config  # noqa: E402
import intents  # noqa: E402
import jukebox  # noqa: E402

SONGS = [(1, "Kesariya", "Arijit Singh", "Brahmastra", "Bollywood", 2022, 268.0, r"D:\songs\a.mp3", "a.mp3", "h1", 1, 1),
         (2, "Tum Hi Ho", "Arijit Singh", "Aashiqui 2", "Bollywood", 2013, 262.0, r"D:\songs\b.mp3", "b.mp3", "h2", 1, 1),
         (3, "Blinding Lights", "The Weeknd", "After Hours", "Pop", 2020, 200.0, r"D:\songs\c.mp3", "c.mp3", "h3", 1, 1)]


class FakeMemory:
    def _sql(self, query, params=None, fetch=None):
        if "count(*)" in query:
            return (len(SONGS), "t")
        if fetch == "all":
            return list(SONGS)
        return None


def library(state="playing", track=1, vol=70):
    with mock.patch.object(config, "MUSIC_SCAN_MIN", 0), mock.patch("builtins.print"):
        jb = jukebox.Jukebox(FakeMemory())
    jb.observe(f"state={state};track={track};pos=30;vol={vol}" if state != "stopped" else "state=stopped;vol=70")
    return jb


class Words:
    """A tiny stand-in for MiniLM: bag-of-words vectors, so similar wordings share words."""

    def __call__(self, texts):
        import numpy as np

        vocab = sorted({w for p in intents.EXAMPLES.values() for s in p for w in s.split()} | {"sec", "tune", "bit"})
        out = []
        for t in texts:
            v = np.array([float(w in t.split()) for w in vocab]) + 1e-6
            out.append(v / np.linalg.norm(v))
        return np.array(out)


class Cleaning(unittest.TestCase):
    def test_wake_word_politeness_and_punctuation_go(self):
        self.assertEqual(intents.normalize("Hey Friday, could you pause the music, please?"), "pause the music")
        self.assertEqual(intents.normalize("Jarvis. What's playing?"), "whats playing")
        self.assertEqual(intents.normalize("Okay, next song, thanks."), "next song")
        self.assertEqual(intents.normalize("Alexa, volume 40."), "volume 40")
        self.assertEqual(intents.normalize("  "), "")

    def test_numbers_in_digits_and_words(self):
        for said, n in (("40", 40), ("40%", 40), ("forty", 40), ("forty five", 45), ("a hundred", 100),
                        ("hundred", 100), ("seven", 7)):
            self.assertEqual(intents.number(said), n, said)
        for said in ("max", "loud", "forty forty", ""):
            self.assertIsNone(intents.number(said), said)


class Phrases(unittest.TestCase):
    def setUp(self):
        self.r = intents.Router()
        self.jb = library()

    def ask(self, said, jb="default"):
        return self.r.route(said, self.jb if jb == "default" else jb)

    def test_pause_in_its_usual_wordings(self):
        for said in ("pause", "Pause.", "Pause the music.", "Hey Friday, pause it please", "hold the song"):
            hit = self.ask(said)
            self.assertIsNotNone(hit, said)
            self.assertEqual((hit.intent, hit.reply, hit.action, hit.how), ("pause", "Paused.", {"music": "pause"}, "phrase"))

    def test_skip_back_restart_stop(self):
        nxt = self.ask("next song")
        self.assertEqual(nxt.intent, "next")
        self.assertEqual(nxt.action["music"], "play")
        self.assertNotEqual(nxt.action["track"], 1)
        self.assertTrue(nxt.reply.startswith("Here's "))
        self.assertEqual(self.ask("skip this one").intent, "next")
        self.assertEqual(self.ask("play it again").reply, "From the top.")
        self.assertEqual(self.ask("stop the music").action, {"music": "stop"})

    def test_volume(self):
        cases = {"volume 40": 40, "Volume 40%.": 40, "set the volume to forty five percent": 45, "volume max": 100,
                 "full volume": 100, "louder": 85, "a bit louder": 85, "turn it down": 55, "quieter": 55,
                 "set it to 30": 30, "volume at a hundred": 100}
        for said, level in cases.items():
            hit = self.ask(said)
            self.assertIsNotNone(hit, said)
            self.assertEqual(hit.action, {"volume": level}, said)
            self.assertEqual(hit.reply, f"Volume {level} percent.", said)
        self.assertIsNone(self.ask("volume 250"))                  # not a volume: let Gemini deal with it

    def test_play_a_song_an_artist_or_queue_it(self):
        hit = self.ask("Play Tum Hi Ho.")
        self.assertEqual((hit.intent, hit.action["track"]), ("play", 2))
        self.assertEqual(hit.reply, "Here's Tum Hi Ho by Arijit Singh.")
        queued = self.ask("play blinding lights next")
        self.assertEqual(queued.action, {})
        self.assertEqual(queued.reply, "Okay, Blinding Lights by The Weeknd up next.")
        self.assertEqual(self.jb.queue[0], 3)
        some = self.ask("play something by arijit")
        self.assertIn(some.action["track"], (1, 2))
        self.assertEqual(self.ask("add kesariya to the queue").reply, "Added Kesariya by Arijit Singh to the queue.")

    def test_play_music_shuffles_or_resumes(self):
        self.assertEqual(self.ask("play some music").action["music"], "play")
        paused = library(state="paused")
        self.assertEqual(self.ask("play", paused).action, {"music": "resume"})
        self.assertEqual(self.ask("resume", paused).reply, "Back on.")

    def test_whats_playing(self):
        self.assertEqual(self.ask("What's this song?").reply, "This is Kesariya by Arijit Singh.")
        self.assertEqual(self.ask("who sings this", library(state="paused")).reply,
                         "That is Kesariya by Arijit Singh, paused.")
        self.assertEqual(self.ask("whats playing", library(state="stopped")).reply, "Nothing's playing right now.")

    def test_nothing_playing_is_answered_without_gemini(self):
        idle = library(state="stopped")
        hit = self.ask("pause", idle)
        self.assertEqual((hit.reply, hit.action), ("Nothing's playing right now.", {}))
        self.assertEqual(self.ask("stop", idle).reply, "Nothing was playing.")

    def test_time_and_date_need_no_speaker(self):
        fixed = datetime.datetime(2026, 10, 24, 16, 5)
        with mock.patch.object(self.r, "_now", return_value=fixed):
            self.assertEqual(self.ask("what time is it", None).reply, "It's 4:05 PM.")
            self.assertEqual(self.ask("What's the date today?", None).reply, "It's Saturday, 24 October.")

    def test_music_without_a_music_speaker_goes_to_gemini(self):
        for said in ("pause", "volume 40", "play kesariya", "whats playing"):
            self.assertIsNone(self.ask(said, None), said)

    def test_everything_else_goes_to_gemini(self):
        for said in ("tell me a joke", "play the latest news", "add milk to my shopping list", "don't stop the music",
                     "what's the weather in Pune", "play a game with me", "remember that my bike is red", ""):
            self.assertIsNone(self.ask(said), said)


class Model(unittest.TestCase):
    def setUp(self):
        for name, value in (("INTENT_MIN_SIMILARITY", 0.6), ("INTENT_MARGIN", 0.05), ("INTENT_MAX_WORDS", 8)):
            p = mock.patch.object(config, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.r = intents.Router(embedder=Words())
        self.jb = library()

    def test_another_wording_of_a_command(self):
        hit = self.r.route("hold the music for a sec", self.jb)
        self.assertEqual(hit.intent, "pause")
        self.assertTrue(re.fullmatch(r"model 0\.\d\d", hit.how), hit.how)

    def test_guards_send_doubtful_ones_to_gemini(self):
        self.assertEqual(self.r.guess("hold the music for a sec")[0], "pause")
        for said in ("dont pause the music", "is the music too loud", "pause the music at 5", "play the music by the pool",
                     "hold the music for a second and then tell me about the history of the song in detail"):
            self.assertIsNone(self.r.guess(intents.normalize(said))[0], said)
        with mock.patch.object(config, "INTENT_MIN_SIMILARITY", 0.99):
            self.assertIsNone(self.r.guess("hold the music for a sec")[0])

    def test_scores_rank_every_command(self):
        ranked = self.r.scores("make it louder")
        self.assertEqual(ranked[0][0], "volume_up")
        self.assertEqual(len(ranked), len(intents.EXAMPLES))

    def test_no_model_means_phrases_only(self):
        r = intents.Router()
        self.assertEqual(r.scores("anything"), [])
        self.assertIsNone(r.route("hold the music for a sec", self.jb))
        self.assertEqual(r.route("pause", self.jb).intent, "pause")


if __name__ == "__main__":
    unittest.main()
