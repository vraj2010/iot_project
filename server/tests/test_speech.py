import os
import tempfile
import threading
import time
import unittest
from unittest import mock

import _path  # noqa: F401

import config
import speech
from speech import SpeechSession, VoiceError
from voicefakes import FakeVoice, wait_until


def audio(voice, text):
    return voice.audio(text)


def collect(session):
    return list(session.pieces())


def datas(session):
    return [d for _tag, d, _first in collect(session)]


class Order(unittest.TestCase):
    def test_audio_comes_out_in_the_order_the_sentences_were_added(self):
        v = FakeVoice(hold=True)
        s = SpeechSession([v], parallel=3)
        s.add("First one.")
        s.add("Second one.")
        v.gate("Second one.").set()                        # the second is ready long before the first
        self.assertTrue(wait_until(lambda: v.done == ["Second one."]))
        self.assertIsNone(s.peek())                        # ...but the first sentence is still being made
        v.gate("First one.").set()
        s.close()
        self.assertEqual(datas(s), [audio(v, "First one."), audio(v, "Second one.")])
        self.assertEqual(v.done, ["Second one.", "First one."])

    def test_several_are_made_at_once_but_not_more_than_allowed(self):
        v = FakeVoice(hold=True)
        s = SpeechSession([v], parallel=2)
        for n in range(4):
            s.add(f"Sentence {n}.")
        self.assertTrue(wait_until(lambda: len(v.calls) == 2))
        time.sleep(0.1)
        self.assertEqual(sorted(v.calls), ["Sentence 0.", "Sentence 1."])      # the other two wait for a place
        v.gate("Sentence 0.").set()
        self.assertTrue(wait_until(lambda: len(v.calls) == 3))
        for n in range(4):
            v.gate(f"Sentence {n}.").set()
        s.close()
        self.assertEqual(len(datas(s)), 4)
        self.assertLessEqual(v.peak, 2)

    def test_audio_is_there_before_the_sentence_is_finished(self):
        v = FakeVoice(pieces=3)
        v.mid_gate["Long one."] = threading.Event()
        s = SpeechSession([v])
        s.add("Long one.")
        self.assertTrue(wait_until(lambda: s.peek() is not None))
        voice, ms = s.peek()
        self.assertIs(voice, v)
        self.assertGreaterEqual(ms, 0)
        it = s.pieces()
        tag, first, is_first = next(it)
        self.assertTrue(is_first)
        self.assertEqual(first, audio(v, "Long one.")[:len(first)])
        self.assertEqual(v.done, [])                       # the rest is still being made
        v.mid_gate["Long one."].set()
        s.close()
        rest = [d for _t, d, f in it]
        self.assertEqual(first + b"".join(rest), audio(v, "Long one."))

    def test_tags_and_the_first_piece_of_each_sentence(self):
        v = FakeVoice(pieces=2)
        s = SpeechSession([v])
        s.add("One sec, boss.", tag="filler")
        s.add("The answer.")
        s.close()
        got = collect(s)
        self.assertEqual([(t, f) for t, _d, f in got], [("filler", True), ("filler", False), (None, True), (None, False)])

    def test_empty_text_is_ignored_and_nothing_can_be_added_after_close(self):
        s = SpeechSession([FakeVoice()])
        self.assertFalse(s.add("   "))
        self.assertTrue(s.add("Hello there."))
        s.close()
        self.assertFalse(s.add("Too late."))
        self.assertEqual(len(collect(s)), 1)

    def test_pieces_waits_for_more_until_closed(self):
        v = FakeVoice()
        s = SpeechSession([v])
        got = []
        t = threading.Thread(target=lambda: got.extend(d for _t, d, _f in s.pieces()), daemon=True)
        t.start()
        s.add("One.")
        self.assertTrue(wait_until(lambda: got == [audio(v, "One.")]))
        self.assertTrue(t.is_alive())                      # still waiting: more may come
        s.add("Two.")
        s.close()
        t.join(3)
        self.assertFalse(t.is_alive())
        self.assertEqual(got, [audio(v, "One."), audio(v, "Two.")])


class Dropping(unittest.TestCase):
    def test_everything_not_yet_sent_can_be_forgotten(self):
        v = FakeVoice(hold=True)
        s = SpeechSession([v])
        s.add("A draft that turns out to be a tool round.")
        s.add("Another one.")
        self.assertTrue(wait_until(lambda: len(v.calls) == 2))
        self.assertEqual(s.drop_unsent(), 2)
        v.gate("A draft that turns out to be a tool round.").set()
        v.gate("Another one.").set()
        self.assertTrue(wait_until(lambda: len(v.stopped) == 2))   # their voices were told to stop
        s.add("The real answer.")
        v.gate("The real answer.").set()
        s.close()
        self.assertEqual(datas(s), [audio(v, "The real answer.")])

    def test_dropped_sentences_do_not_hold_up_the_next_ones(self):
        v = FakeVoice(hold=True)
        s = SpeechSession([v], parallel=1)
        s.add("Draft.")
        self.assertTrue(wait_until(lambda: v.calls == ["Draft."]))
        s.drop_unsent()
        s.add("Answer.")                                   # does not wait for the draft's thread to finish
        self.assertTrue(wait_until(lambda: v.calls == ["Draft.", "Answer."]))
        v.gate("Answer.").set()
        s.close()
        self.assertEqual(datas(s), [audio(v, "Answer.")])
        v.gate("Draft.").set()

    def test_the_sentence_that_is_going_out_stays(self):
        v = FakeVoice(pieces=2)
        s = SpeechSession([v])
        for text in ("Going out now.", "Not yet.", "Nor this."):
            s.add(text)
        self.assertTrue(wait_until(lambda: len(v.done) == 3))
        it = s.pieces()
        _tag, first, _f = next(it)
        self.assertEqual(s.drop_unsent(), 2)
        s.close()
        rest = b"".join(d for _t, d, _f in it)
        self.assertEqual(first + rest, audio(v, "Going out now."))

    def test_abort_wakes_a_reader_that_is_waiting(self):
        s = SpeechSession([FakeVoice()])
        ended = threading.Event()

        def read():
            list(s.pieces())
            ended.set()

        threading.Thread(target=read, daemon=True).start()
        time.sleep(0.05)
        s.abort()
        self.assertTrue(ended.wait(2))
        self.assertFalse(s.add("Nobody is listening."))


class Failing(unittest.TestCase):
    def two_voices(self):
        return FakeVoice("edge"), FakeVoice("local", fmt="pcm")

    def test_the_first_sentence_falls_back_to_the_next_voice(self):
        edge, local = self.two_voices()
        edge.fail["Hello there."] = RuntimeError("offline")
        s = SpeechSession([edge, local])
        s.add("Hello there.")
        self.assertTrue(wait_until(lambda: s.peek() is not None))
        self.assertIs(s.peek()[0], local)
        s.close()
        self.assertEqual(datas(s), [audio(local, "Hello there.")])

    def test_later_sentences_stay_with_the_voice_the_reply_started_in(self):
        edge, local = self.two_voices()
        edge.fail["Hello there."] = RuntimeError("offline")
        s = SpeechSession([edge, local])
        s.add("Hello there.")
        self.assertTrue(wait_until(lambda: s.peek() is not None))
        it = s.pieces()
        next(it)                                           # the reply is now in the local voice
        s.add("And more.")
        s.close()
        self.assertEqual([d for _t, d, _f in it], [audio(local, "And more.")])
        self.assertEqual(edge.calls, ["Hello there."])     # the voice that failed isn't asked again

    def test_a_sentence_made_by_the_other_voice_is_made_again(self):
        edge, local = self.two_voices()
        edge.fail["First."] = RuntimeError("offline")
        edge.hold = False
        s = SpeechSession([edge, local])
        s.add("First.")
        s.add("Second.")                                   # edge makes this one, before the reply has a voice
        self.assertTrue(wait_until(lambda: edge.done == ["Second."] and s.peek() is not None))
        s.close()
        self.assertEqual(datas(s), [audio(local, "First."), audio(local, "Second.")])

    def test_a_sentence_that_cannot_be_made_ends_the_reply(self):
        v = FakeVoice()
        v.fail["Two."] = RuntimeError("edge voice failed: boom")
        s = SpeechSession([v])
        for text in ("One.", "Two.", "Three."):
            s.add(text)
        s.close()
        self.assertEqual(datas(s), [audio(v, "One.")])
        self.assertIn("boom", str(s.error))

    def test_nothing_can_be_said_at_all(self):
        edge, local = self.two_voices()
        edge.fail["Hello."] = RuntimeError("offline")
        local.fail["Hello."] = ValueError("no voices installed")
        s = SpeechSession([edge, local])
        s.add("Hello.")
        self.assertTrue(wait_until(lambda: s.head_error() is not None))
        self.assertIsNone(s.peek())
        message = str(s.head_error())
        self.assertIn("offline", message)
        self.assertIn("no voices installed", message)

    def test_a_voice_that_makes_nothing_is_a_failure(self):
        edge, local = self.two_voices()
        edge.silent.add("Hello.")
        s = SpeechSession([edge, local])
        s.add("Hello.")
        s.close()
        self.assertEqual(datas(s), [audio(local, "Hello.")])

    def test_a_voice_that_breaks_half_way_is_not_replaced_by_another(self):
        edge, local = self.two_voices()
        edge.pieces = 2
        edge.fail_after["Cut off."] = RuntimeError("connection lost")
        s = SpeechSession([edge, local])
        s.add("Cut off.")
        s.add("Never said.")
        s.close()
        got = datas(s)
        self.assertEqual(len(got), 1)                       # what was made before the break is kept
        self.assertTrue(audio(edge, "Cut off.").startswith(got[0]))
        self.assertEqual(local.calls, [])
        self.assertIn("connection lost", str(s.error))

    def test_a_silent_voice_is_given_up_on(self):
        v = FakeVoice(hold=True)
        s = SpeechSession([v], stall_s=0.2)
        s.add("Hello there.")
        t0 = time.time()
        self.assertEqual(collect(s), [])
        self.assertLess(time.time() - t0, 3)
        self.assertIn("stalled", str(s.error))
        v.gate("Hello there.").set()


class Stalling(unittest.TestCase):
    """A voice that takes the request and never answers (a connection that died quietly) must not hold the
    reply up until its own long time-out. Microsoft's voice was the one that did this."""

    def hung_and_working(self):
        edge = FakeVoice("edge", hold=True)                 # waits for a gate nobody opens
        local = FakeVoice("local", fmt="pcm")
        return edge, local

    def test_the_first_sentence_moves_to_the_next_voice_while_the_reply_has_not_begun(self):
        edge, local = self.hung_and_working()
        s = SpeechSession([edge, local], stall_s=0.2)
        s.add("Hello there.")
        t0 = time.time()
        # what a reader does while it waits to see whether there is a reply: look, and ask for a check
        self.assertTrue(wait_until(lambda: (s.poll(), s.peek())[1] is not None))
        self.assertLess(time.time() - t0, 2)
        self.assertIs(s.peek()[0], local)
        s.close()
        self.assertEqual(datas(s), [audio(local, "Hello there.")])
        edge.gate("Hello there.").set()                     # the old request finally comes back: ignored
        self.assertTrue(wait_until(lambda: edge.stopped == ["Hello there."]))
        self.assertEqual(datas(s), [])

    def test_the_voice_that_went_quiet_is_told_so(self):
        edge, local = self.hung_and_working()
        told = []
        edge.stalled = lambda: told.append("edge")
        s = SpeechSession([edge, local], stall_s=0.2)
        s.add("Hello there.")
        self.assertTrue(wait_until(lambda: (s.poll(), s.peek())[1] is not None))
        self.assertEqual(told, ["edge"])
        edge.gate("Hello there.").set()

    def test_a_reader_that_is_already_taking_audio_moves_on_too(self):
        edge, local = self.hung_and_working()
        s = SpeechSession([edge, local], stall_s=0.2)
        s.add("Hello there.")
        s.close()
        self.assertEqual(datas(s), [audio(local, "Hello there.")])
        edge.gate("Hello there.").set()

    def test_with_nothing_to_move_to_the_first_sentence_gets_an_error(self):
        v = FakeVoice(hold=True)
        s = SpeechSession([v], stall_s=0.2)
        s.add("Hello there.")
        self.assertTrue(wait_until(lambda: (s.poll(), s.head_error())[1] is not None))
        self.assertIn("stalled", str(s.head_error()))
        self.assertIsNone(s.peek())
        v.gate("Hello there.").set()

    def test_once_the_reply_has_a_voice_a_stalled_sentence_ends_the_reply(self):
        edge, local = self.hung_and_working()
        edge.hold = False
        s = SpeechSession([edge, local], stall_s=0.2)
        s.add("One.")
        it = s.pieces()
        self.assertEqual(next(it)[1], audio(edge, "One."))
        edge.hold = True
        s.add("Two.")
        s.close()
        self.assertEqual(list(it), [])
        self.assertIn("stalled", str(s.error))
        self.assertEqual(local.calls, [])                   # MP3 and PCM can't be mixed in one reply
        edge.gate("Two.").set()

    def test_a_late_answer_from_the_old_request_leaves_the_count_of_places_alone(self):
        edge, local = self.hung_and_working()
        s = SpeechSession([edge, local], parallel=1, stall_s=0.2)
        s.add("One.")
        s.add("Two.")                                       # has to wait for the only place
        s.close()
        self.assertEqual(datas(s), [audio(local, "One."), audio(local, "Two.")])
        edge.gate("One.").set()
        edge.gate("Two.").set()
        self.assertTrue(wait_until(lambda: edge.active == 0))
        self.assertTrue(wait_until(lambda: s._running == 0))
        self.assertLessEqual(local.peak, 1)


class SharedLock(unittest.TestCase):
    def test_the_caller_can_wait_on_the_same_lock(self):
        cv = threading.Condition()
        v = FakeVoice()
        s = SpeechSession([v], cond=cv)
        self.assertIs(s.cv, cv)
        woke = threading.Event()

        def wait():
            with cv:
                while s.peek() is None:
                    cv.wait(2)
            woke.set()

        threading.Thread(target=wait, daemon=True).start()
        s.add("Wake up.")
        self.assertTrue(woke.wait(2))


class EdgeVoiceCache(unittest.TestCase):
    """Edge's speak(): a line said before comes from the cache, and short lines are kept."""

    def setUp(self):
        import tts

        self.tts = tts
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        for name, value in (("TTS_CACHE_DIR", self.dir.name), ("TTS_CACHE_MAX", 50)):
            p = mock.patch.object(config, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.voice = tts._EdgeVoice.__new__(tts._EdgeVoice)      # without needing edge-tts installed
        self.made = []

        def synth(text, put):
            self.made.append(text)
            put(b"a" * 150)
            put(b"b" * 150)

        self.voice._synth_into = synth

    def speak(self, text):
        got = []
        self.voice.speak(text, got.append)
        return got

    def test_a_short_line_is_made_once_and_then_comes_from_the_cache(self):
        self.assertEqual(self.speak("Done, boss."), [b"a" * 150, b"b" * 150])
        self.assertEqual(self.speak("Done, boss."), [b"a" * 150 + b"b" * 150])
        self.assertEqual(self.made, ["Done, boss."])

    def test_a_long_sentence_is_not_kept(self):
        text = "This sentence is rather long, certainly longer than the eighty characters a cached line may have."
        self.speak(text)
        self.speak(text)
        self.assertEqual(self.made, [text, text])
        self.assertEqual(os.listdir(self.dir.name), [])

    def test_a_sentence_that_was_stopped_is_not_kept(self):
        def stopped(text, put):
            put(b"a" * 150)
            raise speech.Stop()

        self.voice._synth_into = stopped
        with self.assertRaises(speech.Stop):
            self.speak("Never finished.")
        self.assertEqual(os.listdir(self.dir.name), [])


class EdgeVoiceHealth(unittest.TestCase):
    """Does Microsoft's voice look reachable? A line from the voice cache is only a good way to start a reply if
    it does: the cache makes Edge the voice of the whole reply, and if Edge is down the reply ends after that line."""

    def setUp(self):
        import sys
        import types

        import tts

        self.tts = tts
        self.mode = "ok"
        outer = self

        class Communicate:
            def __init__(self, text, voice, rate=None, pitch=None, **kw):
                pass

            async def stream(self):
                if outer.mode == "down":
                    raise OSError("network is unreachable")
                if outer.mode == "ok":
                    yield {"type": "audio", "data": b"a" * 300}

        patcher = mock.patch.dict(sys.modules, {"edge_tts": types.SimpleNamespace(Communicate=Communicate)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        for name, value in (("TTS_CACHE_DIR", self.dir.name), ("TTS_CACHE_MAX", 50)):
            p = mock.patch.object(config, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.voice = tts._EdgeVoice.__new__(tts._EdgeVoice)

    def speak(self, text="Hello there friend."):
        got = []
        self.voice.speak(text, got.append)
        return got

    def test_it_looks_up_until_a_request_fails_and_again_after_one_works(self):
        self.assertTrue(self.voice.looks_up())
        self.mode = "down"
        with self.assertRaises(self.tts.TTSError):
            self.speak()
        self.assertFalse(self.voice.looks_up())
        self.mode = "ok"
        self.assertEqual(self.speak("Another one."), [b"a" * 300])
        self.assertTrue(self.voice.looks_up())

    def test_a_voice_that_returns_no_audio_looks_down(self):
        self.mode = "empty"
        with self.assertRaises(self.tts.TTSError):
            self.speak()
        self.assertFalse(self.voice.looks_up())

    def test_a_line_from_the_cache_says_nothing_about_the_service(self):
        self.mode = "down"
        with self.assertRaises(self.tts.TTSError):
            self.speak("A fresh one.")
        self.tts._cache_store("Done, boss.", b"a" * 300)
        self.assertEqual(self.speak("Done, boss."), [b"a" * 300])
        self.assertFalse(self.voice.looks_up())

    def test_a_sentence_that_was_stopped_says_nothing_either(self):
        def stop(data):
            raise speech.Stop()

        with self.assertRaises(speech.Stop):
            self.voice.speak("Hello there friend.", stop)
        self.assertTrue(self.voice.looks_up())


class LocalVoice(unittest.TestCase):
    def test_one_piece_of_16_bit_audio(self):
        import numpy as np
        import tts

        class Raw:
            name, provider = "raw", 5

            def synth(self, text):
                return np.array([1, -2, 3], dtype=np.int16)

        local = tts._Local(Raw())
        got = []
        local.speak("hi", got.append)
        self.assertEqual(got, [np.array([1, -2, 3], dtype="<i2").tobytes()])
        self.assertEqual(local.fmt, "pcm")


class TtsSession(unittest.TestCase):
    def test_tts_hands_out_a_session_with_its_voices_in_order(self):
        import tts

        t = tts.TTS.__new__(tts.TTS)
        t.edge, t.local = FakeVoice("edge"), FakeVoice("local", fmt="pcm")
        with mock.patch.object(config, "TTS_PARALLEL", 2):
            s = t.session()
        self.assertEqual([v.name for v in s._voices], ["edge", "local"])
        self.assertEqual(s._parallel, 2)

    def test_the_filler_is_only_fine_while_the_first_voice_looks_up(self):
        import tts

        t = tts.TTS.__new__(tts.TTS)
        edge, t.local = FakeVoice("edge"), FakeVoice("local", fmt="pcm")
        t.edge = edge
        self.assertTrue(t.filler_ok())                      # a voice that keeps no track is taken as fine
        edge.looks_up = lambda: False
        self.assertFalse(t.filler_ok())
        t.edge = None                                       # only the laptop's own voice: no cache is involved
        self.assertTrue(t.filler_ok())


if __name__ == "__main__":
    unittest.main()
