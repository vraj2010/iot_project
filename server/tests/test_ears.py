"""The laptop's ears: wake word -> recording -> hand-off to the speaker, and never waking on her own voice."""

import threading
import time
import unittest
from unittest import mock

import numpy as np

import _path  # noqa: F401

import config  # noqa: E402
from ears import endpoint  # noqa: E402
from ears.listener import HANDOFF_S, IDLE, LISTENING, Ears  # noqa: E402
from ears.mic import FRAME  # noqa: E402

FRAME_S = 0.08


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Script:
    """A wake word or a speech detector whose next value the test sets."""

    def __init__(self, value=0.0):
        self.value = value
        self.resets = 0
        self.phrase = "hey jarvis"

    def score(self, frame):
        return self.value

    prob = score

    def reset(self):
        self.resets += 1


class Base(unittest.TestCase):
    def setUp(self):
        for name, value in (("WAKE_THRESHOLD", 0.6), ("WAKE_COOLDOWN_S", 0.8), ("WAKE_DEBUG", False),
                            ("EARS_START_S", 5.0), ("EARS_SILENCE_MS", 800), ("EARS_MAX_S", 12.0),
                            ("EARS_VAD_THRESHOLD", 0.5)):
            p = mock.patch.object(config, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch("builtins.print")
        self.printed = p.start()
        self.addCleanup(p.stop)
        self.clock = Clock()
        self.wake, self.vad = Script(), Script()
        self.ears = Ears(wake=self.wake, vad=self.vad, clock=self.clock)

    def frames(self, n, wake=0.0, speech=0.0):
        self.wake.value, self.vad.value = wake, speech
        for _ in range(n):
            self.ears.feed(np.zeros(FRAME, np.int16))
            self.clock.t += FRAME_S
        self.wake.value = self.vad.value = 0.0

    def poll(self):
        """An idle poll that returns at once: the speaker is connected and idle."""
        return self.ears.wait_for_wake(0)

    def wake_up(self):
        self.poll()
        self.frames(1, wake=0.9)
        wake_id = self.poll()
        self.assertIsNotNone(wake_id)
        return wake_id

    def said(self):
        return " ".join(str(c.args[0]) for c in self.printed.call_args_list if c.args)


class Endpointing(unittest.TestCase):
    def feed(self, ep, probs):
        return [ep.feed(p) for p in probs]

    def test_speech_then_silence_is_done(self):
        ep = endpoint.Endpointer(silence_ms=800)
        out = self.feed(ep, [0.0] * 3 + [0.9] * 6 + [0.1] * 10)
        self.assertEqual(out[:3], [endpoint.WAITING] * 3)
        self.assertEqual(out[8], endpoint.SPEAKING)
        self.assertEqual(out.index(endpoint.DONE), 3 + 6 + 10 - 1)     # 10 quiet frames = 800 ms

    def test_a_blip_is_not_a_question(self):
        ep = endpoint.Endpointer(start_ms=800, min_speech_ms=240)
        out = self.feed(ep, [0.9, 0.9] + [0.0] * 8)                    # 160 ms: the tail of "jarvis"
        self.assertEqual(out[-1], endpoint.NOTHING)

    def test_in_between_scores_keep_a_sentence_going(self):
        ep = endpoint.Endpointer(silence_ms=240)
        out = self.feed(ep, [0.9] * 4 + [0.4] * 10 + [0.0] * 3)         # 0.4: neither speech nor silence
        self.assertNotIn(endpoint.DONE, out[:14])
        self.assertEqual(out[-1], endpoint.DONE)

    def test_too_long_keeps_what_was_said(self):
        ep = endpoint.Endpointer(max_ms=800)
        self.assertEqual(self.feed(ep, [0.9] * 10)[-1], endpoint.TOO_LONG)


class HandOff(Base):
    def test_wake_word_then_question(self):
        wake_id = self.wake_up()
        self.assertEqual(self.ears.state, LISTENING)
        self.frames(6, speech=0.9)                                     # "what's the weather"
        self.assertIsNone(self.ears.question)
        self.frames(10)                                                # 800 ms of silence
        self.assertEqual(self.ears.state, IDLE)
        pcm = self.ears.wait_for_question(wake_id, 0)
        self.assertEqual(pcm.size, 16 * FRAME)
        self.assertIsNone(self.ears.wait_for_question(wake_id, 0))     # handed over once

    def test_an_idle_poll_waits_for_the_wake_word(self):
        got = []
        t = threading.Thread(target=lambda: got.append(self.ears.wait_for_wake(5)))
        t.start()
        self.assertTrue(_until(lambda: self.ears.polling == 1))
        self.frames(1, wake=0.9)
        t.join(2)
        self.assertEqual(got, [1])

    def test_an_awake_poll_waits_for_the_end_of_the_question(self):
        wake_id = self.wake_up()
        got = []
        t = threading.Thread(target=lambda: got.append(self.ears.wait_for_question(wake_id, 5)))
        t.start()
        time.sleep(0.05)
        self.frames(5, speech=0.9)
        self.assertEqual(got, [])
        self.frames(10)
        t.join(2)
        self.assertEqual(got[0].size, 15 * FRAME)

    def test_nothing_said_after_the_wake_word(self):
        wake_id = self.wake_up()
        self.frames(int(5 / FRAME_S) + 1)
        self.assertEqual(self.ears.state, IDLE)
        self.assertIsNone(self.ears.wait_for_question(wake_id, 0))
        self.assertIn("nothing said", self.said())
        self.assertFalse(self.ears.muted)                              # no reply coming: still awake

    def test_no_speaker_no_wake(self):
        self.frames(1, wake=0.9)                                       # nobody has polled
        self.assertEqual(self.ears.state, IDLE)
        self.assertIn("isn't connected", self.said())

    def test_a_wake_word_nobody_picks_up_is_dropped(self):
        self.poll()
        self.frames(1, wake=0.9)
        self.assertEqual(self.ears.state, LISTENING)
        self.frames(int(HANDOFF_S / FRAME_S) + 2, speech=0.9)
        self.assertEqual(self.ears.state, IDLE)
        self.assertIn("didn't answer", self.said())

    def test_the_speaker_moving_on_stops_the_listening(self):
        self.wake_up()
        self.frames(2, speech=0.9)
        self.assertIsNone(self.poll())                                 # idle again: it gave up on that wake
        self.assertEqual(self.ears.state, IDLE)

    def test_an_old_wake_id_gets_nothing(self):
        wake_id = self.wake_up()
        self.assertIsNone(self.ears.wait_for_question(wake_id + 1, 0))

    def test_the_wake_word_model_is_reset_after_it_fires(self):
        self.wake_up()
        before = self.wake.resets
        self.frames(1, speech=0.9)
        self.assertEqual(self.wake.resets, before + 1)

    def test_boot_button_listens_without_the_wake_word(self):
        self.poll()
        self.assertTrue(self.ears.wake_now())
        self.assertEqual((self.ears.state, self.ears.source), (LISTENING, "button"))
        self.assertEqual(self.poll(), 1)
        self.assertFalse(self.ears.wake_now())                         # already listening


class NeverHerOwnVoice(Base):
    def question(self):
        wake_id = self.wake_up()
        self.frames(4, speech=0.9)
        self.frames(10)
        return self.ears.wait_for_question(wake_id, 0)

    def test_deaf_from_the_question_until_the_speaker_polls_again(self):
        self.assertIsNotNone(self.question())
        with self.ears.busy():                                         # the reply is being made and sent
            self.frames(1, wake=0.95)
            self.assertEqual(self.ears.state, IDLE)
        self.frames(1, wake=0.95)                                      # sent, but maybe still playing
        self.assertEqual(self.ears.state, IDLE)
        self.poll()                                                    # finished playing: now listening again...
        self.frames(1, wake=0.95)
        self.assertEqual(self.ears.state, IDLE)                        # ...after the echo has died down
        self.clock.t += config.WAKE_COOLDOWN_S
        self.frames(1, wake=0.95)
        self.assertEqual(self.ears.state, LISTENING)

    def test_typed_and_push_to_talk_turns_mute_too(self):
        self.poll()
        with self.ears.busy():
            self.frames(1, wake=0.95)
            self.assertFalse(self.ears.wake_now())
        self.assertEqual(self.ears.state, IDLE)
        self.assertTrue(self.ears.muted)

    def test_a_turn_answers_waiting_idle_polls_at_once(self):
        got = []
        t = threading.Thread(target=lambda: got.append(self.ears.wait_for_wake(5)))
        t.start()
        self.assertTrue(_until(lambda: self.ears.polling == 1))
        with self.ears.busy():
            t.join(2)
        self.assertEqual(got, [None])

    def test_a_poll_that_arrives_during_a_turn_is_answered_when_it_ends(self):
        got = []
        with self.ears.busy():
            t = threading.Thread(target=lambda: got.append(self.ears.wait_for_wake(5)))
            t.start()
            self.assertTrue(_until(lambda: self.ears.polling == 1))
            self.assertEqual(got, [])
        t.join(2)
        self.assertEqual(got, [None])
        self.assertTrue(self.ears.muted)                               # still deaf: the reply may be playing
        self.poll()                                                    # the next poll is the speaker being done
        self.assertFalse(self.ears.muted)

    def test_a_typed_question_beats_a_wake_word_not_yet_picked_up(self):
        self.poll()
        self.frames(1, wake=0.9)
        with self.ears.busy():
            self.assertEqual(self.ears.state, IDLE)

    def test_status(self):
        s = self.ears.status()
        self.assertEqual((s["wake_word"], s["state"], s["deaf"]), ("hey jarvis", "idle", False))


def _until(cond, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


if __name__ == "__main__":
    unittest.main()
