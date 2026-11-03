import json
import threading
import time
import unittest

import _path  # noqa: F401

import replystream
from replystream import FILLER, StreamingReply
from speech import SpeechSession
from voicefakes import FakeVoice, wait_until


class FakeTransport:
    """The HTTP response, written down. fail_on_write=N makes the Nth write fail like a hung-up socket."""

    def __init__(self, fail_on_write=None):
        self.events = []
        self.fail_on_write = fail_on_write
        self._writes = 0
        self._lock = threading.Lock()

    def begin(self):
        with self._lock:
            self.events.append(("begin",))

    def write(self, data):
        with self._lock:
            self._writes += 1
            if self.fail_on_write == self._writes:
                raise BrokenPipeError("the speaker hung up")
            self.events.append(("write", bytes(data)))

    def end(self):
        with self._lock:
            self.events.append(("end",))

    def kinds(self):
        with self._lock:
            return [e[0] for e in self.events]

    def body(self):
        with self._lock:
            return b"".join(e[1] for e in self.events if e[0] == "write")


class Header:
    """Builds a frame header we can read back, and remembers what it was asked for."""

    def __init__(self):
        self.calls = []

    def __call__(self, voice, text, chat_ms, tts_ms, actions):
        self.calls.append({"voice": voice, "text": text, "chat_ms": chat_ms, "tts_ms": tts_ms, "actions": actions})
        return b"H" + json.dumps({"text": text, "actions": actions, "fmt": voice.fmt}).encode() + b"|"


def parse_body(body):
    """(header dict, audio) of what the speaker received."""
    assert body.startswith(b"H"), body[:20]
    head, _, audio = body[1:].partition(b"|")
    return json.loads(head), audio


class Base(unittest.TestCase):
    def make(self, hold_s=30.0, fillers=("One sec, boss.",), apology="Sorry, boss, something went wrong.",
             actions=None, voice=None, transport=None, voices=None, stall_s=None, **kw):
        self.voice = voice or FakeVoice()
        self.session = SpeechSession(voices or [self.voice], parallel=3, stall_s=stall_s)
        self.transport = transport or FakeTransport()
        self.header = Header()
        self.actions = actions if actions is not None else {}
        self.reply = StreamingReply(self.session, self.transport, self.header, lambda: dict(self.actions),
                                    hold_s=hold_s, fillers=fillers, apology=apology, **kw)
        self.reply.start()
        self.addCleanup(self.reply.close)
        return self.reply

    def audio_of(self, *texts):
        return b"".join(self.voice.audio(t) for t in texts)

    def started(self):
        return "begin" in self.transport.kinds()

    def finish_ok(self):
        self.reply.finish()
        self.reply.wait(3)


class Normal(Base):
    def test_a_plain_reply_goes_out_in_order_behind_one_header(self):
        r = self.make()
        r.sentence("Hello there friend.")
        r.sentence("How are you today?")
        r.round_end(True)
        self.finish_ok()
        self.assertEqual(self.transport.kinds()[0], "begin")
        self.assertEqual(self.transport.kinds()[-1], "end")
        head, audio = parse_body(self.transport.body())
        self.assertEqual(head, {"text": "Hello there friend. How are you today?", "actions": {}, "fmt": "mp3"})
        self.assertEqual(audio, self.audio_of("Hello there friend.", "How are you today?"))
        self.assertEqual(self.transport.kinds().count("begin"), 1)
        self.assertTrue(r.started)
        self.assertFalse(r.dead)
        self.assertIsNone(r.voice_error)
        self.assertEqual(bytes(r.audio), audio)
        self.assertEqual(r.text(), "Hello there friend. How are you today?")

    def test_the_times_in_the_header_and_for_the_log(self):
        r = self.make(offset_ms=700)
        r.sentence("Hello there friend.")
        r.round_end(True)
        self.finish_ok()
        call = self.header.calls[0]
        self.assertGreaterEqual(call["tts_ms"], 0)
        self.assertGreaterEqual(call["chat_ms"], 0)
        self.assertGreaterEqual(r.first_audio_ms, 700)
        self.assertGreaterEqual(r.answer_ms, 700)
        self.assertIs(r.voice, self.voice)

    def test_the_header_time_counts_from_the_sentence_that_is_spoken_not_one_that_was_thrown_away(self):
        r = self.make()
        r.sentence("Let me look that up.")                    # said before the tool call: thrown away
        r.round_end(False, ["get_weather"], slow=False)
        time.sleep(0.4)                                        # the tool runs
        r.sentence("It is sunny today.")
        r.round_end(True)
        self.finish_ok()
        self.assertGreaterEqual(self.header.calls[0]["chat_ms"], 350)

    def test_the_answer_time_is_that_of_the_final_round_not_of_something_said_before_the_tool(self):
        r = self.make(hold_s=0.05)
        r.sentence("Let me look that up.")
        self.assertTrue(wait_until(self.started))              # it was heard...
        r.round_end(False, ["get_weather"], slow=False)
        time.sleep(0.4)                                        # ...and then the tool ran
        r.sentence("It is sunny today.")
        r.round_end(True)
        self.finish_ok()
        self.assertGreaterEqual(r.answer_ms - r.first_audio_ms, 350)

    def test_actions_ride_in_the_header(self):
        r = self.make(actions={"music": "play", "track": 12})
        r.sentence("Here's Kesariya.")
        r.round_end(True)
        self.finish_ok()
        head, _ = parse_body(self.transport.body())
        self.assertEqual(head["actions"], {"music": "play", "track": 12})

    def test_nothing_to_say(self):
        r = self.make()
        self.finish_ok()
        self.assertEqual(self.transport.events, [])
        self.assertFalse(r.started)
        self.assertFalse(r.decided)


class Gate(Base):
    def test_nothing_is_sent_while_the_round_may_still_end_in_a_tool_call(self):
        r = self.make(hold_s=30)
        r.sentence("Hello there friend.")
        self.assertTrue(wait_until(lambda: self.session.peek() is not None))
        time.sleep(0.15)
        self.assertEqual(self.transport.events, [])         # the audio is ready, and held back
        self.assertTrue(r.music_ok())
        r.round_end(True)
        self.assertTrue(wait_until(self.started))
        self.finish_ok()

    def test_a_long_round_starts_anyway_after_the_hold(self):
        r = self.make(hold_s=0.15)
        r.sentence("Hello there friend.")
        self.assertTrue(wait_until(self.started))            # no round_end(): the hold ran out
        self.assertFalse(r.music_ok())                       # from now on music can't be changed
        r.sentence("This one comes later.")
        r.round_end(True)
        self.finish_ok()
        _, audio = parse_body(self.transport.body())
        self.assertEqual(audio, self.audio_of("Hello there friend.", "This one comes later."))

    def test_the_hold_counts_from_the_first_sentence_of_the_round(self):
        r = self.make(hold_s=0.5, fillers=())
        r.sentence("A draft before a tool call.")
        time.sleep(0.3)
        r.round_end(False, ["save_memory"])
        r.sentence("The answer after it.")
        time.sleep(0.3)                                       # 0.6 s since the draft, 0.3 s since the answer
        self.assertEqual(self.transport.events, [])
        self.assertTrue(wait_until(self.started, 2))
        r.round_end(True)
        self.finish_ok()

    def test_music_is_open_until_the_reply_starts(self):
        r = self.make()
        self.assertTrue(r.music_ok())
        r.sentence("Playing it now.")
        r.round_end(True)
        self.assertTrue(wait_until(self.started))
        self.assertFalse(r.music_ok())
        self.finish_ok()


class ToolRounds(Base):
    def test_the_text_before_a_tool_call_is_never_sent(self):
        draft, answer = "Let me look that up for you.", "The answer is forty two."
        r = self.make(fillers=(), voice=FakeVoice(hold=True))
        r.sentence(draft)
        self.assertTrue(wait_until(lambda: self.voice.calls == [draft]))
        r.round_end(False, ["web_search"], slow=False)
        self.voice.gate(draft).set()
        self.assertTrue(wait_until(lambda: self.voice.stopped == [draft]))   # its voice was told to stop
        r.sentence(answer)
        self.voice.gate(answer).set()
        r.round_end(True)
        self.finish_ok()
        head, audio = parse_body(self.transport.body())
        self.assertEqual(head["text"], answer)
        self.assertEqual(audio, self.audio_of(answer))

    def test_a_filler_goes_out_at_once_while_a_slow_tool_runs(self):
        r = self.make()
        r.round_end(False, ["web_search"], slow=True)
        self.assertTrue(wait_until(self.started))            # without waiting for anything else
        self.assertFalse(r.music_ok())
        self.assertEqual(r.filler, "One sec, boss.")
        r.sentence("It is thirty one degrees.")
        r.round_end(True)
        self.finish_ok()
        head, audio = parse_body(self.transport.body())
        self.assertEqual(head["text"], "One sec, boss.")      # that is all that was known when it started
        self.assertEqual(audio, self.audio_of("One sec, boss.", "It is thirty one degrees."))
        self.assertGreaterEqual(r.answer_ms, r.first_audio_ms)

    def test_no_filler_for_a_quick_tool(self):
        r = self.make()
        r.round_end(False, ["save_memory"], slow=False)
        time.sleep(0.1)
        self.assertEqual(self.transport.events, [])
        self.assertIsNone(r.filler)
        r.sentence("Noted, boss.")
        r.round_end(True)
        self.finish_ok()
        self.assertEqual(parse_body(self.transport.body())[1], self.audio_of("Noted, boss."))

    def test_no_filler_when_there_are_none_or_the_reply_has_already_started(self):
        r = self.make(fillers=())
        r.round_end(False, ["web_search"], slow=True)
        self.assertIsNone(r.filler)
        r.sentence("Here it is.")
        r.round_end(True)
        self.finish_ok()

    def test_no_filler_when_the_voice_looks_down(self):
        """A line from the cache would make Edge the voice of the whole reply; if Edge is down, the reply would end
        right after the filler. Better to wait for the answer and let whichever voice works say it."""
        r = self.make(filler_ok=lambda: False)
        r.round_end(False, ["web_search"], slow=True)
        self.assertIsNone(r.filler)
        self.assertTrue(r.music_ok())                         # nothing was decided
        r.sentence("Here it is.")
        r.round_end(True)
        self.finish_ok()
        self.assertEqual(parse_body(self.transport.body())[1], self.audio_of("Here it is."))

    def test_it_remembers_that_tools_were_asked_for(self):
        r = self.make()
        self.assertFalse(r.tools_ran)
        r.sentence("Hello there friend.")
        r.round_end(True)
        self.assertFalse(r.tools_ran)
        r2 = StreamingReply(SpeechSession([FakeVoice()]), FakeTransport(), Header(), dict)
        r2.round_end(False, ["save_memory"], slow=False)
        self.assertTrue(r2.tools_ran)
        r2.close()
        self.finish_ok()

    def test_a_filler_is_not_added_behind_a_reply_that_is_already_going_out(self):
        r = self.make(hold_s=0.05)
        r.sentence("Let me see what I can find about that.")
        self.assertTrue(wait_until(self.started))
        r.round_end(False, ["web_search"], slow=True)        # already speaking: no need
        self.assertIsNone(r.filler)
        r.sentence("Found it.")
        r.round_end(True)
        self.finish_ok()
        self.assertEqual(parse_body(self.transport.body())[1],
                         self.audio_of("Let me see what I can find about that.", "Found it."))

    def test_the_same_filler_is_not_picked_twice_in_a_row(self):
        replystream._last_filler[0] = None
        picked = []
        for _ in range(8):
            r = self.make(fillers=("One sec.", "Checking.", "On it."))
            r.round_end(False, ["web_search"], slow=True)
            picked.append(r.filler)
            r.close()
        self.assertTrue(all(a != b for a, b in zip(picked, picked[1:])), picked)


class Failures(Base):
    def test_the_speaker_hanging_up_cancels_the_turn(self):
        r = self.make(voice=FakeVoice(pieces=4), transport=FakeTransport(fail_on_write=2))
        r.sentence("Hello there friend.")
        r.round_end(True)
        self.assertTrue(wait_until(r.cancelled))              # the brain would stop at its next chunk
        r.finish()
        r.wait(3)
        self.assertTrue(r.dead)
        self.assertNotIn("end", self.transport.kinds())
        self.assertTrue(self.session.aborted)
        r.sentence("Ignored.")                                  # nothing more is made for nobody
        self.assertEqual(self.voice.calls, ["Hello there friend."])

    def test_a_failure_before_the_reply_starts_sends_nothing(self):
        r = self.make()
        r.sentence("Some text that will never be said.")
        r.finish(error=RuntimeError("gemini broke"))
        r.wait(3)
        self.assertEqual(self.transport.events, [])
        self.assertFalse(r.decided)                             # the caller can still answer with an error
        self.assertFalse(r.started)

    def test_a_failure_after_the_reply_started_ends_it_cleanly_with_an_apology(self):
        r = self.make(hold_s=0.05)
        r.sentence("Let me tell you something.")
        self.assertTrue(wait_until(self.started))
        r.finish(error=RuntimeError("gemini broke"))
        r.wait(3)
        self.assertTrue(r.decided)
        self.assertEqual(self.transport.kinds()[-1], "end")
        _, audio = parse_body(self.transport.body())
        self.assertEqual(audio, self.audio_of("Let me tell you something.", "Sorry, boss, something went wrong."))

    def test_no_voice_at_all(self):
        v = FakeVoice()
        v.fail["Hello there friend."] = RuntimeError("no voice installed")
        r = self.make(voice=v)
        r.sentence("Hello there friend.")
        self.assertTrue(wait_until(lambda: r.voice_error is not None))
        r.sentence("More text.")
        r.round_end(True)
        self.finish_ok()
        self.assertEqual(self.transport.events, [])
        self.assertFalse(r.started)
        self.assertIn("no voice installed", str(r.voice_error))
        self.assertEqual(v.calls, ["Hello there friend."])      # no point making the rest

    def test_a_voice_that_dies_part_way_still_finishes_the_response(self):
        v = FakeVoice()
        v.fail["Two."] = RuntimeError("edge voice failed")
        r = self.make(voice=v)
        for text in ("One one one.", "Two.", "Three."):
            r.sentence(text)
        r.round_end(True)
        self.finish_ok()
        self.assertEqual(self.transport.kinds()[-1], "end")
        self.assertEqual(parse_body(self.transport.body())[1], self.audio_of("One one one."))

    def test_a_voice_that_goes_quiet_is_replaced_before_the_reply_begins(self):
        """The old path gave Microsoft's voice 15 s and then used the laptop's; waiting for the voice's own
        time-out here would take 45 s, which is how long the speaker waits before giving up on the laptop."""
        edge = FakeVoice("edge", hold=True)
        local = FakeVoice("local", fmt="pcm")
        r = self.make(voice=local, voices=[edge, local], stall_s=0.2)
        t0 = time.time()
        r.sentence("Hello there friend.")
        r.round_end(True)
        self.assertTrue(wait_until(self.started, timeout=3))
        self.assertLess(time.time() - t0, 2)
        self.finish_ok()
        head, audio = parse_body(self.transport.body())
        self.assertEqual(head["fmt"], "pcm")
        self.assertEqual(audio, local.audio("Hello there friend."))
        self.assertIsNone(r.voice_error)
        edge.gate("Hello there friend.").set()

    def test_a_voice_that_dies_part_way_stops_the_brain_and_the_voice(self):
        v = FakeVoice()
        v.fail["Two."] = RuntimeError("edge voice failed")
        r = self.make(voice=v, hold_s=0.05)                   # the brain is still talking when the voice dies
        for text in ("One one one.", "Two.", "Three."):
            r.sentence(text)
        self.assertTrue(wait_until(r.cancelled))              # the brain stops at its next chunk
        self.assertTrue(wait_until(lambda: self.session.aborted))
        made = list(v.calls)
        r.sentence("Four.")                                     # nothing more is made for a reply that is over
        r.round_end(True)
        r.finish()
        r.wait(3)
        time.sleep(0.05)
        self.assertEqual(v.calls, made)
        self.assertEqual(self.transport.kinds()[-1], "end")
        self.assertFalse(r.dead)                                # the speaker didn't hang up; the voice stopped

    def test_close_always_returns_even_if_the_brain_never_finished(self):
        r = self.make()
        r.sentence("Left hanging.")
        t0 = time.time()
        r.close()
        self.assertLess(time.time() - t0, 3)


if __name__ == "__main__":
    unittest.main()
