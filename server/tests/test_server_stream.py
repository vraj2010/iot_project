"""The whole streamed turn over real HTTP: the server, the brain, the voice and the frame the speaker reads,
with only Gemini, the voice and the database replaced."""

import contextlib
import http.client
import io
import json
import os
import tempfile
import threading
import time
import unittest
import wave
from unittest import mock

import _path  # noqa: F401

import config
import frames
import gemini
import server
from brain import Brain
from brainfakes import FakeJukebox, FakeMemory, script, tool_call
from speech import SpeechSession
from voicefakes import FakeVoice, wait_until

END = ("end", {"role": "assistant", "content": "x"})
LEGACY = {"reply": "Legacy answer.", "ms": 5, "tools": [], "actions": {}, "timing": {}, "episode_id": 7,
          "new_session": False, "previous_id": None}


def tool_then(error, name="get_weather", args='{"place": "Rajkot"}'):
    """A stand-in for gemini.chat_stream: the first round asks for a tool, the next one fails with `error`."""
    calls = []

    def fake(messages, tools=None, **kw):
        calls.append(1)

        def gen():
            if len(calls) == 1:
                yield ("end", {"role": "assistant", "content": "", "tool_calls": [tool_call(name, args)]})
            else:
                raise error
        return gen()

    return fake, calls


class FakeTTS:
    def __init__(self, voice):
        self.voice = voice
        self.name = voice.name
        self.up = True

    def filler_ok(self):
        return self.up

    def session(self, cond=None):
        return SpeechSession([self.voice], parallel=3, cond=cond)

    def start(self, reply):                       # the old whole-reply way
        data = self.voice.audio(reply)
        return self.voice, data[:8], iter([data[8:]]), 5


class FakeAssistant:
    think_stream = server.Assistant.think_stream
    stream_enabled = server.Assistant.stream_enabled
    local = server.Assistant.local
    router = None                                 # no local commands unless a test sets one

    def __init__(self, voice):
        self.turn_lock = threading.Lock()
        self.tts = FakeTTS(voice)
        self.memory = FakeMemory()
        self.brain = Brain(self.memory)
        self.jukebox = FakeJukebox()
        self.stt = object()
        self.stream_fails = 0
        self.heard = "what is the weather"
        self.legacy_calls = []
        self.logged = []

    def hear(self, pcm):
        return self.heard, self.heard, 111

    def think(self, heard, music_state=None):
        self.legacy_calls.append(heard)
        return dict(LEGACY)

    def voice_start(self, reply):
        return self.tts.start(reply)

    def after_turn(self, heard, result):
        self.logged.append((heard, result["reply"]))


def wav(seconds=0.6):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * int(16000 * seconds))
    return buf.getvalue()


class Base(unittest.TestCase):
    def setUp(self):
        for name, value in (("DEVICE_TOKEN", "tok"), ("STREAM_REPLIES", True), ("STREAM_MAX_FAILS", 3),
                            ("TOOL_FILLER", True), ("STREAM_HOLD_MS", 1500), ("SAVE_AUDIO", False),
                            ("LATENCY_LOG", False), ("MAX_REPLY_CHARS", 400), ("MAX_TOOL_ROUNDS", 3),
                            ("WEB_SEARCH", True), ("GEMINI_REASONING", "low"), ("FAST_MUSIC_REPLIES", True)):
            p = mock.patch.object(config, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.voice = FakeVoice(pieces=2)
        self.assistant = FakeAssistant(self.voice)
        old = server.Handler.assistant
        server.Handler.assistant = self.assistant
        self.addCleanup(setattr, server.Handler, "assistant", old)
        self.httpd = server.Server(("127.0.0.1", 0), server.Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def post(self, path, body, headers=None, timeout=10):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        conn.request("POST", path, body=raw, headers={"X-Device-Token": "tok", **(headers or {})})
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, data

    def say(self, text="what is the weather", headers=None):
        return self.post("/say", {"text": text}, headers)

    def stream(self, *rounds, patch=None, **post):
        fake, calls = script(*rounds)
        with mock.patch.object(gemini, "chat_stream", patch or fake):
            result = self.say(**post)
        return result, calls

    def audio_of(self, *texts):
        return b"".join(self.voice.audio(t) for t in texts)


class Streamed(Base):
    def test_say_answers_in_one_frame_with_the_audio_in_order(self):
        (status, body), calls = self.stream([("text", "It is sunny today. Enjoy it!"), END])
        self.assertEqual(status, 200)
        f = frames.parse(body)
        self.assertEqual(f["text"], "It is sunny today. Enjoy it!")
        self.assertEqual(f["format"], frames.FORMAT_MP3)
        self.assertEqual(f["provider"], self.voice.provider)
        self.assertEqual(f["audio"], self.audio_of("It is sunny today.", "Enjoy it!"))
        self.assertTrue(wait_until(lambda: self.assistant.logged))        # logged just after the reply is sent
        self.assertEqual(self.assistant.logged, [("what is the weather", "It is sunny today. Enjoy it!")])
        self.assertEqual(self.assistant.legacy_calls, [])
        self.assertEqual(len(calls), 1)

    def test_ask_with_a_recording_answers_the_question_it_heard(self):
        fake, _ = script([("text", "It is sunny today."), END])
        with mock.patch.object(gemini, "chat_stream", fake):
            status, body = self.post("/ask", wav())
        self.assertEqual(status, 200)
        f = frames.parse(body)
        self.assertEqual((f["heard"], f["text"]), ("what is the weather", "It is sunny today."))
        self.assertEqual(f["stt_ms"], 111)
        self.assertEqual(f["audio"], self.audio_of("It is sunny today."))
        self.assertTrue(wait_until(lambda: self.assistant.logged))
        self.assertEqual(self.assistant.logged[0][0], "what is the weather")

    def test_a_music_command_carries_its_action_in_the_header(self):
        round1 = [("end", {"role": "assistant", "content": "", "tool_calls": [tool_call("play_music", '{"query": "kesariya"}')]})]
        fake, calls = script(round1)
        with mock.patch.object(gemini, "chat_stream", fake):
            status, body = self.post("/say", {"text": "play kesariya"}, {"X-Music-State": "state=stopped"})
        self.assertEqual(status, 200)
        f = frames.parse(body)
        self.assertEqual(f["text"], "Here's Kesariya.")
        self.assertEqual(f["action"], {"music": "play", "track": 12, "title": "Kesariya"})
        self.assertEqual(f["audio"], self.audio_of("Here's Kesariya."))
        self.assertEqual(len(calls), 1)                         # no second Gemini round for a plain music command

    def test_a_slow_tool_is_covered_by_a_filler_and_the_answer_follows(self):
        call = tool_call("get_weather", '{"place": "Rajkot"}', extra_content={"google": {"thought_signature": "SIG"}})
        round1 = [("end", {"role": "assistant", "content": "", "tool_calls": [call]})]
        round2 = [("text", "It is thirty one degrees in Rajkot."), END]
        with mock.patch("weather.is_cached", return_value=False), \
                mock.patch("weather.get_weather", return_value={"temp": 31}):
            (status, body), calls = self.stream(round1, round2)
        self.assertEqual(status, 200)
        f = frames.parse(body)
        # the header says what was known when the reply started: the filler, and maybe the answer if it was quick
        filler = next(line for line in server.FILLERS if f["text"].startswith(line))
        self.assertEqual(f["audio"], self.audio_of(filler, "It is thirty one degrees in Rajkot."))
        self.assertTrue(wait_until(lambda: self.assistant.logged))
        self.assertEqual(self.assistant.logged[-1][1], "It is thirty one degrees in Rajkot.")
        self.assertEqual(calls[1]["messages"][-2]["tool_calls"][0]["extra_content"],
                         {"google": {"thought_signature": "SIG"}})

    def test_no_filler_when_the_voice_looks_down(self):
        self.assistant.tts.up = False
        call = tool_call("get_weather", '{"place": "Rajkot"}')
        round1 = [("end", {"role": "assistant", "content": "", "tool_calls": [call]})]
        round2 = [("text", "It is thirty one degrees in Rajkot."), END]
        with mock.patch("weather.is_cached", return_value=False), \
                mock.patch("weather.get_weather", return_value={"temp": 31}):
            (status, body), _ = self.stream(round1, round2)
        f = frames.parse(body)
        self.assertEqual(f["text"], "It is thirty one degrees in Rajkot.")
        self.assertEqual(f["audio"], self.audio_of("It is thirty one degrees in Rajkot."))

    def test_the_filler_can_be_turned_off(self):
        call = tool_call("get_weather", '{"place": "Rajkot"}')
        round1 = [("end", {"role": "assistant", "content": "", "tool_calls": [call]})]
        round2 = [("text", "It is thirty one degrees in Rajkot."), END]
        with mock.patch("weather.is_cached", return_value=False), \
                mock.patch("weather.get_weather", return_value={"temp": 31}), \
                mock.patch.object(config, "TOOL_FILLER", False):
            (status, body), _ = self.stream(round1, round2)
        f = frames.parse(body)
        self.assertEqual(f["text"], "It is thirty one degrees in Rajkot.")
        self.assertEqual(f["audio"], self.audio_of("It is thirty one degrees in Rajkot."))

    def test_the_header_waits_for_the_round_but_not_for_ever(self):
        """A long round starts anyway after STREAM_HOLD_MS, and then music can't be changed any more."""
        gate = threading.Event()

        def slow(messages, tools=None, **kw):
            def gen():
                yield ("text", "Here is the start of a long answer. ")
                gate.wait(5)                       # Gemini is slow to finish this round
                yield ("text", "And the rest of it, written later.")
                yield END
            return gen()

        results = []
        with mock.patch.object(config, "STREAM_HOLD_MS", 150), mock.patch.object(gemini, "chat_stream", slow):
            t = threading.Thread(target=lambda: results.append(self.say()), daemon=True)
            t.start()
            self.assertTrue(wait_until(lambda: self.voice.done == ["Here is the start of a long answer."]))
            time.sleep(0.4)                        # longer than the hold: the first sentence is on its way already
            self.assertEqual(results, [])          # (the response isn't finished: the rest hasn't been written)
            gate.set()
            t.join(5)
        status, body = results[0]
        f = frames.parse(body)
        self.assertEqual(f["text"], "Here is the start of a long answer.")
        self.assertEqual(f["audio"], self.audio_of("Here is the start of a long answer.",
                                                   "And the rest of it, written later."))


class NoLog(Base):
    def test_a_test_request_is_not_logged_or_learned_from(self):
        fake, _ = script([("text", "It is sunny today."), END], [("text", "Still sunny."), END])
        with mock.patch.object(gemini, "chat_stream", fake):
            self.assertEqual(self.say("first", {"X-No-Log": "1"})[0], 200)
            self.assertEqual(self.say("second")[0], 200)
        self.assertTrue(wait_until(lambda: self.assistant.logged))
        self.assertEqual(self.assistant.logged, [("second", "Still sunny.")])      # only the one that wasn't a test

    def test_the_old_path_honours_it_too(self):
        self.assertEqual(self.say("first", {"X-Stream": "0", "X-No-Log": "1"})[0], 200)
        self.assertEqual(self.say("second", {"X-Stream": "0"})[0], 200)
        self.assertTrue(wait_until(lambda: self.assistant.logged))
        self.assertEqual(self.assistant.logged, [("second", "Legacy answer.")])


class Switching(Base):
    def test_x_stream_zero_uses_the_normal_path(self):
        fake, calls = script([("text", "Never asked."), END])
        with mock.patch.object(gemini, "chat_stream", fake):
            status, body = self.say(headers={"X-Stream": "0"})
        self.assertEqual(status, 200)
        f = frames.parse(body)
        self.assertEqual(f["text"], "Legacy answer.")
        self.assertEqual(f["audio"], self.audio_of("Legacy answer."))
        self.assertEqual(calls, [])
        self.assertEqual(self.assistant.legacy_calls, ["what is the weather"])

    def test_x_stream_one_streams_even_when_it_is_switched_off(self):
        with mock.patch.object(config, "STREAM_REPLIES", False):
            (status, body), calls = self.stream([("text", "It is sunny today."), END], headers={"X-Stream": "1"})
        self.assertEqual(frames.parse(body)["text"], "It is sunny today.")
        self.assertEqual(len(calls), 1)

    def test_switched_off_in_the_settings(self):
        with mock.patch.object(config, "STREAM_REPLIES", False):
            (status, body), calls = self.stream([("text", "Not used."), END])
        self.assertEqual(frames.parse(body)["text"], "Legacy answer.")
        self.assertEqual(calls, [])

    def test_a_bug_in_the_streamed_path_falls_back_and_eventually_switches_it_off(self):
        def broken(*a, **k):
            raise RuntimeError("a bug")

        with mock.patch.object(config, "STREAM_MAX_FAILS", 2):
            for expected in (1, 2):
                (status, body), _ = self.stream(patch=broken)
                self.assertEqual(status, 200)
                self.assertEqual(frames.parse(body)["text"], "Legacy answer.")      # the speaker still got an answer
                self.assertEqual(self.assistant.stream_fails, expected)
            self.assertFalse(self.assistant.stream_enabled())
            seen = []
            (status, body), _ = self.stream(patch=lambda *a, **k: seen.append(1))
            self.assertEqual(seen, [])                          # not even tried any more
            self.assertEqual(frames.parse(body)["text"], "Legacy answer.")

    def test_a_failure_the_normal_path_shares_is_not_blamed_on_streaming(self):
        def broken(*a, **k):
            raise RuntimeError("a bug")

        def also_broken(*a, **k):
            raise RuntimeError("the database is down")

        self.assistant.think = also_broken
        (status, body), _ = self.stream(patch=broken)
        self.assertEqual(status, 500)
        self.assertEqual(self.assistant.stream_fails, 0)       # the normal way fails too: not streaming's fault

    def test_tools_that_have_run_are_not_run_again_by_the_fallback(self):
        fake, _ = tool_then(RuntimeError("a bug"))
        with mock.patch("weather.is_cached", return_value=True), \
                mock.patch("weather.get_weather", return_value={"temp": 31}) as weather:
            (status, body), _ = self.stream(patch=fake)
        self.assertEqual(status, 500)
        self.assertEqual(weather.call_count, 1)
        self.assertEqual(self.assistant.legacy_calls, [])      # asking the question again would repeat the tools
        self.assertEqual(self.assistant.stream_fails, 0)

    def test_a_good_streamed_turn_resets_the_count(self):
        self.assistant.stream_fails = 2
        (status, body), _ = self.stream([("text", "It is sunny today."), END])
        self.assertTrue(wait_until(lambda: self.assistant.stream_fails == 0))

    def test_nothing_heard_is_answered_the_normal_way(self):
        self.assistant.heard = ""
        fake, calls = script([("text", "Never asked."), END])
        with mock.patch.object(gemini, "chat_stream", fake):
            status, body = self.post("/ask", wav())
        f = frames.parse(body)
        self.assertEqual(f["text"], server.NOT_HEARD_REPLY)
        self.assertEqual(calls, [])


class Errors(Base):
    def test_gemini_failing_before_anything_is_said_is_an_error_reply(self):
        def refuse(*a, **k):
            raise gemini.GeminiError("quota exceeded", 429)

        (status, body), _ = self.stream(patch=refuse)
        self.assertEqual(status, 429)
        data = json.loads(body)
        self.assertIn("quota exceeded", data["error"])
        self.assertEqual(self.assistant.stream_fails, 0)       # that's Gemini, not a bug in streaming
        self.assertEqual(self.assistant.legacy_calls, [])      # and asking again at once wouldn't help

    def test_no_voice_means_the_speaker_gets_the_text(self):
        self.voice.fail["It is sunny today."] = RuntimeError("no voice installed")
        (status, body), _ = self.stream([("text", "It is sunny today."), END])
        self.assertEqual(status, 502)
        data = json.loads(body)
        self.assertTrue(data["error"].startswith("voice:"))
        self.assertEqual(data["reply"], "It is sunny today.")

    def test_a_filler_nobody_can_say_and_then_a_failed_turn_still_gets_an_answer(self):
        """The filler counts as decided, but no voice could say it: nothing reached the speaker, so it has to be
        told what went wrong instead of being left to wait for a reply that is never coming."""
        for line in server.FILLERS:
            self.voice.fail[line] = RuntimeError("no voice installed")
        fake, _ = tool_then(gemini.GeminiError("quota exceeded", 429))
        with mock.patch("weather.is_cached", return_value=False), \
                mock.patch("weather.get_weather", return_value={"temp": 31}):
            (status, body), _ = self.stream(patch=fake)
        self.assertEqual(status, 429)
        self.assertIn("quota exceeded", json.loads(body)["error"])

    def test_a_voice_that_dies_part_way_ends_the_reply_and_stops_gemini(self):
        self.voice.fail["Two two two."] = RuntimeError("edge voice failed")
        gate = threading.Event()
        read_to_the_end = []

        def slow(messages, tools=None, **kw):
            def gen():
                yield ("text", "One one one. Two two two. Three three three. ")
                gate.wait(5)                                    # Gemini is still writing when the voice breaks
                yield ("text", "Four four four.")
                yield END
                read_to_the_end.append(True)
            return gen()

        with mock.patch.object(config, "STREAM_HOLD_MS", 100), mock.patch.object(gemini, "chat_stream", slow):
            status, body = self.say()
            gate.set()
            self.assertTrue(wait_until(lambda: not self.assistant.turn_lock.locked()))
        self.assertEqual(status, 200)
        self.assertEqual(frames.parse(body)["audio"], self.audio_of("One one one."))
        self.assertEqual(read_to_the_end, [])                   # Gemini was stopped, not read to the end
        self.assertEqual(self.assistant.legacy_calls, [])
        self.assertEqual(self.assistant.stream_fails, 0)

    def test_a_failure_after_the_reply_has_started_ends_it_with_an_apology(self):
        def breaks(messages, tools=None, **kw):
            def gen():
                yield ("text", "Let me tell you something. ")
                time.sleep(0.4)                        # long enough for the hold to start the reply
                raise gemini.GeminiError("the stream was cut", 502)
            return gen()

        with mock.patch.object(config, "STREAM_HOLD_MS", 100):
            (status, body), _ = self.stream(patch=breaks)
        self.assertEqual(status, 200)
        f = frames.parse(body)
        self.assertEqual(f["audio"], self.audio_of("Let me tell you something.", server.ERROR_LINE))

    def test_a_speaker_that_hangs_up_does_not_wedge_the_server(self):
        self.voice.pieces = 3
        gate = threading.Event()
        self.voice.mid_gate["Here is the first sentence."] = gate
        fake, _ = script([("text", "Here is the first sentence. And here is the second one."), END])
        with mock.patch.object(gemini, "chat_stream", fake):
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
            conn.request("POST", "/say", body=json.dumps({"text": "hi"}).encode(), headers={"X-Device-Token": "tok"})
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            resp.read1(8)
            conn.close()                                # the speaker is gone, mid-reply
            gate.set()
            status, body = self.say()                   # the next question is answered normally
        self.assertEqual(status, 200)


class LatencyLog(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        for target, name, value in ((server, "HERE", __import__("pathlib").Path(self.dir.name)),
                                    (config, "LATENCY_LOG", True)):
            p = mock.patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.path = os.path.join(self.dir.name, "latency.log")
        self.result = {"timing": {"prompt_ms": 12, "gemini_ms": [900, 700], "tool_ms": 300, "fast": False}}

    def lines(self, path=None):
        with open(path or self.path, encoding="utf-8") as f:
            return [line.rstrip("\n").split("\t") for line in f]

    def test_a_normal_turn_and_a_streamed_one_share_the_same_columns(self):
        server._latency("spoken", "what is the weather", 700, self.result, 1500, 4100)
        server._latency("spoken", "what is the weather", 700, self.result, 1500, 3200,
                        {"first_sentence_ms": 900, "ttft_ms": 450, "hedged": False, "filler": "One sec, boss.",
                         "answer_ms": 3900})
        rows = self.lines()
        self.assertEqual(rows[0], list(server.LATENCY_COLUMNS))
        normal, streamed = (dict(zip(rows[0], r)) for r in rows[1:])
        self.assertEqual((normal["stream"], normal["first_audio_ms"], normal["gemini_ms"]), ("0", "4100", "900+700"))
        self.assertEqual((streamed["stream"], streamed["first_sentence_ms"], streamed["ttft_ms"], streamed["filler"],
                          streamed["answer_ms"], streamed["hedged"]), ("1", "900", "450", "1", "3900", "0"))
        self.assertTrue(all(len(r) == len(server.LATENCY_COLUMNS) for r in rows))

    def console(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            server._latency(*args)
        return out.getvalue()

    def test_the_console_line_for_a_turn_that_started_with_a_filler(self):
        line = self.console("spoken", "weather", 700, self.result, 20, 1900,
                            {"first_sentence_ms": 3100, "ttft_ms": 450, "hedged": False,
                             "filler": "One sec, boss.", "answer_ms": 3900})
        self.assertIn("1900 ms to first audio", line)
        self.assertIn("the filler 'One sec, boss.'", line)
        self.assertIn("Gemini's first round 900", line)          # what came before the filler
        self.assertIn("the answer itself from 3900 ms", line)
        self.assertNotIn("to the first sentence", line)          # the first audio was not that sentence

    def test_the_console_line_adds_up_when_the_reply_was_held_back(self):
        line = self.console("spoken", "play something", 700, self.result, 1500, 4000,
                            {"first_sentence_ms": 900, "ttft_ms": 450, "hedged": False, "filler": None,
                             "answer_ms": 4000})
        self.assertIn("held back 900", line)                     # 4000 - 700 - 900 - 1500
        quiet = self.console("spoken", "hi", 700, self.result, 1500, 3200,
                             {"first_sentence_ms": 900, "ttft_ms": 450, "hedged": False, "filler": None,
                              "answer_ms": 3200})
        self.assertNotIn("held back", quiet)

    def test_an_older_log_with_other_columns_is_kept_aside(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("time\tkind\tfirst_audio_ms\told layout\n2026-01-01 00:00:00\tspoken\t3000\tx\n")
        server._latency("typed", "hi", 0, self.result, 1000, 2000)
        rotated = [n for n in os.listdir(self.dir.name) if n != "latency.log"]
        self.assertEqual(len(rotated), 1)
        self.assertTrue(rotated[0].startswith("latency.") and rotated[0].endswith(".log"))
        with open(os.path.join(self.dir.name, rotated[0]), encoding="utf-8") as f:
            self.assertIn("old layout", f.read())
        self.assertEqual(self.lines()[0], list(server.LATENCY_COLUMNS))
        self.assertEqual(len(self.lines()), 2)


if __name__ == "__main__":
    unittest.main()
