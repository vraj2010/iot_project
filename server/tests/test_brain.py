import json
import re
import unittest
from unittest import mock

import _path  # noqa: F401

import brain
import config
import gemini
from brainfakes import FakeJukebox, FakeMemory, Recorder, script, tool_call


def old_for_speech(text: str) -> str:
    """for_speech() exactly as it was before it was split in two."""
    out = re.sub(r"```[\s\S]*?```", " ", text or "")
    out = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", out)
    out = re.sub(r"https?://\S+", " ", out)
    out = re.sub(r"\s*\[\d+(?:,\s*\d+)*\]", "", out)
    out = re.sub(r"[*_#`>|~]", "", out)
    out = re.sub(r"^\s*[-•]\s+", "", out, flags=re.M)
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r",\s*boss\s*,\s*", ", ", out, flags=re.I)
    limit = config.MAX_REPLY_CHARS
    if len(out) > limit:
        cut = out[:limit]
        end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
        out = cut[: end + 1] if end > limit / 2 else cut.rstrip() + "..."
    return out


class ForSpeech(unittest.TestCase):
    def test_same_as_before_the_split(self):
        samples = ["**Bold** and _italic_ with `code`", "See [the docs](http://x.y/z) now [1] ok",
                   "```py\nprint(1)\n``` after", "- one\n- two\n• three", "Yes, boss, I will.", "plain text.",
                   "A. " * 200, "Sentence one is here. " * 30, "x" * 900, ""]
        for limit in (400, 60):
            with mock.patch.object(config, "MAX_REPLY_CHARS", limit):
                for text in samples:
                    self.assertEqual(brain.for_speech(text), old_for_speech(text), (limit, text[:30]))


class Sentences(unittest.TestCase):
    def split(self, *pieces):
        c = brain.SentenceChunker()
        out = []
        for p in pieces:
            out += c.feed(p)
        tail = c.flush()
        return out + ([tail] if tail else [])

    def test_basic(self):
        self.assertEqual(self.split("Hello there. It's sunny today. Enjoy!"),
                         ["Hello there.", "It's sunny today.", "Enjoy!"])

    def test_the_first_sentence_comes_out_before_the_second_is_finished(self):
        c = brain.SentenceChunker()
        self.assertEqual(c.feed("It is twenty five degrees in "), [])
        self.assertEqual(c.feed("Rajkot right now. Light"), ["It is twenty five degrees in Rajkot right now."])
        self.assertEqual(c.flush(), "Light")

    def test_a_full_stop_is_only_believed_once_a_space_follows(self):
        c = brain.SentenceChunker()
        self.assertEqual(c.feed("This sentence is long enough."), [])
        self.assertEqual(c.feed(" Next"), ["This sentence is long enough."])

    def test_same_result_however_the_text_is_cut(self):
        text = "Trains are big in India. They carry millions each day! Want to know more? Just ask."
        want = self.split(text)
        for step in (1, 2, 3, 5, 7):
            pieces = [text[i:i + step] for i in range(0, len(text), step)]
            self.assertEqual(self.split(*pieces), want, step)

    def test_short_sentences_wait_for_the_next_one(self):
        self.assertEqual(self.split("Hi. How are you doing today?"), ["Hi. How are you doing today?"])

    def test_titles_initials_decimals_and_ellipses_do_not_end_a_sentence(self):
        self.assertEqual(self.split("Dr. Rao will see you at 3.5 hours. Bring the form."),
                         ["Dr. Rao will see you at 3.5 hours.", "Bring the form."])
        self.assertEqual(self.split("It was J. K. Rowling who wrote it. Great books."),
                         ["It was J. K. Rowling who wrote it.", "Great books."])
        self.assertEqual(self.split("Well... that is a long story, boss. Sit down."),
                         ["Well... that is a long story, boss.", "Sit down."])
        self.assertEqual(self.split("Use e.g. a spoon for this job. Then stir."),
                         ["Use e.g. a spoon for this job.", "Then stir."])

    def test_questions_and_quotes(self):
        self.assertEqual(self.split('She said "that is fine." Then she left.'),
                         ['She said "that is fine."', "Then she left."])
        self.assertEqual(self.split("Is it raining outside right now? Yes it is."),
                         ["Is it raining outside right now?", "Yes it is."])

    def test_list_items_on_their_own_lines(self):
        self.assertEqual(self.split("You need:\nmilk and eggs\nsome flour and sugar\n"),
                         ["You need:", "milk and eggs", "some flour and sugar"])

    def test_a_run_on_is_cut_at_a_comma(self):
        text = ("This is a very long sentence without any full stop that keeps going and going, "
                "past what anybody would want to wait for before hearing a word of it, because it just does not end "
                "and it keeps on rambling about nothing in particular for a good while longer than it ever should ")
        self.assertGreater(len(text), brain.SentenceChunker.MAX_CHARS)
        out = self.split(text)
        self.assertGreater(len(out), 1)
        self.assertTrue(out[0].endswith(","))
        self.assertLessEqual(len(out[0]), brain.SentenceChunker.MAX_CHARS)
        self.assertEqual(" ".join(out), text.strip())


class Budget(unittest.TestCase):
    def test_sentences_pass_through_under_the_limit(self):
        b = brain.SpeechBudget(100)
        self.assertEqual(b.take("**Hello** there."), "Hello there.")
        self.assertEqual(b.take("Second one."), "Second one.")
        self.assertEqual(b.parts, ["Hello there.", "Second one."])
        self.assertFalse(b.full)

    def test_past_the_limit_it_ends_on_the_last_full_sentence(self):
        b = brain.SpeechBudget(60)
        self.assertTrue(b.take("This is the first sentence, nice."))
        self.assertIsNone(b.take("This second sentence will not fit in here."))
        self.assertTrue(b.full)
        self.assertIsNone(b.take("Nothing more is said."))

    def test_a_first_sentence_that_is_too_long_is_cut_short(self):
        b = brain.SpeechBudget(60)
        out = b.take("This single sentence just goes on and on and on past the limit we allow here.")
        self.assertTrue(out.endswith("..."))
        self.assertLessEqual(len(out), 60)
        self.assertTrue(b.full)

    def test_empty_after_cleaning_is_skipped(self):
        b = brain.SpeechBudget(100)
        self.assertIsNone(b.take("```code```"))
        self.assertEqual(b.parts, [])


def make_brain(city="Rajkot"):
    return brain.Brain(FakeMemory(city))


class AnswerStillWorks(unittest.TestCase):
    """The refactor must not change what answer() does."""

    def test_plain_reply(self):
        with mock.patch.object(gemini, "chat", return_value={"role": "assistant", "content": "**Hi** there, boss."}):
            out = make_brain().answer("hello")
        self.assertEqual(out["reply"], "Hi there, boss.")
        self.assertEqual(out["tools"], [])
        self.assertEqual(out["episode_id"], 7)
        self.assertEqual(len(out["timing"]["gemini_ms"]), 1)

    def test_tool_round_then_answer_passes_thought_signatures_back(self):
        call = tool_call("get_weather", '{"place": "Rajkot"}', extra_content={"google": {"thought_signature": "SIG"}})
        replies = [{"role": "assistant", "content": None, "tool_calls": [call]},
                   {"role": "assistant", "content": "It's 31 degrees."}]
        seen = []

        def chat(messages, tools=None, reasoning=None, **kw):
            seen.append(json.loads(json.dumps(messages)))
            return replies[len(seen) - 1]

        with mock.patch.object(gemini, "chat", chat), \
                mock.patch("weather.get_weather", return_value={"temp": 31}) as weather:
            out = make_brain().answer("weather?")
        weather.assert_called_once_with("Rajkot")
        self.assertEqual(out["reply"], "It's 31 degrees.")
        assistant = seen[1][-2]
        self.assertEqual(assistant["tool_calls"][0]["extra_content"], {"google": {"thought_signature": "SIG"}})
        self.assertEqual(assistant["content"], "")
        self.assertEqual(seen[1][-1]["role"], "tool")

    def test_quick_music_reply_skips_the_second_round(self):
        reply = {"role": "assistant", "content": None, "tool_calls": [tool_call("play_music", '{"query": "kesariya"}')]}
        with mock.patch.object(gemini, "chat", return_value=reply) as chat:
            out = make_brain().answer("play kesariya", FakeJukebox())
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(out["reply"], "Here's Kesariya.")
        self.assertEqual(out["actions"]["music"], "play")
        self.assertTrue(out["timing"]["fast"])

    def test_empty_answer_is_an_error(self):
        with mock.patch.object(gemini, "chat", return_value={"role": "assistant", "content": "  "}):
            with self.assertRaises(gemini.GeminiError):
                make_brain().answer("hello")


class AnswerStream(unittest.TestCase):
    def setUp(self):
        for name, value in (("MAX_REPLY_CHARS", 400), ("MAX_TOOL_ROUNDS", 3), ("WEB_SEARCH", True),
                            ("GEMINI_REASONING", "low"), ("FAST_MUSIC_REPLIES", True)):
            p = mock.patch.object(config, name, value)
            p.start()
            self.addCleanup(p.stop)

    def run_turn(self, *rounds, jukebox=None, sink=None, heard="hello"):
        fake, calls = script(*rounds)
        sink = sink or Recorder()
        with mock.patch.object(gemini, "chat_stream", fake):
            out = make_brain().answer_stream(heard, jukebox, sink)
        return out, sink, calls

    def test_plain_reply_is_spoken_sentence_by_sentence(self):
        end = {"role": "assistant", "content": "x"}
        out, sink, calls = self.run_turn([("text", "Hello there. It's "), ("text", "sunny today. Enjoy!"), ("end", end)])
        self.assertEqual(sink.sentences(), ["Hello there.", "It's sunny today.", "Enjoy!"])
        self.assertEqual(sink.events[-1], ("round_end", True, [], False))
        self.assertEqual(out["reply"], "Hello there. It's sunny today. Enjoy!")
        self.assertEqual(out["episode_id"], 7)
        self.assertTrue(out["timing"]["stream"])
        self.assertEqual(out["timing"]["ttft_ms"], 120)
        self.assertIsNotNone(out["timing"]["first_sentence_ms"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["messages"][-1], {"role": "user", "content": "hello"})
        self.assertTrue(calls[0]["closed"])

    def test_sentences_are_cleaned_for_the_ear(self):
        end = {"role": "assistant", "content": "x"}
        out, sink, _ = self.run_turn([("text", "**Done**, boss. Check [this](http://a.b) out now.\n"), ("end", end)])
        self.assertEqual(sink.sentences(), ["Done, boss.", "Check this out now."])

    def test_a_tool_round_ends_unspoken_and_the_signature_goes_back(self):
        call = tool_call("get_weather", '{"place": "Rajkot"}', extra_content={"google": {"thought_signature": "SIG"}})
        round1 = [("text", "Let me check that for you."),
                  ("end", {"role": "assistant", "content": "Let me check that for you.", "tool_calls": [call]})]
        round2 = [("text", "It's thirty one degrees in Rajkot."), ("end", {"role": "assistant", "content": "x"})]
        with mock.patch("weather.is_cached", return_value=False), \
                mock.patch("weather.get_weather", return_value={"temp": 31}):
            out, sink, calls = self.run_turn(round1, round2)
        # the unfinished line before the call is dropped, and the sink is told the round was not the answer
        self.assertEqual(sink.events, [("round_end", False, ["get_weather"], True),
                                       ("sentence", "It's thirty one degrees in Rajkot."),
                                       ("round_end", True, [], False)])
        self.assertEqual(out["reply"], "It's thirty one degrees in Rajkot.")
        self.assertEqual(out["tools"][0]["tool"], "get_weather")
        assistant = calls[1]["messages"][-2]
        self.assertEqual(assistant["tool_calls"][0]["extra_content"], {"google": {"thought_signature": "SIG"}})
        self.assertEqual(calls[1]["messages"][-1]["role"], "tool")
        self.assertEqual(len(out["timing"]["gemini_ms"]), 2)

    def test_a_finished_line_before_the_call_reaches_the_sink_but_is_not_the_reply(self):
        call = tool_call("get_weather", '{"place": "Rajkot"}')
        round1 = [("text", "Let me check that for you. One moment"),
                  ("end", {"role": "assistant", "content": "x", "tool_calls": [call]})]
        round2 = [("text", "It's thirty one degrees in Rajkot."), ("end", {"role": "assistant", "content": "x"})]
        with mock.patch("weather.is_cached", return_value=True), \
                mock.patch("weather.get_weather", return_value={"temp": 31}):
            out, sink, _ = self.run_turn(round1, round2)
        # the sink has to be able to drop what it was given in a round that turned out to be a tool round
        self.assertEqual(sink.events, [("sentence", "Let me check that for you."),
                                       ("round_end", False, ["get_weather"], False),
                                       ("sentence", "It's thirty one degrees in Rajkot."),
                                       ("round_end", True, [], False)])
        self.assertEqual(out["reply"], "It's thirty one degrees in Rajkot.")

    def test_cached_weather_is_not_slow_but_a_search_is(self):
        weather_call = tool_call("get_weather", "{}")
        search_call = tool_call("web_search", '{"query": "score"}')
        b = make_brain()
        with mock.patch("weather.is_cached", return_value=True):
            self.assertFalse(b._tool_is_slow(weather_call))
        with mock.patch("weather.is_cached", return_value=False):
            self.assertTrue(b._tool_is_slow(weather_call))
            self.assertTrue(b._tool_is_slow(search_call))
        with mock.patch.object(config, "WEB_SEARCH", False):
            self.assertFalse(b._tool_is_slow(search_call))
        self.assertFalse(b._tool_is_slow(tool_call("save_memory", '{"fact": "x"}')))
        self.assertFalse(make_brain(city="")._tool_is_slow(weather_call))

    def test_quick_music_reply_costs_no_second_round(self):
        jb = FakeJukebox()
        round1 = [("end", {"role": "assistant", "content": "", "tool_calls": [tool_call("play_music", '{"query": "kesariya"}')]})]
        out, sink, calls = self.run_turn(round1, jukebox=jb)
        self.assertEqual(len(calls), 1)
        self.assertEqual(sink.events, [("round_end", False, ["play_music"], False),
                                       ("sentence", "Here's Kesariya."), ("round_end", True, [], False)])
        self.assertEqual(out["actions"], {"music": "play", "track": 12, "title": "Kesariya"})
        self.assertTrue(out["timing"]["fast"])
        self.assertEqual(jb.played, ["kesariya"])

    def test_music_cannot_change_once_the_reply_has_started(self):
        jb = FakeJukebox()
        sink = Recorder()
        sink.locked = True
        round1 = [("end", {"role": "assistant", "content": "", "tool_calls": [tool_call("play_music", '{"query": "kesariya"}')]})]
        round2 = [("text", "Sorry, I can't start music right now."), ("end", {"role": "assistant", "content": "x"})]
        out, sink, calls = self.run_turn(round1, round2, jukebox=jb, sink=sink)
        self.assertEqual(jb.played, [])
        self.assertEqual(out["actions"], {})
        self.assertIn("already started", calls[1]["messages"][-1]["content"])
        self.assertEqual(out["reply"], "Sorry, I can't start music right now.")

    def test_a_long_reply_stops_gemini_at_the_limit(self):
        long_text = " ".join(f"This is sentence number {n} of a rather long answer." for n in range(1, 30))
        with mock.patch.object(config, "MAX_REPLY_CHARS", 120):
            out, sink, calls = self.run_turn([("text", long_text[:200]), ("text", long_text[200:]),
                                              ("end", {"role": "assistant", "content": "x"})])
        self.assertLessEqual(len(out["reply"]), 120)
        self.assertTrue(out["reply"].endswith("."))
        self.assertTrue(calls[0]["closed"])
        self.assertEqual(sink.events[-1], ("round_end", True, [], False))
        self.assertEqual(" ".join(sink.sentences()), out["reply"])

    def test_hanging_up_stops_the_turn(self):
        sink = Recorder()
        sink.cancel = True
        with self.assertRaises(brain.TurnCancelled):
            self.run_turn([("text", "Hello there friend."), ("end", {"role": "assistant", "content": "x"})], sink=sink)

    def test_nothing_to_say_is_an_error(self):
        with self.assertRaises(gemini.GeminiError):
            self.run_turn([("text", "  "), ("end", {"role": "assistant", "content": " "})])

    def test_works_with_no_sink_at_all(self):
        fake, _ = script([("text", "Plain answer here."), ("end", {"role": "assistant", "content": "x"})])
        with mock.patch.object(gemini, "chat_stream", fake):
            self.assertEqual(make_brain().answer_stream("hi")["reply"], "Plain answer here.")

    def test_the_last_round_is_sent_without_tools(self):
        calls_made = []
        tc = {"role": "assistant", "content": "", "tool_calls": [tool_call("save_memory", '{"fact": "x"}')]}
        final = [("text", "All done now."), ("end", {"role": "assistant", "content": "x"})]
        with mock.patch.object(config, "MAX_TOOL_ROUNDS", 1):
            out, sink, calls = self.run_turn([("end", tc)], final)
        self.assertIsNotNone(calls[0]["tools"])
        self.assertIsNone(calls[1]["tools"])
        self.assertEqual(out["reply"], "All done now.")


if __name__ == "__main__":
    unittest.main()
