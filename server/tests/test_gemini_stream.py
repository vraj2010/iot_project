import json
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import _path  # noqa: F401
import requests

import config
import gemini
from fakes import FakeResponse, FakeSession, delta, sse

MSG = [{"role": "user", "content": "hi"}]


def call_chunk(**call):
    return {"choices": [{"delta": {"tool_calls": [call]}}]}


class SseParsing(unittest.TestCase):
    def test_events_survive_being_cut_anywhere(self):
        raw = sse(delta("naïve café 🙂 ok"), delta(" more"), {"choices": [{"delta": {}, "finish_reason": "stop"}]})
        want = list(gemini._iter_sse([raw]))
        self.assertEqual(len(want), 3)
        for cut in range(1, len(raw)):
            self.assertEqual(list(gemini._iter_sse([raw[:cut], raw[cut:]])), want, f"cut at {cut}")

    def test_comments_crlf_and_junk_are_ignored(self):
        raw = (b": keep-alive\r\n\r\nevent: message\r\ndata: {\"a\": 1}\r\n\r\ndata: not json\r\n\r\n"
               b"data: {\"b\": 2}\r\n\r\ndata: [DONE]\r\n\r\ndata: {\"never\": 3}\r\n")
        self.assertEqual(list(gemini._iter_sse([raw])), [{"a": 1}, {"b": 2}])

    def test_last_event_without_a_final_newline(self):
        self.assertEqual(list(gemini._iter_sse([b'data: {"a": 1}'])), [{"a": 1}])


class Assembling(unittest.TestCase):
    def build(self, *chunks):
        acc = gemini._Acc()
        text = [acc.add(c) for c in chunks]
        return acc.message(), text

    def test_text(self):
        msg, text = self.build(delta("Hello"), delta(" there."))
        self.assertEqual(msg, {"role": "assistant", "content": "Hello there."})
        self.assertEqual(text, ["Hello", " there."])

    def test_a_tool_call_in_pieces_keeps_gemini_signature(self):
        msg, _ = self.build(
            call_chunk(index=0, id="c1", type="function", function={"name": "get_weather", "arguments": '{"pla'},
                       extra_content={"google": {"thought_signature": "SIG=="}}),
            call_chunk(index=0, function={"arguments": 'ce": "Rajkot"}'}))
        call = msg["tool_calls"][0]
        self.assertEqual(call["function"], {"name": "get_weather", "arguments": '{"place": "Rajkot"}'})
        self.assertEqual(call["id"], "c1")
        self.assertEqual(call["extra_content"], {"google": {"thought_signature": "SIG=="}})

    def test_parallel_calls_all_numbered_zero_are_still_two_calls(self):
        msg, _ = self.build(
            call_chunk(index=0, id="a", function={"name": "get_weather", "arguments": '{"place": "A"}'}),
            call_chunk(index=0, id="b", function={"name": "web_search", "arguments": '{"query": "x"}'}))
        self.assertEqual([c["function"]["name"] for c in msg["tool_calls"]], ["get_weather", "web_search"])

    def test_whole_calls_without_index_or_id(self):
        msg, _ = self.build(call_chunk(function={"name": "a", "arguments": "{}"}),
                            call_chunk(function={"name": "b", "arguments": '{"x": 1}'}))
        self.assertEqual([c["function"]["name"] for c in msg["tool_calls"]], ["a", "b"])

    def test_call_without_arguments_gets_an_empty_object(self):
        msg, _ = self.build(call_chunk(index=0, id="c", function={"name": "get_weather"}))
        self.assertEqual(msg["tool_calls"][0]["function"]["arguments"], "{}")

    def test_name_repeated_in_a_continuation_does_not_split_the_call(self):
        msg, _ = self.build(call_chunk(index=0, id="c", function={"name": "w", "arguments": '{"pl'}),
                            call_chunk(index=0, id="c", function={"name": "w", "arguments": 'ace": "x"}'}))
        self.assertEqual(len(msg["tool_calls"]), 1)
        self.assertEqual(msg["tool_calls"][0]["function"]["arguments"], '{"place": "x"}')

    def test_an_error_event_raises(self):
        with self.assertRaises(gemini.GeminiError):
            gemini._Acc().add({"error": {"message": "quota"}})


class ChatStream(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, gemini, "_reasoning_ok", gemini._reasoning_ok)
        for name, value in (("GEMINI_HEDGE_S", 3.0), ("GEMINI_TIMEOUT", 5.0), ("GEMINI_API_KEY", "k"),
                            ("GEMINI_BASE", "http://gemini.test/v1beta")):
            p = mock.patch.object(config, name, value)
            p.start()
            self.addCleanup(p.stop)

    def run_stream(self, first, fresh=None, **kw):
        """Run chat_stream with `first` as the session in use and `fresh` as the one a retry opens."""
        info = {}
        p1 = mock.patch.object(gemini, "_session", first)
        p2 = mock.patch.object(gemini, "_new_session", lambda: fresh)
        with p1, p2:
            events = list(gemini.chat_stream(MSG, info=info, **kw))
        return events, info

    def test_streams_words_then_the_whole_message(self):
        session = FakeSession(FakeResponse([sse(delta("Hi"), delta(" there."))]))
        events, info = self.run_stream(session)
        self.assertEqual(events, [("text", "Hi"), ("text", " there."),
                                  ("end", {"role": "assistant", "content": "Hi there."})])
        self.assertFalse(info["hedged"])
        self.assertIn("ttft_ms", info)
        sent = session.calls[0]
        self.assertTrue(sent["json"]["stream"])
        self.assertTrue(sent["url"].endswith("/openai/chat/completions"))

    def test_hedge_asks_again_when_nothing_arrives_and_keeps_the_faster_connection(self):
        stuck_resp = FakeResponse(hang=True)
        stuck = FakeSession(stuck_resp)
        fast = FakeSession(FakeResponse([sse(delta("Hello."))]))
        events, info = self.run_stream(stuck, fast, hedge_s=0.05)
        self.assertEqual(events[-1], ("end", {"role": "assistant", "content": "Hello."}))
        self.assertTrue(info["hedged"])
        self.assertTrue(stuck_resp.closed.wait(2), "the stuck request should have been cancelled")
        self.assertEqual(len(fast.calls), 1)

    def test_the_fresh_connection_replaces_the_stale_one_for_next_time(self):
        stuck, fast = FakeSession(FakeResponse(hang=True)), FakeSession(FakeResponse([sse(delta("Hello."))]))
        with mock.patch.object(gemini, "_session", stuck), mock.patch.object(gemini, "_new_session", lambda: fast):
            list(gemini.chat_stream(MSG, hedge_s=0.05))
            self.assertIs(gemini._session, fast)
        for _ in range(40):
            if stuck.closed:
                break
            time.sleep(0.05)
        self.assertTrue(stuck.closed, "the stale session should be closed")

    def test_a_failed_connection_is_retried_at_once(self):
        broken = FakeSession(requests.ConnectionError("reset"))
        fast = FakeSession(FakeResponse([sse(delta("Back."))]))
        events, info = self.run_stream(broken, fast, hedge_s=30)
        self.assertEqual(events[-1][1]["content"], "Back.")
        self.assertTrue(info["hedged"])

    def test_a_bad_request_is_not_retried(self):
        bad = FakeSession(FakeResponse(status=400, text='{"error": "bad"}'))
        fresh = FakeSession()
        with self.assertRaises(gemini.GeminiError) as ctx:
            self.run_stream(bad, fresh)
        self.assertIn("400", str(ctx.exception))
        self.assertEqual(fresh.calls, [])

    def test_a_rate_limit_is_not_retried(self):
        limited = FakeSession(FakeResponse(status=429))
        fresh = FakeSession()
        with self.assertRaises(gemini.GeminiError) as ctx:
            self.run_stream(limited, fresh)
        self.assertEqual(ctx.exception.status, 429)
        self.assertEqual(fresh.calls, [])

    def test_a_model_that_refuses_reasoning_effort_is_asked_again_without_it(self):
        session = FakeSession(FakeResponse(status=400, text="Unknown name reasoning_effort"),
                              FakeResponse([sse(delta("Fine."))]))
        events, _ = self.run_stream(session, reasoning="low")
        self.assertEqual(events[-1][1]["content"], "Fine.")
        self.assertIn("reasoning_effort", session.calls[0]["json"])
        self.assertNotIn("reasoning_effort", session.calls[1]["json"])
        self.assertFalse(gemini._reasoning_ok)

    def test_a_server_that_ignores_stream_still_works(self):
        reply = {"choices": [{"message": {"role": "assistant", "content": "Plain.",
                                          "tool_calls": [{"id": "x", "type": "function",
                                                          "function": {"name": "get_weather", "arguments": "{}"}}]}}]}
        session = FakeSession(FakeResponse(ctype="application/json", json_data=reply))
        events, _ = self.run_stream(session)
        self.assertEqual(events[0], ("text", "Plain."))
        self.assertEqual(events[-1][1]["tool_calls"][0]["function"]["name"], "get_weather")

    def test_an_empty_stream_is_an_error(self):
        session = FakeSession(FakeResponse([sse(delta(role="assistant"))]))
        with self.assertRaises(gemini.GeminiError):
            self.run_stream(session)

    def test_closing_early_cancels_the_request(self):
        resp = FakeResponse([sse(delta("One."), delta(" Two."), delta(" Three."))], gap=0.02)
        with mock.patch.object(gemini, "_session", FakeSession(resp)):
            gen = gemini.chat_stream(MSG)
            self.assertEqual(next(gen), ("text", "One."))
            gen.close()
        self.assertTrue(resp.closed.wait(2))

    def test_a_stream_that_stalls_after_starting_raises(self):
        class Stalls(FakeResponse):
            def iter_content(self, chunk_size=None):
                yield sse(delta("Start"))[:-len(b"data: [DONE]\n\n")]
                raise requests.ConnectionError("dropped")

        with self.assertRaises(gemini.GeminiError):
            self.run_stream(FakeSession(Stalls()))


class _SseServer(ThreadingHTTPServer):
    """A pretend Gemini on this computer that answers like the real one: chunked server-sent events for a
    streamed request, plain JSON otherwise, over connections that stay open. It counts the TCP connections."""

    daemon_threads = True

    def __init__(self):
        self.connections = 0
        self.rest_of_answer = threading.Event()               # clear it to hold the end of a streamed answer back
        self.rest_of_answer.set()

        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self):
                super().setup()
                outer.connections += 1

            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for number, event in enumerate((delta("Hello there friend. "), delta("Bye."))):
                        if number:
                            outer.rest_of_answer.wait(5)
                        data = b"data: " + json.dumps(event).encode() + b"\n\n"
                        self.wfile.write(b"%X\r\n%s\r\n" % (len(data), data))
                    data = b"data: [DONE]\n\n"
                    self.wfile.write(b"%X\r\n%s\r\n" % (len(data), data))
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    out = json.dumps({"choices": [{"message": {"role": "assistant", "content": "hi"}}]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)

        super().__init__(("127.0.0.1", 0), Handler)

    def handle_error(self, request, client_address):          # a client that hangs up mid-answer is part of the tests
        pass


class ConnectionReuse(unittest.TestCase):
    """The connection to Google is kept for the next request: opening a new one costs a TCP and a TLS handshake."""

    def setUp(self):
        self.server = _SseServer()
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        env = mock.patch.dict(os.environ, {"NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"})
        env.start()
        self.addCleanup(env.stop)
        for name, value in (("GEMINI_BASE", f"http://127.0.0.1:{self.server.server_address[1]}"),
                            ("GEMINI_API_KEY", "test-key"), ("GEMINI_HEDGE_S", 0)):
            patch = mock.patch.object(config, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        session = gemini._new_session()
        patch = mock.patch.object(gemini, "_session", session)
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(session.close)

    def opened_by(self, call) -> int:
        before = self.server.connections
        call()
        return self.server.connections - before

    def test_streamed_answers_leave_the_connection_for_the_next_request(self):
        self.assertEqual(self.opened_by(lambda: gemini.chat(MSG)), 1)              # the first request opens it
        for number in (1, 2, 3):
            events = []
            self.assertEqual(self.opened_by(lambda: events.extend(gemini.chat_stream(MSG))), 0,
                             f"streamed answer {number} opened a new connection")
            self.assertEqual(events[-1][1]["content"], "Hello there friend. Bye.")
        self.assertEqual(self.opened_by(lambda: gemini.chat(MSG)), 0)              # and an ordinary request too

    def test_stopping_early_throws_that_connection_away(self):
        """Half an answer is still on the wire: that connection can't be used for anything else."""
        gemini.chat(MSG)
        self.server.rest_of_answer.clear()
        gen = gemini.chat_stream(MSG)
        self.assertEqual(next(gen), ("text", "Hello there friend. "))
        gen.close()
        self.server.rest_of_answer.set()
        self.assertEqual(self.opened_by(lambda: gemini.chat(MSG)), 1)


if __name__ == "__main__":
    unittest.main()
