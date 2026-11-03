"""The benchmark and the network check: the parts that can be tested without Google or Microsoft."""

import http.server
import io
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import _path  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import bench_latency  # noqa: E402
import config  # noqa: E402
import diag_network  # noqa: E402
import gemini  # noqa: E402
from brainfakes import always  # noqa: E402
from test_server_stream import END, Base  # noqa: E402
from voicefakes import wait_until  # noqa: E402


def sample(mode, first, done=None, ok=True, prompt="q"):
    s = bench_latency.Sample(mode, prompt)
    s.status = 200 if ok else 502
    s.first_ms = first if ok else None
    s.done_ms = done if done is not None else (first + 1000 if ok else None)
    s.error = "" if ok else "boom"
    return s


class BenchStats(unittest.TestCase):
    def test_percentile_is_nearest_rank(self):
        values = list(range(1, 11))
        self.assertEqual(bench_latency.percentile(values, 90), 9)
        self.assertEqual(bench_latency.percentile(values, 100), 10)
        self.assertEqual(bench_latency.percentile([5], 90), 5)
        self.assertIsNone(bench_latency.percentile([], 90))

    def test_summary_counts_only_the_ones_that_worked(self):
        samples = [sample("normal", 4000), sample("normal", 5000), sample("normal", 0, ok=False),
                   sample("streamed", 3000), sample("streamed", 2000)]
        s = bench_latency.summarize(samples)
        self.assertEqual((s["normal"]["asked"], s["normal"]["worked"]), (3, 2))
        self.assertEqual(s["normal"]["first_median"], 4500)
        self.assertEqual(s["streamed"]["first_median"], 2500)
        self.assertEqual(s["streamed"]["first_worst"], 3000)

    def test_report_says_how_much_sooner(self):
        lines = []
        bench_latency.report([sample("normal", 4000), sample("streamed", 3000), sample("normal", 0, ok=False)],
                             lines.append)
        text = "\n".join(lines)
        self.assertIn("1000 ms sooner when streamed (25 %)", text)
        self.assertIn("FAILED", text)
        self.assertIn("1 of 3 questions failed", text)


class BenchAgainstTheServer(Base):
    def run_bench(self, *args):
        lines = []
        code = bench_latency.main(["--port", str(self.port), "--gap", "0", *args], lines.append)
        return code, "\n".join(lines)

    def test_it_asks_both_ways_and_reports_both(self):
        fake, calls = always(("text", "It is sunny today."), END)
        with mock.patch.object(gemini, "chat_stream", fake):
            code, text = self.run_bench("--runs", "2", "--limit", "2")
        self.assertEqual(code, 0, text)
        self.assertIn("streaming on by default", text)
        summary_lines = [line for line in text.splitlines() if line.startswith(("normal", "streamed"))
                         and "/" in line]
        self.assertEqual(len(summary_lines), 2)
        self.assertTrue(all("4/4" in line for line in summary_lines), summary_lines)
        self.assertEqual(self.assistant.legacy_calls.count("How are you doing today?"), 2)   # the "normal" half
        self.assertEqual(len(self.assistant.legacy_calls), 4)                                   # and nothing else
        self.assertEqual(len(calls), 4)                    # the streamed half: 2 questions x 2 runs
        self.assertEqual(self.assistant.stream_fails, 0)
        self.assertEqual(self.assistant.logged, [])        # none of it went into the history or the memory

    def test_a_recording_goes_to_ask_and_the_timings_can_be_saved(self):
        from test_server_stream import wav

        fake, _ = always(("text", "It is sunny today."), END)
        with tempfile.TemporaryDirectory() as d:
            wav_path, csv_path = os.path.join(d, "q.wav"), os.path.join(d, "t.csv")
            with open(wav_path, "wb") as f:
                f.write(wav())
            with mock.patch.object(gemini, "chat_stream", fake):
                code, text = self.run_bench("--runs", "1", "--limit", "1", "--wav", wav_path, "--csv", csv_path)
            self.assertEqual(code, 0, text)
            with open(csv_path, encoding="utf-8") as f:
                rows = f.read().splitlines()
        self.assertEqual(rows[0], "way,question,status,first_sound_ms,done_ms,bytes,error")
        self.assertEqual(len(rows), 1 + 2 * 2)             # one question and one recording, each both ways
        self.assertTrue(any("q.wav" in r for r in rows))

    def test_a_server_that_is_not_there(self):
        lines = []
        code = bench_latency.main(["--port", "1", "--gap", "0"], lines.append)
        self.assertEqual(code, 2)
        self.assertIn("Is server.py running?", "\n".join(lines))


class Page(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class Probe(unittest.TestCase):
    def setUp(self):
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Page)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/x"

    def test_a_brand_new_connection_is_timed_in_steps(self):
        p = diag_network.probe(self.url)
        self.assertIsNone(p["error"])
        self.assertEqual(p["status"], 200)
        self.assertIsNone(p["tls"])                        # plain http: no handshake
        for key in ("dns", "tcp", "first_byte", "total"):
            self.assertGreaterEqual(p[key], 0)
        self.assertGreaterEqual(p["total"], p["first_byte"])

    def test_the_middle_value_of_several(self):
        p = diag_network.probe_median(self.url, None, 3)
        self.assertEqual(p["status"], 200)
        self.assertEqual(p["failed"], 0)

    def test_a_closed_port_is_an_error_not_a_crash(self):
        p = diag_network.probe("http://127.0.0.1:1/x")
        self.assertIn("Error", p["error"])
        lines = []
        diag_network.describe_probe("nothing", p, lines.append)
        self.assertIn("FAILED", lines[0])

    def test_a_probe_is_described_in_words(self):
        lines = []
        diag_network.describe_probe("Google", diag_network.probe(self.url), lines.append)
        self.assertIn("DNS", lines[0])
        self.assertIn("brand-new connection costs about", lines[1])

    def test_the_idle_test_runs_through_its_pauses(self):
        class Session:
            def get(self, url, headers=None, timeout=None):
                return types.SimpleNamespace(close=lambda: None)

        waited, lines = [], []
        results = diag_network.idle_test(self.url, None, [0, 5, 60], lines.append, sleep=waited.append,
                                         session=Session())
        self.assertEqual([g for g, _ in results], [0, 5, 60])
        self.assertEqual(waited, [5, 60])
        self.assertIn("no slowdown up to 60 s", "\n".join(lines))


class Verdicts(unittest.TestCase):
    def test_no_slowdown(self):
        v = diag_network.idle_verdict(100, [(0, 110), (60, 130), (120, 120)])
        self.assertIn("no slowdown up to 120 s", v[0])

    def test_the_first_slow_pause_is_named_and_the_keepalive_suggested(self):
        v = diag_network.idle_verdict(100, [(0, 110), (30, 120), (60, 4200), (120, 9000)])
        self.assertIn("after 60 s of silence", v[0])
        self.assertIn("GEMINI_KEEPALIVE_S=36", v[1])

    def test_a_failure_counts_as_a_slowdown(self):
        v = diag_network.idle_verdict(100, [(0, 110), (15, "ConnectionError")])
        self.assertIn("after 15 s", v[0])
        self.assertIn("GEMINI_KEEPALIVE_S=9", v[1])

    def test_a_tiny_keepalive_is_never_suggested(self):
        self.assertIn("GEMINI_KEEPALIVE_S=5", diag_network.idle_verdict(100, [(5, 9000)])[1])

    def test_the_voice_verdicts(self):
        self.assertIn("fine", diag_network.edge_verdict({"short sentence": 900})[0])
        self.assertIn("Microsoft's service", diag_network.edge_verdict({"short sentence": 2600})[0])
        slow = diag_network.edge_verdict({"short sentence": 1000, "parallel": 2600})
        self.assertTrue(any("TTS_PARALLEL=2" in line for line in slow))
        quiet = diag_network.edge_verdict({"short sentence": 1000, "parallel": 1100})
        self.assertFalse(any("TTS_PARALLEL" in line for line in quiet))


class EdgeSection(unittest.TestCase):
    def test_it_reports_alone_and_at_once(self):
        lines = []
        with mock.patch.dict(sys.modules, {"edge_tts": types.ModuleType("edge_tts")}), \
                mock.patch.object(config, "TTS_PARALLEL", 3):
            stats = diag_network.edge_section(lines.append, repeat=2, once=lambda text: (900.0, 1500.0, 9000),
                                              parallel=lambda text, n: [(1000.0, 1600.0, 9000)] * n)
        self.assertEqual(stats, {"short sentence": 900.0, "longer sentence": 900.0, "parallel": 1000.0})
        text = "\n".join(lines)
        self.assertIn("3 at once", text)
        self.assertIn("3 of 3 worked", text)

    def test_a_voice_that_fails_is_reported_not_raised(self):
        def broken(text):
            raise RuntimeError("no network")

        lines = []
        with mock.patch.dict(sys.modules, {"edge_tts": types.ModuleType("edge_tts")}):
            diag_network.edge_section(lines.append, repeat=1, once=broken,
                                      parallel=lambda text, n: ["RuntimeError: no network"] * n)
        self.assertIn("FAILED - RuntimeError: no network", "\n".join(lines))
        self.assertIn("all failed", "\n".join(lines))


class Main(unittest.TestCase):
    def test_unknown_parts_are_refused(self):
        lines = []
        self.assertEqual(diag_network.main(["--only", "bogus"], lines.append), 2)
        self.assertIn("bogus", lines[0])

    def test_gaps(self):
        self.assertEqual(diag_network.parse_gaps("0, 5,60"), [0.0, 5.0, 60.0])

    def test_without_a_key_the_google_parts_are_skipped(self):
        lines = []
        with mock.patch.object(config, "GEMINI_API_KEY", ""):
            self.assertEqual(diag_network.main(["--only", "gemini"], lines.append), 0)
        self.assertIn("GEMINI_API_KEY is not set", lines[0])


if __name__ == "__main__":
    unittest.main()
