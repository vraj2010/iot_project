"""How long does a question take to come back as sound? Compares the streamed reply with the normal one.

    python tools/bench_latency.py                    3 rounds of 8 questions, each asked both ways
    python tools/bench_latency.py --runs 5           more rounds: steadier numbers
    python tools/bench_latency.py --gap 60           wait a minute between questions (the connections go cold)
    python tools/bench_latency.py --tools            also questions that need the weather or a web search
    python tools/bench_latency.py --wav question.wav also send a recording to /ask (that includes Whisper)
    python tools/bench_latency.py --host 192.168.1.20   the server is on another PC

It talks to the running server the way the speaker does, with the same token, and times each reply:
"first sound" is how long until the first bit of the reply reaches the speaker, "done" is until the last bit has.
The two ways are asked in turns (X-Stream: 1 and 0), so a slow minute on the network hits both the same.
The questions are sent with X-No-Log, so they don't end up in the conversation history or in memory.

Questions that make Gemini search the web or look up the weather are not in the default set: how long
those take depends on the other website far more than on anything here.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import frames  # noqa: E402

PROMPTS = [
    "How are you doing today?",
    "Tell me a fun fact about octopuses.",
    "What's the capital of Australia?",
    "Explain what a black hole is.",
    "Give me three tips for a good night's sleep.",
    "Who wrote Pride and Prejudice?",
    "What is twelve times eight?",
    "Tell me a short joke.",
]
TOOL_PROMPTS = [
    "What's the weather like today?",
    "Who won the most recent cricket world cup?",
]


class Sample:
    def __init__(self, mode: str, prompt: str) -> None:
        self.mode = mode                  # "streamed" or "normal"
        self.prompt = prompt
        self.status: int | None = None
        self.first_ms: float | None = None
        self.done_ms: float | None = None
        self.size = 0
        self.text = ""
        self.error = ""

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.first_ms is not None

    def row(self) -> list:
        return [self.mode, self.prompt, self.status, "" if self.first_ms is None else round(self.first_ms),
                "" if self.done_ms is None else round(self.done_ms), self.size, self.error]


def timed_post(host: str, port: int, path: str, body: bytes, headers: dict, timeout: float = 90.0):
    """(status, body, ms until the first bytes of the body, ms until the end)."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    t0 = time.perf_counter()
    try:
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        first = resp.read1(1 << 16) if hasattr(resp, "read1") else resp.read(1)
        t_first = (time.perf_counter() - t0) * 1000
        data = first + resp.read()
        return resp.status, data, t_first, (time.perf_counter() - t0) * 1000
    finally:
        conn.close()


def ask(host: str, port: int, mode: str, prompt: str, wav: bytes | None, music: bool) -> Sample:
    sample = Sample(mode, prompt)
    headers = {"X-Device-Token": config.DEVICE_TOKEN, "X-Stream": "1" if mode == "streamed" else "0",
               "X-No-Log": "1"}
    if music:
        headers["X-Music-State"] = "state=stopped"
    if wav is None:
        path, body = "/say", json.dumps({"text": prompt}).encode()
        headers["Content-Type"] = "application/json"
    else:
        path, body = "/ask", wav
        headers["Content-Type"] = "audio/wav"
    try:
        sample.status, data, sample.first_ms, sample.done_ms = timed_post(host, port, path, body, headers)
    except OSError as exc:
        sample.error = f"{type(exc).__name__}: {exc}"
        return sample
    sample.size = len(data)
    try:
        if sample.status == 200:
            sample.text = frames.parse(data)["text"]
        else:
            sample.error = str(json.loads(data).get("error", ""))[:120]
    except (ValueError, KeyError, UnicodeDecodeError, IndexError) as exc:
        sample.error = f"unreadable reply ({type(exc).__name__})"
    return sample


def percentile(values: list[float], p: float) -> float | None:
    values = sorted(values)
    if not values:
        return None
    return values[max(0, math.ceil(p / 100 * len(values)) - 1)]


def summarize(samples: list[Sample]) -> dict:
    """Per mode: how many worked, and median / 90th percentile / worst of first sound and done."""
    out = {}
    for mode in ("normal", "streamed"):
        mine = [s for s in samples if s.mode == mode]
        good = [s for s in mine if s.ok]
        first = [s.first_ms for s in good]
        done = [s.done_ms for s in good if s.done_ms is not None]
        out[mode] = {
            "asked": len(mine), "worked": len(good),
            "first_median": statistics.median(first) if first else None, "first_p90": percentile(first, 90),
            "first_worst": max(first) if first else None,
            "done_median": statistics.median(done) if done else None, "done_p90": percentile(done, 90),
            "done_worst": max(done) if done else None,
        }
    return out


def fmt(v: float | None) -> str:
    return "     -" if v is None else f"{v:6.0f}"


def report(samples: list[Sample], out=print) -> None:
    out("")
    out(f"{'question':<42} {'way':<9} {'first sound':>11} {'done':>7}")
    for s in samples:
        shown = fmt(s.first_ms) if s.ok else "  FAILED"
        out(f"{s.prompt[:41]:<42} {s.mode:<9} {shown:>11} {fmt(s.done_ms) if s.ok else '':>7}"
            f"{'  ' + s.error if s.error else ''}")
    summary = summarize(samples)
    out("")
    out("                first sound (ms)                done (ms)")
    out("             worked  median    p90   worst     median    p90   worst")
    for mode in ("normal", "streamed"):
        m = summary[mode]
        out(f"{mode:<9} {m['worked']:>3}/{m['asked']:<3}  {fmt(m['first_median'])} {fmt(m['first_p90'])} "
            f"{fmt(m['first_worst'])}    {fmt(m['done_median'])} {fmt(m['done_p90'])} {fmt(m['done_worst'])}")
    n, st = summary["normal"], summary["streamed"]
    if n["first_median"] and st["first_median"]:
        diff = st["first_median"] - n["first_median"]
        out("")
        out(f"First sound, median: {abs(diff):.0f} ms {'sooner' if diff < 0 else 'later'} when streamed "
            f"({abs(diff) / n['first_median'] * 100:.0f} %).")
        if n["first_p90"] and st["first_p90"]:
            d90 = st["first_p90"] - n["first_p90"]
            out(f"First sound, slowest 10 %: {abs(d90):.0f} ms {'sooner' if d90 < 0 else 'later'} when streamed.")
    failed = [s for s in samples if not s.ok]
    if failed:
        out(f"{len(failed)} of {len(samples)} questions failed; the first reason was: {failed[0].error or failed[0].status}")


def read_wav(path: str) -> bytes:
    return Path(path).read_bytes()


def health(host: str, port: int) -> dict | None:
    try:
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        data = json.loads(resp.read() or b"{}")
        conn.close()
        return data if resp.status == 200 else None
    except (OSError, ValueError):
        return None


def main(argv: list[str] | None = None, out=print) -> int:
    ap = argparse.ArgumentParser(description="Time the server's replies, streamed against normal")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=config.HTTP_PORT)
    ap.add_argument("--runs", type=int, default=3, help="how many times to go through the questions")
    ap.add_argument("--gap", type=float, default=2.0, help="seconds to wait between questions")
    ap.add_argument("--limit", type=int, default=0, help="only use the first N questions")
    ap.add_argument("--tools", action="store_true", help="also ask questions that need the weather or a web search")
    ap.add_argument("--wav", metavar="FILE", help="also send this recording to /ask each round")
    ap.add_argument("--no-music", action="store_true", help="don't pretend to be a speaker that can play music")
    ap.add_argument("--csv", metavar="FILE", help="save every timing here")
    args = ap.parse_args(argv)

    if not config.DEVICE_TOKEN:
        out("DEVICE_TOKEN is not set in .env: the server wouldn't accept the questions.")
        return 2
    info = health(args.host, args.port)
    if info is None:
        out(f"Nothing answered at http://{args.host}:{args.port}/health. Is server.py running?")
        return 2
    out(f"Server {args.host}:{args.port}: model {info.get('model')}, voice {info.get('voice')}, "
        f"streaming {'on' if info.get('stream') else 'off'} by default")
    prompts = list(PROMPTS) + (TOOL_PROMPTS if args.tools else [])
    if args.limit:
        prompts = prompts[:args.limit]
    wav = read_wav(args.wav) if args.wav else None

    jobs = [(p, None) for p in prompts] + ([(Path(args.wav).name, wav)] if wav is not None else [])
    samples: list[Sample] = []
    total = len(jobs) * args.runs * 2
    done = 0
    for run in range(args.runs):
        for index, (prompt, audio) in enumerate(jobs):
            order = ("normal", "streamed") if (run + index) % 2 == 0 else ("streamed", "normal")
            for mode in order:
                sample = ask(args.host, args.port, mode, prompt, audio, not args.no_music)
                samples.append(sample)
                done += 1
                out(f"  [{done}/{total}] {mode:<8} {prompt[:40]:<40} "
                    f"{('%d ms' % sample.first_ms) if sample.ok else 'FAILED'}")
                if args.gap > 0 and done < total:
                    time.sleep(args.gap)
    report(samples, out)
    if args.csv:
        import csv

        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["way", "question", "status", "first_sound_ms", "done_ms", "bytes", "error"])
            w.writerows(s.row() for s in samples)
        out(f"Every timing is in {args.csv}")
    return 0 if all(s.ok for s in samples) else 1


if __name__ == "__main__":
    sys.exit(main())
