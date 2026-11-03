"""Where does the time go on this laptop's network? Run it on the laptop that runs server.py.

    python tools/diag_network.py                 a quick look at Google (Gemini), Open-Meteo and the Edge voice
    python tools/diag_network.py --idle          also find out how long an idle connection to Google survives (~8 minutes)
    python tools/diag_network.py --idle --gaps 10,60,120,300      pauses to try, in seconds
    python tools/diag_network.py --only gemini,idle                just those parts (gemini, weather, edge, idle)

What it measures
  * Gemini and Open-Meteo: DNS lookup, TCP connect, TLS handshake and time to the first byte of a request, so you
    can see what a brand-new connection costs
  * --idle: the same request again and again on ONE kept-alive connection, after pauses of different lengths. If it is
    suddenly much slower after a pause, something on the way (the router, a VPN, the antivirus) drops idle
    connections after about that long. That is where the 10-30 second stalls in latency.log come from, and
    GEMINI_KEEPALIVE_S has to be shorter than the pause that broke it
  * the Edge voice: time to the first sound of a short and a longer sentence, alone and three at a time
    (three at a time is what TTS_PARALLEL does)

Nothing here changes any setting, and the API key is never printed.
"""

from __future__ import annotations

import argparse
import socket
import ssl
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

SHORT = "Good morning, boss."
LONG = "It is a mild evening in Rajkot, with a light breeze from the west and a little rain on the way after midnight."
DEFAULT_GAPS = [0, 5, 15, 30, 60, 120, 240]


# ---------------------------------------------------------------- one brand-new connection, in steps

def probe(url: str, headers: dict | None = None, timeout: float = 15.0) -> dict:
    """One request on a brand-new connection. Milliseconds for: dns, tcp, tls (https only), first_byte
    (from sending the request to the first byte back) and total; plus status. `error` if it failed."""
    u = urlparse(url)
    https = u.scheme == "https"
    host, port = u.hostname, u.port or (443 if https else 80)
    path = (u.path or "/") + (f"?{u.query}" if u.query else "")
    out: dict = {"url": url, "error": None, "status": None}
    sock = None
    try:
        t0 = time.perf_counter()
        ip = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0][4][0]
        t1 = time.perf_counter()
        sock = socket.create_connection((ip, port), timeout=timeout)
        t2 = time.perf_counter()
        if https:
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
        t3 = time.perf_counter()
        request = [f"GET {path} HTTP/1.1", f"Host: {host}", "User-Agent: friday-diag", "Accept: */*",
                   "Connection: close"] + [f"{k}: {v}" for k, v in (headers or {}).items()]
        sock.sendall(("\r\n".join(request) + "\r\n\r\n").encode())
        data = sock.recv(65536)
        t4 = time.perf_counter()
        sock.settimeout(5)
        try:
            while True:
                more = sock.recv(65536)
                if not more:
                    break
                data += more
        except OSError:
            pass
        t5 = time.perf_counter()
        out.update(dns=(t1 - t0) * 1000, tcp=(t2 - t1) * 1000, tls=(t3 - t2) * 1000 if https else None,
                   first_byte=(t4 - t3) * 1000, total=(t5 - t0) * 1000,
                   status=int(data.split(b" ", 2)[1]) if data.startswith(b"HTTP/") else None)
    except (OSError, ValueError, IndexError) as exc:        # ssl.SSLError is an OSError too
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
    return out


def probe_median(url: str, headers: dict | None, repeat: int) -> dict:
    """probe() a few times and keep the middle value of each step."""
    runs = [probe(url, headers) for _ in range(max(1, repeat))]
    good = [r for r in runs if not r["error"]]
    if not good:
        return runs[0]
    out = dict(good[0])
    for key in ("dns", "tcp", "tls", "first_byte", "total"):
        values = [r[key] for r in good if r.get(key) is not None]
        out[key] = statistics.median(values) if values else None
    out["failed"] = len(runs) - len(good)
    return out


def describe_probe(name: str, p: dict, out) -> None:
    if p["error"]:
        out(f"  {name}: FAILED - {p['error']}")
        return
    parts = [f"DNS {p['dns']:.0f}", f"TCP {p['tcp']:.0f}"]
    if p.get("tls") is not None:
        parts.append(f"TLS {p['tls']:.0f}")
    parts.append(f"first byte {p['first_byte']:.0f}")
    out(f"  {name}: {' + '.join(parts)} ms = {p['total']:.0f} ms in all (HTTP {p['status']})")
    setup = p["dns"] + p["tcp"] + (p.get("tls") or 0)
    out(f"      a brand-new connection costs about {setup:.0f} ms before the request is even sent; "
        f"a kept-alive one skips that")


# ---------------------------------------------------------------- idle connections

def timed_get(session, url: str, headers: dict | None, timeout: float = 30.0) -> float | str:
    t0 = time.perf_counter()
    try:
        r = session.get(url, headers=headers, timeout=(10.0, timeout))
        r.close()
    except Exception as exc:        # requests.RequestException and friends: report, don't stop
        return f"{type(exc).__name__}"
    return (time.perf_counter() - t0) * 1000


def idle_test(url: str, headers: dict | None, gaps: list[float], out, sleep=time.sleep, session=None) -> list:
    """Same request on one kept-alive connection after each pause. Returns [(gap, ms or error text)]."""
    if session is None:
        from httpsession import new_session

        session = new_session()
    first = timed_get(session, url, headers)               # opens the connection
    base = [timed_get(session, url, headers) for _ in range(3)]
    baseline = statistics.median([b for b in base if isinstance(b, float)] or [0.0])
    out(f"  new connection: {first if isinstance(first, str) else f'{first:.0f} ms'}; "
        f"reused straight away: {baseline:.0f} ms (that is the normal speed)")
    results = []
    for gap in gaps:
        if gap:
            out(f"  waiting {gap:g} s ...")
            sleep(gap)
        took = timed_get(session, url, headers)
        results.append((gap, took))
        out(f"  after {gap:g} s of silence: {took if isinstance(took, str) else f'{took:.0f} ms'}")
    for line in idle_verdict(baseline, results):
        out(line)
    return results


def idle_verdict(baseline_ms: float, results: list) -> list[str]:
    """Plain words about which pause (if any) made the connection go bad."""
    slow = max(baseline_ms * 3, baseline_ms + 400)
    bad = [(gap, took) for gap, took in results if isinstance(took, str) or took > slow]
    if not results:
        return []
    if not bad:
        top = max(gap for gap, _ in results)
        return [f"  -> no slowdown up to {top:g} s of silence: idle connections are not what is making replies slow here "
                f"(if the stalls in latency.log still happen, look at the Gemini side or the weather lookups)."]
    gap = bad[0][0]
    keep = max(5, int(gap * 0.6))
    return [f"  -> the connection became slow or died after {gap:g} s of silence. Something between this laptop and Google "
            f"(the router, a VPN, an antivirus) drops idle connections.",
            f"     Keep it warm: GEMINI_KEEPALIVE_S={keep} in .env (it is 40 now). Pings are free."]


# ---------------------------------------------------------------- the Edge voice

def edge_once(text: str):
    """(ms to the first sound, ms to the end, bytes) for one sentence."""
    import asyncio

    import edge_tts

    async def run():
        t0 = time.perf_counter()
        first = None
        size = 0
        com = edge_tts.Communicate(text, config.EDGE_VOICE, rate=config.EDGE_RATE, pitch=config.EDGE_PITCH)
        async for chunk in com.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                if first is None:
                    first = (time.perf_counter() - t0) * 1000
                size += len(chunk["data"])
        return first, (time.perf_counter() - t0) * 1000, size

    return asyncio.run(asyncio.wait_for(run(), timeout=config.EDGE_TIMEOUT + 15))


def edge_parallel(text: str, n: int):
    """The same sentence made n times at once: [(first ms, total ms, bytes) or an error text]."""
    import threading

    results: list = [None] * n

    def work(i: int) -> None:
        try:
            results[i] = edge_once(text)
        except Exception as exc:
            results[i] = f"{type(exc).__name__}: {exc}"

    threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def edge_section(out, repeat: int = 3, once=edge_once, parallel=edge_parallel) -> dict:
    stats = {}
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        out("  edge-tts is not installed (pip install edge-tts): skipping")
        return stats
    for label, text in (("short sentence", SHORT), ("longer sentence", LONG)):
        firsts, totals = [], []
        for _ in range(repeat):
            try:
                first, total, _size = once(text)
            except Exception as exc:
                out(f"  {label}: FAILED - {type(exc).__name__}: {exc}")
                break
            firsts.append(first)
            totals.append(total)
        if firsts:
            stats[label] = statistics.median(firsts)
            out(f"  {label} ({len(text)} chars), alone: first sound {statistics.median(firsts):.0f} ms "
                f"(best {min(firsts):.0f}, worst {max(firsts):.0f}), all of it {statistics.median(totals):.0f} ms")
    n = max(2, config.TTS_PARALLEL)
    got = parallel(SHORT, n)
    good = [r for r in got if isinstance(r, tuple)]
    if good:
        firsts = [r[0] for r in good]
        stats["parallel"] = statistics.median(firsts)
        out(f"  short sentence, {n} at once: first sound {statistics.median(firsts):.0f} ms "
            f"(worst {max(firsts):.0f}); {len(good)} of {n} worked")
    else:
        out(f"  {n} at once: all failed - {got[0]}")
    for line in edge_verdict(stats):
        out(line)
    return stats


def edge_verdict(stats: dict) -> list[str]:
    lines = []
    short = stats.get("short sentence")
    if short is not None:
        if short > 1800:
            lines.append(f"  -> even a three-word sentence takes {short:.0f} ms to start sounding. That is Microsoft's "
                         f"service from here, not the length of the reply; short lines are served from the voice "
                         f"cache, everything else pays this once per reply.")
        else:
            lines.append(f"  -> the Edge voice starts in about {short:.0f} ms from here, which is fine.")
    if short is not None and stats.get("parallel") is not None and stats["parallel"] > short * 1.4 + 200:
        lines.append(f"  -> making several sentences at once is slower for each ({stats['parallel']:.0f} ms against "
                     f"{short:.0f} ms): try TTS_PARALLEL=2 or 1.")
    return lines


# ---------------------------------------------------------------- main

def parse_gaps(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def main(argv: list[str] | None = None, out=print) -> int:
    ap = argparse.ArgumentParser(description="Time the connections Friday depends on")
    ap.add_argument("--only", default="gemini,weather,edge", help="parts to run: gemini, weather, edge, idle")
    ap.add_argument("--idle", action="store_true", help="also run the idle-connection test (takes minutes)")
    ap.add_argument("--gaps", default=",".join(str(g) for g in DEFAULT_GAPS), help="pauses for --idle, in seconds")
    ap.add_argument("--repeat", type=int, default=3, help="repeat each quick probe this many times")
    args = ap.parse_args(argv)
    parts = {p.strip() for p in args.only.split(",") if p.strip()}
    if args.idle:
        parts.add("idle")
    unknown = parts - {"gemini", "weather", "edge", "idle"}
    if unknown:
        out(f"Unknown part(s): {', '.join(sorted(unknown))}")
        return 2

    gemini_url = f"{config.GEMINI_BASE}/models/{config.GEMINI_MODEL}"
    headers = None
    if parts & {"gemini", "idle"}:
        if not config.GEMINI_API_KEY:
            out("GEMINI_API_KEY is not set in .env: skipping the Gemini parts.")
            parts -= {"gemini", "idle"}
        else:
            import gemini

            headers = gemini._native_headers()

    if "gemini" in parts:
        out("Gemini (a request that asks Google for the model's details: no tokens used)")
        describe_probe("Google", probe_median(gemini_url, headers, args.repeat), out)
    if "weather" in parts:
        out("Weather (Open-Meteo)")
        describe_probe("forecast", probe_median(
            "https://api.open-meteo.com/v1/forecast?latitude=22.3&longitude=70.8&current=temperature_2m",
            None, args.repeat), out)
    if "edge" in parts:
        out("Voice (Microsoft Edge read-aloud)")
        edge_section(out, args.repeat)
    if "idle" in parts:
        gaps = parse_gaps(args.gaps)
        total = sum(gaps)
        out(f"Idle connections to Google: pauses of {', '.join(f'{g:g}' for g in gaps)} s (about {total / 60:.0f} minutes)")
        idle_test(gemini_url, headers, gaps, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
