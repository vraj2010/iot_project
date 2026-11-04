"""NEXA Orb: Cyberpunk / JARVIS-style visual dashboard and control console.

Run it in a separate terminal alongside server.py:

    python orb.py

It provides:
  - 3D Animated glowing interactive Orb visualizer
  - Real-time synchronized terminal output and latency telemetry
  - Interactive Web Control: Push-to-Talk button, quick command chips, text input bar
  - Status badges for Database (nexa-pg), Gemini, Edge-TTS, and Mic Wake Word

Options:
    --port 8777        port to serve dashboard (default: 8777)
    --no-browser       don't auto-launch browser window
    --full             launch in fullscreen
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import config  # reads server/.env (DATABASE_URL, HTTP_PORT, DEVICE_TOKEN...)

HERE = Path(__file__).resolve().parent
PAGE = HERE / "orb.html"
LATENCY_LOG = HERE / "latency.log"


class Turns:
    """A tiny read-only view of the turns table in PostgreSQL."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = None

    def _query(self, sql: str, args: tuple = ()):
        import psycopg

        with self._lock:
            try:
                if self._conn is None or self._conn.closed:
                    self._conn = psycopg.connect(config.DATABASE_URL, autocommit=True, connect_timeout=3)
                return self._conn.execute(sql, args).fetchall()
            except Exception:
                try:
                    if self._conn is not None:
                        self._conn.close()
                finally:
                    self._conn = None
                raise

    def latest_id(self) -> int:
        try:
            return int(self._query("select coalesce(max(id), 0) from turns")[0][0])
        except Exception:
            return 0

    def since(self, after: int) -> list:
        try:
            return self._query(
                "select id, role, content, tools, ms, created_at from turns where id > %s order by id limit 25",
                (after,),
            )
        except Exception:
            return []


db = Turns()
_lat_cache: dict = {"key": None, "rows": []}
_last_user = {"text": ""}


def latency_rows() -> list[dict]:
    try:
        st = LATENCY_LOG.stat()
    except OSError:
        return []
    key = (st.st_mtime_ns, st.st_size)
    if _lat_cache["key"] != key:
        try:
            lines = LATENCY_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return _lat_cache["rows"]
        head = lines[0].split("\t") if lines else []
        _lat_cache["rows"] = [dict(zip(head, ln.split("\t"))) for ln in lines[1:] if ln.strip()]
        _lat_cache["key"] = key
    return _lat_cache["rows"]


def _num(v) -> int | None:
    try:
        return sum(int(float(p)) for p in str(v).split("+") if p.strip())
    except (TypeError, ValueError):
        return None


def timing_of(r: dict) -> dict:
    return {k: _num(r.get(k)) for k in ("stt_ms", "gemini_ms", "tool_ms", "tts_ms", "first_audio_ms")}


def timing_for(heard: str, rows: list[dict]) -> dict | None:
    said = (heard or "")[:80].replace("\t", " ")
    for r in reversed(rows[-200:]):
        if r.get("kind") != "local" and r.get("said", "") == said:
            return timing_of(r)
    return None


_server_state = {
    "online": False,
    "model": "Neural Core",
    "voice": "edge",
    "facts": 0,
    "phase": "idle",
    "phase_text": 'System Idle // Listening for "Hey Nexa"',
    "ears": None,
    "last_check": 0.0,
}
_state_lock = threading.Lock()


def get_server_state() -> dict:
    with _state_lock:
        return dict(_server_state)


def _state_sync_worker() -> None:
    conn = None
    while True:
        try:
            if conn is None:
                conn = http.client.HTTPConnection("127.0.0.1", config.HTTP_PORT, timeout=1.5)
            conn.request("GET", "/health", headers={"Connection": "keep-alive"})
            resp = conn.getresponse()
            if resp.status == 200:
                raw = resp.read()
                data = json.loads(raw.decode("utf-8"))
                phase = data.get("phase", "idle")
                ears = data.get("ears")
                if isinstance(ears, dict) and ears.get("state") == "listening":
                    phase = "listening"
                    text = "Listening to you..."
                elif phase == "thinking" or (isinstance(ears, dict) and ears.get("deaf")):
                    phase = "thinking"
                    text = "Thinking..."
                elif phase == "speaking":
                    text = "Nexa Replying..."
                else:
                    phase = "idle"
                    text = 'System Idle // Listening for "Hey Nexa"'

                with _state_lock:
                    _server_state.update({
                        "online": True,
                        "model": "Neural Core",
                        "voice": data.get("voice", "edge"),
                        "facts": data.get("facts", 0),
                        "ears": ears,
                        "phase": phase,
                        "phase_text": text,
                        "last_check": time.time(),
                    })
            else:
                resp.read()
        except Exception:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
            with _state_lock:
                _server_state["online"] = False
                _server_state["phase"] = "idle"
                _server_state["phase_text"] = "Connecting to Nexa Core..."
        time.sleep(0.15)


def trigger_wake_core() -> bool:
    try:
        conn = http.client.HTTPConnection("127.0.0.1", config.HTTP_PORT, timeout=3.0)
        headers = {"Content-Type": "application/octet-stream"}
        if config.DEVICE_TOKEN:
            headers["X-Device-Token"] = config.DEVICE_TOKEN
        conn.request("POST", "/listen/wake", b"", headers=headers)
        r = conn.getresponse()
        r.read()
        conn.close()
        return r.status in (200, 202)
    except Exception:
        return False


def trigger_say_core(text: str) -> bool:
    try:
        conn = http.client.HTTPConnection("127.0.0.1", config.HTTP_PORT, timeout=30.0)
        payload = json.dumps({"text": text}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if config.DEVICE_TOKEN:
            headers["X-Device-Token"] = config.DEVICE_TOKEN
        conn.request("POST", "/say", payload, headers=headers)
        r = conn.getresponse()
        r.read()
        conn.close()
        return r.status == 200
    except Exception:
        return False


def server_health() -> dict:
    """Fast non-blocking read of server health."""
    return get_server_state()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args) -> None:
        pass

    def _json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            try:
                body = PAGE.read_bytes()
            except OSError:
                return self._json({"error": "orb.html is missing next to orb.py"}, 500)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if url.path == "/api/health":
            return self._json(server_health())

        if url.path == "/api/info":
            return self._json({"wake": bool(config.WAKE_ENABLED), "wake_model": config.WAKE_MODEL})

        if url.path == "/api/turns":
            q = parse_qs(url.query)

            def arg(name: str) -> int:
                try:
                    return int(q.get(name, ["-1"])[0])
                except ValueError:
                    return -1

            after, lat = arg("after"), arg("lat")
            lrows = latency_rows()
            st = get_server_state()
            out: dict = {
                "turns": [],
                "locals": [],
                "lat": len(lrows),
                "phase": st["phase"],
                "phase_text": st["phase_text"],
                "online": st["online"],
            }
            if 0 <= lat <= len(lrows):
                out["locals"] = [
                    {"said": r.get("said", ""), "timing": timing_of(r)}
                    for r in lrows[lat:]
                    if r.get("kind") == "local"
                ]

            try:
                if after < 0:
                    out.update(db=True, last=db.latest_id())
                    return self._json(out)
                rows = db.since(after)
            except Exception as exc:
                out.update(db=False, error=str(exc)[:200], last=after)
                return self._json(out)

            for r in rows:
                t = {
                    "id": r[0],
                    "role": r[1],
                    "text": r[2],
                    "tools": [x.get("tool") for x in (r[3] or []) if isinstance(x, dict) and x.get("tool")],
                    "ms": r[4],
                    "at": r[5].isoformat() if hasattr(r[5], "isoformat") else str(r[5]),
                }
                if t["role"] == "user":
                    _last_user["text"] = t["text"]
                else:
                    t["heard"] = _last_user["text"]
                    t["timing"] = timing_for(_last_user["text"], lrows)
                out["turns"].append(t)

            out.update(db=True, last=out["turns"][-1]["id"] if out["turns"] else after)
            return self._json(out)

        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""

        # Trigger listen / wake on the server
        if url.path == "/api/wake":
            ok = trigger_wake_core()
            return self._json({"ok": ok})

        # Dispatch typed prompt to Nexa
        if url.path == "/api/say":
            try:
                data = json.loads(body.decode("utf-8")) if body else {}
                text = str(data.get("text", "")).strip()
                if not text:
                    return self._json({"error": "empty text"}, 400)
                ok = trigger_say_core(text)
                return self._json({"ok": ok})
            except Exception as exc:
                return self._json({"ok": False, "error": str(exc)}, 500)

        self._json({"error": "not found"}, 404)


def find_browser() -> str | None:
    candidates = [
        os.path.expandvars(p)
        for p in (
            r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
            r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe",
            r"%LocalAppData%\Google\Chrome\Application\chrome.exe",
            r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe",
            r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe",
        )
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return shutil.which("chrome") or shutil.which("msedge")


def open_window(url: str, full: bool) -> None:
    exe = find_browser()
    if not exe:
        webbrowser.open(url)
        return
    profile = os.path.join(tempfile.gettempdir(), "nexa_orb_profile")
    args = [
        exe,
        f"--app={url}",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--no-default-browser-check",
        "--autoplay-policy=no-user-gesture-required",
        "--window-size=1280,780",
    ]
    if full:
        args.append("--start-fullscreen")
    subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    ap = argparse.ArgumentParser(description="NEXA Orb: Sci-Fi Visual Dashboard & Terminal Console")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--no-browser", action="store_true", help="don't launch browser window automatically")
    ap.add_argument("--full", action="store_true", help="start fullscreen")
    args = ap.parse_args()

    threading.Thread(target=_state_sync_worker, daemon=True, name="nexa-state-watcher").start()

    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        raise SystemExit(f"[orb] can't use port {args.port}: {exc}\n      Is the orb already running? Try --port 8778")

    url = f"http://127.0.0.1:{args.port}/"
    print(f"\n[nexa-orb] Interactive Cyber Dashboard running at: {url}")
    print(f"[nexa-orb] Synchronized with PostgreSQL (port 5433) and Nexa Core (port {config.HTTP_PORT})")
    print("[nexa-orb] Press Ctrl+C to close.\n")

    if not args.no_browser:
        threading.Timer(0.6, open_window, (url, args.full)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[nexa-orb] shutting down")


if __name__ == "__main__":
    main()
