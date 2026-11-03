"""Does Gemini answer a CHAT request from this laptop, and which kind?  Run it where server.py runs.

    python tools/diag_chat.py                       your model, 4 ways, then a few other Flash models
    python tools/diag_chat.py --models a,b          test exactly these models instead of the auto list
    python tools/diag_chat.py --timeout 30          wait longer before calling a request hung (default 20 s)

What it tells you
  * chat, reasoning low        exactly what Friday sends for a normal reply (OpenAI-compatible endpoint)
  * chat, no reasoning         the same without reasoning_effort: if THIS answers and the one above hangs,
                               reasoning_effort is what the model is choking on
  * chat, streamed             how Friday streams spoken replies; reports time to the first words
  * native generateContent     Google's own endpoint, not the compatibility one
  * other models               the same plain request on other Flash models your key can use, fastest first

Each request is a few tokens ("say ok"). The API key is never printed.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

BASE = config.GEMINI_BASE
KEY = config.GEMINI_API_KEY
SKIP = ("embedding", "tts", "image", "imagen", "veo", "live", "audio", "aqa", "robotics",
        "computer-use", "gemma", "learnlm", "deep-research", "customtools")


def _short(text: str, n: int = 110) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _error_text(r: requests.Response) -> str:
    try:
        data = r.json()
        data = data[0] if isinstance(data, list) and data else data
        return (data.get("error") or {}).get("message") or r.text
    except ValueError:
        return r.text


def _chat_body(model: str, reasoning: str | None, stream: bool) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": "Say ok."}], "max_tokens": 40}
    if reasoning:
        body["reasoning_effort"] = reasoning
    if stream:
        body["stream"] = True
    return body


def run_chat(model: str, reasoning: str | None, timeout: float) -> dict:
    t0 = time.time()
    try:
        r = requests.post(f"{BASE}/openai/chat/completions", json=_chat_body(model, reasoning, False),
                          headers={"Authorization": f"Bearer {KEY}"}, timeout=(5.0, timeout))
        secs = time.time() - t0
        if r.status_code != 200:
            return {"ok": False, "secs": secs, "note": f"HTTP {r.status_code}: {_short(_error_text(r))}"}
        text = (((r.json().get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
        return {"ok": True, "secs": secs, "note": f"HTTP 200, replied: {_short(text, 40)!r}"}
    except requests.Timeout:
        return {"ok": False, "secs": time.time() - t0, "note": f"HUNG: nothing in {timeout:g} s"}
    except requests.RequestException as exc:
        return {"ok": False, "secs": time.time() - t0, "note": f"connection error: {_short(exc)}"}


def run_stream(model: str, reasoning: str | None, timeout: float) -> dict:
    t0 = time.time()
    try:
        r = requests.post(f"{BASE}/openai/chat/completions", json=_chat_body(model, reasoning, True),
                          headers={"Authorization": f"Bearer {KEY}"}, timeout=(5.0, timeout), stream=True)
        if r.status_code != 200:
            return {"ok": False, "secs": time.time() - t0, "note": f"HTTP {r.status_code}: {_short(_error_text(r))}"}
        for line in r.iter_lines():
            if line.startswith(b"data:") and b"[DONE]" not in line and b'"content"' in line:
                r.close()
                return {"ok": True, "secs": time.time() - t0, "note": "HTTP 200, first words arrived"}
        return {"ok": False, "secs": time.time() - t0, "note": "stream ended with no words"}
    except requests.Timeout:
        return {"ok": False, "secs": time.time() - t0, "note": f"HUNG: nothing in {timeout:g} s"}
    except requests.RequestException as exc:
        return {"ok": False, "secs": time.time() - t0, "note": f"connection error: {_short(exc)}"}


def run_native(model: str, timeout: float) -> dict:
    t0 = time.time()
    body = {"contents": [{"role": "user", "parts": [{"text": "Say ok."}]}],
            "generationConfig": {"maxOutputTokens": 40}}
    try:
        r = requests.post(f"{BASE}/models/{model}:generateContent", json=body,
                          headers={"x-goog-api-key": KEY}, timeout=(5.0, timeout))
        secs = time.time() - t0
        if r.status_code != 200:
            return {"ok": False, "secs": secs, "note": f"HTTP {r.status_code}: {_short(_error_text(r))}"}
        return {"ok": True, "secs": secs, "note": "HTTP 200"}
    except requests.Timeout:
        return {"ok": False, "secs": time.time() - t0, "note": f"HUNG: nothing in {timeout:g} s"}
    except requests.RequestException as exc:
        return {"ok": False, "secs": time.time() - t0, "note": f"connection error: {_short(exc)}"}


def other_models() -> list[str]:
    """Flash models this key can call generateContent on."""
    try:
        r = requests.get(f"{BASE}/models", params={"pageSize": 200}, headers={"x-goog-api-key": KEY},
                         timeout=(5.0, 10.0))
        r.raise_for_status()
        names = []
        for m in r.json().get("models", []):
            name = str(m.get("name", "")).removeprefix("models/")
            if ("flash" in name and "generateContent" in (m.get("supportedGenerationMethods") or [])
                    and not any(word in name for word in SKIP)):
                names.append(name)
        return sorted(set(names))
    except (requests.RequestException, ValueError) as exc:
        print(f"(could not list models: {_short(exc)})")
        return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="comma-separated models to test instead of the automatic list")
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()
    if not KEY:
        sys.exit("GEMINI_API_KEY is not set in .env")

    mine = config.GEMINI_MODEL
    print(f"Friday's model: {mine}   reasoning: {config.GEMINI_REASONING or '(default)'}   "
          f"waiting up to {args.timeout:g} s per request\n")

    jobs: list[tuple[str, object]] = [
        (f"{mine}: chat, reasoning {config.GEMINI_REASONING or 'default'}",
         lambda: run_chat(mine, config.GEMINI_REASONING or None, args.timeout)),
        (f"{mine}: chat, no reasoning", lambda: run_chat(mine, None, args.timeout)),
        (f"{mine}: chat, streamed", lambda: run_stream(mine, config.GEMINI_REASONING or None, args.timeout)),
        (f"{mine}: native generateContent", lambda: run_native(mine, args.timeout)),
    ]
    extra = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else other_models()
    for m in extra:
        if m != mine:
            jobs.append((f"{m}: chat, no reasoning", lambda m=m: run_chat(m, None, args.timeout)))

    results: dict[int, dict] = {}

    def work(i: int, fn) -> None:
        results[i] = fn()

    threads = [threading.Thread(target=work, args=(i, fn), daemon=True) for i, (_, fn) in enumerate(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    width = max(len(label) for label, _ in jobs)
    for i, (label, _) in enumerate(jobs):
        res = results.get(i) or {"ok": False, "secs": 0.0, "note": "no result"}
        print(f"{'OK  ' if res['ok'] else 'FAIL'} {label:<{width}}  {res['secs']:5.1f} s  {res['note']}")
        if i == 3 and len(jobs) > 4:
            print(f"\nOther Flash models, same plain request:")


if __name__ == "__main__":
    main()
