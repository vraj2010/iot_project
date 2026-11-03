"""Which Gemini limit are you hitting, and when does it reset?  Run it where server.py runs.

    python tools/diag_quota.py                       your model plus a few others, one tiny request each
    python tools/diag_quota.py --models a,b          exactly these models

Google's 429 reply says which limit was hit (per minute, per day, tokens...) and how long to wait, but Friday
only prints "rate limit hit". This asks once per model and prints Google's own explanation. Each probe is a
single tiny request, so it costs next to nothing. The API key is never printed.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

DEFAULT_OTHERS = ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview",
                  "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]


def _details(r: requests.Response) -> tuple[str, list[dict], str]:
    """(message, quota violations, retry delay) out of a Google error reply."""
    try:
        data = r.json()
    except ValueError:
        return r.text[:200], [], ""
    data = data[0] if isinstance(data, list) and data else data
    err = (data or {}).get("error") or {}
    violations: list[dict] = []
    retry = ""
    for item in err.get("details") or []:
        kind = str(item.get("@type", ""))
        if kind.endswith("QuotaFailure"):
            violations += [v for v in item.get("violations") or [] if isinstance(v, dict)]
        elif kind.endswith("RetryInfo"):
            retry = str(item.get("retryDelay", ""))
    return str(err.get("message", ""))[:300], violations, retry


def probe(model: str, out: dict) -> None:
    t0 = time.time()
    body = {"model": model, "messages": [{"role": "user", "content": "Say ok."}], "max_tokens": 20}
    try:
        r = requests.post(f"{config.GEMINI_BASE}/openai/chat/completions", json=body,
                          headers={"Authorization": f"Bearer {config.GEMINI_API_KEY}"}, timeout=(5.0, 25.0))
    except requests.Timeout:
        out[model] = {"status": "HUNG", "secs": time.time() - t0, "message": "no answer in 25 s", "viol": [], "retry": ""}
        return
    except requests.RequestException as exc:
        out[model] = {"status": "ERR", "secs": time.time() - t0, "message": str(exc)[:200], "viol": [], "retry": ""}
        return
    message, viol, retry = ("", [], "") if r.status_code == 200 else _details(r)
    out[model] = {"status": r.status_code, "secs": time.time() - t0, "message": message, "viol": viol, "retry": retry}


def pacific_reset_local() -> str:
    """When Google's daily quotas next reset (midnight Pacific), in your own time zone."""
    try:
        from zoneinfo import ZoneInfo
        la = ZoneInfo("America/Los_Angeles")
        reset = (datetime.now(la) + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        mine = reset.astimezone(ZoneInfo(config.TIMEZONE))
        return mine.strftime("%a %H:%M") + f" {config.TIMEZONE}"
    except Exception:
        return "midnight Pacific time"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="comma-separated models to probe instead of the default list")
    args = ap.parse_args()
    if not config.GEMINI_API_KEY:
        sys.exit("GEMINI_API_KEY is not set in .env")

    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else \
        [config.GEMINI_MODEL] + [m for m in DEFAULT_OTHERS if m != config.GEMINI_MODEL]
    print(f"Friday's model: {config.GEMINI_MODEL}\n")

    results: dict[str, dict] = {}
    threads = [threading.Thread(target=probe, args=(m, results), daemon=True) for m in models]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    daily = False
    for m in models:
        res = results.get(m) or {"status": "?", "secs": 0.0, "message": "no result", "viol": [], "retry": ""}
        mark = "  <- Friday uses this" if m == config.GEMINI_MODEL else ""
        print(f"{'OK  ' if res['status'] == 200 else 'FAIL'} {m}: {res['status']} in {res['secs']:.1f} s{mark}")
        if res["status"] == 429:
            for v in res["viol"]:
                qid = v.get("quotaId", "") or v.get("quotaMetric", "")
                dims = v.get("quotaDimensions") or {}
                print(f"       limit: {qid}   model: {dims.get('model', '?')}   allowed: {v.get('quotaValue', '?')}")
                daily = daily or "perday" in qid.lower().replace("_", "").replace("-", "")
            if not res["viol"]:
                print(f"       {res['message']}")
            if res["retry"]:
                print(f"       Google says retry after: {res['retry']}")
        elif res["status"] != 200:
            print(f"       {res['message']}")

    print()
    if daily:
        print(f"A DAILY limit is used up on at least one model. Daily limits reset at midnight Pacific "
              f"(next: {pacific_reset_local()}). Waiting a few minutes will not help; use a model marked OK "
              f"above, turn billing on, or cut the requests per turn (AUTO_MEMORY=0).")


if __name__ == "__main__":
    main()
