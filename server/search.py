"""Web search for the web_search tool.

  1. Google Search grounding through Gemini: best answers, but Google only allows
     it on keys with billing enabled for Gemini 3.x models.
  2. DuckDuckGo (ddgs package): free, no key. Returns a handful of snippets and
     Gemini answers from them.

SEARCH_PROVIDER=auto asks both at once and prefers Google. If Google refuses (no
billing, quota), it isn't asked again for an hour.
"""

from __future__ import annotations

import re
import time

import config
import gemini

_google_off_until = 0.0
_NEWSY = re.compile(r"\b(news|latest|today|tonight|yesterday|this week|score|match|won|price|rate|weather)\b", re.I)


def _google(query: str) -> dict:
    return {"source": "google", "answer": gemini.search(query)}


def _duckduckgo(query: str) -> dict:
    try:
        from ddgs import DDGS
    except ImportError:
        raise RuntimeError("ddgs is not installed (pip install ddgs)") from None

    ddgs = DDGS(timeout=config.SEARCH_TIMEOUT)
    results = []
    if _NEWSY.search(query):                       # fresh news first for "today / latest / score..."
        try:
            for r in ddgs.news(query, region=config.SEARCH_REGION, timelimit="w", max_results=3):
                results.append({"title": r.get("title", ""), "snippet": (r.get("body") or "")[:300],
                                "date": r.get("date", "")})
        except Exception:
            pass
    for r in ddgs.text(query, region=config.SEARCH_REGION, max_results=5):
        results.append({"title": r.get("title", ""), "snippet": (r.get("body") or "")[:300]})
    if not results:
        raise RuntimeError("no results")
    return {
        "source": "duckduckgo",
        "results": results[:7],
        "how_to_answer": "Answer from these results in one or two short spoken sentences. "
                         "If they don't answer the question, say you couldn't find it.",
    }


def web_search(query: str) -> dict:
    """Google (if allowed) and DuckDuckGo are asked at the same time: Google's answer
    is used when it comes, DuckDuckGo's when Google refuses or is slow."""
    global _google_off_until
    query = (query or "").strip()[:300]
    if not query:
        return {"error": "empty query"}
    provider = config.SEARCH_PROVIDER
    use_google = provider in ("auto", "google") and time.time() >= _google_off_until
    use_ddg = provider in ("auto", "duckduckgo")
    if not use_google and not use_ddg:
        return {"error": "web search is turned off"}

    from concurrent.futures import ThreadPoolExecutor, wait

    pool = ThreadPoolExecutor(max_workers=2)
    google = pool.submit(_google, query) if use_google else None
    ddg = pool.submit(_duckduckgo, query) if use_ddg else None
    pool.shutdown(wait=False)
    errors = []

    if google is not None:
        # Google is better, but only worth waiting for a little after DuckDuckGo has answered.
        wait([f for f in (google, ddg) if f], return_when="ALL_COMPLETED" if ddg is None else "FIRST_COMPLETED",
             timeout=config.GEMINI_TIMEOUT + 10)
        if ddg is not None and ddg.done() and not google.done():
            wait([google], timeout=config.SEARCH_GOOGLE_GRACE)
        if google.done():
            try:
                return google.result()
            except gemini.GeminiError as exc:
                errors.append(f"google: {exc}")
                if exc.status != 504 and provider == "auto":   # not a one-off timeout: stop asking for a while
                    _google_off_until = time.time() + 3600
                    print(f"[search] Google grounding refused ({str(exc)[:120]}); using DuckDuckGo for the next hour")
            except Exception as exc:
                errors.append(f"google: {exc}")
    if ddg is not None:
        try:
            return ddg.result(timeout=config.SEARCH_TIMEOUT + 5)
        except Exception as exc:
            errors.append(f"duckduckgo: {exc}")
    return {"error": "; ".join(errors) or "no results"}


if __name__ == "__main__":
    # python search.py "usd to inr today"  -> tries each route and prints exactly what happens
    import json
    import sys

    q = " ".join(sys.argv[1:]) or "US dollar to Indian rupee exchange rate today"
    print(f"query: {q}\n")
    for name, fn in (("google", _google), ("duckduckgo", _duckduckgo)):
        t0 = time.time()
        try:
            out = fn(q)
            print(f"[{name}] OK in {time.time() - t0:.1f} s\n{json.dumps(out, indent=2, ensure_ascii=False)[:1200]}\n")
        except Exception as exc:
            print(f"[{name}] FAILED in {time.time() - t0:.1f} s: {type(exc).__name__}: {exc}\n")
