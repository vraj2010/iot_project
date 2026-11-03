"""Every call to Google's Gemini API lives here.

  chat()    OpenAI-compatible chat endpoint, with function tools (same endpoint the Worker uses)
  chat_stream()  the same, streamed: the first words arrive while the rest is still being written
  search()  native generateContent with the built-in Google Search tool
  embed()   native embedContent, for memory similarity and Cortex question embeddings
  warm()    keep the connection to Google open and alive (see keepalive_start)
"""

from __future__ import annotations

import json
import math
import queue
import re
import threading
import time

import requests

import config
from httpsession import new_session as _new_session


class GeminiError(RuntimeError):
    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status
        self.http: int | None = None     # Google's own HTTP status, when it answered at all


_session = _new_session()       # keeps the TLS connection to Google open between calls
_session_lock = threading.Lock()
_last_used = time.time()


def _get_session() -> requests.Session:
    return _session


def _replace_session(new: requests.Session | None = None) -> None:
    """Use `new` (or a fresh session) from now on and close the old one, whose connection can't be trusted."""
    global _session
    with _session_lock:
        old, _session = _session, (new or _new_session())
    threading.Thread(target=old.close, daemon=True, name="gemini-close").start()


def _touch() -> None:
    global _last_used
    _last_used = time.time()


def _post(url: str, body: dict, headers: dict, timeout: float | None = None,
          session: requests.Session | None = None) -> dict:
    """`session`: another kept-alive session to send it on (Cortex has its own); default the shared one."""
    try:
        r = (session or _get_session()).post(url, json=body, headers=headers, timeout=timeout or config.GEMINI_TIMEOUT)
    except requests.Timeout:
        raise GeminiError("gemini timed out", 504) from None
    except requests.RequestException as exc:
        raise GeminiError(f"could not reach gemini: {exc}", 502) from None
    if session is None:
        _touch()
    if r.status_code == 429:
        raise GeminiError("gemini rate limit hit, try again shortly", 429)
    if r.status_code != 200:
        # Pass Gemini's own message through: it names a wrong model or key.
        raise GeminiError(f"gemini {r.status_code}: {r.text[:300]}", 502)
    try:
        return r.json()
    except ValueError:
        raise GeminiError("gemini sent a non-JSON reply", 502) from None


def _native_headers() -> dict:
    return {"x-goog-api-key": config.GEMINI_API_KEY}


# ---------------------------------------------------------------- chat

_reasoning_ok = True   # set False if this model refuses reasoning_effort, so we stop sending it


def _chat_request(messages: list[dict], tools: list[dict] | None, model: str | None, max_tokens: int,
                  json_mode: bool, reasoning: str | None) -> tuple[str, dict, dict]:
    """(url, headers, body) of one chat completion."""
    body: dict = {
        "model": model or config.GEMINI_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = tools
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if reasoning and _reasoning_ok:
        body["reasoning_effort"] = reasoning
    return (f"{config.GEMINI_BASE}/openai/chat/completions",
            {"Authorization": f"Bearer {config.GEMINI_API_KEY}"}, body)


def chat(messages: list[dict], tools: list[dict] | None = None, model: str | None = None,
         max_tokens: int = 1024, json_mode: bool = False, reasoning: str | None = None) -> dict:
    """One chat completion. Returns the assistant message dict (content and/or tool_calls).
    reasoning: low | medium | high, how long Gemini thinks first (less = faster)."""
    global _reasoning_ok
    url, headers, body = _chat_request(messages, tools, model, max_tokens, json_mode, reasoning)
    try:
        data = _post(url, body, headers)
    except GeminiError as exc:
        if "reasoning_effort" not in body or "400" not in str(exc):
            raise
        print(f"[gemini] this model won't take reasoning_effort={reasoning}; using its default ({exc})")
        _reasoning_ok = False
        body.pop("reasoning_effort")
        data = _post(url, body, headers)
    try:
        return data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise GeminiError("gemini returned no message", 502) from None


def chat_json(prompt: str, model: str | None = None) -> dict | list | None:
    """Ask for a JSON answer; returns the parsed value or None."""
    msg = chat([{"role": "user", "content": prompt}], model=model, max_tokens=800, json_mode=True)
    text = (msg.get("content") or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except ValueError:
        return None


# ---------------------------------------------------------------- streaming chat

_DONE = object()


def _sse_data(line: bytes):
    """One line of a server-sent-events stream -> a parsed JSON object, _DONE, or None."""
    line = line.strip()
    if not line.startswith(b"data:"):
        return None
    payload = line[5:].strip()
    if payload == b"[DONE]":
        return _DONE
    try:
        obj = json.loads(payload.decode("utf-8", errors="replace"))
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def _iter_sse(chunks):
    """The JSON events of a server-sent-events stream, however its bytes are cut into pieces."""
    buf = b""
    for chunk in chunks:
        if not chunk:
            continue
        buf += chunk
        while True:
            nl = buf.find(b"\n")
            if nl < 0:
                break
            line, buf = buf[:nl], buf[nl + 1:]
            event = _sse_data(line)
            if event is _DONE:
                return
            if event is not None:
                yield event
    if buf.strip():
        event = _sse_data(buf)
        if event is not None and event is not _DONE:
            yield event


def _json_complete(text: str) -> bool:
    try:
        json.loads(text or "")
        return True
    except ValueError:
        return False


def _merge(into: dict, extra: dict) -> None:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(into.get(key), dict):
            _merge(into[key], value)
        else:
            into[key] = value


class _Acc:
    """Rebuilds the assistant message from stream chunks, in the shape chat() returns it:
    the text, and the tool calls with everything Gemini attached to them (its thought
    signatures live in extra_content and must be sent back unchanged)."""

    def __init__(self) -> None:
        self.text: list[str] = []
        self.calls: dict[int, dict] = {}
        self.finish: str | None = None

    def add(self, chunk: dict) -> str:
        """Take in one chunk; returns the new text it carried ("" if none)."""
        if chunk.get("error") and not chunk.get("choices"):
            err = chunk["error"]
            detail = err.get("message") if isinstance(err, dict) else err
            raise GeminiError(f"gemini stream error: {detail}"[:300], 502)
        choices = chunk.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return ""
        choice = choices[0]
        delta = choice.get("delta") or choice.get("message") or {}
        if choice.get("finish_reason"):
            self.finish = choice["finish_reason"]
        for call in delta.get("tool_calls") or []:
            self._add_call(call)
        text = delta.get("content")
        if isinstance(text, str) and text:
            self.text.append(text)
            return text
        return ""

    def _add_call(self, tc: dict) -> None:
        fn = tc.get("function") or {}
        name, args = fn.get("name") or "", fn.get("arguments") or ""
        idx = tc.get("index")
        if not isinstance(idx, int):               # Gemini sometimes sends whole calls with no index
            idx = len(self.calls) if (name or tc.get("id")) else max(self.calls, default=0)
        call = self.calls.get(idx)
        if call is not None and name and call["function"]["name"] and (
                (tc.get("id") or "") != call["id"] or _json_complete(call["function"]["arguments"])):
            idx, call = max(self.calls) + 1, None  # numbered 0 again, but it's a different call
        if call is None:
            call = self.calls[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
        if tc.get("id"):
            call["id"] = tc["id"]
        if tc.get("type"):
            call["type"] = tc["type"]
        if name:
            call["function"]["name"] = name
        if args:
            call["function"]["arguments"] += args
        for key, value in tc.items():
            if key in ("index", "id", "type", "function"):
                continue
            if isinstance(value, dict) and isinstance(call.get(key), dict):
                _merge(call[key], value)
            else:
                call[key] = value

    def message(self) -> dict:
        msg: dict = {"role": "assistant", "content": "".join(self.text)}
        if self.calls:
            calls = [self.calls[i] for i in sorted(self.calls)]
            for call in calls:
                call["function"]["arguments"] = call["function"]["arguments"] or "{}"
            msg["tool_calls"] = calls
        return msg


def _whole_reply_events(r):
    """The server ignored stream=true and sent one ordinary JSON reply: pass it on as a single event."""
    try:
        data = r.json()
    except ValueError:
        raise GeminiError("gemini sent a non-JSON reply", 502) from None
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise GeminiError("gemini returned no message", 502) from None
    yield {"choices": [{"delta": message, "finish_reason": "stop"}]}


_HAND_BACK_WAIT_S = 0.25                                       # most a finished answer waits for its connection to be put back


def _hand_back(r, chunks) -> None:
    """Read what is left of an answer that has ended (after [DONE] that is only the end-of-body
    marker) and close it. Closing a response that was not read to its end closes the connection
    too, and the next request would pay for a new TCP and TLS handshake."""
    try:
        if chunks is not None:
            for _ in chunks:
                pass
    except Exception:
        pass
    finally:
        try:
            r.close()
        except Exception:
            pass


class _Attempt:
    """One streamed request, read on its own thread; what it receives goes to a shared queue."""

    def __init__(self, number: int, fresh: bool, out: "queue.Queue") -> None:
        self.number = number
        self.fresh = fresh                                     # on a brand new connection
        self.session = _new_session() if fresh else _get_session()
        self.out = out
        self.cancelled = threading.Event()
        self.response = None
        self.failed = False
        self.finished = False                                  # the whole answer has been read
        self.settled = threading.Event()                       # the thread is done with its connection
        self._lock = threading.Lock()

    def start(self, url: str, body: dict, headers: dict) -> None:
        threading.Thread(target=self._run, args=(url, body, headers), daemon=True,
                         name=f"gemini-stream-{self.number}").start()

    def cancel(self) -> None:
        with self._lock:
            self.cancelled.set()
            if self.finished:                                  # a complete answer: _run puts its connection back
                return
            response = self.response
        if response is not None:
            try:
                response.close()
            except Exception:
                pass

    def _finish(self) -> bool:
        """The answer ended normally. False if it was cancelled meanwhile (and its connection closed)."""
        with self._lock:
            if self.cancelled.is_set():
                return False
            self.finished = True
            return True

    def _put(self, kind: str, value=None) -> None:
        if not self.cancelled.is_set():
            self.out.put((self, kind, value))

    def _run(self, url: str, body: dict, headers: dict) -> None:
        try:
            r = self.session.post(url, json=body, headers=headers, timeout=(5.0, config.GEMINI_TIMEOUT),
                                  stream=True)
            self.response = r
            if self.cancelled.is_set():
                r.close()
                return
            if r.status_code == 429:
                raise GeminiError("gemini rate limit hit, try again shortly", 429)
            if r.status_code != 200:
                err = GeminiError(f"gemini {r.status_code}: {r.text[:300]}", 502)
                err.http = r.status_code
                raise err
            chunks = None
            if "text/event-stream" in r.headers.get("Content-Type", ""):
                chunks = r.iter_content(chunk_size=None)           # chunks as they arrive, not in blocks
                events = _iter_sse(chunks)
            else:
                events = _whole_reply_events(r)
            for event in events:
                if self.cancelled.is_set():
                    return
                self._put("chunk", event)
            if not self._finish():
                return
            self._put("done")
            _hand_back(r, chunks)                                  # after "done": the caller doesn't wait for this
        except requests.Timeout:
            self._put("error", GeminiError("gemini timed out", 504))
        except requests.RequestException as exc:
            self._put("error", GeminiError(f"could not reach gemini: {exc}", 502))
        except GeminiError as exc:
            self._put("error", exc)
        except Exception as exc:
            self._put("error", GeminiError(f"gemini stream failed: {exc}", 502))
        finally:
            self.settled.set()


def _retryable(exc: GeminiError) -> bool:
    """A failure of the connection or of Google's servers, as opposed to a bad request or a rate limit."""
    http = getattr(exc, "http", None)
    return exc.status != 429 and (http is None or http >= 500)


def _stream_once(url: str, headers: dict, body: dict, hedge_s: float | None, info: dict):
    hedge = config.GEMINI_HEDGE_S if hedge_s is None else hedge_s
    out: queue.Queue = queue.Queue()
    attempts: list[_Attempt] = []
    acc = _Acc()
    t0 = time.time()
    winner = None
    info.setdefault("hedged", False)

    def launch(fresh: bool) -> None:
        attempt = _Attempt(len(attempts), fresh, out)
        attempts.append(attempt)
        attempt.start(url, body, headers)

    launch(False)
    try:
        while True:
            if winner is None and len(attempts) == 1 and hedge > 0:
                wait = t0 + hedge - time.time()               # until it's time to ask a second time
            elif winner is None:
                wait = t0 + config.GEMINI_TIMEOUT + 5 - time.time()
            else:
                wait = config.GEMINI_TIMEOUT + 5               # between two pieces of an answer
            try:
                attempt, kind, value = out.get(timeout=max(0.0, wait))
            except queue.Empty:
                if winner is None and len(attempts) == 1 and hedge > 0 and time.time() - t0 < config.GEMINI_TIMEOUT:
                    print(f"[gemini] nothing after {hedge:g} s; asking again on a fresh connection")
                    info["hedged"] = True
                    launch(True)
                    continue
                raise GeminiError("gemini timed out", 504)
            if winner is not None and attempt is not winner:
                continue
            if kind == "error":
                if attempt is winner:
                    raise value
                attempt.failed = True
                if all(a.failed for a in attempts):
                    if len(attempts) < 2 and _retryable(value):
                        print(f"[gemini] {value}; trying again on a fresh connection")
                        info["hedged"] = True
                        launch(True)
                        continue
                    raise value
                continue
            if winner is None:                                 # the first thing heard from Gemini
                winner = attempt
                info["ttft_ms"] = int((time.time() - t0) * 1000)
                for other in attempts:
                    if other is not attempt:
                        other.cancel()
                if attempt.fresh:                              # the old connection was the problem
                    _replace_session(attempt.session)
            if kind == "chunk":
                text = acc.add(value)
                if text:
                    yield ("text", text)
                continue
            message = acc.message()                            # kind == "done"
            if not message["content"] and not message.get("tool_calls"):
                raise GeminiError("gemini returned no message", 502)
            info["ms"] = int((time.time() - t0) * 1000)
            info["finish"] = acc.finish
            yield ("end", message)
            return
    finally:
        for attempt in attempts:
            attempt.cancel()
        if winner is not None and winner.finished:             # it is putting its connection back: a moment, no more
            winner.settled.wait(_HAND_BACK_WAIT_S)
        _touch()


def chat_stream(messages: list[dict], tools: list[dict] | None = None, model: str | None = None,
                max_tokens: int = 1024, reasoning: str | None = None, hedge_s: float | None = None,
                info: dict | None = None):
    """Like chat(), but streamed. Yields ("text", words) as Gemini writes them and finally
    ("end", message) with the same message chat() would have returned.

    No answer after GEMINI_HEDGE_S seconds (or a failed connection): the request is sent
    again on a fresh connection and whichever answers first is used, which is what stops a
    connection the router forgot from costing 15 s. `info` gets ttft_ms, ms and hedged.
    Close the generator to stop early (the request is cancelled)."""
    global _reasoning_ok
    info = info if info is not None else {}
    for attempt in range(2):
        url, headers, body = _chat_request(messages, tools, model, max_tokens, False, reasoning)
        body["stream"] = True
        spoke = False
        try:
            for event in _stream_once(url, headers, body, hedge_s, info):
                spoke = True
                yield event
            return
        except GeminiError as exc:
            if spoke or attempt or "reasoning_effort" not in body or "400" not in str(exc):
                raise
            print(f"[gemini] this model won't take reasoning_effort={reasoning}; using its default ({exc})")
            _reasoning_ok = False


# ---------------------------------------------------------------- keeping the connection warm

def warm() -> bool:
    """One cheap request (the model's details: no tokens, no cost) so the connection to Google
    is open and known to be alive. A dead connection is replaced. True if Google answered."""
    try:
        r = _get_session().get(f"{config.GEMINI_BASE}/models/{config.GEMINI_MODEL}", headers=_native_headers(),
                               timeout=(5.0, 8.0))
        _touch()
        return r.status_code < 500
    except requests.RequestException:
        _replace_session()
        return False


_keepalive_started = False


def keepalive_start() -> None:
    """Background thread: connect to Google now, and again whenever the line has been quiet for
    GEMINI_KEEPALIVE_S, so the first question after a long silence isn't the one that finds
    out the connection died."""
    global _keepalive_started
    if _keepalive_started or config.GEMINI_KEEPALIVE_S <= 0:
        return
    _keepalive_started = True

    def loop() -> None:
        t0 = time.time()
        ok = warm()
        print(f"[gemini] connection {'ready' if ok else 'not ready yet'} ({int((time.time() - t0) * 1000)} ms)")
        while True:
            time.sleep(max(5.0, config.GEMINI_KEEPALIVE_S / 2))
            if time.time() - _last_used >= config.GEMINI_KEEPALIVE_S:
                warm()

    threading.Thread(target=loop, name="gemini-keepalive", daemon=True).start()


# ---------------------------------------------------------------- web search

def search(query: str) -> str:
    """Answer `query` from a live Google Search, in a couple of plain sentences."""
    prompt = (
        "Search the web and answer for a voice assistant in at most three short plain sentences, "
        "with numbers and dates spelled out the way a person would say them. "
        f"Say when the information is from if it matters. Question: {query}"
    )
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
    }
    url = f"{config.GEMINI_BASE}/models/{config.GEMINI_SEARCH_MODEL}:generateContent"
    level = config.GEMINI_REASONING
    if level and _reasoning_ok:                       # think briefly: a search answer needs little
        body["generationConfig"] = {"thinkingConfig": {"thinkingLevel": level}}
    try:
        data = _post(url, body, _native_headers(), timeout=config.GEMINI_TIMEOUT + 10)
    except GeminiError as exc:
        if "generationConfig" not in body or "400" not in str(exc) or "thinking" not in str(exc).lower():
            raise
        body.pop("generationConfig")
        data = _post(url, body, _native_headers(), timeout=config.GEMINI_TIMEOUT + 10)
    parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
    text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
    if not text:
        raise GeminiError("search returned nothing", 502)
    return text


# ---------------------------------------------------------------- embeddings

def embed(text: str, task_type: str | None = None, dims: int | None = None, model: str | None = None,
          session: requests.Session | None = None, timeout: float = 10) -> list[float]:
    """Unit-length embedding of `text` (EMBED_DIMS numbers, or `dims`).
    Memory calls it with no options, as it always has. Cortex asks for task_type="RETRIEVAL_QUERY",
    dims=768 and model="gemini-embedding-001" (what its documents were embedded with), on its own session."""
    body = {
        "content": {"parts": [{"text": text[:2000]}]},
        "outputDimensionality": dims or config.EMBED_DIMS,
    }
    if task_type:
        body["taskType"] = task_type
    data = _post(f"{config.GEMINI_BASE}/models/{model or config.GEMINI_EMBED_MODEL}:embedContent",
                 body, _native_headers(), timeout=timeout, session=session)
    values = (data.get("embedding") or {}).get("values")
    if not isinstance(values, list) or not values:
        raise GeminiError("no embedding in the reply", 502)
    # Shortened embeddings are not unit length; normalise so a dot product is cosine similarity.
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]
