"""Gemini embeddings over REST (rules: docs/cortex.md, "Embeddings").

gemini-embedding-001, outputDimensionality 768, then L2-normalised (shortened vectors are not
unit length). Documents: RETRIEVAL_DOCUMENT with title=filename, batched <= 100 per
:batchEmbedContents call. Questions: RETRIEVAL_QUERY, one :embedContent call.
"""
import math
import re

import httpx

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DIMS = 768
MAX_BATCH = 100


class EmbedError(RuntimeError):
    """Gemini refused or failed; `transient` errors are worth retrying later."""

    def __init__(self, message: str, status: int | None = None, transient: bool = False,
                 retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.transient = transient
        self.retry_after = retry_after


class RateLimited(EmbedError):
    """HTTP 429: free-tier quota; back off and call again. `daily`: the per-day quota is used up
    (waiting seconds won't help)."""

    def __init__(self, message: str, status: int | None = 429, transient: bool = True,
                 retry_after: float | None = None, daily: bool = False):
        super().__init__(message, status=status, transient=transient, retry_after=retry_after)
        self.daily = daily


DAILY_RETRY_THRESHOLD_S = 60.0


def _is_daily(data: dict | None, retry_after: float | None) -> bool:
    if retry_after is not None and retry_after > DAILY_RETRY_THRESHOLD_S:
        return True
    for detail in ((data or {}).get("error") or {}).get("details") or []:
        if not isinstance(detail, dict):
            continue
        for v in detail.get("violations") or []:
            if isinstance(v, dict) and any("perday" in str(v.get(k, "")).lower()
                                           for k in ("quotaId", "quotaMetric")):
                return True
    return False


def l2_normalise(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    if not norm or not math.isfinite(norm):
        raise EmbedError("Gemini returned a zero or invalid embedding")
    return [v / norm for v in values]


def _retry_after(resp: httpx.Response, data: dict | None) -> float | None:
    header = resp.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    for detail in ((data or {}).get("error") or {}).get("details") or []:
        delay = detail.get("retryDelay") if isinstance(detail, dict) else None
        if isinstance(delay, str):
            m = re.fullmatch(r"\s*([\d.]+)s\s*", delay)
            if m:
                return float(m.group(1))
    return None


class Gemini:
    def __init__(self, api_key: str, model: str = "gemini-embedding-001", dims: int = DIMS,
                 client: httpx.Client | None = None):
        self.api_key = api_key
        self.model = model.removeprefix("models/")
        self.dims = dims
        # One client per instance: keeps the TLS connection to Google alive between calls.
        self.client = client or httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))

    # --- requests -------------------------------------------------------------------------
    def _request(self, text: str, task_type: str, title: str | None = None) -> dict:
        req = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
            "outputDimensionality": self.dims,
        }
        if title:
            req["title"] = title
        return req

    def _post(self, method: str, body: dict, timeout: float | None = None) -> dict:
        url = f"{API_BASE}/models/{self.model}:{method}"
        kwargs = {}
        if timeout is not None:  # cap this call so the whole function stays inside its deadline
            t = max(1.0, min(60.0, timeout))
            kwargs["timeout"] = httpx.Timeout(t, connect=min(10.0, t))
        try:
            resp = self.client.post(url, json=body, headers={"x-goog-api-key": self.api_key}, **kwargs)
        except httpx.TimeoutException as exc:
            raise EmbedError(f"Gemini timed out ({exc.__class__.__name__})", transient=True) from exc
        except httpx.HTTPError as exc:
            raise EmbedError(f"Gemini unreachable: {exc}", transient=True) from exc
        try:
            data = resp.json()
        except ValueError:
            data = None
        if resp.status_code == 429:
            retry_after = _retry_after(resp, data)
            daily = _is_daily(data, retry_after)
            msg = ((data or {}).get("error") or {}).get("message") or ""
            raise RateLimited(("Gemini daily quota used up" if daily else "Gemini rate limit (429)")
                              + (f": {msg[:200]}" if msg else ""), retry_after=retry_after, daily=daily)
        if resp.status_code >= 400:
            msg = ((data or {}).get("error") or {}).get("message") or resp.text[:300]
            raise EmbedError(f"Gemini HTTP {resp.status_code}: {msg}", status=resp.status_code,
                             transient=resp.status_code in (500, 502, 503, 504),
                             retry_after=_retry_after(resp, data))
        if not isinstance(data, dict):
            raise EmbedError("Gemini returned a non-JSON response")
        return data

    def _vector(self, emb: dict) -> list[float]:
        values = (emb or {}).get("values")
        if not isinstance(values, list) or len(values) != self.dims:
            got = len(values) if isinstance(values, list) else "no"
            raise EmbedError(f"Gemini returned {got} dimensions, expected {self.dims}")
        return l2_normalise([float(v) for v in values])

    # --- public ---------------------------------------------------------------------------
    def embed_documents(self, texts: list[str], title: str, timeout: float | None = None) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), MAX_BATCH):
            batch = texts[i:i + MAX_BATCH]
            data = self._post("batchEmbedContents", {
                "requests": [self._request(t, "RETRIEVAL_DOCUMENT", title) for t in batch]}, timeout=timeout)
            embs = data.get("embeddings") or []
            if len(embs) != len(batch):
                raise EmbedError(f"Gemini returned {len(embs)} embeddings for {len(batch)} texts")
            out.extend(self._vector(e) for e in embs)
        return out

    def embed_query(self, text: str) -> list[float]:
        data = self._post("embedContent", self._request(text, "RETRIEVAL_QUERY"))
        return self._vector(data.get("embedding"))


def to_pgvector(values: list[float]) -> str:
    """Text form accepted by `::extensions.vector` (no pgvector Python package needed)."""
    return "[" + ",".join(format(v, ".8g") for v in values) + "]"
