"""Resumable ingest (docs/cortex.md, "Ingest is resumable").

One call: take the document's lease -> download -> parse -> chunk (deterministic) -> skip
chunk_index values already stored -> embed in batches -> insert (searchable = false) -> update
progress, until the time budget is used up. When every chunk is stored, ONE transaction makes them
all searchable and marks the document 'ready', so a half-ingested document is never searched.

Returns {"status": "processing", done, total} to be called again, or "ready" / "failed".
- Another call holds the lease -> {"status": "processing", "busy": true, "retry_after": 10}, no work.
- Gemini 429 -> back off briefly, halve the batch for the rest of the call (min 10), go on; if it
  keeps happening, return "processing" with retry_after. Daily quota used up -> "processing" with
  paused="quota" (the page stops; try later).
- Gemini/Storage 5xx, timeouts, a dropped DB connection -> "processing" with retry_after.
- Anything else (bad file, no text, Gemini refusal) -> "failed" with the reason.
"""
import hashlib
import logging
import time
from collections import OrderedDict
from typing import Callable

import psycopg

from . import db
from .chunk import CHUNKER_VERSION, Chunk, chunk_pages
from .embed import MAX_BATCH, EmbedError, RateLimited, to_pgvector
from .parse import UnsupportedFileType, parse
from .storage import NotFound, StorageError

log = logging.getLogger("cortex.ingest")

# Parsed chunks of recently ingested files, so the next call on a warm instance skips the
# download and parse. Keyed by storage_path (unique per upload, the object never changes).
_CACHE: "OrderedDict[str, tuple[list[Chunk], int | None, str]]" = OrderedDict()
_CACHE_MAX = 2

MAX_BATCH_CHARS = 60_000      # per Gemini call: non-Latin text has far more tokens per char
MIN_BATCH = 10
MAX_BACKOFF_S = 5.0           # longest sleep inside a call after a 429
MAX_429_PER_CALL = 3          # consecutive 429s before handing back to the page
TRANSIENT_RETRY_S = 10.0
LEASE_S = 290                 # a crashed call frees the document after this
HARD_DEADLINE_S = 285.0       # no network call is allowed to run past this (Vercel stops at 300)
QUOTA_MESSAGE = "Paused: Gemini quota used up, try later"


class IngestFailure(Exception):
    """A permanent problem with this document (bad file, no text, Gemini refusal...)."""


class Transient(Exception):
    """Worth retrying later: the page calls again after retry_after seconds."""

    def __init__(self, message: str, retry_after: float = TRANSIENT_RETRY_S):
        super().__init__(message)
        self.retry_after = retry_after


def clear_cache() -> None:
    _CACHE.clear()


def clear_cache_key(key: str) -> None:
    _CACHE.pop(key, None)


def _cached_parse(doc: dict, storage, timeout: float) -> tuple[list[Chunk], int | None, str]:
    key = doc["storage_path"]
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    try:
        data = storage.download(key, timeout=timeout)
    except NotFound as exc:
        raise IngestFailure("The file is not in storage (did the upload finish?)") from exc
    except StorageError as exc:
        if getattr(exc, "transient", False):
            raise Transient(str(exc)) from exc
        raise IngestFailure(str(exc)) from exc
    sha = hashlib.sha256(data).hexdigest()
    try:
        parsed = parse(doc["filename"], data)
    except UnsupportedFileType as exc:
        raise IngestFailure(str(exc)) from exc
    except Exception as exc:  # corrupt / encrypted files
        raise IngestFailure(f"Could not read the file: {type(exc).__name__}: {exc}") from exc
    chunks = chunk_pages(parsed.pages)
    _CACHE[key] = (chunks, parsed.page_count, sha)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return chunks, parsed.page_count, sha


# --- SQL ---------------------------------------------------------------------------------------
def _take_lease(document_id: int) -> bool:
    row = db.fetch_one(
        "update cortex.documents set ingest_lease_until = now() + make_interval(secs => %s)"
        " where id = %s and (ingest_lease_until is null or ingest_lease_until < now()) returning id",
        (LEASE_S, document_id))
    return row is not None


def _release_lease(document_id: int) -> None:
    try:
        db.execute("update cortex.documents set ingest_lease_until = null where id = %s", (document_id,))
    except Exception:
        log.warning("could not release the ingest lease of document %s (it expires by itself)", document_id)


def _start(document_id: int, chunker_version: int | None) -> None:
    """Reset to processing: nothing of this document is searchable until it is ready again; chunks
    made by another chunker version are thrown away."""
    def go(conn):
        with conn.transaction():
            conn.execute("update cortex.chunks set searchable = false where document_id = %s and searchable",
                         (document_id,))
            if chunker_version != CHUNKER_VERSION:
                conn.execute("delete from cortex.chunks where document_id = %s", (document_id,))
            conn.execute("update cortex.documents set status = 'processing', error = null, chunker_version = %s,"
                         " chunk_count = (select count(*) from cortex.chunks where document_id = %s),"
                         " ready_at = null where id = %s", (CHUNKER_VERSION, document_id, document_id))
    db.run(go)


def _count(document_id: int) -> int:
    row = db.fetch_one("select count(*) as n from cortex.chunks where document_id = %s", (document_id,))
    return int(row["n"])


def _insert(document_id: int, batch: list[Chunk], vectors: list[list[float]]) -> int:
    """One statement per batch (searchable stays false); returns the document's chunk count."""
    def go(conn):
        with conn.transaction():
            conn.execute(
                "insert into cortex.chunks (document_id, chunk_index, page, content, embedding)"
                " select %s, t.i, t.p, t.c, t.e::extensions.vector"
                " from unnest(%s::int[], %s::int[], %s::text[], %s::text[]) as t(i, p, c, e)"
                " on conflict (document_id, chunk_index) do nothing",
                (document_id, [c.chunk_index for c in batch], [c.page for c in batch],
                 [c.content for c in batch], [to_pgvector(v) for v in vectors]))
            n = conn.execute("select count(*) as n from cortex.chunks where document_id = %s",
                             (document_id,)).fetchone()["n"]
            conn.execute("update cortex.documents set chunk_count = %s where id = %s", (n, document_id))
            return int(n)
    return db.run(go)


def _finish(document_id: int, total: int) -> int | None:
    """All chunks searchable + status 'ready' in ONE transaction. None if chunks are missing."""
    def go(conn):
        with conn.transaction():
            n = conn.execute("select count(*) as n from cortex.chunks where document_id = %s",
                             (document_id,)).fetchone()["n"]
            if n < total:
                return None
            conn.execute("update cortex.chunks set searchable = true where document_id = %s", (document_id,))
            conn.execute("update cortex.documents set status = 'ready', error = null, chunk_count = %s,"
                         " ready_at = now() where id = %s", (n, document_id))
            return int(n)
    return db.run(go)


def _note(document_id: int, message: str) -> None:
    """A message shown with a document that is still 'processing' (e.g. paused for quota)."""
    try:
        db.execute("update cortex.documents set error = %s where id = %s and status = 'processing'",
                   (message[:1000], document_id))
    except Exception:
        log.exception("could not note document %s", document_id)


def _mark_failed(document_id: int, reason: str) -> None:
    try:
        db.execute("update cortex.documents set status = 'failed', error = %s where id = %s",
                   (reason[:1000], document_id))
    except Exception:
        log.exception("could not mark document %s failed", document_id)


def _safe_count(document_id: int) -> int:
    try:
        return _count(document_id)
    except Exception:
        return 0


def _next_batch(todo: list[Chunk], start: int, max_n: int) -> list[Chunk]:
    """Up to max_n chunks (and <= MAX_BATCH_CHARS characters, at least one chunk)."""
    batch: list[Chunk] = []
    chars = 0
    for c in todo[start:start + max_n]:
        if batch and chars + len(c.content) > MAX_BATCH_CHARS:
            break
        batch.append(c)
        chars += len(c.content)
    return batch


# --- the call ----------------------------------------------------------------------------------
def ingest(document_id: int, storage, embedder, budget_s: float = 240.0,
           clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> dict:
    t0 = clock()
    doc = db.fetch_one("select id, filename, storage_path, status, chunk_total, chunker_version"
                       " from cortex.documents where id = %s", (document_id,))
    if doc is None:
        return {"document_id": document_id, "status": "missing", "error": "No such document"}

    def result(status: str, done: int, total: int | None, **extra) -> dict:
        out = {"document_id": document_id, "status": status, "done": done, "total": total,
               "elapsed_ms": int((clock() - t0) * 1000)}
        out.update(extra)
        return out

    def time_left() -> float:
        return HARD_DEADLINE_S - (clock() - t0)

    if doc["status"] == "ready":
        n = _count(document_id)
        return result("ready", n, doc["chunk_total"] or n, duplicate_of=[])

    if not _take_lease(document_id):
        return result("processing", _safe_count(document_id), doc["chunk_total"], busy=True,
                      retry_after=TRANSIENT_RETRY_S, note="Another call is processing this document")
    total = doc["chunk_total"]
    done = 0
    dupes: list[dict] = []
    try:
        _start(document_id, doc["chunker_version"])
        chunks, page_count, sha = _cached_parse(doc, storage, timeout=time_left())
        if not chunks:
            raise IngestFailure("No extractable text found (scanned or empty file?)")
        total = len(chunks)
        db.execute("update cortex.documents set sha256 = %s, pages = %s, chunk_total = %s where id = %s",
                   (sha, page_count, total, document_id))
        dupes = [{"id": d["id"], "filename": d["filename"], "status": d["status"]} for d in db.fetch_all(
            "select id, filename, status from cortex.documents where sha256 = %s and id <> %s order by id",
            (sha, document_id))]

        have = {r["chunk_index"] for r in db.fetch_all(
            "select chunk_index from cortex.chunks where document_id = %s", (document_id,))}
        todo = [c for c in chunks if c.chunk_index not in have]
        done = total - len(todo)
        batch_n = MAX_BATCH
        last_batch_s = 0.0
        hits_429 = 0
        pos = 0
        while pos < len(todo):
            if clock() - t0 + last_batch_s > budget_s:
                return result("processing", done, total, duplicate_of=dupes)
            batch = _next_batch(todo, pos, batch_n)
            b0 = clock()
            try:
                vectors = embedder.embed_documents([c.content for c in batch], title=doc["filename"],
                                                   timeout=time_left() - 5)
            except RateLimited as exc:
                if getattr(exc, "daily", False):
                    _note(document_id, QUOTA_MESSAGE)
                    retry = max(exc.retry_after or 3600.0, 60.0)
                    return result("processing", done, total, duplicate_of=dupes, paused="quota",
                                  retry_after=round(retry, 1), note=QUOTA_MESSAGE)
                hits_429 += 1
                batch_n = max(MIN_BATCH, batch_n // 2)
                wait = exc.retry_after if exc.retry_after is not None else 2.0 * hits_429
                if (hits_429 >= MAX_429_PER_CALL or wait > MAX_BACKOFF_S
                        or clock() - t0 + wait + last_batch_s > budget_s):
                    log.warning("document %s: %s; handing back for %.0fs", document_id, exc, wait)
                    return result("processing", done, total, duplicate_of=dupes,
                                  retry_after=round(max(wait, 1.0), 1), note=str(exc))
                sleep(wait)
                continue
            except EmbedError as exc:
                if not exc.transient:
                    raise IngestFailure(str(exc)) from exc
                raise Transient(str(exc), retry_after=min(exc.retry_after or TRANSIENT_RETRY_S, 60.0)) from exc
            hits_429 = 0
            done = _insert(document_id, batch, vectors)
            pos += len(batch)
            last_batch_s = clock() - b0

        finished = _finish(document_id, total)
        if finished is None:  # should not happen while we hold the lease
            return result("processing", _count(document_id), total, duplicate_of=dupes)
        clear_cache_key(doc["storage_path"])
        return result("ready", finished, total, duplicate_of=dupes)
    except Transient as exc:
        log.warning("document %s: transient: %s", document_id, exc)
        return result("processing", _safe_count(document_id), total, duplicate_of=dupes,
                      retry_after=round(exc.retry_after, 1), note=str(exc))
    except psycopg.OperationalError as exc:
        log.warning("document %s: database unavailable: %s", document_id, exc)
        db.reset()
        return result("processing", done, total, duplicate_of=dupes, retry_after=TRANSIENT_RETRY_S,
                      note="Database connection dropped; retrying")
    except IngestFailure as exc:
        _mark_failed(document_id, str(exc))
        return result("failed", _safe_count(document_id), total, error=str(exc))
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        log.exception("ingest failed for document %s", document_id)
        _mark_failed(document_id, reason)
        return result("failed", _safe_count(document_id), total, error=reason)
    finally:
        _release_lease(document_id)
