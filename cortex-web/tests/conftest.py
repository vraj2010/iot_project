"""Shared fixtures: a real local Postgres + pgvector database, fake Gemini, fake Storage.

Database: CORTEX_TEST_DATABASE_URL (default: local socket /tmp, port 55432, user cortex,
database cortex_web_test). The schema is rebuilt from sql/schema.sql once per session.
DB tests are skipped when that database cannot be reached.
"""
import hashlib
import math
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.embed import RateLimited, l2_normalise  # noqa: E402
from core.storage import NotFound  # noqa: E402

TEST_DSN = os.environ.get(
    "CORTEX_TEST_DATABASE_URL",
    "host=/tmp port=55432 user=cortex dbname=cortex_web_test sslmode=disable",
)
SECRET = "test-session-secret-0123456789abcdef"
PASSWORD = "correct horse battery staple"


# --- fakes --------------------------------------------------------------------------------
class FakeStorage:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.signed: list[str] = []

    def create_upload_url(self, path):
        self.signed.append(path)
        return {"url": f"https://fake.supabase.co/storage/v1/object/upload/sign/cortex-docs/{path}?token=tok",
                "token": "tok"}

    def put(self, path, data: bytes):  # what the browser's PUT would do
        self.objects[path] = data

    def download(self, path, timeout=None):
        self.downloads = getattr(self, "downloads", 0) + 1
        if path not in self.objects:
            raise NotFound("download: object not found", status=404)
        return self.objects[path]

    def signed_download_url(self, path, expires_in=300):
        return f"https://fake.supabase.co/storage/v1/object/sign/cortex-docs/{path}?token=dl"

    def delete(self, path):
        self.deleted.append(path)
        self.objects.pop(path, None)


_WORD = re.compile(r"[a-z0-9]+")


def fake_vector(text: str, dims: int = 768) -> list[float]:
    """Deterministic bag-of-words vector: texts sharing words are similar."""
    v = [0.0] * dims
    for w in _WORD.findall(text.lower()):
        h = int.from_bytes(hashlib.sha256(w.encode()).digest()[:4], "big")
        v[h % dims] += 1.0
    if not any(v):
        v[0] = 1.0
    return l2_normalise(v)


class FakeEmbedder:
    """Stands in for core.embed.Gemini. fail_on: {call_number: exception} (1-based)."""

    def __init__(self, fail_on=None, on_call=None):
        self.calls: list[dict] = []
        self.fail_on = fail_on or {}
        self.on_call = on_call

    def embed_documents(self, texts, title, timeout=None):
        assert len(texts) <= 100, "batches must be <= 100"
        self.calls.append({"kind": "documents", "n": len(texts), "title": title,
                           "chars": sum(len(t) for t in texts), "timeout": timeout})
        if self.on_call:
            self.on_call()
        exc = self.fail_on.get(len(self.calls))
        if exc:
            raise exc
        return [fake_vector(t) for t in texts]

    def embed_query(self, text):
        self.calls.append({"kind": "query", "text": text})
        return fake_vector(text)


# --- sample documents -----------------------------------------------------------------------
TOPICS = ["photosynthesis", "mitochondria", "enzymes", "osmosis", "genetics", "ecology",
          "respiration", "proteins", "hormones", "neurons"]


def page_text(page_no: int, sentences: int = 22) -> str:
    topic = TOPICS[page_no % len(TOPICS)]
    out = [f"Chapter {page_no}: {topic.title()}."]
    for i in range(sentences):
        out.append(f"Sentence {i} on page {page_no} explains how {topic} works in living cells "
                   f"and why biology students should remember detail number {i * 7 + page_no}.")
    return " ".join(out)


def make_pdf(pages: int, sentences: int = 22) -> bytes:
    import pymupdf

    doc = pymupdf.open()
    for p in range(1, pages + 1):
        page = doc.new_page()
        rect = pymupdf.Rect(36, 36, page.rect.width - 36, page.rect.height - 36)
        page.insert_textbox(rect, page_text(p, sentences), fontsize=6)
    data = doc.tobytes()
    doc.close()
    return data


# --- fixtures ---------------------------------------------------------------------------------
def assert_safe_test_dsn(dsn: str) -> None:
    """The fixture drops schema `cortex`: only ever on a local database whose name says test."""
    from psycopg.conninfo import conninfo_to_dict

    info = conninfo_to_dict(dsn)
    host = (info.get("host") or "").strip()
    local = host == "" or host.startswith("/") or all(
        h.strip() in ("localhost", "127.0.0.1", "::1") for h in host.split(","))
    if not local or "test" not in (info.get("dbname") or ""):
        raise RuntimeError(f"refusing to rebuild schema cortex in {info.get('dbname')!r} on {host!r}: "
                           "tests need a LOCAL database whose name contains 'test'")


@pytest.fixture(scope="session")
def database():
    psycopg = pytest.importorskip("psycopg")
    assert_safe_test_dsn(TEST_DSN)
    try:
        conn = psycopg.connect(TEST_DSN, autocommit=True)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"test database not reachable: {exc}")
    conn.execute("drop schema if exists cortex cascade")
    conn.execute("create schema if not exists extensions")
    conn.execute((ROOT / "sql" / "schema.sql").read_text())
    conn.close()
    return TEST_DSN


@pytest.fixture
def clean_db(database):
    import psycopg

    from core import db, ingest

    with psycopg.connect(database, autocommit=True) as conn:
        conn.execute("truncate cortex.answers, cortex.chunks, cortex.documents, cortex.login_failures"
                     " restart identity cascade")
    db.set_dsn(database)
    ingest.clear_cache()
    yield database
    db.reset()


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("SESSION_SECRET", SECRET)
    monkeypatch.setenv("UPLOAD_PASSWORD", PASSWORD)
    monkeypatch.setenv("MAX_UPLOAD_MB", "50")
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    for name in ("SUPABASE_URL", "SUPABASE_SECRET_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fakes():
    return FakeStorage(), FakeEmbedder()


@pytest.fixture
def client(env, clean_db, fakes, monkeypatch):
    from fastapi.testclient import TestClient

    from api import index
    from api.index import app, set_services

    monkeypatch.setattr(index, "LOGIN_FAIL_DELAY_S", 0.0)

    storage, embedder = fakes
    set_services(storage=storage, embedder=embedder)
    with TestClient(app, base_url="https://testserver") as c:
        yield c
    set_services()


@pytest.fixture
def authed(client):
    r = client.post("/api/login", json={"password": PASSWORD})
    assert r.status_code == 200, r.text
    return client


def upload(c, storage, filename: str, data: bytes, subject=None) -> int:
    """What the page does: ask for a signed URL, then PUT the bytes (here: into the fake)."""
    r = c.post("/api/upload-url", json={"filename": filename, "size": len(data), "subject": subject})
    assert r.status_code == 200, r.text
    body = r.json()
    storage.put(body["path"], data)
    return body["document_id"]


def assert_unit(v):
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-6)


__all__ = ["FakeStorage", "FakeEmbedder", "RateLimited", "make_pdf", "page_text", "upload", "fake_vector"]
