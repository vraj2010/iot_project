"""Upload URL validation, resumable ingest against the real local pgvector DB (fake Gemini and
Storage), search with citations, delete cascade, answers, file links, routing."""
import re

import psycopg
import pytest
from conftest import FakeEmbedder, RateLimited, make_pdf, upload

from core import ingest as ingest_mod
from core.embed import EmbedError


def q(dsn, sql, params=None):
    with psycopg.connect(dsn) as conn:
        return conn.execute(sql, params).fetchall()


# --- upload-url ------------------------------------------------------------------------------
def test_upload_url_validation(authed, clean_db, fakes):
    storage, _ = fakes
    bad = [({"filename": "virus.exe", "size": 10}, 400), ({"filename": "noext", "size": 10}, 400),
           ({"filename": "big.pdf", "size": 50 * 1024 * 1024 + 1}, 413), ({"filename": "empty.txt", "size": 0}, 400),
           ({"filename": "a.pdf"}, 422)]
    for body, code in bad:
        assert authed.post("/api/upload-url", json=body).status_code == code, body
    assert q(clean_db, "select count(*) from cortex.documents")[0][0] == 0

    r = authed.post("/api/upload-url", json={"filename": "C:\\Users\\me\\Biology Notes – ch1 (final).PDF",
                                             "size": 1234, "subject": "  Biology "})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["filename"] == "Biology Notes – ch1 (final).PDF"
    assert re.fullmatch(r"docs/[0-9a-f-]{36}/Biology_Notes_ch1_final\.pdf", b["path"]), b["path"]
    assert b["content_type"] == "application/pdf"
    assert b["upload_url"].startswith("https://fake.supabase.co/") and b["token"] == "tok"
    row = q(clean_db, "select filename, subject, storage_path, bytes, status from cortex.documents")[0]
    assert row == ("Biology Notes – ch1 (final).PDF", "Biology", b["path"], 1234, "uploaded")
    docx = authed.post("/api/upload-url", json={"filename": "a.docx", "size": 5}).json()
    assert docx["content_type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert authed.post("/api/upload-url", json={"filename": "a.txt", "size": 5}).json()["content_type"] == "text/plain"


def test_upload_url_storage_failure_removes_row(authed, clean_db, fakes):
    storage, _ = fakes

    def boom(path):
        raise RuntimeError("bucket missing")

    storage.create_upload_url = boom
    r = authed.post("/api/upload-url", json={"filename": "a.pdf", "size": 10})
    assert r.status_code == 502
    assert q(clean_db, "select count(*) from cortex.documents")[0][0] == 0


# --- ingest --------------------------------------------------------------------------------------
def _chunks(dsn, doc_id):
    return q(dsn, "select chunk_index, page, content, embedding::text from cortex.chunks"
                  " where document_id = %s order by chunk_index", (doc_id,))


def test_ingest_one_call_ready(authed, clean_db, fakes):
    storage, embedder = fakes
    pdf = make_pdf(40)
    doc = upload(authed, storage, "biology.pdf", pdf, subject="Biology")
    r = authed.post("/api/ingest", json={"document_id": doc}).json()
    assert r["status"] == "ready", r
    assert r["done"] == r["total"] > 100                    # more than one Gemini batch
    assert all(c["n"] <= 100 and c["title"] == "biology.pdf" for c in embedder.calls)
    row = q(clean_db, "select status, pages, chunk_total, chunk_count, sha256, ready_at is not null, error"
                      " from cortex.documents where id = %s", (doc,))[0]
    import hashlib
    assert row == ("ready", 40, r["total"], r["total"], hashlib.sha256(pdf).hexdigest(), True, None)
    chunks = _chunks(clean_db, doc)
    assert [c[0] for c in chunks] == list(range(r["total"]))
    assert {c[1] for c in chunks} == set(range(1, 41))
    # stored vectors are unit length
    norm = q(clean_db, "select max(abs(extensions.vector_norm(embedding) - 1)) from cortex.chunks")[0][0]
    assert norm < 1e-5
    # calling again on a ready document does nothing
    calls = len(embedder.calls)
    again = authed.post("/api/ingest", json={"document_id": doc}).json()
    assert again["status"] == "ready" and len(embedder.calls) == calls


def test_ingest_resumes_across_calls_identically(authed, clean_db, fakes, monkeypatch):
    storage, _ = fakes
    pdf = make_pdf(40)
    # reference: one uninterrupted call
    ref = upload(authed, storage, "biology.pdf", pdf)
    assert authed.post("/api/ingest", json={"document_id": ref}).json()["status"] == "ready"
    reference = [(i, p, c, e) for i, p, c, e in _chunks(clean_db, ref)]

    # now with a fake clock: every embedding batch "takes" 10 s, budget 15 s -> one batch per call
    from api import index

    now = [0.0]
    slow = FakeEmbedder(on_call=lambda: now.__setitem__(0, now[0] + 10))
    index.set_services(storage=storage, embedder=slow)
    real = ingest_mod.ingest
    monkeypatch.setattr(ingest_mod, "ingest", lambda doc_id, s, e, budget_s=240: real(
        doc_id, s, e, budget_s=15, clock=lambda: now[0], sleep=lambda s: None))

    doc = upload(authed, storage, "biology.pdf", pdf)
    statuses = []
    for _ in range(10):
        r = authed.post("/api/ingest", json={"document_id": doc}).json()
        statuses.append((r["status"], r["done"], r["total"]))
        if r["status"] == "ready":
            break
        ingest_mod.clear_cache()   # also exercise a cold instance: download + parse again
        mid = q(clean_db, "select status, chunk_count from cortex.documents where id = %s", (doc,))[0]
        assert mid == ("processing", r["done"])
    first = slow.calls[0]["n"]
    assert statuses[0][0] == "processing" and statuses[0][1] == first
    assert statuses[-1][0] == "ready" and len(statuses) >= 2
    assert sum(c["n"] for c in slow.calls) == statuses[-1][2]               # nothing embedded twice
    assert all(c["chars"] <= ingest_mod.MAX_BATCH_CHARS for c in slow.calls)
    assert _chunks(clean_db, doc) == reference
    dup = authed.post("/api/ingest", json={"document_id": doc}).json()
    assert dup["status"] == "ready"


def test_ingest_429_halves_batch_and_continues(authed, clean_db, fakes, monkeypatch):
    storage, _ = fakes
    from api import index

    limited = FakeEmbedder(fail_on={2: RateLimited("Gemini rate limit (429)", retry_after=1.0)})
    index.set_services(storage=storage, embedder=limited)
    slept = []
    real = ingest_mod.ingest
    monkeypatch.setattr(ingest_mod, "ingest", lambda doc_id, s, e, budget_s=240: real(
        doc_id, s, e, budget_s=budget_s, sleep=slept.append))
    doc = upload(authed, storage, "bio.pdf", make_pdf(40))
    r = authed.post("/api/ingest", json={"document_id": doc}).json()
    assert r["status"] == "ready" and r["done"] == r["total"]      # carried on in the same call
    assert slept == [1.0]
    sizes = [c["n"] for c in limited.calls]
    assert sizes[2] <= 50 and all(n <= 50 for n in sizes[2:])     # halved after the 429
    assert sum(sizes) - sizes[1] == r["total"]                     # the refused batch was redone


def test_ingest_repeated_or_long_429_hands_back(authed, clean_db, fakes, monkeypatch):
    storage, _ = fakes
    from api import index

    long_wait = FakeEmbedder(fail_on={2: RateLimited("Gemini rate limit (429)", retry_after=20.0)})
    index.set_services(storage=storage, embedder=long_wait)
    slept = []
    real = ingest_mod.ingest
    monkeypatch.setattr(ingest_mod, "ingest", lambda doc_id, s, e, budget_s=240: real(
        doc_id, s, e, budget_s=budget_s, sleep=slept.append))
    doc = upload(authed, storage, "bio.pdf", make_pdf(40))
    r = authed.post("/api/ingest", json={"document_id": doc}).json()
    assert r["status"] == "processing" and r["retry_after"] == 20.0 and "paused" not in r
    assert slept == [] and r["done"] == long_wait.calls[0]["n"]
    assert q(clean_db, "select status, error from cortex.documents where id = %s", (doc,))[0] == ("processing", None)

    always = FakeEmbedder(fail_on={i: RateLimited("Gemini rate limit (429)", retry_after=None) for i in range(1, 9)})
    index.set_services(storage=storage, embedder=always)
    r = authed.post("/api/ingest", json={"document_id": doc}).json()
    assert r["status"] == "processing" and r["retry_after"] >= 1
    assert len(always.calls) == ingest_mod.MAX_429_PER_CALL and slept == [2.0, 4.0]
    sizes = [c["n"] for c in always.calls]
    assert sizes[0] < 100 and always.calls[0]["chars"] <= ingest_mod.MAX_BATCH_CHARS  # capped by characters
    assert sizes[1:] == [50, 25]                                   # then halved after each 429 (min 10)

    index.set_services(storage=storage, embedder=FakeEmbedder())
    assert authed.post("/api/ingest", json={"document_id": doc}).json()["status"] == "ready"


def test_ingest_daily_quota_pauses(authed, clean_db, fakes):
    storage, _ = fakes
    from api import index

    quota = FakeEmbedder(fail_on={1: RateLimited("Gemini daily quota used up", retry_after=3000.0, daily=True)})
    index.set_services(storage=storage, embedder=quota)
    doc = upload(authed, storage, "bio.pdf", make_pdf(5))
    r = authed.post("/api/ingest", json={"document_id": doc}).json()
    assert r["status"] == "processing" and r["paused"] == "quota" and r["retry_after"] == 3000.0
    assert "quota used up" in r["note"]
    row = q(clean_db, "select status, error, ingest_lease_until from cortex.documents where id = %s", (doc,))[0]
    assert row == ("processing", ingest_mod.QUOTA_MESSAGE, None)  # shown in the table; lease released
    index.set_services(storage=storage, embedder=FakeEmbedder())
    r = authed.post("/api/ingest", json={"document_id": doc}).json()
    assert r["status"] == "ready"
    assert q(clean_db, "select error from cortex.documents where id = %s", (doc,))[0][0] is None


def test_ingest_failures_are_marked_and_retry_works(authed, clean_db, fakes):
    storage, embedder = fakes
    from api import index

    # permanent Gemini error -> failed with the reason
    bad = FakeEmbedder(fail_on={1: EmbedError("Gemini HTTP 400: API key not valid", status=400)})
    index.set_services(storage=storage, embedder=bad)
    doc = upload(authed, storage, "notes.txt", ("Osmosis moves water across membranes. " * 50).encode())
    r = authed.post("/api/ingest", json={"document_id": doc}).json()
    assert r["status"] == "failed" and "API key not valid" in r["error"]
    assert q(clean_db, "select status, error from cortex.documents where id = %s", (doc,))[0] == (
        "failed", "Gemini HTTP 400: API key not valid")
    # fixed -> calling again resumes and finishes
    index.set_services(storage=storage, embedder=embedder)
    r = authed.post("/api/ingest", json={"document_id": doc}).json()
    assert r["status"] == "ready"
    assert q(clean_db, "select pages, error from cortex.documents where id = %s", (doc,))[0] == (None, None)
    assert {p for _, p, _, _ in _chunks(clean_db, doc)} == {None}

    # no text
    blank = upload(authed, storage, "blank.txt", b"   \n\n  ")
    r = authed.post("/api/ingest", json={"document_id": blank}).json()
    assert r["status"] == "failed" and "No extractable text" in r["error"]
    # never uploaded to storage
    r = authed.post("/api/upload-url", json={"filename": "ghost.pdf", "size": 10}).json()
    out = authed.post("/api/ingest", json={"document_id": r["document_id"]}).json()
    assert out["status"] == "failed" and "not in storage" in out["error"]
    # corrupt PDF
    junk = upload(authed, storage, "junk.pdf", b"not a pdf at all")
    out = authed.post("/api/ingest", json={"document_id": junk}).json()
    assert out["status"] == "failed" and "Could not read the file" in out["error"]
    assert authed.post("/api/ingest", json={"document_id": 99999}).status_code == 404


def test_ingest_reports_duplicates(authed, clean_db, fakes):
    storage, _ = fakes
    data = ("Mitochondria are the powerhouse of the cell. " * 40).encode()
    a = upload(authed, storage, "a.txt", data)
    b = upload(authed, storage, "b.txt", data)
    assert authed.post("/api/ingest", json={"document_id": a}).json()["duplicate_of"] == []
    r = authed.post("/api/ingest", json={"document_id": b}).json()
    assert r["status"] == "ready"                            # not blocked
    assert r["duplicate_of"] == [{"id": a, "filename": "a.txt", "status": "ready"}]


# --- documents, search, delete -----------------------------------------------------------------
def test_documents_search_and_delete_cascade(authed, clean_db, fakes):
    storage, _ = fakes
    bio = upload(authed, storage, "biology.pdf", make_pdf(12), subject="Biology")
    notes = upload(authed, storage, "chemistry.txt",
                   b"Covalent bonds share electron pairs between atoms. Ionic bonds transfer electrons. " * 20,
                   subject="Chemistry")
    for d in (bio, notes):
        assert authed.post("/api/ingest", json={"document_id": d}).json()["status"] == "ready"
    pending = upload(authed, storage, "later.txt", b"Enzymes speed reactions. " * 10)

    docs = authed.get("/api/documents").json()["documents"]
    assert [d["id"] for d in docs] == [pending, notes, bio]        # newest first
    assert docs[0]["status"] == "uploaded" and docs[0]["progress"] == 0.0
    assert docs[2]["progress"] == 1.0 and docs[2]["pages"] == 12 and docs[2]["subject"] == "Biology"
    assert "storage_path" not in docs[0]

    r = authed.post("/api/search", json={"query": "how does osmosis work in living cells"}).json()
    assert 1 <= len(r["passages"]) <= 4
    top = r["passages"][0]
    assert top["n"] == 1 and top["filename"] == "biology.pdf" and "osmosis" in top["content"].lower()
    assert isinstance(top["page"], int) and 0 < top["similarity"] <= 1 and top["chunk_id"]
    assert set(r["timings"]) == {"embed_ms", "search_ms", "total_ms"}
    r = authed.post("/api/search", json={"query": "ionic bonds electrons", "document": "chemistry", "limit": 3}).json()
    assert r["passages"] and {p["filename"] for p in r["passages"]} == {"chemistry.txt"}
    assert r["passages"][0]["page"] is None
    assert authed.post("/api/search", json={"query": "ionic", "document": "nothing-like-this"}).json()["passages"] == []

    # file links: PDF opens at the page, TXT has no page fragment
    loc = authed.get(f"/api/file/{bio}?page=3", follow_redirects=False)
    assert loc.status_code == 302 and loc.headers["location"].endswith("?token=dl#page=3")
    loc = authed.get(f"/api/file/{notes}?page=3", follow_redirects=False)
    assert loc.headers["location"].endswith("?token=dl")
    assert authed.get("/api/file/424242", follow_redirects=False).status_code == 404

    # delete: row, chunks (cascade) and the storage object
    path = q(clean_db, "select storage_path from cortex.documents where id = %s", (bio,))[0][0]
    before = q(clean_db, "select count(*) from cortex.chunks where document_id = %s", (bio,))[0][0]
    assert before > 0
    r = authed.delete(f"/api/documents/{bio}")
    assert r.status_code == 200 and r.json()["storage_deleted"] is True
    assert q(clean_db, "select count(*) from cortex.chunks where document_id = %s", (bio,))[0][0] == 0
    assert q(clean_db, "select count(*) from cortex.documents where id = %s", (bio,))[0][0] == 0
    assert storage.deleted == [path] and path not in storage.objects
    assert authed.delete(f"/api/documents/{bio}").status_code == 404
    r = authed.post("/api/search", json={"query": "osmosis living cells"}).json()
    assert all(p["filename"] != "biology.pdf" for p in r["passages"])


def test_search_reports_gemini_trouble(authed, clean_db, fakes):
    storage, _ = fakes
    from api import index

    class Down(FakeEmbedder):
        def embed_query(self, text):
            raise RateLimited("Gemini rate limit (429)", status=429, transient=True)

    index.set_services(storage=storage, embedder=Down())
    r = authed.post("/api/search", json={"query": "x"})
    assert r.status_code == 503 and "429" in r.json()["detail"]


def test_answers_newest_first(authed, clean_db):
    with psycopg.connect(clean_db, autocommit=True) as conn:
        conn.execute("insert into cortex.answers (question, answer, citations, mode, latency) values"
                     " ('old?', 'old.', '[]', 'tool', null),"
                     " ('What is osmosis?', 'Water moving across a membrane [1].',"
                     "  '[{\"n\":1,\"document_id\":5,\"filename\":\"biology.pdf\",\"page\":3,\"chunk_id\":9,"
                     "\"excerpt\":\"Osmosis is...\",\"similarity\":0.82}]', 'inject', '{\"embed_ms\": 180}')")
    rows = authed.get("/api/answers?limit=20").json()["answers"]
    assert [r["question"] for r in rows] == ["What is osmosis?", "old?"]
    assert rows[0]["citations"][0]["page"] == 3 and rows[0]["latency"] == {"embed_ms": 180}
    assert len(authed.get("/api/answers?limit=1").json()["answers"]) == 1
    assert authed.get("/api/answers?limit=0").status_code == 422


def test_rewritten_path_is_restored(authed, clean_db):
    # what Vercel's rewrite /api/:path* -> /api/index?__route=:path* may hand the function
    r = authed.get("/api/index?__route=documents")
    assert r.status_code == 200 and r.json() == {"documents": []}
    r = authed.get("/api/index?__route=answers&limit=5")
    assert r.status_code == 200
    r = authed.get("/api/documents?__route=documents")     # original path kept: param just dropped
    assert r.status_code == 200


def test_db_connection_is_reused_and_recovers(authed, clean_db):
    from core import db

    authed.get("/api/documents")
    first = db._conn
    authed.get("/api/documents")
    assert db._conn is first                                  # warm reuse
    first.close()                                             # e.g. the pooler dropped it
    assert authed.get("/api/documents").status_code == 200
    assert db._conn is not first and not db._conn.closed
    # a connection killed server-side is detected on use and retried once
    pid = db._conn.info.backend_pid
    with psycopg.connect(clean_db, autocommit=True) as admin:
        admin.execute("select pg_terminate_backend(%s)", (pid,))
    assert authed.get("/api/documents").status_code == 200
