"""Searchability, lease, chunker version, transient errors, the document_id search filter, DOCX order."""
import io
import threading
import time

import psycopg
import pytest
from conftest import FakeEmbedder, assert_safe_test_dsn, fake_vector, make_pdf, upload

from core import chunk as chunk_mod
from core import ingest as ingest_mod
from core.embed import EmbedError, to_pgvector
from core.storage import StorageError


def q(dsn, sql, params=None):
    with psycopg.connect(dsn) as conn:
        return conn.execute(sql, params).fetchall()


def flags(dsn, doc):
    return {r[0] for r in q(dsn, "select searchable from cortex.chunks where document_id = %s", (doc,))}


def budgeted(monkeypatch, now, budget=15, sleep=None):
    real = ingest_mod.ingest
    monkeypatch.setattr(ingest_mod, "ingest", lambda doc_id, s, e, budget_s=240: real(
        doc_id, s, e, budget_s=budget, clock=lambda: now[0], sleep=sleep or (lambda s: None)))


# --- searchable ------------------------------------------------------------------------------------
def test_document_is_unsearchable_until_ready(authed, clean_db, fakes, monkeypatch):
    storage, _ = fakes
    from api import index

    now = [0.0]
    index.set_services(storage=storage, embedder=FakeEmbedder(on_call=lambda: now.__setitem__(0, now[0] + 10)))
    budgeted(monkeypatch, now)
    doc = upload(authed, storage, "osmosis.pdf", make_pdf(40))
    r = authed.post("/api/ingest", json={"document_id": doc}).json()
    assert r["status"] == "processing" and r["done"] > 0
    assert flags(clean_db, doc) == {False}
    # nothing of an unfinished document is found
    found = authed.post("/api/search", json={"query": "osmosis living cells"}).json()["passages"]
    assert found == []
    while r["status"] != "ready":
        r = authed.post("/api/ingest", json={"document_id": doc}).json()
    assert flags(clean_db, doc) == {True}
    assert authed.post("/api/search", json={"query": "osmosis living cells"}).json()["passages"]

    # a retry of a failed document (or any reset to processing) makes everything unsearchable again
    with psycopg.connect(clean_db, autocommit=True) as conn:
        conn.execute("update cortex.documents set status = 'failed', error = 'x' where id = %s", (doc,))
        conn.execute("delete from cortex.chunks where document_id = %s and chunk_index >= 110", (doc,))
    index.set_services(storage=storage, embedder=FakeEmbedder(fail_on={1: EmbedError("Gemini HTTP 503", status=503,
                                                                                    transient=True)}))
    r = authed.post("/api/ingest", json={"document_id": doc}).json()
    assert r["status"] == "processing" and r["retry_after"] > 0          # transient, not failed
    assert flags(clean_db, doc) == {False}
    assert q(clean_db, "select status from cortex.documents where id = %s", (doc,))[0][0] == "processing"
    index.set_services(storage=storage, embedder=FakeEmbedder())
    assert authed.post("/api/ingest", json={"document_id": doc}).json()["status"] == "ready"
    assert flags(clean_db, doc) == {True}


def test_unfinished_near_duplicates_cannot_crowd_out_ready_chunks(authed, clean_db):
    query = "what does the manual say about calibrating the sensor"
    qv = fake_vector(query)
    near = fake_vector(query + " drift offset")                      # similar, not identical
    with psycopg.connect(clean_db, autocommit=True) as conn:
        busy = conn.execute("insert into cortex.documents (filename, storage_path, status)"
                            " values ('draft.pdf', 'docs/a/draft.pdf', 'processing') returning id").fetchone()[0]
        ready = conn.execute("insert into cortex.documents (filename, storage_path, status, ready_at)"
                             " values ('manual.pdf', 'docs/b/manual.pdf', 'ready', now()) returning id").fetchone()[0]
        conn.execute("insert into cortex.chunks (document_id, chunk_index, page, content, embedding)"
                     " select %s, g, 1, 'unfinished copy ' || g, %s::extensions.vector"
                     " from generate_series(0, 1999) g", (busy, to_pgvector(qv)))
        conn.execute("insert into cortex.chunks (document_id, chunk_index, page, content, embedding, searchable)"
                     " values (%s, 0, 7, 'Calibrate the sensor by zeroing the offset first.', %s::extensions.vector,"
                     " true)", (ready, to_pgvector(near)))
        conn.execute("analyze cortex.chunks")
    r = authed.post("/api/search", json={"query": query}).json()
    assert [(p["filename"], p["page"]) for p in r["passages"]] == [("manual.pdf", 7)]
    # and when the planner is forced onto the HNSW index
    with psycopg.connect(clean_db, autocommit=True) as conn:
        conn.execute("set enable_seqscan = off")
        rows = conn.execute("select filename from cortex.search(%s::extensions.vector, %s, 4)",
                            (to_pgvector(qv), query)).fetchall()
    assert rows == [("manual.pdf",)]


def test_same_passage_in_two_ready_documents_comes_back_once(authed, clean_db, fakes):
    storage, _ = fakes
    text = ("Mitochondria are the powerhouse of the cell and make ATP by respiration. " * 3).encode()
    a = upload(authed, storage, "a.txt", text)
    b = upload(authed, storage, "b.txt", text)
    for d in (a, b):
        assert authed.post("/api/ingest", json={"document_id": d}).json()["status"] == "ready"
    passages = authed.post("/api/search", json={"query": "mitochondria powerhouse ATP"}).json()["passages"]
    contents = [p["content"] for p in passages]
    assert len(contents) == 1 and len(set(contents)) == len(contents)


# --- lease -----------------------------------------------------------------------------------------
def test_busy_document_is_left_alone(authed, clean_db, fakes):
    storage, embedder = fakes
    doc = upload(authed, storage, "bio.pdf", make_pdf(5))
    with psycopg.connect(clean_db, autocommit=True) as conn:
        conn.execute("update cortex.documents set ingest_lease_until = now() + interval '1 minute' where id = %s", (doc,))
    r = authed.post("/api/ingest", json={"document_id": doc}).json()
    assert r["status"] == "processing" and r["busy"] is True and r["retry_after"] == 10
    assert embedder.calls == [] and getattr(storage, "downloads", 0) == 0
    with psycopg.connect(clean_db, autocommit=True) as conn:              # an expired lease is taken over
        conn.execute("update cortex.documents set ingest_lease_until = now() - interval '1 second' where id = %s", (doc,))
    assert authed.post("/api/ingest", json={"document_id": doc}).json()["status"] == "ready"
    assert q(clean_db, "select ingest_lease_until from cortex.documents where id = %s", (doc,))[0][0] is None


def test_concurrent_calls_embed_each_batch_once(authed, clean_db, fakes):
    storage, _ = fakes
    doc = upload(authed, storage, "bio.pdf", make_pdf(40))
    slow = FakeEmbedder(on_call=lambda: time.sleep(0.3))
    results = []

    def run():
        results.append(ingest_mod.ingest(doc, storage, slow, budget_s=60))

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
        time.sleep(0.05)
    for t in threads:
        t.join()
    assert sorted(r["status"] for r in results) == ["processing", "ready"]
    assert [r for r in results if r["status"] == "processing"][0]["busy"] is True
    total = [r for r in results if r["status"] == "ready"][0]["total"]
    assert sum(c["n"] for c in slow.calls) == total                         # nothing embedded twice


# --- chunker version, transient errors ---------------------------------------------------------------
def test_chunker_version_change_restarts_the_document(authed, clean_db, fakes, monkeypatch):
    storage, _ = fakes
    from api import index

    now = [0.0]
    index.set_services(storage=storage, embedder=FakeEmbedder(on_call=lambda: now.__setitem__(0, now[0] + 10)))
    pdf = make_pdf(40)
    doc = upload(authed, storage, "bio.pdf", pdf)
    with monkeypatch.context() as m:
        budgeted(m, now)
        assert authed.post("/api/ingest", json={"document_id": doc}).json()["status"] == "processing"
    assert q(clean_db, "select chunker_version from cortex.documents where id = %s", (doc,))[0][0] == \
        chunk_mod.CHUNKER_VERSION
    with psycopg.connect(clean_db, autocommit=True) as conn:  # pretend those chunks came from an older chunker
        conn.execute("update cortex.documents set chunker_version = 0 where id = %s", (doc,))
        conn.execute("update cortex.chunks set content = 'stale' where document_id = %s", (doc,))
    index.set_services(storage=storage, embedder=FakeEmbedder())
    r = authed.post("/api/ingest", json={"document_id": doc}).json()
    assert r["status"] == "ready"
    stored = [c for (c,) in q(clean_db, "select content from cortex.chunks where document_id = %s"
                                        " order by chunk_index", (doc,))]
    from core.parse import parse
    assert stored == [c.content for c in chunk_mod.chunk_pages(parse("bio.pdf", pdf).pages)]


def test_storage_and_database_hiccups_are_transient(authed, clean_db, fakes, monkeypatch):
    storage, _ = fakes
    doc = upload(authed, storage, "bio.pdf", make_pdf(5))

    def down(path, timeout=None):
        raise StorageError("download: Supabase Storage HTTP 503: busy", status=503, transient=True)

    with monkeypatch.context() as m:
        m.setattr(storage, "download", down)
        r = authed.post("/api/ingest", json={"document_id": doc}).json()
    assert r["status"] == "processing" and r["retry_after"] == 10 and "503" in r["note"]
    assert q(clean_db, "select status from cortex.documents where id = %s", (doc,))[0][0] == "processing"

    def dropped(*a, **k):
        raise psycopg.OperationalError("server closed the connection unexpectedly")

    with monkeypatch.context() as m:
        m.setattr(ingest_mod, "_insert", dropped)
        r = authed.post("/api/ingest", json={"document_id": doc}).json()
    assert r["status"] == "processing" and r["retry_after"] == 10
    assert authed.post("/api/ingest", json={"document_id": doc}).json()["status"] == "ready"


# --- search by document_id ----------------------------------------------------------------------------
def test_search_document_id_is_exact(authed, clean_db, fakes):
    storage, _ = fakes
    body = b"Photosynthesis turns light into chemical energy in chloroplasts. " * 10
    one = upload(authed, storage, "notes.txt", body)
    two = upload(authed, storage, "old notes.txt", body + b" Extra line about chloroplasts and light.")
    for d in (one, two):
        assert authed.post("/api/ingest", json={"document_id": d}).json()["status"] == "ready"
    for d in (one, two):
        ps = authed.post("/api/search", json={"query": "photosynthesis light", "document_id": d}).json()["passages"]
        assert ps and {p["document_id"] for p in ps} == {d}
        assert [p["n"] for p in ps] == list(range(1, len(ps) + 1))
    pending = upload(authed, storage, "later.txt", body)
    assert authed.post("/api/search", json={"query": "light", "document_id": pending}).json()["passages"] == []
    assert authed.post("/api/search", json={"query": "light", "document_id": 98765}).json()["passages"] == []


# --- DOCX reading order, test-DB guard -------------------------------------------------------------------
def test_docx_keeps_reading_order_and_separates_headings():
    import docx

    from core.chunk import chunk_page_text
    from core.parse import parse

    d = docx.Document()
    d.add_heading("Cell Biology", level=1)
    d.add_paragraph("Cells are the basic unit of life.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Organelle", "Job"
    t.cell(1, 0).text, t.cell(1, 1).text = "Ribosome", "Makes proteins"
    d.add_paragraph("After the table comes this paragraph.")
    buf = io.BytesIO()
    d.save(buf)
    text = parse("x.docx", buf.getvalue()).pages[0].text
    assert text == ("Cell Biology\n\nCells are the basic unit of life.\n\nOrganelle | Job\nRibosome | Makes proteins"
                    "\n\nAfter the table comes this paragraph.")
    assert chunk_page_text(text)[0].startswith("Cell Biology\nCells are")   # heading is not glued to a sentence


def test_test_database_guard():
    assert_safe_test_dsn("host=/tmp port=55432 user=cortex dbname=cortex_web_test")
    assert_safe_test_dsn("postgresql://postgres@localhost:5432/my_test_db")
    for bad in ("postgresql://postgres.ref:pw@aws-0-ap-south-1.pooler.supabase.com:6543/postgres",
                "postgresql://postgres@db.example.com/cortex_test",
                "host=localhost dbname=postgres"):
        with pytest.raises(RuntimeError, match="refusing"):
            assert_safe_test_dsn(bad)
