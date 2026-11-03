"""Cortex against a real PostgreSQL + pgvector with cortex-web/sql/schema.sql applied: documents and chunks
with known vectors go in, and cortex.search() (the SQL function, through this module) must find the right ones.

Skipped unless CORTEX_TEST_DATABASE_URL points at a SCRATCH database (its cortex tables are emptied!), e.g.
    createdb friday_cortex_test && psql -d friday_cortex_test -c "create schema extensions" \
        && psql -d friday_cortex_test -f ../../cortex-web/sql/schema.sql
    CORTEX_TEST_DATABASE_URL="postgresql://me@/friday_cortex_test?host=/tmp&sslmode=disable" python -m unittest test_cortex_db
Gemini is never called: the question's embedding is faked."""

import math
import os
import socket
import threading
import time
import unittest
from unittest import mock

import _path  # noqa: F401

import brain
import config
import cortex
import gemini
from brainfakes import FakeMemory

URL = os.environ.get("CORTEX_TEST_DATABASE_URL", "")
TCP = os.environ.get("CORTEX_TEST_PG_TCP", "")      # host:port of the same server over TCP, for the proxy test
DIMS = 768


class DroppingProxy:
    """TCP proxy in front of PostgreSQL. drop=True: it keeps accepting and acknowledging traffic but passes
    nothing on in either direction, like a network path that swallows packets."""

    def __init__(self, target: str) -> None:
        host, port = target.rsplit(":", 1)
        self.target = (host, int(port))
        self.drop = False
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(16)
        self.port = self.srv.getsockname()[1]
        self.socks = []
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                client, _ = self.srv.accept()
            except OSError:
                return
            upstream = socket.create_connection(self.target)
            self.socks += [client, upstream]
            for a, b in ((client, upstream), (upstream, client)):
                threading.Thread(target=self._pump, args=(a, b), daemon=True).start()

    def _pump(self, src, dst):
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                if not self.drop:
                    dst.sendall(data)
        except OSError:
            pass

    def close(self):
        self.srv.close()
        for s in self.socks:
            try:
                s.close()
            except OSError:
                pass


def vec(**parts):
    """A unit vector with weights on the given axes: vec(a0=1, a1=1) -> (e0 + e1) / sqrt 2."""
    v = [0.0] * DIMS
    for name, w in parts.items():
        v[int(name[1:])] = float(w)
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def lit(v):
    return "[" + ",".join(repr(x) for x in v) + "]"


@unittest.skipUnless(URL, "set CORTEX_TEST_DATABASE_URL to a scratch PostgreSQL with pgvector and the Cortex schema")
class RealDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psycopg

        cls.admin = psycopg.connect(URL, autocommit=True)
        a = cls.admin
        a.execute("truncate cortex.answers, cortex.chunks, cortex.documents restart identity cascade")
        ids = {}
        for key, name, subject, status in (("bio", "biology-notes.pdf", "Biology", "ready"),
                                           ("router", "Router_Manual.txt", None, "ready"),
                                           ("draft", "draft-chemistry.pdf", "Chemistry", "processing")):
            ids[key] = a.execute("insert into cortex.documents (filename, subject, storage_path, status) "
                                 "values (%s, %s, %s, %s) returning id",
                                 (name, subject, f"docs/{key}/{name}", status)).fetchone()[0]
        cls.ids = ids
        # (searchable is what ingest flips to true when it marks the document ready)
        chunks = [("bio", 0, 3, "Mitochondria are the powerhouse of the cell.", vec(a0=1), True),
                  ("bio", 1, 4, "Ribosomes build proteins from amino acids.", vec(a0=1, a1=1), True),
                  ("bio", 2, 5, "Mitochondria chunk of a ready document not flipped yet.", vec(a0=1), False),
                  ("router", 0, None, "Hold the reset button for ten seconds.", vec(a1=1), True),
                  ("router", 1, None, "The status light blinks amber while it updates.", vec(a0=1, a2=3), True),
                  ("draft", 0, 1, "Mitochondria draft that isn't ready yet.", vec(a0=1), False)]
        cls.chunk = {}
        for doc, idx, page, content, v, searchable in chunks:
            cls.chunk[(doc, idx)] = a.execute(
                "insert into cortex.chunks (document_id, chunk_index, page, content, embedding, searchable) "
                "values (%s, %s, %s, %s, %s::extensions.vector, %s) returning id",
                (ids[doc], idx, page, content, lit(v), searchable)).fetchone()[0]

    @classmethod
    def tearDownClass(cls):
        for c in list(cortex._search_pool.queue) + [cortex._aux]:
            c.close()
        cls.admin.close()

    def setUp(self):
        for name, value in (("CORTEX_DATABASE_URL", URL), ("CORTEX_TOP_K", 4), ("CORTEX_MIN_SIMILARITY", 0.5),
                            ("CORTEX_TIMEOUT", 5.0), ("CORTEX_MAX_CHARS", 1200), ("CORTEX_INJECT", True),
                            ("CORTEX_PREFETCH", True), ("CORTEX_LOG_ANSWERS", True), ("MAX_TOOL_ROUNDS", 3),
                            ("WEB_SEARCH", True), ("MAX_REPLY_CHARS", 400)):
            p = mock.patch.object(config, name, value)
            p.start()
            self.addCleanup(p.stop)
        cortex._reset_for_tests()
        self.addCleanup(cortex._reset_for_tests)
        self.query_vec = vec(a0=1)
        p = mock.patch.object(gemini, "embed", lambda *a, **k: list(self.query_vec))
        p.start()
        self.addCleanup(p.stop)

    def search(self, q, document=None):
        out = cortex.search(q, document)
        self.assertNotIn("error", out, out.get("error"))
        return out["passages"]

    def test_vector_search_finds_the_nearest_ready_chunks_above_min_similarity(self):
        found = self.search("what are mitochondria")
        self.assertEqual([p["chunk_id"] for p in found], [self.chunk[("bio", 0)], self.chunk[("bio", 1)]])
        self.assertEqual([p["n"] for p in found], [1, 2])
        self.assertAlmostEqual(found[0]["similarity"], 1.0, places=3)
        self.assertAlmostEqual(found[1]["similarity"], 1 / math.sqrt(2), places=3)
        self.assertEqual((found[0]["filename"], found[0]["page"], found[0]["document_id"]),
                         ("biology-notes.pdf", 3, self.ids["bio"]))
        self.assertTrue(found[0]["keyword_hit"])                 # "mitochondria" is in its text too
        self.assertNotIn(self.chunk[("draft", 0)], [p["chunk_id"] for p in found])   # processing: never searched
        self.assertNotIn(self.chunk[("bio", 2)], [p["chunk_id"] for p in found])     # not searchable yet

    def test_the_same_passage_uploaded_twice_comes_back_once(self):
        a = self.admin
        copy = a.execute("insert into cortex.documents (filename, subject, storage_path, status) values "
                         "('biology-notes (1).pdf', 'Biology', 'docs/copy/x.pdf', 'ready') returning id").fetchone()[0]
        try:
            a.execute("insert into cortex.chunks (document_id, chunk_index, page, content, embedding, searchable) "
                      "values (%s, 0, 3, 'Mitochondria are the powerhouse of the cell.', %s::extensions.vector, true)",
                      (copy, lit(vec(a0=1))))
            texts = [p["text"] for p in self.search("what are mitochondria")]
            self.assertEqual(texts.count("Mitochondria are the powerhouse of the cell."), 1)
        finally:
            a.execute("delete from cortex.documents where id = %s", (copy,))

    def test_statement_timeouts_are_set_when_connecting(self):
        cortex._reset_for_tests()
        with mock.patch.object(config, "CORTEX_TIMEOUT", 2.5):
            self.search("warm")
            c = cortex._search_pool.get()
            try:
                self.assertEqual(c.execute("show statement_timeout")[0]["statement_timeout"], "2500ms")
            finally:
                cortex._search_pool.put(c)
        self.assertEqual(cortex._aux.execute("show statement_timeout")[0]["statement_timeout"], "5s")

    def test_a_lower_min_similarity_lets_more_in(self):
        with mock.patch.object(config, "CORTEX_MIN_SIMILARITY", 0.2):
            found = self.search("what are mitochondria")
        self.assertIn(self.chunk[("router", 1)], [p["chunk_id"] for p in found])     # similarity 0.316
        self.assertNotIn(self.chunk[("router", 0)], [p["chunk_id"] for p in found])  # similarity 0

    def test_a_filename_filter_searches_just_that_document(self):
        with mock.patch.object(config, "CORTEX_MIN_SIMILARITY", 0.0):
            found = self.search("status light", "router")
        self.assertEqual({p["filename"] for p in found}, {"Router_Manual.txt"})
        self.assertEqual(found[0]["chunk_id"], self.chunk[("router", 1)])

    def test_nothing_similar_enough_is_an_empty_list(self):
        self.query_vec = vec(a5=1)
        self.assertEqual(self.search("quantum chromodynamics"), [])

    def test_the_catalog_lists_ready_documents_only(self):
        self.assertTrue(cortex.refresh())
        self.assertEqual([d["filename"] for d in cortex.catalog()], ["Router_Manual.txt", "biology-notes.pdf"])
        self.assertEqual(cortex.words(), frozenset({"biology", "router"}))
        self.assertEqual(cortex.status(), 2)

    def test_a_dropped_connection_is_reopened_transparently(self):
        self.search("warm it up")
        killed = self.admin.execute("select count(pg_terminate_backend(pid)) from pg_stat_activity "
                                    "where application_name = 'friday-cortex'").fetchone()[0]
        self.assertGreaterEqual(killed, 1)
        time.sleep(0.2)
        self.assertEqual(len(self.search("what are mitochondria")), 2)

    def test_the_answer_log(self):
        cortex.log_answer("what are mitochondria", "The powerhouse of the cell.",
                          [{"n": 1, "document_id": self.ids["bio"], "filename": "biology-notes.pdf", "page": 3}],
                          "inject", {"rag_ms": 300, "embed_ms": 250, "search_ms": 12})
        row = self.admin.execute("select question, answer, citations, mode, latency from cortex.answers "
                                 "order by id desc limit 1").fetchone()
        self.assertEqual(row[0:2], ("what are mitochondria", "The powerhouse of the cell."))
        self.assertEqual(row[2][0]["page"], 3)
        self.assertEqual((row[3], row[4]["search_ms"]), ("inject", 12))

    def test_a_whole_inject_turn_against_the_real_database(self):
        self.assertTrue(cortex.refresh())
        seen = []

        def chat(messages, tools=None, **kw):
            seen.append(messages[0]["content"])
            return {"role": "assistant", "content": "Your biology notes, page three, say they power the cell [1]."}

        with mock.patch.object(gemini, "chat", chat):
            out = brain.Brain(FakeMemory()).answer("what do my biology notes say about mitochondria")
        self.assertEqual(len(seen), 1)
        self.assertIn('<passage n="1" source="biology notes (biology-notes.pdf), page 3">\nMitochondria are the '
                      'powerhouse', seen[0])
        self.assertEqual(out["reply"], "Your biology notes, page three, say they power the cell.")
        self.assertEqual(out["citations"][0]["chunk_id"], self.chunk[("bio", 0)])
        self.assertEqual(out["cortex"]["mode"], "inject")

    @unittest.skipUnless(TCP, "set CORTEX_TEST_PG_TCP=host:port (the same server over TCP) for the proxy test")
    def test_a_path_that_drops_traffic_never_wedges_cortex(self):
        proxy = DroppingProxy(TCP)
        self.addCleanup(proxy.close)
        db = URL.split("?")[0].rsplit("/", 1)[1]
        url = f"postgresql://{os.environ.get('CORTEX_TEST_PG_USER', 'cortex')}@127.0.0.1:{proxy.port}/{db}?sslmode=disable"
        for target, name, value in ((config, "CORTEX_DATABASE_URL", url), (config, "CORTEX_TIMEOUT", 0.5),
                                    (cortex, "AUX_TIMEOUT", 0.6)):
            p = mock.patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)
        cortex._reset_for_tests()
        self.assertTrue(cortex.refresh())
        self.assertEqual(len(self.search("what are mitochondria")), 2)      # connections open, through the proxy
        proxy.drop = True                                                   # the line goes dead, silently
        for q in ("what are ribosomes", "what is a cell"):
            t0 = time.monotonic()
            out = cortex.search(q)
            self.assertLess(time.monotonic() - t0, 1.0)
            self.assertTrue(out.get("timed_out"), out)
        t0 = time.monotonic()
        self.assertFalse(cortex.refresh())
        self.assertLess(time.monotonic() - t0, 1.2)
        self.assertEqual(cortex.status(), "unreachable")
        self.assertEqual(cortex.Turn("what do my notes say").route, "down")
        proxy.drop = False                                                  # the line comes back
        self.assertTrue(cortex.refresh())
        self.assertEqual(len(self.search("what are mitochondria again")), 2)   # replacements, not "all busy"
        self.assertEqual(cortex._search_pool.qsize(), cortex.SEARCH_CONNS)

    def test_timing(self):
        self.search("warm")
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            cortex._db_search(self.query_vec, "what are mitochondria", None)
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        print(f"\n[measure] cortex.search round trip on this database: median {times[10]:.1f} ms, worst {times[-1]:.1f} ms")
        self.assertLess(times[10], 200)


if __name__ == "__main__":
    unittest.main()
