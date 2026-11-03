"""SQL used by the API (documents, search, answers). Ingest's SQL lives in core.ingest."""
from . import db
from .embed import to_pgvector

DOC_COLUMNS = ("id, filename, subject, storage_path, bytes, sha256, pages, chunk_total, chunk_count, "
               "status, error, uploaded_at, ready_at")


def create_document(filename: str, subject: str | None, storage_path: str, size: int) -> dict:
    return db.fetch_one(
        "insert into cortex.documents (filename, subject, storage_path, bytes, status)"
        " values (%s, %s, %s, %s, 'uploaded') returning " + DOC_COLUMNS,
        (filename, subject, storage_path, size))


def get_document(document_id: int) -> dict | None:
    return db.fetch_one(f"select {DOC_COLUMNS} from cortex.documents where id = %s", (document_id,))


def list_documents(limit: int = 500) -> list[dict]:
    return db.fetch_all(
        f"select {DOC_COLUMNS} from cortex.documents order by uploaded_at desc, id desc limit %s", (limit,))


def delete_document(document_id: int) -> dict | None:
    """Deletes the row (chunks go with it: on delete cascade); returns what was deleted."""
    return db.fetch_one(
        "delete from cortex.documents where id = %s returning id, filename, storage_path", (document_id,))


def search(embedding: list[float], query: str, match_count: int, filename_filter: str | None,
           min_similarity: float = 0.0) -> list[dict]:
    return db.fetch_all(
        "select chunk_id, document_id, filename, subject, page, chunk_index, content,"
        " similarity, keyword_hit, score"
        " from cortex.search(%s::extensions.vector, %s, %s, %s, %s)",
        (to_pgvector(embedding), query, match_count, filename_filter, min_similarity))


def recent_answers(limit: int) -> list[dict]:
    return db.fetch_all(
        "select id, question, answer, citations, mode, latency, created_at"
        " from cortex.answers order by created_at desc, id desc limit %s", (limit,))


def ready_filename(document_id: int) -> str | None:
    row = db.fetch_one("select filename from cortex.documents where id = %s and status = 'ready'", (document_id,))
    return row["filename"] if row else None


# --- login throttle: at most LOGIN_MAX_FAILURES failed logins per LOGIN_WINDOW_S, all clients ---
LOGIN_MAX_FAILURES = 10
LOGIN_WINDOW_S = 15 * 60


def login_locked_for() -> int:
    """Seconds until another login may be tried (0 = not locked)."""
    row = db.fetch_one(
        "select greatest(1, ceil(extract(epoch from (at + make_interval(secs => %s) - now()))))::int as wait"
        " from cortex.login_failures where at > now() - make_interval(secs => %s)"
        " order by at desc offset %s limit 1",
        (LOGIN_WINDOW_S, LOGIN_WINDOW_S, LOGIN_MAX_FAILURES - 1))
    return int(row["wait"]) if row else 0


def record_login_failure() -> None:
    def go(conn):
        with conn.transaction():
            conn.execute("insert into cortex.login_failures default values")
            conn.execute("delete from cortex.login_failures where at < now() - interval '1 day'")
    db.run(go)
