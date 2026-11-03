"""One psycopg connection per warm function instance, reused while healthy.

DATABASE_URL is Supabase's transaction pooler (port 6543): server-side prepared statements do
not survive between transactions there, so prepare_threshold=None. autocommit=True: every
statement is its own transaction unless a block opens one. A lock serialises use of the shared
connection in case the runtime runs requests concurrently in one instance.
"""
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

import psycopg
from psycopg.rows import dict_row

from . import config

T = TypeVar("T")

_lock = threading.RLock()
_conn: psycopg.Connection | None = None
_dsn_override: str | None = None  # tests point this at a local database


def set_dsn(dsn: str | None) -> None:
    global _dsn_override
    with _lock:
        _dsn_override = dsn
        reset()


def _connect() -> psycopg.Connection:
    dsn = _dsn_override or config.database_url()
    kwargs: dict[str, Any] = {}
    if "sslmode" not in dsn:
        kwargs["sslmode"] = "require"
    return psycopg.connect(dsn, autocommit=True, prepare_threshold=None, row_factory=dict_row,
                           connect_timeout=10, application_name="cortex-web", **kwargs)


def reset() -> None:
    global _conn
    with _lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
        _conn = None


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """The shared connection, held under the lock for the block."""
    global _conn
    with _lock:
        if _conn is None or _conn.closed or _conn.broken:
            reset()
            _conn = _connect()
        try:
            yield _conn
        except psycopg.OperationalError:
            reset()
            raise


def run(fn: Callable[[psycopg.Connection], T]) -> T:
    """Run fn(conn); if a reused connection turns out dead, reconnect and try once more."""
    with _lock:
        fresh = _conn is None or _conn.closed or _conn.broken
        try:
            with connection() as conn:
                return fn(conn)
        except psycopg.OperationalError:
            if fresh:
                raise
        with connection() as conn:
            return fn(conn)


def fetch_all(sql: str, params: tuple | dict | None = None) -> list[dict]:
    return run(lambda c: c.execute(sql, params).fetchall())


def fetch_one(sql: str, params: tuple | dict | None = None) -> dict | None:
    return run(lambda c: c.execute(sql, params).fetchone())


def execute(sql: str, params: tuple | dict | None = None) -> int:
    return run(lambda c: c.execute(sql, params).rowcount)
