"""
Shared database module for AI SCOUT.

Supports two backends:
  - **SQLite** (default) — thread-local connection cache with WAL mode.
  - **PostgreSQL** — thread-safe connection pool via psycopg2 (activated by
    setting the ``DATABASE_URL`` env var to a ``postgresql://`` DSN).

All other modules should import ``get_connection`` / ``init_schema`` from here
instead of creating their own ``_connect`` helpers.

The module transparently translates the ``?`` placeholder style used by SQLite
to ``%s`` for PostgreSQL, and rewrites common DDL differences (AUTOINCREMENT →
SERIAL).  The few DML statements that use ``INSERT OR IGNORE`` / ``INSERT OR
REPLACE`` have been rewritten in their respective modules to standard
``ON CONFLICT`` syntax that both backends understand.
"""

import os
import re
import sqlite3
import threading
import logging
from contextlib import contextmanager
from typing import Optional

from app.core.config import DATABASE_URL, SCOUT_DB_PATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def is_postgres() -> bool:
    """Return True when the application is configured to use PostgreSQL."""
    return DATABASE_URL.startswith(("postgresql://", "postgres://"))


# ---------------------------------------------------------------------------
# SQLite — thread-local connection cache
# ---------------------------------------------------------------------------

_sqlite_local = threading.local()


def _get_sqlite_connection(db_path: str) -> sqlite3.Connection:
    """Return a cached-per-thread SQLite connection with WAL enabled."""
    key = f"_conn_{os.path.abspath(db_path)}"
    conn: Optional[sqlite3.Connection] = getattr(_sqlite_local, key, None)
    if conn is None:
        conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        setattr(_sqlite_local, key, conn)
    return conn


# ---------------------------------------------------------------------------
# PostgreSQL — ThreadedConnectionPool (lazy-initialised singleton)
# ---------------------------------------------------------------------------

_pg_pool = None
_pg_pool_lock = threading.Lock()


def _get_pg_pool():
    """Lazily create and return the global psycopg2 connection pool."""
    global _pg_pool
    if _pg_pool is None:
        with _pg_pool_lock:
            if _pg_pool is None:
                import psycopg2.pool  # type: ignore[import-untyped]
                _pg_pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=2, maxconn=20, dsn=DATABASE_URL,
                )
                logger.info("PostgreSQL connection pool initialised.")
    return _pg_pool


# ---------------------------------------------------------------------------
# PostgreSQL connection adapter
# ---------------------------------------------------------------------------

class _PgCursorWrapper:
    """Thin wrapper around a psycopg2 cursor that translates ``?`` → ``%s``."""

    def __init__(self, cursor):
        self._cur = cursor

    # -- query translation --------------------------------------------------

    @staticmethod
    def _translate(sql: str) -> str:
        return sql.replace("?", "%s")

    # -- public interface (mirrors sqlite3.Cursor) --------------------------

    def execute(self, sql: str, params=None):
        sql = self._translate(sql)
        self._cur.execute(sql, params or ())
        return self  # allow chaining / assignment to cursor

    def executemany(self, sql: str, seq_of_params):
        sql = self._translate(sql)
        self._cur.executemany(sql, seq_of_params)
        return self

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    @property
    def lastrowid(self):
        # psycopg2 does not natively populate lastrowid for non-serial
        # inserts.  Callers that need the id should append RETURNING and
        # use fetchone() instead.
        return getattr(self._cur, "lastrowid", None)

    @property
    def rowcount(self):
        return self._cur.rowcount

    @property
    def description(self):
        return self._cur.description


class _PgConnectionWrapper:
    """
    Makes a psycopg2 connection behave enough like ``sqlite3.Connection``
    that existing module code works unchanged.
    """

    def __init__(self, raw_conn):
        self._conn = raw_conn

    def execute(self, sql: str, params=None):
        import psycopg2.extras  # type: ignore[import-untyped]
        cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        wrapper = _PgCursorWrapper(cur)
        wrapper.execute(sql, params)
        return wrapper

    def executemany(self, sql: str, seq_of_params):
        import psycopg2.extras  # type: ignore[import-untyped]
        cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        wrapper = _PgCursorWrapper(cur)
        wrapper.executemany(sql, seq_of_params)
        return wrapper

    def executescript(self, sql: str):
        """Emulate sqlite3's ``executescript`` by splitting on ``;``."""
        import psycopg2.extras  # type: ignore[import-untyped]
        cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                cur.execute(stmt)

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        pass  # pool manages lifecycle

    # Context-manager support (transaction boundary)
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._conn.rollback()
        else:
            self._conn.commit()
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@contextmanager
def get_connection(db_path: str = None):
    """
    Context manager that yields a database connection.

    * **SQLite** (default): thread-local cached connection, WAL mode.
    * **PostgreSQL**: pooled connection from ``psycopg2.pool``.

    Usage::

        with get_connection() as conn:
            conn.execute("INSERT INTO ...", (val,))
    """
    if is_postgres():
        pool = _get_pg_pool()
        raw = pool.getconn()
        wrapper = _PgConnectionWrapper(raw)
        try:
            yield wrapper
            raw.commit()
        except Exception:
            raw.rollback()
            raise
        finally:
            pool.putconn(raw)
    else:
        path = db_path or SCOUT_DB_PATH
        conn = _get_sqlite_connection(path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def get_raw_connection(db_path: str = None):
    """
    Return a *raw* connection for callers that need manual transaction
    control (e.g. ``BEGIN IMMEDIATE`` in SQLite, ``SELECT … FOR UPDATE``
    in PostgreSQL).

    **Caller is responsible for commit / rollback / close.**
    """
    if is_postgres():
        pool = _get_pg_pool()
        raw = pool.getconn()
        return _PgConnectionWrapper(raw), lambda: pool.putconn(raw)
    else:
        path = db_path or SCOUT_DB_PATH
        conn = sqlite3.connect(path, timeout=30, check_same_thread=False,
                               isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn, conn.close


# ---------------------------------------------------------------------------
# DDL helpers
# ---------------------------------------------------------------------------

_DDL_AUTOINCREMENT_RE = re.compile(
    r"(\w+)\s+INTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT",
    re.IGNORECASE,
)


def _translate_ddl(sql: str) -> str:
    """Translate SQLite DDL to PostgreSQL-compatible DDL."""
    return _DDL_AUTOINCREMENT_RE.sub(r"\1 SERIAL PRIMARY KEY", sql)


def init_schema(schema_sql: str, db_path: str = None) -> None:
    """
    Execute DDL statements to initialise database tables.

    For SQLite uses ``executescript``; for PostgreSQL splits on ``;``
    and translates dialect differences.
    """
    if is_postgres():
        translated = _translate_ddl(schema_sql)
        with get_connection() as conn:
            conn.executescript(translated)
    else:
        path = db_path or SCOUT_DB_PATH
        conn = _get_sqlite_connection(path)
        conn.executescript(schema_sql)


def insert_returning_id(conn, sql: str, params, id_column: str = "id") -> int:
    """
    Execute an INSERT and return the auto-generated *id_column*.

    - **SQLite**: uses ``cursor.lastrowid``.
    - **PostgreSQL**: appends ``RETURNING <id_column>`` and fetches the value.
    """
    if is_postgres():
        translated = sql.replace("?", "%s")
        if "RETURNING" not in translated.upper():
            translated = translated.rstrip().rstrip(";") + f" RETURNING {id_column}"
        cur = conn.execute(translated, params)
        row = cur.fetchone()
        return row[id_column] if isinstance(row, dict) else row[0]
    else:
        cur = conn.execute(sql, params)
        return cur.lastrowid
