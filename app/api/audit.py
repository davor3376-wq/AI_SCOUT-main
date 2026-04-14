"""
Audit logger — records every significant API action to SQLite.

Every row captures:
  timestamp   UTC ISO-8601
  username    who did it (from auth)
  action      verb + resource  e.g. "launch_mission", "download_pdf"
  resource    job_id / user_id / etc.
  details     optional free-text context
  ip_address  client IP
  status      "ok" | "denied" | "error"

The audit table lives in the same SQLite file as jobs so there's one backup
target and one restore operation.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from app.core.config import SCOUT_DB_PATH
from app.core.database import get_connection, init_schema

logger = logging.getLogger(__name__)

_AUDIT_SCHEMA = """
    CREATE TABLE IF NOT EXISTS audit_logs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT NOT NULL,
        username    TEXT,
        action      TEXT NOT NULL,
        resource    TEXT,
        details     TEXT,
        ip_address  TEXT,
        status      TEXT NOT NULL DEFAULT 'ok'
    );
    CREATE INDEX IF NOT EXISTS idx_audit_ts
        ON audit_logs(timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_audit_user
        ON audit_logs(username);
    CREATE INDEX IF NOT EXISTS idx_audit_action
        ON audit_logs(action);
"""


def init_audit_db(db_path: str) -> None:
    init_schema(_AUDIT_SCHEMA, db_path)


class AuditLogger:
    def __init__(self, db_path: str = SCOUT_DB_PATH):
        self.db_path = db_path
        init_audit_db(db_path)

    def log(
        self,
        action: str,
        username: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[str] = None,
        ip_address: Optional[str] = None,
        status: str = "ok",
    ) -> None:
        try:
            with get_connection(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO audit_logs "
                    "(timestamp, username, action, resource, details, ip_address, status) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (
                        datetime.now(timezone.utc).isoformat(),
                        username,
                        action,
                        resource,
                        details,
                        ip_address,
                        status,
                    ),
                )
        except Exception as exc:
            # Never let audit failures crash the main request
            logger.error(f"Audit write failed: {exc}")

    def query(
        self,
        username: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list:
        clauses = []
        params: list = []
        if username:
            clauses.append("username=?")
            params.append(username)
        if action:
            clauses.append("action=?")
            params.append(action)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params += [limit, offset]
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT * FROM audit_logs {where} "
                f"ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                params,
            ).fetchall()
        return [dict(r) for r in rows]
