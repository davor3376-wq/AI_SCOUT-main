import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import uuid

from app.core.config import SCOUT_DB_PATH
from app.core.database import get_connection, init_schema

logger = logging.getLogger("JobManager")

_JOBS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS jobs (
        id          TEXT PRIMARY KEY,
        status      TEXT NOT NULL DEFAULT 'PENDING',
        created_at  TEXT NOT NULL,
        recurrence  TEXT,
        sensor      TEXT,
        data        TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_jobs_status
        ON jobs(status);
    CREATE INDEX IF NOT EXISTS idx_jobs_created_at
        ON jobs(created_at DESC);
"""


def _init_db(db_path: str) -> None:
    init_schema(_JOBS_SCHEMA, db_path)


class JobManager:
    """
    Manages job lifecycle backed by SQLite.

    The full job dict is stored as JSON in the `data` column so we never
    lose fields. Indexed columns (id, status, created_at, recurrence, sensor)
    exist solely for fast filtering — they are always kept in sync with `data`.
    """

    def __init__(self, db_path: str = SCOUT_DB_PATH):
        self.db_path = db_path
        _init_db(self.db_path)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def create_job(self, metadata: Dict[str, Any]) -> str:
        job_id = f"JOB-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now(timezone.utc).isoformat()
        job: Dict[str, Any] = {
            "id": job_id,
            "status": "PENDING",
            "created_at": now,
            "recurrence": metadata.get("recurrence"),
            "last_run": None,
            **metadata,
            "results": {
                "raw_files": [],
                "processed_files": [],
                "pdf_report": None,
            },
        }
        with get_connection(self.db_path) as conn:
            conn.execute(
                "INSERT INTO jobs (id, status, created_at, recurrence, sensor, data) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    "PENDING",
                    now,
                    metadata.get("recurrence"),
                    metadata.get("sensor"),
                    json.dumps(job),
                ),
            )
        logger.info(f"Created job {job_id}")
        return job_id

    def update_job_status(
        self,
        job_id: str,
        status: str,
        results: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT data FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if not row:
                logger.error(f"Job {job_id} not found for update.")
                return
            job = json.loads(row["data"])
            job["status"] = status
            job["updated_at"] = now
            if results:
                current = job.get("results", {})
                current.update(results)
                job["results"] = current
            if error:
                job["error"] = error
            conn.execute(
                "UPDATE jobs SET status=?, data=? WHERE id=?",
                (status, json.dumps(job), job_id),
            )
        logger.info(f"Updated job {job_id} → {status}")

    def update_last_run(self, job_id: str, last_run: datetime) -> None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT data FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if not row:
                return
            job = json.loads(row["data"])
            job["last_run"] = last_run.isoformat()
            conn.execute(
                "UPDATE jobs SET data=? WHERE id=?", (json.dumps(job), job_id)
            )

    def delete_job(self, job_id: str) -> bool:
        with get_connection(self.db_path) as conn:
            cur = conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        deleted = cur.rowcount > 0
        if deleted:
            logger.info(f"Deleted job {job_id}")
        return deleted

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT data FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        return json.loads(row["data"]) if row else None

    def list_jobs(self) -> List[Dict[str, Any]]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT data FROM jobs ORDER BY created_at DESC"
            ).fetchall()
        return [json.loads(r["data"]) for r in rows]

    def list_recurring_jobs(self) -> List[Dict[str, Any]]:
        """Return only jobs with a non-null recurrence (e.g. DAILY)."""
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT data FROM jobs WHERE recurrence IS NOT NULL "
                "ORDER BY created_at DESC"
            ).fetchall()
        return [json.loads(r["data"]) for r in rows]
