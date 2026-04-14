"""Initial schema — all core tables.

Creates:
  users             — authentication (manager.py)
  jobs              — mission lifecycle (job_manager.py)
  audit_logs        — immutable action log (audit.py)
  client_profiles   — per-client credit & geofence (usage_controller.py)
  credit_ledger     — immutable credit ledger (usage_controller.py)
  daily_surface_log — daily km² consumption (usage_controller.py)
  usage_log         — per-job governance audit (usage_controller.py)

Revision ID: 0001
Revises:
Create Date: 2026-04-14 00:00:00.000000
"""

from typing import Sequence, Union
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          TEXT PRIMARY KEY,
            username    TEXT UNIQUE NOT NULL,
            email       TEXT UNIQUE,
            hashed_pw   TEXT NOT NULL,
            role        TEXT NOT NULL DEFAULT 'analyst',
            api_key     TEXT UNIQUE NOT NULL,
            created_at  TEXT NOT NULL,
            is_active   INTEGER NOT NULL DEFAULT 1
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key)"
    )

    op.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id          TEXT PRIMARY KEY,
            status      TEXT NOT NULL DEFAULT 'PENDING',
            created_at  TEXT NOT NULL,
            recurrence  TEXT,
            sensor      TEXT,
            data        TEXT NOT NULL
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC)"
    )

    op.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            username    TEXT,
            action      TEXT NOT NULL,
            resource    TEXT,
            details     TEXT,
            ip_address  TEXT,
            status      TEXT NOT NULL DEFAULT 'ok'
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_logs(timestamp DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs(username)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action)"
    )

    op.execute("""
        CREATE TABLE IF NOT EXISTS client_profiles (
            user_id               TEXT PRIMARY KEY,
            username              TEXT UNIQUE NOT NULL,
            credit_balance        REAL NOT NULL DEFAULT 0.0,
            daily_surface_cap_km2 REAL NOT NULL DEFAULT 10000.0,
            licensed_geofence     TEXT
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS credit_ledger (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    TEXT    NOT NULL,
            job_id     TEXT    NOT NULL,
            entry_type TEXT    NOT NULL,
            amount     REAL    NOT NULL,
            note       TEXT,
            created_at TEXT    NOT NULL
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_ledger_user "
        "ON credit_ledger(user_id, created_at DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_ledger_job ON credit_ledger(job_id)"
    )

    op.execute("""
        CREATE TABLE IF NOT EXISTS daily_surface_log (
            user_id          TEXT NOT NULL,
            date             TEXT NOT NULL,
            surface_used_km2 REAL NOT NULL DEFAULT 0.0,
            PRIMARY KEY (user_id, date)
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS usage_log (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id              TEXT    NOT NULL UNIQUE,
            client_id           TEXT    NOT NULL,
            username            TEXT    NOT NULL,
            sensor              TEXT    NOT NULL,
            bbox_area_km2       REAL    NOT NULL,
            temporal_depth_days INTEGER NOT NULL,
            credits_consumed    REAL    NOT NULL DEFAULT 0.0,
            timestamp           TEXT    NOT NULL,
            status              TEXT    NOT NULL DEFAULT 'PENDING'
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_usage_client "
        "ON usage_log(client_id, timestamp DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_usage_job ON usage_log(job_id)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS usage_log")
    op.execute("DROP TABLE IF EXISTS daily_surface_log")
    op.execute("DROP TABLE IF EXISTS credit_ledger")
    op.execute("DROP TABLE IF EXISTS client_profiles")
    op.execute("DROP TABLE IF EXISTS audit_logs")
    op.execute("DROP TABLE IF EXISTS jobs")
    op.execute("DROP TABLE IF EXISTS users")
