"""Add MFA columns to users table.

Adds:
  mfa_secret  TEXT    — base32 TOTP secret (NULL until user sets up MFA)
  mfa_enabled INTEGER — 0 = disabled (default), 1 = enforced on login

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-14 00:00:00.000000
"""

from typing import Sequence, Union
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE users ADD COLUMN mfa_secret TEXT")
    op.execute("ALTER TABLE users ADD COLUMN mfa_enabled INTEGER NOT NULL DEFAULT 0")


def downgrade() -> None:
    # SQLite does not support DROP COLUMN directly before 3.35.
    # For PostgreSQL we can drop cleanly; for SQLite we recreate the table.
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TABLE users DROP COLUMN IF EXISTS mfa_secret")
        op.execute("ALTER TABLE users DROP COLUMN IF EXISTS mfa_enabled")
    else:
        op.execute("""
            CREATE TABLE users_backup AS
            SELECT id, username, email, hashed_pw, role, api_key, created_at, is_active
            FROM users
        """)
        op.execute("DROP TABLE users")
        op.execute("""
            CREATE TABLE users (
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
        op.execute("INSERT INTO users SELECT * FROM users_backup")
        op.execute("DROP TABLE users_backup")
        op.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        op.execute("CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key)")
