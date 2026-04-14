"""
Alembic environment — resolves the database URL from the same env vars
that the application uses (DATABASE_URL for PostgreSQL, or
sqlite:///<SCOUT_DB_PATH> for SQLite).

Raw-SQL migrations (op.execute) are used throughout to stay consistent
with the existing codebase and avoid coupling to an ORM metadata layer.
"""

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# ── Load .env (if present) before reading env vars ─────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Alembic Config object ──────────────────────────────────────────────────
config = context.config

# Interpret the config file for Python logging if present.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Resolve the database URL from env vars ─────────────────────────────────
_database_url = os.environ.get("DATABASE_URL", "")
_db_path = os.environ.get("SCOUT_DB_PATH", "jobs.db")

if _database_url.startswith(("postgresql://", "postgres://")):
    _url = _database_url
else:
    # Normalise to absolute path so Alembic can be run from any CWD.
    _abs = os.path.abspath(_db_path)
    _url = f"sqlite:///{_abs}"

config.set_main_option("sqlalchemy.url", _url)

# No ORM metadata — we use raw SQL in migration scripts.
target_metadata = None


# ── Migration runners ──────────────────────────────────────────────────────

def run_migrations_offline() -> None:
    """Run migrations without a live DB connection (generates SQL script)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
