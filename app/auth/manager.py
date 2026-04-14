"""
User & token management for AI SCOUT.

Two auth paths are supported:
  1. JWT Bearer token  — issued by POST /auth/login
  2. Per-user API key  — prefixed sk-, issued on user creation / rotation

Security model:
  - SCOUT_SECRET_KEY **must** be set in production.
  - If it is not set the system runs in dev-open mode (all requests are
    treated as the built-in "dev/admin" user) and a warning is logged on
    every startup.
  - Passwords are bcrypt-hashed; keys are 256-bit URL-safe random strings.

Roles (least → most privilege):
  viewer   — read-only (GET /jobs, GET /tiles, download PDF)
  analyst  — viewer + launch missions
  admin    — analyst + manage users, view audit log, trigger cleanup
"""

import re
import uuid
import secrets
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import bcrypt as _bcrypt
from jose import JWTError, jwt

from app.core.config import (
    SECRET_KEY,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
    MIN_PASSWORD_LENGTH,
    SCOUT_DB_PATH,
)
from app.core.database import get_connection, init_schema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ALGORITHM = "HS256"

ROLES = {"viewer", "analyst", "admin"}


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

class WeakPasswordError(ValueError):
    """Raised when a password does not meet complexity requirements."""


def validate_password(password: str) -> None:
    """
    Enforce minimum password complexity.

    Rules:
      - At least MIN_PASSWORD_LENGTH characters (default 10).
      - At least one uppercase letter.
      - At least one lowercase letter.
      - At least one digit.

    Raises WeakPasswordError with a human-readable message on failure.
    """
    issues: list[str] = []
    if len(password) < MIN_PASSWORD_LENGTH:
        issues.append(f"at least {MIN_PASSWORD_LENGTH} characters")
    if not re.search(r"[A-Z]", password):
        issues.append("an uppercase letter")
    if not re.search(r"[a-z]", password):
        issues.append("a lowercase letter")
    if not re.search(r"\d", password):
        issues.append("a digit")
    if issues:
        raise WeakPasswordError(
            "Password must contain " + ", ".join(issues) + "."
        )


def _hash_password(password: str) -> str:
    return _bcrypt.hashpw(password.encode("utf-8"), _bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, hashed: str) -> bool:
    return _bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_AUTH_SCHEMA = """
    CREATE TABLE IF NOT EXISTS users (
        id          TEXT PRIMARY KEY,
        username    TEXT UNIQUE NOT NULL,
        email       TEXT UNIQUE,
        hashed_pw   TEXT NOT NULL,
        role        TEXT NOT NULL DEFAULT 'analyst',
        api_key     TEXT UNIQUE NOT NULL,
        created_at  TEXT NOT NULL,
        is_active   INTEGER NOT NULL DEFAULT 1,
        mfa_secret  TEXT,
        mfa_enabled INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_users_username
        ON users(username);
    CREATE INDEX IF NOT EXISTS idx_users_api_key
        ON users(api_key);
"""


def init_auth_db(db_path: str) -> None:
    init_schema(_AUTH_SCHEMA, db_path)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------
def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    if not SECRET_KEY:
        raise RuntimeError("SCOUT_SECRET_KEY is not set — cannot issue tokens.")
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Issue a long-lived refresh token (separate `type` claim prevents misuse as access token)."""
    if not SECRET_KEY:
        raise RuntimeError("SCOUT_SECRET_KEY is not set — cannot issue tokens.")
    payload = data.copy()
    payload["type"] = "refresh"
    payload["exp"] = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    if not SECRET_KEY:
        return None
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


# ---------------------------------------------------------------------------
# UserManager
# ---------------------------------------------------------------------------
class UserManager:
    def __init__(self, db_path: str = SCOUT_DB_PATH):
        self.db_path = db_path
        init_auth_db(db_path)

    # ---- reads ----

    def count_users(self) -> int:
        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()
        return row["n"]

    def get_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username=?", (username,)
            ).fetchone()
        return dict(row) if row else None

    def get_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id=?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_by_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE api_key=? AND is_active=1", (api_key,)
            ).fetchone()
        return dict(row) if row else None

    def list_users(self) -> list:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, username, email, role, created_at, is_active FROM users"
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- writes ----

    def create_user(
        self,
        username: str,
        password: str,
        role: str = "analyst",
        email: Optional[str] = None,
    ) -> Dict[str, Any]:
        if role not in ROLES:
            raise ValueError(f"Invalid role '{role}'. Must be one of {ROLES}.")
        validate_password(password)
        user_id = str(uuid.uuid4())
        hashed = _hash_password(password)
        api_key = f"sk-{secrets.token_urlsafe(32)}"
        now = datetime.now(timezone.utc).isoformat()
        with get_connection(self.db_path) as conn:
            conn.execute(
                "INSERT INTO users (id, username, email, hashed_pw, role, api_key, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (user_id, username, email, hashed, role, api_key, now),
            )
        logger.info(f"Created user '{username}' with role '{role}'")
        return self.get_by_username(username)  # type: ignore[return-value]

    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        user = self.get_by_username(username)
        if not user or not user["is_active"]:
            return None
        if not _verify_password(password, user["hashed_pw"]):
            return None
        return user

    def deactivate_user(self, user_id: str) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                "UPDATE users SET is_active=0 WHERE id=?", (user_id,)
            )

    def rotate_api_key(self, username: str) -> str:
        new_key = f"sk-{secrets.token_urlsafe(32)}"
        with get_connection(self.db_path) as conn:
            conn.execute(
                "UPDATE users SET api_key=? WHERE username=?", (new_key, username)
            )
        logger.info(f"Rotated API key for '{username}'")
        return new_key

    def rotate_all_api_keys(self) -> int:
        """
        Rotate every active user's API key at once.
        Used by the emergency-wipe endpoint to invalidate stolen credentials.
        Returns the number of keys rotated.
        """
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT username FROM users WHERE is_active=1"
            ).fetchall()
            count = 0
            for row in rows:
                new_key = f"sk-{secrets.token_urlsafe(32)}"
                conn.execute(
                    "UPDATE users SET api_key=? WHERE username=?",
                    (new_key, row["username"]),
                )
                count += 1
        logger.warning(f"Emergency key rotation: rotated {count} API keys")
        return count

    # ---- MFA ----

    def setup_mfa(self, username: str) -> str:
        """
        Generate a fresh TOTP secret and persist it (not yet *enabled*).
        The user must confirm with a valid code before MFA is enforced.
        Returns the raw base32 secret.
        """
        from app.auth.mfa import generate_secret
        secret = generate_secret()
        with get_connection(self.db_path) as conn:
            conn.execute(
                "UPDATE users SET mfa_secret=?, mfa_enabled=0 WHERE username=?",
                (secret, username),
            )
        logger.info(f"MFA secret generated for '{username}' (not yet enabled)")
        return secret

    def enable_mfa(self, username: str, secret: str) -> None:
        """
        Persist *secret* and set mfa_enabled=1 for *username*.
        Caller must have already verified a valid TOTP code.
        """
        with get_connection(self.db_path) as conn:
            conn.execute(
                "UPDATE users SET mfa_secret=?, mfa_enabled=1 WHERE username=?",
                (secret, username),
            )
        logger.info(f"MFA enabled for '{username}'")

    def disable_mfa(self, username: str) -> None:
        """Turn off MFA and clear the stored secret for *username*."""
        with get_connection(self.db_path) as conn:
            conn.execute(
                "UPDATE users SET mfa_secret=NULL, mfa_enabled=0 WHERE username=?",
                (username,),
            )
        logger.info(f"MFA disabled for '{username}'")
