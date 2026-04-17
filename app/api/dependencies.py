"""
Shared FastAPI dependencies for auth and role enforcement.

Auth resolution order (first match wins):
  1. Bearer JWT token in Authorization header
  2. X-API-Key header (per-user key prefixed sk-)
  3. Legacy SCOUT_API_KEY env var (treated as admin — backward compat)
  4. Dev-open mode: if SCOUT_SECRET_KEY is unset, every request is admitted
     as a synthetic "dev/admin" user and a warning is logged once.

Role hierarchy:  viewer < analyst < admin
"""

import os
import logging
from typing import Dict, Any, Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader

from app.auth.manager import UserManager, decode_token
from app.core.config import ENVIRONMENT

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("SCOUT_DB_PATH", "jobs.db")
LEGACY_API_KEY = os.environ.get("SCOUT_API_KEY", "")


def _secret_key_set() -> bool:
    return bool(os.environ.get("SCOUT_SECRET_KEY", ""))

_dev_mode_warned = False

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Synthetic user returned in dev-open mode
_DEV_USER: Dict[str, Any] = {
    "id": "dev",
    "username": "dev",
    "role": "admin",
    "is_active": 1,
    "api_key": "",
}

ROLE_ORDER = {"viewer": 0, "analyst": 1, "admin": 2}


async def get_current_user(
    token: Optional[str] = Security(oauth2_scheme),
    x_api_key: Optional[str] = Security(api_key_header),
) -> Dict[str, Any]:
    global _dev_mode_warned
    um = UserManager(db_path=DB_PATH)

    # 1. JWT Bearer token
    if token:
        payload = decode_token(token)
        if payload:
            user = um.get_by_username(payload.get("sub", ""))
            if user and user["is_active"]:
                return user

    # 2. Per-user API key (sk- prefix)
    if x_api_key and x_api_key.startswith("sk-"):
        user = um.get_by_api_key(x_api_key)
        if user:
            return user
        # API key was provided but is invalid - reject even in dev mode
        logger.warning(f"Invalid API key attempt: {x_api_key[:12]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 3. Legacy single shared API key (backward compat — admin level)
    if LEGACY_API_KEY and x_api_key == LEGACY_API_KEY:
        return {**_DEV_USER, "username": "legacy-key", "api_key": LEGACY_API_KEY}

    # 4. Dev-open mode — no secret key configured AND no credentials were provided
    if not _secret_key_set() and not x_api_key:
        if ENVIRONMENT == "production":
            # Never silently admit requests in production without a secret key.
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Server misconfiguration: SCOUT_SECRET_KEY is not set.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not _dev_mode_warned:
            logger.warning(
                "SCOUT_SECRET_KEY is not set — running in dev-open mode. "
                "All requests are admitted as admin. Set SCOUT_SECRET_KEY to enable auth."
            )
            _dev_mode_warned = True
        return _DEV_USER

    # Debug logging for auth failures
    logger.warning(
        f"Auth failed: token={'present' if token else 'missing'}, "
        f"api_key={'present' if x_api_key else 'missing'}, "
        f"legacy_key_set={'yes' if LEGACY_API_KEY else 'no'}, "
        f"secret_key_set={'yes' if _secret_key_set() else 'no'}"
    )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_role(*roles: str):
    """
    Dependency factory.  Usage:
        @app.get("/admin/...", dependencies=[Depends(require_role("admin"))])
    or:
        async def endpoint(user = Depends(require_role("admin", "analyst"))):
    """
    min_level = min(ROLE_ORDER.get(r, 99) for r in roles)

    async def _check(user: Dict[str, Any] = Depends(get_current_user)):
        if ROLE_ORDER.get(user.get("role", ""), -1) < min_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of: {list(roles)}",
            )
        return user

    return _check
