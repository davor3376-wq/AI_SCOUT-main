"""
Authentication endpoints.

  POST /auth/setup          First-run only: create the initial admin user.
  POST /auth/login          Exchange username+password (+ TOTP if MFA enabled) for a JWT.
  GET  /auth/me             Return the calling user's profile.
  POST /auth/rotate-key     Issue a new API key for the calling user.
  POST /auth/refresh        Exchange refresh token for new token pair.
  POST /auth/users          Admin only: create additional users.
  GET  /auth/users          Admin only: list all users.
  DELETE /auth/users/{id}   Admin only: deactivate a user.
  POST /auth/mfa/setup      Begin MFA enrollment — returns secret + QR code.
  POST /auth/mfa/confirm    Verify a TOTP code to activate MFA.
  DELETE /auth/mfa          Disable MFA for the calling user.
"""

import os
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.auth.manager import UserManager, WeakPasswordError, create_access_token, create_refresh_token, decode_token
from app.auth.mfa import verify_totp, totp_uri, totp_qr_png_b64
from app.api.dependencies import get_current_user, require_role
from app.api.usage_controller import UsageController

router = APIRouter(prefix="/auth", tags=["auth"])
_limiter = Limiter(key_func=get_remote_address)

DB_PATH = os.environ.get("SCOUT_DB_PATH", "jobs.db")


def _um() -> UserManager:
    return UserManager(db_path=DB_PATH)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "analyst"
    email: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    role: str
    mfa_required: bool = False


class RefreshRequest(BaseModel):
    refresh_token: str


class MFAConfirmRequest(BaseModel):
    secret: str
    totp_code: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/setup", status_code=201)
@_limiter.limit("3/minute")
async def setup_first_admin(request: Request, body: UserCreate):
    """
    One-time endpoint: bootstrap the first admin account.
    Returns 409 if any users already exist.
    """
    um = _um()
    if um.count_users() > 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Setup already complete. Use POST /auth/users to add more accounts.",
        )
    body.role = "admin"  # first user is always admin
    try:
        user = um.create_user(body.username, body.password, body.role, body.email)
    except WeakPasswordError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "api_key": user["api_key"],
        "message": "Admin created. Store your api_key — it will not be shown again.",
    }


@router.post("/login", response_model=TokenResponse)
@_limiter.limit("5/minute")
async def login(
    request: Request,
    form: OAuth2PasswordRequestForm = Depends(),
    totp_code: Optional[str] = Form(default=None),
):
    um = _um()
    user = um.authenticate(form.username, form.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if user.get("mfa_enabled") and user.get("mfa_secret"):
        if not totp_code:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="MFA code required. Include totp_code in your login request.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not verify_totp(user["mfa_secret"], totp_code):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid MFA code",
                headers={"WWW-Authenticate": "Bearer"},
            )
    token_data = {"sub": user["username"], "role": user["role"]}
    access = create_access_token(token_data)
    refresh = create_refresh_token(token_data)
    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
        "role": user["role"],
        "mfa_required": bool(user.get("mfa_enabled")),
    }


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    return {k: v for k, v in current_user.items() if k not in ("hashed_pw", "api_key")}


@router.get("/credits")
async def credits(current_user: dict = Depends(get_current_user)):
    """
    Return the calling user's credit balance, daily surface quota,
    and the 10 most recent usage events.

    Accessible to all authenticated roles (viewer / analyst / admin).
    """
    uc = UsageController(db_path=DB_PATH)
    username = current_user["username"]

    # Lazily provision a profile row on first call so existing users
    # who pre-date the governance layer are handled gracefully.
    uc.ensure_profile(current_user["id"], username)

    profile = uc.get_profile(username)
    return {
        "username":             username,
        "credit_balance":       profile["credit_balance"],
        "daily_surface_cap_km2": profile["daily_surface_cap_km2"],
        "daily_surface_used_km2": uc.get_daily_surface_used(username),
        "recent_usage":         uc.get_usage_summary(username, limit=10),
    }


@router.post("/refresh", response_model=TokenResponse)
@_limiter.limit("10/minute")
async def refresh_token(request: Request, body: RefreshRequest):
    """Exchange a valid refresh token for a new access + refresh token pair."""
    payload = decode_token(body.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    um = _um()
    user = um.get_by_username(payload.get("sub", ""))
    if not user or not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated",
        )
    token_data = {"sub": user["username"], "role": user["role"]}
    return {
        "access_token": create_access_token(token_data),
        "refresh_token": create_refresh_token(token_data),
        "token_type": "bearer",
        "role": user["role"],
    }


@router.post("/rotate-key")
async def rotate_key(current_user: dict = Depends(get_current_user)):
    new_key = _um().rotate_api_key(current_user["username"])
    return {"api_key": new_key}


@router.post("/users", status_code=201)
async def create_user(
    body: UserCreate,
    _admin: dict = Depends(require_role("admin")),
):
    um = _um()
    try:
        user = um.create_user(body.username, body.password, body.role, body.email)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "api_key": user["api_key"],
    }


@router.get("/users")
async def list_users(_admin: dict = Depends(require_role("admin"))):
    return _um().list_users()


@router.delete("/users/{user_id}", status_code=204)
async def deactivate_user(
    user_id: str,
    _admin: dict = Depends(require_role("admin")),
):
    um = _um()
    if not um.get_by_id(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    um.deactivate_user(user_id)


# ---------------------------------------------------------------------------
# MFA endpoints
# ---------------------------------------------------------------------------

@router.post("/mfa/setup", status_code=200)
async def mfa_setup(current_user: dict = Depends(get_current_user)):
    """
    Begin MFA enrollment for the authenticated user.
    Returns the TOTP secret, an otpauth:// URI, and a base64 QR-code PNG.
    MFA is NOT yet active — call POST /auth/mfa/confirm to activate it.
    """
    um = _um()
    secret = um.setup_mfa(current_user["username"])
    uri = totp_uri(secret, current_user["username"])
    qr_b64 = totp_qr_png_b64(secret, current_user["username"])
    return {
        "secret": secret,
        "totp_uri": uri,
        "qr_png_base64": qr_b64,
        "message": (
            "Scan the QR code with your authenticator app, then call "
            "POST /auth/mfa/confirm with the generated code to activate MFA."
        ),
    }


@router.post("/mfa/confirm", status_code=200)
@_limiter.limit("5/minute")
async def mfa_confirm(
    request: Request,
    body: MFAConfirmRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Verify a TOTP code and activate MFA for the authenticated user.
    The *secret* in the request body must match the one returned by /mfa/setup.
    """
    if not verify_totp(body.secret, body.totp_code):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid TOTP code — check your authenticator app clock.",
        )
    _um().enable_mfa(current_user["username"], body.secret)
    return {"mfa_enabled": True, "message": "MFA is now active on your account."}


@router.delete("/mfa", status_code=200)
async def mfa_disable(current_user: dict = Depends(get_current_user)):
    """
    Disable MFA for the authenticated user and clear the stored secret.
    Admins may disable MFA for any user via DELETE /auth/users/{id}/mfa.
    """
    _um().disable_mfa(current_user["username"])
    return {"mfa_enabled": False, "message": "MFA has been disabled."}
