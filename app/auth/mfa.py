"""
TOTP-based MFA helpers for AI SCOUT.

Uses RFC 6238 TOTP (same algorithm as Google Authenticator, Authy, etc.).
Secrets are base32-encoded and stored in the users table.

Usage:
  secret = generate_secret()
  uri    = totp_uri(secret, username="alice", issuer="AI SCOUT")
  ok     = verify_totp(secret, user_supplied_code)
"""

import base64
import io
from typing import Optional

import pyotp
import qrcode


# Allow ±1 time-step (30 s each) to handle clock drift on field devices.
_VALID_WINDOW = 1


def generate_secret() -> str:
    """Return a fresh random base32 TOTP secret."""
    return pyotp.random_base32()


def verify_totp(secret: str, code: str) -> bool:
    """
    Verify a 6-digit TOTP code against *secret*.
    Accepts the current period and one period either side of it.
    Returns False for empty / malformed codes without raising.
    """
    if not code or not code.strip().isdigit():
        return False
    totp = pyotp.TOTP(secret)
    return totp.verify(code.strip(), valid_window=_VALID_WINDOW)


def totp_uri(secret: str, username: str, issuer: str = "AI SCOUT") -> str:
    """
    Return an otpauth:// URI suitable for encoding into a QR code.
    Most authenticator apps accept this format.
    """
    totp = pyotp.TOTP(secret)
    return totp.provisioning_uri(name=username, issuer_name=issuer)


def totp_qr_png_b64(secret: str, username: str, issuer: str = "AI SCOUT") -> str:
    """
    Render the otpauth:// URI as a QR code PNG and return it as a
    base64-encoded string (suitable for embedding in a data: URI).
    """
    uri = totp_uri(secret, username, issuer)
    img = qrcode.make(uri)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
