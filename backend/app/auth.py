"""Authentication helpers — password hashing and signed tokens.

Implemented with the Python standard library only:
  * Passwords: PBKDF2-HMAC-SHA256 with a per-user random salt.
  * Tokens:   compact HMAC-SHA256 signed tokens (JWT-like) carrying the
              user id and an expiry. No external JWT library required.

Set GAMESENSE_SECRET_KEY in backend/.env for a stable signing key across
restarts. If unset, a key is generated at import time (tokens then become
invalid on restart, which is fine for local dev).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Optional

from fastapi import Depends, Header, HTTPException

from app import db

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_SECRET = os.environ.get("GAMESENSE_SECRET_KEY", "").strip() or secrets.token_hex(32)
_TOKEN_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days
_PBKDF2_ROUNDS = 200_000


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${_PBKDF2_ROUNDS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, rounds, salt_hex, digest_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(rounds)
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def create_token(user_id: str) -> str:
    payload = {"sub": user_id, "exp": int(time.time()) + _TOKEN_TTL_SECONDS}
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = _b64e(hmac.new(_SECRET.encode(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def decode_token(token: str) -> Optional[str]:
    """Return the user id if the token is valid and unexpired, else None."""
    try:
        body, sig = token.split(".")
        expected = _b64e(hmac.new(_SECRET.encode(), body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(_b64d(body))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload.get("sub")
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------
def user_id_from_header(authorization: Optional[str]) -> Optional[str]:
    """Parse a 'Bearer <token>' header into a user id (or None)."""
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return decode_token(parts[1])


def get_user_row(user_id: str) -> Optional[dict]:
    with db.cursor() as cur:
        cur.execute(
            "SELECT id, email, display_name, avatar_url, created_at FROM users WHERE id = ?",
            (user_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def current_user(authorization: Optional[str] = Header(default=None)) -> dict:
    """Required-auth dependency: 401 if no valid token."""
    user_id = user_id_from_header(authorization)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required.")
    user = get_user_row(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists.")
    return user


def optional_user(authorization: Optional[str] = Header(default=None)) -> Optional[dict]:
    """Optional-auth dependency: returns the user dict or None."""
    user_id = user_id_from_header(authorization)
    if not user_id:
        return None
    return get_user_row(user_id)
