"""API routes for accounts, the global leaderboard, and leagues."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from app import auth, db
from app.services import achievements, connections, scoring
from app.services.storage import MEDIA_ROOT

router = APIRouter()

_ALLOWED_AVATAR_EXT = {".jpg", ".jpeg", ".png", ".webp"}
_MAX_AVATAR_BYTES = 5 * 1024 * 1024


def _valid_email(value: str) -> str:
    value = (value or "").strip().lower()
    if "@" not in value or "." not in value.split("@")[-1] or len(value) < 5:
        raise ValueError("A valid email address is required.")
    return value


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class SignupRequest(BaseModel):
    email: str
    display_name: str = Field(min_length=1, max_length=40)
    password: str = Field(min_length=6, max_length=128)

    @field_validator("email")
    @classmethod
    def _check_email(cls, v: str) -> str:
        return _valid_email(v)


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user: dict


class CreateLeagueRequest(BaseModel):
    name: str = Field(min_length=1, max_length=60)


class JoinLeagueRequest(BaseModel):
    invite_code: str = Field(min_length=4, max_length=12)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
@router.post("/auth/signup", response_model=AuthResponse)
def signup(payload: SignupRequest) -> AuthResponse:
    email = payload.email.lower().strip()
    with db.cursor() as cur:
        cur.execute("SELECT 1 FROM users WHERE email = ?", (email,))
        if cur.fetchone():
            raise HTTPException(status_code=409, detail="An account with that email already exists.")
        user_id = str(uuid.uuid4())
        try:
            pw_hash = auth.hash_password(payload.password)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        cur.execute(
            "INSERT INTO users (id, email, display_name, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, email, payload.display_name.strip(), pw_hash, datetime.now(timezone.utc).isoformat()),
        )
    user = auth.get_user_row(user_id)
    return AuthResponse(token=auth.create_token(user_id), user=user)


@router.post("/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest) -> AuthResponse:
    email = payload.email.lower().strip()
    with db.cursor() as cur:
        cur.execute(
            "SELECT id, email, display_name, avatar_url, password_hash, created_at FROM users WHERE email = ?",
            (email,),
        )
        row = cur.fetchone()
    if not row or not auth.verify_password(payload.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    user = {k: row[k] for k in ("id", "email", "display_name", "avatar_url", "created_at")}
    return AuthResponse(token=auth.create_token(row["id"]), user=user)


@router.get("/auth/me")
def me(user: dict = Depends(auth.current_user)) -> dict:
    return user


# ---------------------------------------------------------------------------
# Session history
# ---------------------------------------------------------------------------
@router.get("/history")
def history(user: dict = Depends(auth.current_user)) -> dict:
    return {"uploads": scoring.uploads_for_user(user["id"])}


# ---------------------------------------------------------------------------
# Profile (career stats + badges + recent sessions)
# ---------------------------------------------------------------------------
@router.get("/profile")
def profile(user: dict = Depends(auth.current_user)) -> dict:
    data = achievements.profile(user["id"])
    data["user"] = user
    data["recent_sessions"] = scoring.uploads_for_user(user["id"])[:5]
    data["follow_counts"] = connections.follow_counts(user["id"])
    return data


# ---------------------------------------------------------------------------
# Profile picture
# ---------------------------------------------------------------------------
@router.post("/profile/avatar")
async def upload_avatar(
    file: UploadFile = File(...), user: dict = Depends(auth.current_user)
) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_AVATAR_EXT:
        raise HTTPException(status_code=400, detail="Use a JPG, PNG, or WEBP image.")
    data = await file.read()
    if len(data) > _MAX_AVATAR_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large (max 5 MB).")

    avatars_dir = MEDIA_ROOT / "avatars"
    avatars_dir.mkdir(parents=True, exist_ok=True)
    # Remove any previous avatar for this user (different extension).
    for old in avatars_dir.glob(f"{user['id']}.*"):
        try:
            old.unlink()
        except OSError:
            pass
    (avatars_dir / f"{user['id']}{ext}").write_bytes(data)

    # Cache-bust so the new image shows immediately.
    url = f"/media/avatars/{user['id']}{ext}?v={int(time.time())}"
    with db.cursor() as cur:
        cur.execute("UPDATE users SET avatar_url = ? WHERE id = ?", (url, user["id"]))
    return {"avatar_url": url}


# ---------------------------------------------------------------------------
# Following
# ---------------------------------------------------------------------------
@router.get("/users/search")
def search_users(q: str = "", user: dict = Depends(auth.current_user)) -> dict:
    return {"users": connections.search_users(user["id"], q)}


@router.post("/follow/{target_id}")
def follow(target_id: str, user: dict = Depends(auth.current_user)) -> dict:
    try:
        connections.follow(user["id"], target_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"status": "following", "target_id": target_id}


@router.post("/unfollow/{target_id}")
def unfollow(target_id: str, user: dict = Depends(auth.current_user)) -> dict:
    connections.unfollow(user["id"], target_id)
    return {"status": "unfollowed", "target_id": target_id}


@router.get("/following")
def following(user: dict = Depends(auth.current_user)) -> dict:
    return {
        "following": connections.list_following(user["id"]),
        "followers": connections.list_followers(user["id"]),
        "counts": connections.follow_counts(user["id"]),
    }


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------
@router.get("/leaderboard")
def leaderboard(
    sort: str = "points",
    scope: str = "global",
    authorization: str | None = Header(default=None),
) -> dict:
    viewer_id = auth.user_id_from_header(authorization)
    entries = scoring.global_leaderboard(sort, viewer_id=viewer_id, scope=scope)
    return {"sort": sort, "scope": scope, "entries": entries}


# ---------------------------------------------------------------------------
# Leagues
# ---------------------------------------------------------------------------
@router.post("/leagues")
def create_league(payload: CreateLeagueRequest, user: dict = Depends(auth.current_user)) -> dict:
    try:
        return scoring.create_league(user["id"], payload.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/leagues")
def my_leagues(user: dict = Depends(auth.current_user)) -> dict:
    return {"leagues": scoring.leagues_for_user(user["id"])}


@router.get("/leagues/preview")
def league_preview(code: str) -> dict:
    """Public preview of a league by invite code, for share links."""
    code = (code or "").strip().upper()
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT l.id, l.name, l.invite_code,
                   (SELECT COUNT(*) FROM league_members m WHERE m.league_id = l.id) AS member_count
            FROM leagues l WHERE l.invite_code = ?
            """,
            (code,),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No league found for that invite code.")
    return dict(row)


@router.post("/leagues/join")
def join_league(payload: JoinLeagueRequest, user: dict = Depends(auth.current_user)) -> dict:
    try:
        return scoring.join_league_by_code(user["id"], payload.invite_code)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/leagues/{league_id}")
def league_detail(league_id: str, sort: str = "points", user: dict = Depends(auth.current_user)) -> dict:
    league = scoring.get_league(league_id)
    if not league:
        raise HTTPException(status_code=404, detail="League not found.")
    if not scoring.is_member(league_id, user["id"]):
        raise HTTPException(status_code=403, detail="You are not a member of this league.")
    return {"league": league, "sort": sort, "entries": scoring.league_leaderboard(league_id, sort)}


@router.post("/leagues/{league_id}/leave")
def leave_league(league_id: str, user: dict = Depends(auth.current_user)) -> dict:
    if not scoring.get_league(league_id):
        raise HTTPException(status_code=404, detail="League not found.")
    scoring.leave_league(league_id, user["id"])
    return {"status": "left", "league_id": league_id}
