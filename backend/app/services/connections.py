"""Following graph: one-directional follow relationships + user search."""

from __future__ import annotations

from datetime import datetime, timezone

from app import db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def follow(follower_id: str, following_id: str) -> None:
    if follower_id == following_id:
        raise ValueError("You can't follow yourself.")
    with db.cursor() as cur:
        cur.execute("SELECT 1 FROM users WHERE id = ?", (following_id,))
        if not cur.fetchone():
            raise LookupError("User not found.")
        cur.execute(
            "INSERT OR IGNORE INTO follows (follower_id, following_id, created_at) VALUES (?, ?, ?)",
            (follower_id, following_id, _now()),
        )


def unfollow(follower_id: str, following_id: str) -> None:
    with db.cursor() as cur:
        cur.execute(
            "DELETE FROM follows WHERE follower_id = ? AND following_id = ?",
            (follower_id, following_id),
        )


def is_following(follower_id: str, following_id: str) -> bool:
    with db.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM follows WHERE follower_id = ? AND following_id = ?",
            (follower_id, following_id),
        )
        return cur.fetchone() is not None


def follow_counts(user_id: str) -> dict:
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS c FROM follows WHERE follower_id = ?", (user_id,))
        following = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) AS c FROM follows WHERE following_id = ?", (user_id,))
        followers = cur.fetchone()["c"]
    return {"following": following, "followers": followers}


def following_ids(user_id: str) -> list[str]:
    with db.cursor() as cur:
        cur.execute("SELECT following_id FROM follows WHERE follower_id = ?", (user_id,))
        return [r["following_id"] for r in cur.fetchall()]


def _decorate(rows: list[dict], viewer_id: str) -> list[dict]:
    """Attach is_following + follower counts for the viewer's perspective."""
    following = set(following_ids(viewer_id))
    out = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "display_name": r["display_name"],
                "avatar_url": r.get("avatar_url"),
                "is_following": r["id"] in following,
                "is_self": r["id"] == viewer_id,
            }
        )
    return out


def search_users(viewer_id: str, query: str, limit: int = 20) -> list[dict]:
    query = (query or "").strip()
    if len(query) < 2:
        return []
    like = f"%{query}%"
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT id, display_name, avatar_url
            FROM users
            WHERE (display_name LIKE ? OR email LIKE ?) AND id != ?
            ORDER BY display_name COLLATE NOCASE
            LIMIT ?
            """,
            (like, like, viewer_id, limit),
        )
        rows = [dict(r) for r in cur.fetchall()]
    return _decorate(rows, viewer_id)


def list_following(viewer_id: str, of_user_id: str | None = None) -> list[dict]:
    target = of_user_id or viewer_id
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT u.id, u.display_name, u.avatar_url
            FROM follows f JOIN users u ON u.id = f.following_id
            WHERE f.follower_id = ?
            ORDER BY u.display_name COLLATE NOCASE
            """,
            (target,),
        )
        rows = [dict(r) for r in cur.fetchall()]
    return _decorate(rows, viewer_id)


def list_followers(viewer_id: str, of_user_id: str | None = None) -> list[dict]:
    target = of_user_id or viewer_id
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT u.id, u.display_name, u.avatar_url
            FROM follows f JOIN users u ON u.id = f.follower_id
            WHERE f.following_id = ?
            ORDER BY u.display_name COLLATE NOCASE
            """,
            (target,),
        )
        rows = [dict(r) for r in cur.fetchall()]
    return _decorate(rows, viewer_id)
