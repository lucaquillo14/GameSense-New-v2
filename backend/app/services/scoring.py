"""Points scoring + leaderboard / league queries.

Points model = upload activity + performance bonus.

  * Every completed analysis grants a flat participation award (BASE_POINTS).
  * On top of that, a performance bonus is added based on the metrics the
    clip produced, so faster / more powerful / cleaner-technique clips are
    worth more.

A user's total score is the sum of points across all their completed uploads.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app import db

BASE_POINTS = 100

# Bonus weights per metric. Tuned so a strong clip roughly doubles the base.
SPEED_WEIGHT = 3.0       # ~ 30 km/h sprint  -> +90
POWER_WEIGHT = 2.0       # ~ 90 km/h shot    -> +180
TECHNIQUE_WEIGHT = 1.5   # ~ 100 score       -> +150


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_points(max_speed_kmh: float, shot_power_kmh: float, technique_score: float) -> int:
    bonus = (
        max_speed_kmh * SPEED_WEIGHT
        + shot_power_kmh * POWER_WEIGHT
        + technique_score * TECHNIQUE_WEIGHT
    )
    return int(round(BASE_POINTS + bonus))


def extract_metrics(record: dict) -> dict:
    """Pull the three competition metrics out of a finished video record.

    Works across the three analysis modes — whichever metrics are present
    are captured; the rest default to 0.
    """
    speed = 0.0
    power = 0.0
    technique = 0.0

    results = record.get("results") or {}
    if isinstance(results, dict):
        # Metrics (speed mode)
        speed = float(results.get("max_speed_kmh") or results.get("top_speed_kmh") or 0.0)
        # ShotMetrics (shot power mode)
        power = float(results.get("peak_shot_speed_kmh") or 0.0)

    shooting = record.get("shooting_result") or {}
    if isinstance(shooting, dict):
        technique = float(shooting.get("technique_score") or 0.0)
        if not power:
            power = float(shooting.get("shot_power_kmh") or 0.0)

    return {
        "max_speed_kmh": round(speed, 2),
        "shot_power_kmh": round(power, 2),
        "technique_score": round(technique, 2),
    }


def record_upload_score(user_id: str, record: dict) -> bool:
    """Idempotently record a completed upload's score for a user.

    Returns True if a new row was inserted, False if it already existed.
    """
    video_id = record.get("video_id")
    if not video_id:
        return False

    with db.cursor() as cur:
        cur.execute("SELECT 1 FROM uploads WHERE video_id = ?", (video_id,))
        if cur.fetchone():
            return False

        metrics = extract_metrics(record)
        points = compute_points(
            metrics["max_speed_kmh"], metrics["shot_power_kmh"], metrics["technique_score"]
        )
        cur.execute(
            """
            INSERT INTO uploads
              (video_id, user_id, mode, filename, max_speed_kmh,
               shot_power_kmh, technique_score, points, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                video_id,
                user_id,
                record.get("mode"),
                record.get("filename"),
                metrics["max_speed_kmh"],
                metrics["shot_power_kmh"],
                metrics["technique_score"],
                points,
                _now(),
            ),
        )
    return True


def uploads_for_user(user_id: str) -> list[dict]:
    """Return a user's completed analyses, newest first (session history)."""
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT video_id, mode, filename, max_speed_kmh, shot_power_kmh,
                   technique_score, points, created_at
            FROM uploads
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        )
        return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Leaderboard queries
# ---------------------------------------------------------------------------
_LEADERBOARD_SQL = """
    SELECT
        u.id              AS user_id,
        u.display_name    AS display_name,
        u.avatar_url      AS avatar_url,
        COUNT(up.video_id)         AS uploads,
        COALESCE(SUM(up.points), 0)        AS total_points,
        COALESCE(MAX(up.max_speed_kmh), 0) AS best_speed_kmh,
        COALESCE(MAX(up.shot_power_kmh), 0) AS best_power_kmh,
        COALESCE(MAX(up.technique_score), 0) AS best_technique,
        s.tier AS tier
    FROM users u
    LEFT JOIN uploads up ON up.user_id = u.id
    LEFT JOIN subscriptions s ON s.user_id = u.id
    {where}
    GROUP BY u.id
"""

# Map a stored tier onto the public badge id shown on the leaderboard.
_TIER_BADGE = {"pro": "pro", "elite": "elite"}

_SORT_COLUMNS = {
    "points": "total_points",
    "speed": "best_speed_kmh",
    "power": "best_power_kmh",
    "technique": "best_technique",
    "uploads": "uploads",
}


def _rank_rows(rows: list[dict], sort_key: str) -> list[dict]:
    col = _SORT_COLUMNS.get(sort_key, "total_points")
    rows.sort(key=lambda r: (r[col], r["total_points"]), reverse=True)
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
        r["badge"] = _TIER_BADGE.get((r.pop("tier", None) or "free"))
    return rows


def global_leaderboard(sort: str = "points", viewer_id: str | None = None,
                        scope: str = "global") -> list[dict]:
    if scope == "following" and viewer_id:
        # Rank only the people the viewer follows, plus themselves.
        from app.services import connections

        ids = set(connections.following_ids(viewer_id)) | {viewer_id}
        placeholders = ",".join("?" for _ in ids)
        sql = _LEADERBOARD_SQL.format(where=f"WHERE u.id IN ({placeholders})")
        params = tuple(ids)
    else:
        sql = _LEADERBOARD_SQL.format(where="")
        params = ()
    with db.cursor() as cur:
        cur.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
    return _rank_rows(rows, sort)


def league_leaderboard(league_id: str, sort: str = "points") -> list[dict]:
    sql = _LEADERBOARD_SQL.format(
        where="WHERE u.id IN (SELECT user_id FROM league_members WHERE league_id = ?)"
    )
    with db.cursor() as cur:
        cur.execute(sql, (league_id,))
        rows = [dict(r) for r in cur.fetchall()]
    return _rank_rows(rows, sort)


# ---------------------------------------------------------------------------
# League management
# ---------------------------------------------------------------------------
def _gen_invite_code() -> str:
    # 6-char human-friendly code (no ambiguous chars)
    import secrets as _secrets

    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(_secrets.choice(alphabet) for _ in range(6))


def create_league(owner_id: str, name: str) -> dict:
    name = (name or "").strip()
    if not name:
        raise ValueError("League name is required.")
    league_id = str(uuid.uuid4())
    with db.cursor() as cur:
        # Ensure a unique invite code.
        for _ in range(10):
            code = _gen_invite_code()
            cur.execute("SELECT 1 FROM leagues WHERE invite_code = ?", (code,))
            if not cur.fetchone():
                break
        cur.execute(
            "INSERT INTO leagues (id, name, owner_id, invite_code, created_at) VALUES (?, ?, ?, ?, ?)",
            (league_id, name, owner_id, code, _now()),
        )
        cur.execute(
            "INSERT INTO league_members (league_id, user_id, joined_at) VALUES (?, ?, ?)",
            (league_id, owner_id, _now()),
        )
    return get_league(league_id)


def get_league(league_id: str) -> dict | None:
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT l.id, l.name, l.owner_id, l.invite_code, l.created_at,
                   (SELECT COUNT(*) FROM league_members m WHERE m.league_id = l.id) AS member_count
            FROM leagues l WHERE l.id = ?
            """,
            (league_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def join_league_by_code(user_id: str, code: str) -> dict:
    code = (code or "").strip().upper()
    with db.cursor() as cur:
        cur.execute("SELECT id FROM leagues WHERE invite_code = ?", (code,))
        row = cur.fetchone()
        if not row:
            raise LookupError("No league found for that invite code.")
        league_id = row["id"]
        cur.execute(
            "INSERT OR IGNORE INTO league_members (league_id, user_id, joined_at) VALUES (?, ?, ?)",
            (league_id, user_id, _now()),
        )
    return get_league(league_id)


def leagues_for_user(user_id: str) -> list[dict]:
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT l.id, l.name, l.owner_id, l.invite_code, l.created_at,
                   (SELECT COUNT(*) FROM league_members m WHERE m.league_id = l.id) AS member_count
            FROM leagues l
            JOIN league_members lm ON lm.league_id = l.id
            WHERE lm.user_id = ?
            ORDER BY l.created_at DESC
            """,
            (user_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def is_member(league_id: str, user_id: str) -> bool:
    with db.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM league_members WHERE league_id = ? AND user_id = ?",
            (league_id, user_id),
        )
        return cur.fetchone() is not None


def leave_league(league_id: str, user_id: str) -> None:
    with db.cursor() as cur:
        cur.execute(
            "DELETE FROM league_members WHERE league_id = ? AND user_id = ?",
            (league_id, user_id),
        )
