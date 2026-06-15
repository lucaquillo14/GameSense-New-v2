"""Badges & achievements.

Everything here is derived live from the `uploads` table, so a user's badges
are always consistent with their actual analyses — no separate achievement
table to keep in sync. Each badge declares a metric and a target; it is
"earned" once the user's best value for that metric reaches the target, and
progress toward locked badges is reported so the UI can show a progress bar.
"""

from __future__ import annotations

from app import db

# Metric keys map to columns produced by _career_stats below.
#   uploads        -> number of completed analyses
#   best_speed_kmh -> highest top speed across clips
#   best_power_kmh -> highest shot power across clips
#   best_technique -> highest technique score
#   total_points   -> lifetime points
#
# tier drives the badge colour in the UI (bronze / silver / gold).
# icon is a lucide-react icon name rendered on the frontend.
BADGES: list[dict] = [
    # --- Activity / uploads ---
    {"id": "first_upload", "name": "First Steps", "description": "Upload your first clip.",
     "category": "Activity", "tier": "bronze", "icon": "Footprints", "metric": "uploads", "target": 1, "unit": ""},
    {"id": "uploads_5", "name": "Warming Up", "description": "Analyse 5 clips.",
     "category": "Activity", "tier": "bronze", "icon": "Upload", "metric": "uploads", "target": 5, "unit": ""},
    {"id": "uploads_10", "name": "Regular", "description": "Analyse 10 clips.",
     "category": "Activity", "tier": "silver", "icon": "CalendarCheck", "metric": "uploads", "target": 10, "unit": ""},
    {"id": "uploads_25", "name": "Committed", "description": "Analyse 25 clips.",
     "category": "Activity", "tier": "gold", "icon": "Flame", "metric": "uploads", "target": 25, "unit": ""},
    {"id": "uploads_50", "name": "Veteran", "description": "Analyse 50 clips.",
     "category": "Activity", "tier": "gold", "icon": "Crown", "metric": "uploads", "target": 50, "unit": ""},

    # --- Speed ---
    {"id": "speed_20", "name": "Quick Feet", "description": "Hit 20 km/h top speed.",
     "category": "Speed", "tier": "bronze", "icon": "Gauge", "metric": "best_speed_kmh", "target": 20, "unit": "km/h"},
    {"id": "speed_25", "name": "Sprinter", "description": "Hit 25 km/h top speed.",
     "category": "Speed", "tier": "silver", "icon": "Gauge", "metric": "best_speed_kmh", "target": 25, "unit": "km/h"},
    {"id": "speed_30", "name": "Speed Demon", "description": "Hit 30 km/h top speed.",
     "category": "Speed", "tier": "gold", "icon": "Zap", "metric": "best_speed_kmh", "target": 30, "unit": "km/h"},
    {"id": "speed_35", "name": "Lightning", "description": "Hit 35 km/h top speed.",
     "category": "Speed", "tier": "gold", "icon": "Zap", "metric": "best_speed_kmh", "target": 35, "unit": "km/h"},

    # --- Shot power ---
    {"id": "power_60", "name": "Striker", "description": "Strike a shot at 60 km/h.",
     "category": "Shot power", "tier": "bronze", "icon": "Target", "metric": "best_power_kmh", "target": 60, "unit": "km/h"},
    {"id": "power_80", "name": "Cannon", "description": "Strike a shot at 80 km/h.",
     "category": "Shot power", "tier": "silver", "icon": "Target", "metric": "best_power_kmh", "target": 80, "unit": "km/h"},
    {"id": "power_100", "name": "Rocket Launcher", "description": "Strike a shot at 100 km/h.",
     "category": "Shot power", "tier": "gold", "icon": "Crosshair", "metric": "best_power_kmh", "target": 100, "unit": "km/h"},

    # --- Technique ---
    {"id": "tech_60", "name": "Clean Strike", "description": "Score 60 on technique.",
     "category": "Technique", "tier": "bronze", "icon": "Sparkles", "metric": "best_technique", "target": 60, "unit": ""},
    {"id": "tech_75", "name": "Technician", "description": "Score 75 on technique.",
     "category": "Technique", "tier": "silver", "icon": "Sparkles", "metric": "best_technique", "target": 75, "unit": ""},
    {"id": "tech_90", "name": "Maestro", "description": "Score 90 on technique.",
     "category": "Technique", "tier": "gold", "icon": "Award", "metric": "best_technique", "target": 90, "unit": ""},

    # --- Points ---
    {"id": "points_500", "name": "Rising Star", "description": "Earn 500 career points.",
     "category": "Points", "tier": "bronze", "icon": "Star", "metric": "total_points", "target": 500, "unit": "pts"},
    {"id": "points_2000", "name": "Pro", "description": "Earn 2,000 career points.",
     "category": "Points", "tier": "silver", "icon": "Trophy", "metric": "total_points", "target": 2000, "unit": "pts"},
    {"id": "points_5000", "name": "Legend", "description": "Earn 5,000 career points.",
     "category": "Points", "tier": "gold", "icon": "Medal", "metric": "total_points", "target": 5000, "unit": "pts"},
]


def career_stats(user_id: str) -> dict:
    """Aggregate a user's lifetime stats from their uploads."""
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT
                COUNT(*)                              AS uploads,
                COALESCE(SUM(points), 0)              AS total_points,
                COALESCE(MAX(max_speed_kmh), 0)       AS best_speed_kmh,
                COALESCE(MAX(shot_power_kmh), 0)      AS best_power_kmh,
                COALESCE(MAX(technique_score), 0)     AS best_technique
            FROM uploads
            WHERE user_id = ?
            """,
            (user_id,),
        )
        row = cur.fetchone()
    stats = dict(row) if row else {}
    # Normalise types / rounding for display.
    return {
        "uploads": int(stats.get("uploads") or 0),
        "total_points": int(stats.get("total_points") or 0),
        "best_speed_kmh": round(float(stats.get("best_speed_kmh") or 0.0), 1),
        "best_power_kmh": round(float(stats.get("best_power_kmh") or 0.0), 1),
        "best_technique": round(float(stats.get("best_technique") or 0.0), 1),
    }


def evaluate_badges(stats: dict) -> list[dict]:
    """Return every badge annotated with earned state and progress."""
    out: list[dict] = []
    for b in BADGES:
        current = float(stats.get(b["metric"], 0) or 0)
        target = float(b["target"])
        progress = 1.0 if target <= 0 else min(current / target, 1.0)
        out.append(
            {
                "id": b["id"],
                "name": b["name"],
                "description": b["description"],
                "category": b["category"],
                "tier": b["tier"],
                "icon": b["icon"],
                "unit": b["unit"],
                "target": b["target"],
                "current": round(current, 1),
                "earned": current >= target,
                "progress": round(progress, 3),
            }
        )
    return out


def profile(user_id: str) -> dict:
    stats = career_stats(user_id)
    badges = evaluate_badges(stats)
    return {
        "stats": stats,
        "badges": badges,
        "earned_count": sum(1 for b in badges if b["earned"]),
        "total_count": len(badges),
    }
