"""Membership tiers, usage metering, and feature gating.

The single source of truth for what each tier (free / pro / elite) is allowed
to do. All limits are enforced server-side:

  * Usage limits are enforced by counting rows in `analysis_usage` inside a
    rolling weekly/monthly window, so resets happen automatically — there is
    no scheduled reset job to run or to drift.
  * Feature access (history, heatmaps, reports, AI insights, …) is decided by
    `has_feature()` and applied at the relevant endpoints.

Stripe only ever flips a user's `tier`/`status` via the billing webhook; this
module owns everything downstream of that.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from app import db

# ---------------------------------------------------------------------------
# Tier catalogue
# ---------------------------------------------------------------------------
# `limits` maps a usage *kind* to a window + cap:
#   kind "sprint"    -> sprint / shot-power analyses
#   kind "technique" -> shooting-technique analyses
#   kind "analysis"  -> any analysis (combined)
# A tier with no entry for a kind imposes no cap for it. `None` cap = unlimited.
TIERS: dict[str, dict] = {
    "free": {
        "name": "Free",
        "price_monthly": 0,
        "badge": None,
        "tagline": "Get started and climb the leaderboards.",
        "limits": {
            "sprint": {"window": "week", "max": 1},
            "technique": {"window": "week", "max": 1},
        },
        "features": {"leaderboards", "leagues"},
        "highlights": [
            "1 sprint analysis per week",
            "1 shooting-technique analysis per week",
            "Global leaderboard & leagues",
        ],
    },
    "pro": {
        "name": "Pro",
        "price_monthly": 999,  # $9.99 — display only; Stripe price is authoritative
        "badge": "pro",
        "tagline": "Full analytics for serious players.",
        "limits": {
            "analysis": {"window": "month", "max": 20},
        },
        "features": {
            "leaderboards",
            "leagues",
            "history",
            "heatmaps",
            "reports",
            "advanced_analytics",
        },
        "highlights": [
            "20 analyses per month",
            "Performance history",
            "Heatmaps & advanced analytics",
            "Downloadable reports",
        ],
    },
    "elite": {
        "name": "Elite",
        "price_monthly": 1999,  # $19.99 — display only
        "badge": "elite",
        "tagline": "Everything, unlimited, first.",
        "limits": {},  # unlimited
        "features": {
            "leaderboards",
            "leagues",
            "history",
            "heatmaps",
            "reports",
            "advanced_analytics",
            "ai_insights",
            "tactical_analysis",
            "priority_processing",
            "premium_badge",
            "exclusive_leaderboards",
            "early_access",
        },
        "highlights": [
            "Unlimited analyses",
            "AI insights & tactical analysis",
            "Priority processing",
            "Premium badge & exclusive leaderboards",
            "Early access to new features",
        ],
    },
}

DEFAULT_TIER = "free"
PAID_TIERS = ("pro", "elite")


def is_valid_tier(tier: str) -> bool:
    return tier in TIERS


# ---------------------------------------------------------------------------
# Analysis categories
# ---------------------------------------------------------------------------
def category_for_mode(mode: str | None) -> str:
    """Map an analysis mode onto a usage category."""
    if (mode or "") == "shooting_technique":
        return "technique"
    return "sprint"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _window_start(window: str) -> datetime:
    """Start of the current rolling window in UTC."""
    now = _now()
    if window == "week":
        # ISO week — reset Monday 00:00 UTC.
        monday = now - timedelta(days=now.weekday())
        return monday.replace(hour=0, minute=0, second=0, microsecond=0)
    if window == "month":
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    # Fallback: treat unknown windows as "all time".
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Subscription record
# ---------------------------------------------------------------------------
def get_subscription(user_id: str) -> dict:
    """Return the user's subscription, creating a free default if absent."""
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT user_id, tier, status, stripe_customer_id, stripe_subscription_id,
                   current_period_end, cancel_at_period_end, updated_at
            FROM subscriptions WHERE user_id = ?
            """,
            (user_id,),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO subscriptions (user_id, tier, status, updated_at) VALUES (?, 'free', 'active', ?)",
                (user_id, _now_iso()),
            )
            return {
                "user_id": user_id,
                "tier": "free",
                "status": "active",
                "stripe_customer_id": None,
                "stripe_subscription_id": None,
                "current_period_end": None,
                "cancel_at_period_end": False,
                "updated_at": _now_iso(),
            }
    data = dict(row)
    data["cancel_at_period_end"] = bool(data.get("cancel_at_period_end"))
    return data


def effective_tier(user_id: str) -> str:
    """The tier the user actually gets right now.

    A past-due / unpaid subscription falls back to free so we never grant paid
    features without payment.
    """
    sub = get_subscription(user_id)
    tier = sub.get("tier", "free")
    status = sub.get("status", "active")
    if tier in PAID_TIERS and status not in ("active", "trialing", "canceled_pending"):
        return "free"
    return tier if tier in TIERS else "free"


def tier_badge(user_id: str) -> Optional[str]:
    return TIERS.get(effective_tier(user_id), {}).get("badge")


def upsert_subscription(
    user_id: str,
    *,
    tier: Optional[str] = None,
    status: Optional[str] = None,
    stripe_customer_id: Optional[str] = None,
    stripe_subscription_id: Optional[str] = None,
    current_period_end: Optional[str] = None,
    cancel_at_period_end: Optional[bool] = None,
) -> dict:
    """Create or patch a subscription row. Only provided fields are changed."""
    get_subscription(user_id)  # ensure a row exists
    fields: list[str] = []
    params: list[object] = []
    for col, val in (
        ("tier", tier),
        ("status", status),
        ("stripe_customer_id", stripe_customer_id),
        ("stripe_subscription_id", stripe_subscription_id),
        ("current_period_end", current_period_end),
    ):
        if val is not None:
            fields.append(f"{col} = ?")
            params.append(val)
    if cancel_at_period_end is not None:
        fields.append("cancel_at_period_end = ?")
        params.append(1 if cancel_at_period_end else 0)
    fields.append("updated_at = ?")
    params.append(_now_iso())
    params.append(user_id)
    with db.cursor() as cur:
        cur.execute(f"UPDATE subscriptions SET {', '.join(fields)} WHERE user_id = ?", params)
    return get_subscription(user_id)


def find_user_by_customer(stripe_customer_id: str) -> Optional[str]:
    with db.cursor() as cur:
        cur.execute(
            "SELECT user_id FROM subscriptions WHERE stripe_customer_id = ?",
            (stripe_customer_id,),
        )
        row = cur.fetchone()
    return row["user_id"] if row else None


# ---------------------------------------------------------------------------
# Feature gating
# ---------------------------------------------------------------------------
def features_for(tier: str) -> set[str]:
    return set(TIERS.get(tier, TIERS["free"]).get("features", set()))


def has_feature(user_id: str, feature: str) -> bool:
    return feature in features_for(effective_tier(user_id))


# ---------------------------------------------------------------------------
# Usage metering
# ---------------------------------------------------------------------------
def _count_usage(user_id: str, window: str, category: Optional[str]) -> int:
    start = _window_start(window).isoformat()
    sql = "SELECT COUNT(*) AS n FROM analysis_usage WHERE user_id = ? AND created_at >= ?"
    params: list[object] = [user_id, start]
    if category is not None:
        sql += " AND category = ?"
        params.append(category)
    with db.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    return int(row["n"]) if row else 0


def _limit_for(tier: str, mode: str | None) -> tuple[Optional[str], Optional[dict]]:
    """Return (kind, limit-spec) that governs this mode for the tier, or (None, None)."""
    limits = TIERS.get(tier, {}).get("limits", {})
    if not limits:
        return None, None
    category = category_for_mode(mode)
    # Prefer a category-specific limit, then a combined "analysis" limit.
    if category in limits:
        return category, limits[category]
    if "analysis" in limits:
        return "analysis", limits["analysis"]
    return None, None


def check_quota(user_id: str, mode: str | None) -> dict:
    """Decide whether the user may run an analysis of `mode` right now.

    Returns a dict: { allowed, reason, tier, used, limit, window, category }.
    """
    tier = effective_tier(user_id)
    category = category_for_mode(mode)
    kind, spec = _limit_for(tier, mode)

    if spec is None:
        return {
            "allowed": True,
            "reason": None,
            "tier": tier,
            "used": 0,
            "limit": None,
            "window": None,
            "category": category,
        }

    window = spec["window"]
    cap = spec["max"]
    # For the combined "analysis" kind we count every category; otherwise count
    # just this category.
    count_category = None if kind == "analysis" else category
    used = _count_usage(user_id, window, count_category)
    allowed = used < cap

    reason = None
    if not allowed:
        nice_window = "this week" if window == "week" else "this month"
        if tier == "free":
            reason = (
                f"You've used your {cap} free "
                f"{'sprint' if category == 'sprint' else 'technique'} "
                f"analysis {nice_window}. Upgrade to Pro for 20 analyses a month, "
                f"or Elite for unlimited."
            )
        else:
            reason = (
                f"You've reached your {cap} analyses {nice_window}. "
                f"Upgrade to Elite for unlimited analyses."
            )

    return {
        "allowed": allowed,
        "reason": reason,
        "tier": tier,
        "used": used,
        "limit": cap,
        "window": window,
        "category": category,
    }


def record_analysis(user_id: str, mode: str | None, video_id: str | None = None) -> None:
    """Log one analysis run for usage metering."""
    with db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO analysis_usage (id, user_id, video_id, mode, category, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                user_id,
                video_id,
                mode,
                category_for_mode(mode),
                _now_iso(),
            ),
        )


def usage_summary(user_id: str) -> dict:
    """A frontend-friendly snapshot of the user's tier, limits, and usage."""
    tier = effective_tier(user_id)
    spec = TIERS.get(tier, TIERS["free"])
    limits = spec.get("limits", {})

    meters = []
    if not limits:
        meters.append({"label": "Analyses", "used": 0, "limit": None, "window": "unlimited"})
    else:
        for kind, lim in limits.items():
            window = lim["window"]
            count_category = None if kind == "analysis" else kind
            used = _count_usage(user_id, window, count_category)
            label = {
                "sprint": "Sprint analyses",
                "technique": "Technique analyses",
                "analysis": "Analyses",
            }.get(kind, kind.title())
            meters.append({"label": label, "used": used, "limit": lim["max"], "window": window})

    return {
        "tier": tier,
        "tier_name": spec.get("name", tier.title()),
        "badge": spec.get("badge"),
        "features": sorted(features_for(tier)),
        "meters": meters,
    }


# ---------------------------------------------------------------------------
# Billing history
# ---------------------------------------------------------------------------
def record_billing_event(
    user_id: str,
    *,
    event_type: str,
    amount: int = 0,
    currency: str = "usd",
    status: str | None = None,
    description: str | None = None,
    invoice_url: str | None = None,
) -> None:
    with db.cursor() as cur:
        cur.execute(
            """
            INSERT INTO billing_events
              (id, user_id, type, amount, currency, status, description, invoice_url, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                user_id,
                event_type,
                amount,
                currency,
                status,
                description,
                invoice_url,
                _now_iso(),
            ),
        )


def billing_history(user_id: str, limit: int = 50) -> list[dict]:
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT id, type, amount, currency, status, description, invoice_url, created_at
            FROM billing_events WHERE user_id = ?
            ORDER BY created_at DESC LIMIT ?
            """,
            (user_id, limit),
        )
        return [dict(r) for r in cur.fetchall()]


def public_tier_catalog() -> list[dict]:
    """The pricing catalogue for the marketing/pricing page."""
    out = []
    for key, spec in TIERS.items():
        out.append(
            {
                "id": key,
                "name": spec["name"],
                "price_monthly": spec["price_monthly"],
                "badge": spec.get("badge"),
                "tagline": spec.get("tagline", ""),
                "highlights": spec.get("highlights", []),
                "features": sorted(spec.get("features", set())),
            }
        )
    return out
