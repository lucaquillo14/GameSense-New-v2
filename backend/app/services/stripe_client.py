"""Thin Stripe configuration + lazy client.

Stripe is optional at import time so the app still boots (and all non-billing
features keep working) on a server where billing isn't configured yet. Billing
endpoints call `require_stripe()` and return a clean 503 if it's missing.

Environment variables:
  STRIPE_SECRET_KEY        sk_live_... / sk_test_...
  STRIPE_WEBHOOK_SECRET    whsec_...
  STRIPE_PRICE_PRO         price_...  (recurring monthly price for Pro)
  STRIPE_PRICE_ELITE       price_...  (recurring monthly price for Elite)
  FRONTEND_URL             https://app.example.com  (for redirect URLs)
"""

from __future__ import annotations

import os
from typing import Optional

try:  # pragma: no cover - import guard
    import stripe as _stripe
except ImportError:  # stripe not installed
    _stripe = None


def _key() -> str:
    return os.environ.get("STRIPE_SECRET_KEY", "").strip()


def webhook_secret() -> str:
    return os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()


def frontend_url() -> str:
    return os.environ.get("FRONTEND_URL", "http://localhost:3000").strip().rstrip("/")


# tier -> Stripe price id
def price_for_tier(tier: str) -> Optional[str]:
    return {
        "pro": os.environ.get("STRIPE_PRICE_PRO", "").strip() or None,
        "elite": os.environ.get("STRIPE_PRICE_ELITE", "").strip() or None,
    }.get(tier)


def tier_for_price(price_id: str) -> Optional[str]:
    mapping = {
        os.environ.get("STRIPE_PRICE_PRO", "").strip(): "pro",
        os.environ.get("STRIPE_PRICE_ELITE", "").strip(): "elite",
    }
    mapping.pop("", None)
    return mapping.get(price_id)


def is_configured() -> bool:
    return _stripe is not None and bool(_key())


def get_stripe():
    """Return the configured stripe module, or None if unavailable."""
    if _stripe is None or not _key():
        return None
    _stripe.api_key = _key()
    return _stripe
