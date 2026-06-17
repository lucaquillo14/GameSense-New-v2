"""Billing & subscription API: Stripe checkout, webhooks, and management.

Endpoints:
  GET  /billing/plans          public tier catalogue (pricing page)
  GET  /billing/subscription   current user's subscription + usage
  GET  /billing/history        current user's billing history
  POST /billing/checkout       create a Stripe Checkout session for a tier
  POST /billing/portal         open the Stripe customer billing portal
  POST /billing/cancel         cancel at period end
  POST /billing/resume         undo a pending cancellation
  POST /billing/webhook        Stripe webhook receiver (no auth; signature-verified)
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app import auth
from app.services import stripe_client, subscriptions

router = APIRouter(prefix="/billing", tags=["billing"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class CheckoutRequest(BaseModel):
    tier: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_stripe():
    stripe = stripe_client.get_stripe()
    if stripe is None:
        raise HTTPException(
            status_code=503,
            detail="Billing is not configured on this server. Set STRIPE_SECRET_KEY.",
        )
    return stripe


def _ensure_customer(stripe, user: dict) -> str:
    """Return the user's Stripe customer id, creating one if needed."""
    sub = subscriptions.get_subscription(user["id"])
    if sub.get("stripe_customer_id"):
        return sub["stripe_customer_id"]
    customer = stripe.Customer.create(
        email=user.get("email"),
        name=user.get("display_name"),
        metadata={"user_id": user["id"]},
    )
    subscriptions.upsert_subscription(user["id"], stripe_customer_id=customer.id)
    return customer.id


def _iso_from_unix(ts) -> str | None:
    if not ts:
        return None
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Public catalogue
# ---------------------------------------------------------------------------
@router.get("/plans")
def plans() -> dict:
    return {
        "plans": subscriptions.public_tier_catalog(),
        "billing_enabled": stripe_client.is_configured(),
    }


# ---------------------------------------------------------------------------
# Current subscription + usage
# ---------------------------------------------------------------------------
@router.get("/subscription")
def subscription(user: dict = Depends(auth.current_user)) -> dict:
    sub = subscriptions.get_subscription(user["id"])
    summary = subscriptions.usage_summary(user["id"])
    return {
        "tier": summary["tier"],
        "tier_name": summary["tier_name"],
        "badge": summary["badge"],
        "status": sub["status"],
        "current_period_end": sub["current_period_end"],
        "cancel_at_period_end": sub["cancel_at_period_end"],
        "features": summary["features"],
        "meters": summary["meters"],
        "billing_enabled": stripe_client.is_configured(),
    }


@router.get("/history")
def history(user: dict = Depends(auth.current_user)) -> dict:
    return {"events": subscriptions.billing_history(user["id"])}


# ---------------------------------------------------------------------------
# Checkout
# ---------------------------------------------------------------------------
@router.post("/checkout")
def checkout(payload: CheckoutRequest, user: dict = Depends(auth.current_user)) -> dict:
    tier = (payload.tier or "").strip().lower()
    if tier not in subscriptions.PAID_TIERS:
        raise HTTPException(status_code=400, detail="Choose a paid plan (pro or elite).")

    stripe = _require_stripe()
    price_id = stripe_client.price_for_tier(tier)
    if not price_id:
        raise HTTPException(
            status_code=503,
            detail=f"No Stripe price configured for the {tier} plan.",
        )

    customer_id = _ensure_customer(stripe, user)
    base = stripe_client.frontend_url()
    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{base}/billing?status=success&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{base}/pricing?status=cancelled",
            allow_promotion_codes=True,
            metadata={"user_id": user["id"], "tier": tier},
            subscription_data={"metadata": {"user_id": user["id"], "tier": tier}},
        )
    except Exception as exc:  # pragma: no cover - network/stripe errors
        raise HTTPException(status_code=502, detail=f"Stripe checkout failed: {exc}")
    return {"url": session.url, "id": session.id}


# ---------------------------------------------------------------------------
# Customer portal (manage card, invoices, upgrade/downgrade, cancel)
# ---------------------------------------------------------------------------
@router.post("/portal")
def portal(user: dict = Depends(auth.current_user)) -> dict:
    stripe = _require_stripe()
    sub = subscriptions.get_subscription(user["id"])
    customer_id = sub.get("stripe_customer_id")
    if not customer_id:
        raise HTTPException(status_code=400, detail="No billing account yet. Subscribe first.")
    base = stripe_client.frontend_url()
    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=f"{base}/billing",
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=f"Could not open billing portal: {exc}")
    return {"url": session.url}


# ---------------------------------------------------------------------------
# Cancel / resume
# ---------------------------------------------------------------------------
@router.post("/cancel")
def cancel(user: dict = Depends(auth.current_user)) -> dict:
    stripe = _require_stripe()
    sub = subscriptions.get_subscription(user["id"])
    sub_id = sub.get("stripe_subscription_id")
    if not sub_id:
        raise HTTPException(status_code=400, detail="No active subscription to cancel.")
    try:
        stripe.Subscription.modify(sub_id, cancel_at_period_end=True)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=f"Could not cancel: {exc}")
    subscriptions.upsert_subscription(user["id"], cancel_at_period_end=True)
    return {"status": "cancel_scheduled", "cancel_at_period_end": True}


@router.post("/resume")
def resume(user: dict = Depends(auth.current_user)) -> dict:
    stripe = _require_stripe()
    sub = subscriptions.get_subscription(user["id"])
    sub_id = sub.get("stripe_subscription_id")
    if not sub_id:
        raise HTTPException(status_code=400, detail="No subscription to resume.")
    try:
        stripe.Subscription.modify(sub_id, cancel_at_period_end=False)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=f"Could not resume: {exc}")
    subscriptions.upsert_subscription(user["id"], cancel_at_period_end=False)
    return {"status": "active", "cancel_at_period_end": False}


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------
@router.post("/webhook")
async def webhook(request: Request) -> dict:
    stripe = stripe_client.get_stripe()
    if stripe is None:
        raise HTTPException(status_code=503, detail="Billing not configured.")

    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    secret = stripe_client.webhook_secret()

    try:
        if secret:
            event = stripe.Webhook.construct_event(payload, sig, secret)
        else:
            # No signing secret set (e.g. local dev) — parse without verification.
            import json

            event = json.loads(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid webhook: {exc}")

    event_type = event["type"]
    obj = event["data"]["object"]

    try:
        _handle_event(stripe, event_type, obj)
    except Exception as exc:  # never 500 back to Stripe on a handled type
        print(f"[GameSense] Webhook handling error for {event_type}: {exc}")

    return {"received": True}


def _resolve_user_id(stripe, obj) -> str | None:
    """Find our user id from a Stripe object via metadata or customer id."""
    meta = obj.get("metadata") or {}
    if meta.get("user_id"):
        return meta["user_id"]
    customer_id = obj.get("customer")
    if customer_id:
        uid = subscriptions.find_user_by_customer(customer_id)
        if uid:
            return uid
        # Fall back to the customer's metadata.
        try:
            customer = stripe.Customer.retrieve(customer_id)
            return (customer.get("metadata") or {}).get("user_id")
        except Exception:
            return None
    return None


def _apply_subscription(stripe, user_id: str, sub_obj) -> None:
    """Sync a Stripe Subscription object onto our subscription row."""
    status = sub_obj.get("status", "active")
    cancel_at_period_end = bool(sub_obj.get("cancel_at_period_end"))
    period_end = _iso_from_unix(sub_obj.get("current_period_end"))

    # Tier from the first line item's price.
    tier = None
    items = (sub_obj.get("items") or {}).get("data") or []
    if items:
        price_id = (items[0].get("price") or {}).get("id")
        if price_id:
            tier = stripe_client.tier_for_price(price_id)
    if tier is None:
        tier = (sub_obj.get("metadata") or {}).get("tier")

    # An ended/canceled subscription drops the user back to free.
    if status in ("canceled", "incomplete_expired", "unpaid"):
        subscriptions.upsert_subscription(
            user_id,
            tier="free",
            status="active",
            stripe_subscription_id=sub_obj.get("id"),
            current_period_end=period_end,
            cancel_at_period_end=False,
        )
        return

    subscriptions.upsert_subscription(
        user_id,
        tier=tier if tier in subscriptions.PAID_TIERS else None,
        status=status,
        stripe_subscription_id=sub_obj.get("id"),
        current_period_end=period_end,
        cancel_at_period_end=cancel_at_period_end,
    )


def _handle_event(stripe, event_type: str, obj) -> None:
    if event_type == "checkout.session.completed":
        user_id = _resolve_user_id(stripe, obj)
        if not user_id:
            return
        # Persist the customer id and pull the freshly-created subscription.
        if obj.get("customer"):
            subscriptions.upsert_subscription(user_id, stripe_customer_id=obj["customer"])
        sub_id = obj.get("subscription")
        if sub_id:
            sub_obj = stripe.Subscription.retrieve(sub_id, expand=["items.data.price"])
            _apply_subscription(stripe, user_id, sub_obj)

    elif event_type in ("customer.subscription.updated", "customer.subscription.created"):
        user_id = _resolve_user_id(stripe, obj)
        if user_id:
            _apply_subscription(stripe, user_id, obj)

    elif event_type == "customer.subscription.deleted":
        user_id = _resolve_user_id(stripe, obj)
        if user_id:
            subscriptions.upsert_subscription(
                user_id, tier="free", status="active", cancel_at_period_end=False
            )

    elif event_type == "invoice.payment_succeeded":
        user_id = _resolve_user_id(stripe, obj)
        if user_id:
            subscriptions.record_billing_event(
                user_id,
                event_type="invoice",
                amount=int(obj.get("amount_paid") or 0),
                currency=obj.get("currency") or "usd",
                status="paid",
                description=obj.get("description") or "Subscription payment",
                invoice_url=obj.get("hosted_invoice_url"),
            )

    elif event_type == "invoice.payment_failed":
        user_id = _resolve_user_id(stripe, obj)
        if user_id:
            subscriptions.record_billing_event(
                user_id,
                event_type="invoice",
                amount=int(obj.get("amount_due") or 0),
                currency=obj.get("currency") or "usd",
                status="failed",
                description="Payment failed",
                invoice_url=obj.get("hosted_invoice_url"),
            )
