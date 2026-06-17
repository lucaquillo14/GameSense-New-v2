import { API_BASE } from "./api";
import { authHeader, type MembershipTier, type TierBadge } from "./socialApi";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export type Plan = {
  id: MembershipTier;
  name: string;
  price_monthly: number; // cents
  badge: TierBadge;
  tagline: string;
  highlights: string[];
  features: string[];
};

export type UsageMeter = {
  label: string;
  used: number;
  limit: number | null; // null = unlimited
  window: string;
};

export type SubscriptionInfo = {
  tier: MembershipTier;
  tier_name: string;
  badge: TierBadge;
  status: string;
  current_period_end: string | null;
  cancel_at_period_end: boolean;
  features: string[];
  meters: UsageMeter[];
  billing_enabled: boolean;
};

export type BillingEvent = {
  id: string;
  type: string;
  amount: number; // cents
  currency: string;
  status: string | null;
  description: string | null;
  invoice_url: string | null;
  created_at: string;
};

// ---------------------------------------------------------------------------
// Fetch helper (mirrors socialApi.request, kept local to avoid an export churn)
// ---------------------------------------------------------------------------
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      cache: "no-store",
      headers: {
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...authHeader(),
        ...(init?.headers ?? {}),
      },
    });
  } catch {
    throw new Error("Could not reach the server. Is the backend running on port 8000?");
  }
  if (!response.ok) {
    let message = `Request failed (HTTP ${response.status}).`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      /* no JSON body */
    }
    const err = new Error(message) as Error & { status?: number };
    err.status = response.status;
    throw err;
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------
export async function getPlans(): Promise<{ plans: Plan[]; billing_enabled: boolean }> {
  return request("/billing/plans");
}

export async function getSubscription(): Promise<SubscriptionInfo> {
  return request("/billing/subscription");
}

export async function getBillingHistory(): Promise<BillingEvent[]> {
  const res = await request<{ events: BillingEvent[] }>("/billing/history");
  return res.events;
}

/** Start Stripe Checkout for a paid tier and redirect the browser to it. */
export async function startCheckout(tier: "pro" | "elite"): Promise<void> {
  const res = await request<{ url: string }>("/billing/checkout", {
    method: "POST",
    body: JSON.stringify({ tier }),
  });
  if (res.url) window.location.href = res.url;
}

/** Open the Stripe customer billing portal. */
export async function openPortal(): Promise<void> {
  const res = await request<{ url: string }>("/billing/portal", { method: "POST" });
  if (res.url) window.location.href = res.url;
}

export async function cancelSubscription(): Promise<void> {
  await request("/billing/cancel", { method: "POST" });
}

export async function resumeSubscription(): Promise<void> {
  await request("/billing/resume", { method: "POST" });
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------
export function formatPrice(cents: number): string {
  if (!cents) return "Free";
  return `$${(cents / 100).toFixed(2).replace(/\.00$/, "")}`;
}

export function formatAmount(cents: number, currency = "usd"): string {
  const symbol = currency.toLowerCase() === "usd" ? "$" : "";
  return `${symbol}${(cents / 100).toFixed(2)}${symbol ? "" : ` ${currency.toUpperCase()}`}`;
}
