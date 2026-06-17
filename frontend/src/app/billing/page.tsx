"use client";

import { AppShell } from "@/components/AppShell";
import { TierBadge } from "@/components/TierBadge";
import {
  cancelSubscription,
  formatAmount,
  getBillingHistory,
  getSubscription,
  openPortal,
  resumeSubscription,
  type BillingEvent,
  type SubscriptionInfo,
} from "@/lib/billingApi";
import { getStoredUser } from "@/lib/socialApi";
import {
  AlertCircle,
  ArrowUpRight,
  CheckCircle2,
  CreditCard,
  Crown,
  Infinity as InfinityIcon,
  Loader2,
  Receipt,
  XCircle,
} from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

function formatDate(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

export default function BillingPage() {
  const [sub, setSub] = useState<SubscriptionInfo | null>(null);
  const [history, setHistory] = useState<BillingEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [signedOut, setSignedOut] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);

  async function load() {
    const [s, h] = await Promise.all([
      getSubscription(),
      getBillingHistory().catch(() => [] as BillingEvent[]),
    ]);
    setSub(s);
    setHistory(h);
  }

  useEffect(() => {
    if (!getStoredUser()) {
      setSignedOut(true);
      setLoading(false);
      return;
    }
    // Surface the post-checkout success banner.
    const params = new URLSearchParams(window.location.search);
    if (params.get("status") === "success") {
      setNotice("Payment successful — your membership is now active. It may take a moment to sync.");
    }
    load()
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load billing."))
      .finally(() => setLoading(false));
  }, []);

  async function act(name: string, fn: () => Promise<void>, after?: () => void) {
    setBusy(name);
    setError(null);
    try {
      await fn();
      if (after) after();
      else await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setBusy(null);
    }
  }

  if (signedOut) {
    return (
      <AppShell>
        <section className="analytics-grid hero-glow mx-auto max-w-2xl px-5 py-20 text-center fade-in">
          <CreditCard size={36} className="mx-auto mb-4 text-cyan-400" />
          <h1 className="display text-3xl text-[#eef2ff]">Sign in to manage billing</h1>
          <Link href="/login?next=/billing" className="btn-primary mt-6 inline-flex px-6 py-3 text-sm">
            Sign in
          </Link>
        </section>
      </AppShell>
    );
  }

  return (
    <AppShell>
      <section className="mx-auto max-w-3xl px-5 py-12 fade-in">
        <div className="mb-8">
          <h1 className="display text-4xl text-[#eef2ff]">Membership &amp; billing</h1>
          <p className="mt-3 text-sm text-[#6b7a99]">
            Manage your plan, track usage, and view past invoices.
          </p>
        </div>

        {notice && (
          <div className="mb-5 flex items-center gap-2 rounded-xl border border-emerald-400/25 bg-emerald-400/8 px-4 py-3 text-sm text-emerald-300">
            <CheckCircle2 size={16} className="shrink-0" />
            {notice}
          </div>
        )}
        {error && (
          <div className="mb-5 flex items-center gap-2 rounded-xl border border-red-500/25 bg-red-500/8 px-4 py-3 text-sm text-red-300">
            <AlertCircle size={16} className="shrink-0" />
            {error}
          </div>
        )}

        {loading ? (
          <div className="flex items-center justify-center gap-2 py-20 text-[#6b7a99]">
            <Loader2 size={20} className="animate-spin text-cyan-500" /> Loading…
          </div>
        ) : sub ? (
          <>
            {/* Current plan */}
            <div className="card relative overflow-hidden p-6">
              <div className="pointer-events-none absolute -right-10 -top-10 h-40 w-40 rounded-full bg-cyan-500/8 blur-3xl" />
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <p className="data-label">Current plan</p>
                  <div className="mt-2 flex items-center gap-2.5">
                    <span className="font-display text-2xl font-bold text-[#eef2ff]">
                      {sub.tier_name}
                    </span>
                    <TierBadge badge={sub.badge} size="md" />
                  </div>
                  <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-[#6b7a99]">
                    <span className="inline-flex items-center gap-1">
                      <span
                        className={`h-1.5 w-1.5 rounded-full ${
                          sub.status === "active" || sub.status === "trialing"
                            ? "bg-emerald-400"
                            : "bg-amber-400"
                        }`}
                      />
                      {sub.status}
                    </span>
                    {sub.current_period_end && (
                      <span>
                        {sub.cancel_at_period_end ? "Ends" : "Renews"}{" "}
                        {formatDate(sub.current_period_end)}
                      </span>
                    )}
                  </div>
                </div>

                {sub.tier === "free" ? (
                  <Link
                    href="/pricing"
                    className="btn-primary inline-flex items-center gap-1.5 px-4 py-2.5 text-sm"
                  >
                    <Crown size={15} /> Upgrade
                  </Link>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {sub.billing_enabled && (
                      <button
                        type="button"
                        onClick={() => act("portal", openPortal, () => {})}
                        disabled={busy !== null}
                        className="inline-flex items-center gap-1.5 rounded-xl border border-white/[0.08] bg-[#09090f] px-4 py-2.5 text-sm font-medium text-[#eef2ff] transition-colors hover:border-white/[0.14] disabled:opacity-60"
                      >
                        {busy === "portal" ? (
                          <Loader2 size={14} className="animate-spin" />
                        ) : (
                          <CreditCard size={14} />
                        )}
                        Manage card &amp; invoices
                      </button>
                    )}
                    <Link
                      href="/pricing"
                      className="inline-flex items-center gap-1.5 rounded-xl border border-white/[0.08] bg-[#09090f] px-4 py-2.5 text-sm font-medium text-[#eef2ff] transition-colors hover:border-white/[0.14]"
                    >
                      Change plan <ArrowUpRight size={14} />
                    </Link>
                  </div>
                )}
              </div>

              {/* Cancel / resume */}
              {sub.tier !== "free" && (
                <div className="mt-5 border-t border-white/[0.06] pt-4">
                  {sub.cancel_at_period_end ? (
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <p className="text-sm text-amber-300">
                        Your plan is set to cancel at the end of the period.
                      </p>
                      <button
                        type="button"
                        onClick={() => act("resume", resumeSubscription)}
                        disabled={busy !== null}
                        className="inline-flex items-center gap-1.5 rounded-lg bg-cyan-500/15 px-3 py-1.5 text-xs font-semibold text-cyan-300 hover:bg-cyan-500/25 disabled:opacity-60"
                      >
                        {busy === "resume" && <Loader2 size={12} className="animate-spin" />}
                        Resume plan
                      </button>
                    </div>
                  ) : (
                    <button
                      type="button"
                      onClick={() => act("cancel", cancelSubscription)}
                      disabled={busy !== null}
                      className="inline-flex items-center gap-1.5 text-xs font-medium text-[#6b7a99] transition-colors hover:text-red-300 disabled:opacity-60"
                    >
                      {busy === "cancel" && <Loader2 size={12} className="animate-spin" />}
                      Cancel subscription
                    </button>
                  )}
                </div>
              )}
            </div>

            {/* Usage meters */}
            <div className="mt-5">
              <h2 className="data-label mb-3">Usage this period</h2>
              <div className="grid gap-3 sm:grid-cols-2">
                {sub.meters.map((m) => {
                  const unlimited = m.limit === null;
                  const pct = unlimited ? 0 : Math.min(100, Math.round((m.used / Math.max(m.limit!, 1)) * 100));
                  const maxed = !unlimited && m.used >= (m.limit ?? 0);
                  return (
                    <div key={m.label} className="card p-5">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-[#c4d0f0]">{m.label}</span>
                        <span className="stat-value text-sm text-[#eef2ff]">
                          {unlimited ? (
                            <span className="inline-flex items-center gap-1 text-cyan-300">
                              <InfinityIcon size={15} /> Unlimited
                            </span>
                          ) : (
                            `${m.used} / ${m.limit}`
                          )}
                        </span>
                      </div>
                      {!unlimited && (
                        <>
                          <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
                            <div
                              className={`h-full rounded-full ${
                                maxed
                                  ? "bg-amber-400"
                                  : "bg-gradient-to-r from-cyan-500 to-blue-500"
                              }`}
                              style={{ width: `${Math.max(pct, 3)}%` }}
                            />
                          </div>
                          <p className="mt-1.5 text-[11px] text-[#6b7a99]">
                            Resets {m.window === "week" ? "weekly" : "monthly"}
                            {maxed ? " · limit reached" : ""}
                          </p>
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Billing history */}
            <div className="mt-8">
              <h2 className="data-label mb-3 flex items-center gap-1.5">
                <Receipt size={13} /> Billing history
              </h2>
              {history.length === 0 ? (
                <div className="card p-8 text-center text-sm text-[#6b7a99]">
                  No invoices yet. Charges will appear here after your first payment.
                </div>
              ) : (
                <div className="card divide-y divide-white/[0.05]">
                  {history.map((ev) => (
                    <div key={ev.id} className="flex items-center justify-between gap-3 px-5 py-3.5">
                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium text-[#eef2ff]">
                          {ev.description || "Subscription payment"}
                        </p>
                        <p className="text-xs text-[#3a4560]">{formatDate(ev.created_at)}</p>
                      </div>
                      <div className="flex shrink-0 items-center gap-3">
                        <span className="stat-value text-sm text-[#eef2ff]">
                          {formatAmount(ev.amount, ev.currency)}
                        </span>
                        {ev.status === "paid" ? (
                          <CheckCircle2 size={15} className="text-emerald-400" />
                        ) : ev.status === "failed" ? (
                          <XCircle size={15} className="text-red-400" />
                        ) : null}
                        {ev.invoice_url && (
                          <a
                            href={ev.invoice_url}
                            target="_blank"
                            rel="noreferrer"
                            className="text-cyan-400 hover:text-cyan-300"
                            title="View invoice"
                          >
                            <ArrowUpRight size={15} />
                          </a>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </>
        ) : null}
      </section>
    </AppShell>
  );
}
