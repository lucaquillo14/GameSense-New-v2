"use client";

import { AppShell } from "@/components/AppShell";
import { getPlans, startCheckout, formatPrice, type Plan } from "@/lib/billingApi";
import { getStoredUser, type MembershipTier, type User } from "@/lib/socialApi";
import { AlertCircle, Check, Crown, Gem, Loader2, Sparkles, Zap } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

const TIER_ACCENT: Record<string, { icon: React.ReactNode; ring: string; cta: string }> = {
  free: {
    icon: <Zap size={18} />,
    ring: "border-white/[0.08]",
    cta: "bg-white/[0.06] text-[#eef2ff] hover:bg-white/[0.1]",
  },
  pro: {
    icon: <Gem size={18} />,
    ring: "border-cyan-500/40 shadow-[0_0_40px_rgba(6,182,212,0.12)]",
    cta: "btn-primary",
  },
  elite: {
    icon: <Crown size={18} />,
    ring: "border-amber-300/35",
    cta: "bg-gradient-to-r from-amber-400 to-yellow-300 text-amber-950 font-bold hover:brightness-105",
  },
};

const TIER_ORDER: Record<MembershipTier, number> = { free: 0, pro: 1, elite: 2 };

export default function PricingPage() {
  const router = useRouter();
  const [plans, setPlans] = useState<Plan[]>([]);
  const [billingEnabled, setBillingEnabled] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [me, setMe] = useState<User | null>(null);
  const [checkoutBusy, setCheckoutBusy] = useState<string | null>(null);

  useEffect(() => {
    setMe(getStoredUser());
    getPlans()
      .then((d) => {
        setPlans(d.plans);
        setBillingEnabled(d.billing_enabled);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load plans."))
      .finally(() => setLoading(false));
  }, []);

  async function choose(plan: Plan) {
    if (plan.id === "free") {
      router.push(me ? "/" : "/login?next=/");
      return;
    }
    if (!me) {
      router.push("/login?next=/pricing");
      return;
    }
    setError(null);
    setCheckoutBusy(plan.id);
    try {
      await startCheckout(plan.id as "pro" | "elite");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not start checkout.");
      setCheckoutBusy(null);
    }
  }

  const currentTier = me?.tier ?? "free";

  return (
    <AppShell>
      <section className="analytics-grid hero-glow mx-auto max-w-6xl px-5 py-16 fade-in">
        <div className="mb-12 text-center">
          <div className="chip mx-auto mb-5 w-fit">
            <Sparkles size={11} />
            Membership
          </div>
          <h1 className="display text-5xl text-[#eef2ff]">
            Train like a <span className="gradient-text">pro</span>
          </h1>
          <p className="mx-auto mt-4 max-w-xl text-lg text-[#6b7a99]">
            Unlock deeper analytics, more analyses, and AI insights. Cancel anytime.
          </p>
        </div>

        {error && (
          <div className="mx-auto mb-6 flex max-w-md items-center gap-2 rounded-xl border border-red-500/25 bg-red-500/8 px-4 py-3 text-sm text-red-300">
            <AlertCircle size={16} className="shrink-0" />
            {error}
          </div>
        )}

        {!billingEnabled && !loading && (
          <div className="mx-auto mb-8 flex max-w-md items-center gap-2 rounded-xl border border-amber-400/25 bg-amber-400/8 px-4 py-3 text-sm text-amber-300">
            <AlertCircle size={16} className="shrink-0" />
            Checkout isn&apos;t live yet on this server. Plans are shown for preview.
          </div>
        )}

        {loading ? (
          <div className="flex items-center justify-center gap-2 py-20 text-[#6b7a99]">
            <Loader2 size={20} className="animate-spin text-cyan-500" /> Loading plans…
          </div>
        ) : (
          <div className="grid items-start gap-6 md:grid-cols-3">
            {plans.map((plan) => {
              const accent = TIER_ACCENT[plan.id] ?? TIER_ACCENT.free;
              const isCurrent = currentTier === plan.id;
              const isPopular = plan.id === "pro";
              const isElite = plan.id === "elite";
              const isDowngrade = TIER_ORDER[plan.id] < TIER_ORDER[currentTier];
              return (
                <div key={plan.id} className={`relative ${isPopular ? "md:-mt-4 md:mb-4" : ""}`}>
                  {/* Animated gradient glow border on the popular tier */}
                  {isPopular && (
                    <div className="pointer-events-none absolute -inset-px rounded-[1.05rem] bg-[linear-gradient(110deg,#22d3ee,#3b82f6,#8b5cf6,#22d3ee)] bg-[length:220%_auto] opacity-70 blur-[7px] [animation:gradient-pan_6s_linear_infinite]" />
                  )}

                  <div className={`card relative flex h-full flex-col overflow-hidden p-6 ${accent.ring}`}>
                    {/* top accent */}
                    <div
                      className={`absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent to-transparent ${
                        isElite ? "via-amber-300/60" : isPopular ? "via-cyan-400/70" : "via-white/15"
                      }`}
                    />
                    {/* corner glow */}
                    <div
                      className={`pointer-events-none absolute -right-10 -top-10 h-36 w-36 rounded-full blur-3xl ${
                        isElite ? "bg-amber-400/10" : isPopular ? "bg-cyan-500/12" : "bg-white/[0.03]"
                      }`}
                    />

                    {isPopular && (
                      <span className="absolute -top-0 right-4 rounded-b-lg bg-cyan-500 px-3 py-1 text-[10px] font-bold uppercase tracking-widest text-[#04121f] shadow-[0_0_18px_rgba(6,182,212,0.5)]">
                        Most popular
                      </span>
                    )}

                    <div className="mb-5 flex items-center gap-2.5">
                      <span
                        className={`grid h-10 w-10 place-items-center rounded-xl ${
                          isElite
                            ? "bg-amber-400/15 text-amber-300"
                            : isPopular
                              ? "bg-cyan-500/15 text-cyan-300"
                              : "bg-white/[0.06] text-[#9fb0d0]"
                        }`}
                      >
                        {accent.icon}
                      </span>
                      <span className="font-display text-xl font-bold text-[#eef2ff]">{plan.name}</span>
                    </div>

                    <div className="mb-1 flex items-end gap-1.5">
                      <span className="stat-value text-5xl font-bold text-[#eef2ff]">
                        {formatPrice(plan.price_monthly)}
                      </span>
                      {plan.price_monthly > 0 && <span className="mb-1.5 text-sm text-[#6b7a99]">/mo</span>}
                    </div>
                    <p className="mb-6 text-sm text-[#6b7a99]">{plan.tagline}</p>

                    <button
                      type="button"
                      disabled={isCurrent || checkoutBusy !== null}
                      onClick={() => choose(plan)}
                      className={`mb-6 flex w-full items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-semibold transition-all disabled:cursor-not-allowed disabled:opacity-60 ${accent.cta}`}
                    >
                      {checkoutBusy === plan.id ? <Loader2 size={16} className="animate-spin" /> : null}
                      {isCurrent
                        ? "Current plan"
                        : plan.id === "free"
                          ? "Get started"
                          : isDowngrade
                            ? `Switch to ${plan.name}`
                            : `Upgrade to ${plan.name}`}
                    </button>

                    <ul className="space-y-3 border-t border-white/[0.06] pt-5">
                      {plan.highlights.map((h) => (
                        <li key={h} className="flex items-start gap-2.5 text-sm text-[#c4d0f0]">
                          <span
                            className={`mt-0.5 grid h-4 w-4 shrink-0 place-items-center rounded-full ${
                              isElite ? "bg-amber-300/15 text-amber-300" : "bg-cyan-400/15 text-cyan-300"
                            }`}
                          >
                            <Check size={11} strokeWidth={3} />
                          </span>
                          {h}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        <p className="mt-10 text-center text-xs text-[#3a4560]">
          Secure payments by Stripe · Cancel anytime · Prices in GBP
        </p>
      </section>
    </AppShell>
  );
}
