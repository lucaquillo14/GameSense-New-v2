"use client";

import { AppShell } from "@/components/AppShell";
import { getHistory, getStoredUser, type HistoryItem } from "@/lib/socialApi";
import { AlertCircle, ArrowRight, Clock, Crown, Gauge, Loader2, Lock } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

const MODE_LABELS: Record<string, string> = {
  max_speed: "Max Speed",
  max_shot_power: "Max Shot Power",
  shooting_technique: "Shooting Technique",
};

function formatDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function HistoryPage() {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [signedOut, setSignedOut] = useState(false);
  const [locked, setLocked] = useState(false);

  useEffect(() => {
    if (!getStoredUser()) { setSignedOut(true); setLoading(false); return; }
    getHistory()
      .then((data) => { setItems(data); setError(null); })
      .catch((err) => {
        const msg = err instanceof Error ? err.message : "Failed to load history.";
        if (msg.toLowerCase().includes("auth")) setSignedOut(true);
        else if (msg.toLowerCase().includes("pro feature") || msg.toLowerCase().includes("upgrade")) setLocked(true);
        else setError(msg);
      })
      .finally(() => setLoading(false));
  }, []);

  if (locked) {
    return (
      <AppShell>
        <section className="analytics-grid hero-glow mx-auto max-w-2xl px-5 py-20 text-center fade-in">
          <span className="mx-auto mb-5 grid h-16 w-16 place-items-center rounded-2xl bg-cyan-500/12 text-cyan-300">
            <Lock size={28} />
          </span>
          <h1 className="display text-3xl text-[#eef2ff]">Performance history is a Pro feature</h1>
          <p className="mx-auto mt-3 max-w-sm text-sm text-[#6b7a99]">
            Upgrade to Pro to keep a full record of every session, with heatmaps, advanced
            analytics, and downloadable reports.
          </p>
          <Link href="/pricing" className="btn-primary mt-8 inline-flex items-center gap-2 px-6 py-3 text-sm">
            <Crown size={16} /> View plans
          </Link>
        </section>
      </AppShell>
    );
  }

  if (signedOut) {
    return (
      <AppShell>
        <section className="analytics-grid hero-glow mx-auto max-w-2xl px-5 py-20 text-center fade-in">
          <div className="mx-auto mb-5 grid h-16 w-16 place-items-center rounded-2xl bg-gradient-to-br from-cyan-500 to-blue-600 text-[#04121f] shadow-[0_0_28px_rgba(6,182,212,0.45)]">
            <Clock size={28} />
          </div>
          <h1 className="display text-3xl text-[#eef2ff]">Sign in to see your history</h1>
          <p className="mx-auto mt-3 max-w-sm text-sm text-[#6b7a99]">
            Your past analyses are saved to your account when you upload while signed in.
          </p>
          <Link href="/login" className="btn-primary mt-8 inline-flex items-center gap-2 px-6 py-3 text-sm">
            Sign in
          </Link>
        </section>
      </AppShell>
    );
  }

  return (
    <AppShell>
      <section className="analytics-grid mx-auto max-w-3xl px-5 py-12 fade-in">

        <div className="mb-10">
          <div className="chip mb-5 w-fit">
            <Clock size={11} />
            Session log
          </div>
          <h1 className="display text-4xl text-[#eef2ff]">History</h1>
          <p className="mt-3 text-sm text-[#6b7a99]">
            Every clip you&apos;ve analysed, newest first.
          </p>
        </div>

        {loading ? (
          <div className="flex items-center justify-center gap-2 py-20 text-[#6b7a99]">
            <Loader2 size={20} className="animate-spin text-cyan-500" /> Loading…
          </div>
        ) : error ? (
          <div className="flex items-center gap-2 rounded-xl border border-red-500/25 bg-red-500/8 px-4 py-3 text-sm text-red-300">
            <AlertCircle size={16} />
            {error}
          </div>
        ) : items.length === 0 ? (
          <div className="card p-10 text-center text-[#6b7a99]">
            No sessions yet. Upload a clip while signed in and it&apos;ll appear here.
          </div>
        ) : (
          <div className="space-y-3">
            {items.map((item) => (
              <Link
                key={item.video_id}
                href={`/results/${item.video_id}`}
                className="card card-hover flex items-center justify-between gap-4 p-5"
              >
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="inline-flex items-center gap-1.5 rounded-md border border-cyan-500/20 bg-cyan-500/8 px-2 py-0.5 text-xs font-semibold text-cyan-400">
                      <Gauge size={11} />
                      {MODE_LABELS[item.mode ?? ""] ?? "Analysis"}
                    </span>
                  </div>

                  <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1">
                    {item.max_speed_kmh > 0 && (
                      <span>
                        <span className="stat-value text-sm text-[#eef2ff]">
                          {item.max_speed_kmh.toFixed(1)}
                        </span>
                        <span className="ml-1 text-xs text-[#6b7a99]">km/h top speed</span>
                      </span>
                    )}
                    {item.shot_power_kmh > 0 && (
                      <span>
                        <span className="stat-value text-sm text-[#eef2ff]">
                          {item.shot_power_kmh.toFixed(1)}
                        </span>
                        <span className="ml-1 text-xs text-[#6b7a99]">km/h shot power</span>
                      </span>
                    )}
                    {item.technique_score > 0 && (
                      <span>
                        <span className="stat-value text-sm text-[#eef2ff]">
                          {item.technique_score.toFixed(0)}
                        </span>
                        <span className="ml-1 text-xs text-[#6b7a99]">technique</span>
                      </span>
                    )}
                  </div>

                  <p className="mt-2 truncate text-xs text-[#3a4560]">
                    {formatDate(item.created_at)}
                    {item.filename ? ` · ${item.filename}` : ""}
                  </p>
                </div>

                <div className="flex shrink-0 items-center gap-3">
                  <div className="text-right">
                    <div className="stat-value text-xl font-bold neon-text-cyan">{item.points}</div>
                    <div className="data-label mt-0.5">pts</div>
                  </div>
                  <ArrowRight size={15} className="text-[#3a4560]" />
                </div>
              </Link>
            ))}
          </div>
        )}
      </section>
    </AppShell>
  );
}
