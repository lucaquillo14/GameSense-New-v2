"use client";

import { AppShell } from "@/components/AppShell";
import { getHistory, getStoredUser, type HistoryItem } from "@/lib/socialApi";
import { AlertCircle, ArrowRight, Clock, Gauge, Loader2 } from "lucide-react";
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

  useEffect(() => {
    if (!getStoredUser()) {
      setSignedOut(true);
      setLoading(false);
      return;
    }
    getHistory()
      .then((data) => {
        setItems(data);
        setError(null);
      })
      .catch((err) => {
        const msg = err instanceof Error ? err.message : "Failed to load history.";
        if (msg.toLowerCase().includes("auth")) setSignedOut(true);
        else setError(msg);
      })
      .finally(() => setLoading(false));
  }, []);

  if (signedOut) {
    return (
      <AppShell>
        <section className="mx-auto max-w-2xl px-5 py-20 text-center fade-in">
          <Clock size={40} className="mx-auto mb-4 text-[#3b82f6]" />
          <h1 className="text-2xl font-semibold text-[#f1f5f9]">Sign in to see your history</h1>
          <p className="mt-2 text-sm text-[#64748b]">
            Your past analyses are saved to your account when you upload while signed in.
          </p>
          <Link href="/login" className="btn-primary mt-6 inline-flex px-5 py-2.5">
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
          <h1 className="text-3xl font-semibold tracking-tight text-[#f1f5f9]">Session history</h1>
          <p className="mt-2 text-sm text-[#64748b]">Every clip you&apos;ve analysed, newest first.</p>
        </div>

        {loading ? (
          <div className="flex items-center justify-center gap-2 py-20 text-[#64748b]">
            <Loader2 size={20} className="animate-spin" /> Loading…
          </div>
        ) : error ? (
          <div className="flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
            <AlertCircle size={16} />
            {error}
          </div>
        ) : items.length === 0 ? (
          <div className="card p-10 text-center text-[#64748b]">
            No sessions yet. Upload a clip while signed in and it&apos;ll appear here.
          </div>
        ) : (
          <div className="space-y-3">
            {items.map((item) => (
              <Link
                key={item.video_id}
                href={`/results/${item.video_id}`}
                className="card flex items-center justify-between gap-4 p-5 transition-colors hover:border-[#3b82f6]/40"
              >
                <div className="min-w-0">
                  <div className="flex items-center gap-2 text-[#f1f5f9]">
                    <Gauge size={15} className="shrink-0 text-[#3b82f6]" />
                    <span className="font-medium">{MODE_LABELS[item.mode ?? ""] ?? "Analysis"}</span>
                  </div>
                  <div className="mt-1 truncate text-xs text-[#64748b]">
                    {formatDate(item.created_at)}
                    {item.filename ? ` · ${item.filename}` : ""}
                  </div>
                  <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-[#94a3b8]">
                    {item.max_speed_kmh > 0 && <span>Top speed {item.max_speed_kmh.toFixed(1)} km/h</span>}
                    {item.shot_power_kmh > 0 && <span>Shot power {item.shot_power_kmh.toFixed(1)} km/h</span>}
                    {item.technique_score > 0 && <span>Technique {item.technique_score.toFixed(0)}</span>}
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-3">
                  <div className="text-right">
                    <div className="text-lg font-bold tabular-nums text-[#3b82f6]">{item.points}</div>
                    <div className="text-xs text-[#64748b]">points</div>
                  </div>
                  <ArrowRight size={16} className="text-[#64748b]" />
                </div>
              </Link>
            ))}
          </div>
        )}
      </section>
    </AppShell>
  );
}
