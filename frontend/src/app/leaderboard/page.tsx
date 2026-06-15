"use client";

import { AppShell } from "@/components/AppShell";
import { LeaderboardTable } from "@/components/LeaderboardTable";
import {
  getLeaderboard,
  getStoredUser,
  type LeaderboardEntry,
  type LeaderboardScope,
  type SortKey,
  type User,
} from "@/lib/socialApi";
import { AlertCircle, Globe, Loader2, Trophy, Users } from "lucide-react";
import { useEffect, useState } from "react";

export default function LeaderboardPage() {
  const [sort, setSort] = useState<SortKey>("points");
  const [scope, setScope] = useState<LeaderboardScope>("global");
  const [entries, setEntries] = useState<LeaderboardEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [me, setMe] = useState<User | null>(null);
  useEffect(() => setMe(getStoredUser()), []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getLeaderboard(sort, scope)
      .then((data) => {
        if (!cancelled) { setEntries(data); setError(null); }
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load leaderboard.");
      })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [sort, scope]);

  return (
    <AppShell>
      <section className="analytics-grid mx-auto max-w-4xl px-5 py-12 fade-in">

        {/* Header */}
        <div className="mb-10">
          <div className="chip mb-5 w-fit">
            <Trophy size={11} />
            Global rankings
          </div>
          <h1 className="display text-4xl text-[#eef2ff]">Leaderboard</h1>
          <p className="mt-3 text-sm text-[#6b7a99]">
            Earn points for every clip you upload, plus bonuses for speed, shot power, and
            technique.
          </p>
        </div>

        {/* Scope toggle */}
        {me && (
          <div className="mb-6 inline-flex rounded-xl border border-white/[0.07] bg-[#09090f] p-1 text-sm">
            <button
              type="button"
              onClick={() => setScope("global")}
              className={`flex items-center gap-1.5 rounded-lg px-4 py-2 font-semibold transition-all ${
                scope === "global"
                  ? "bg-cyan-500 text-[#04121f] shadow-[0_0_14px_rgba(6,182,212,0.4)]"
                  : "text-[#6b7a99] hover:text-[#eef2ff]"
              }`}
            >
              <Globe size={13} /> Global
            </button>
            <button
              type="button"
              onClick={() => setScope("following")}
              className={`flex items-center gap-1.5 rounded-lg px-4 py-2 font-semibold transition-all ${
                scope === "following"
                  ? "bg-cyan-500 text-[#04121f] shadow-[0_0_14px_rgba(6,182,212,0.4)]"
                  : "text-[#6b7a99] hover:text-[#eef2ff]"
              }`}
            >
              <Users size={13} /> Following
            </button>
          </div>
        )}

        {loading ? (
          <div className="flex items-center justify-center gap-2 py-20 text-[#6b7a99]">
            <Loader2 size={20} className="animate-spin text-cyan-500" /> Loading rankings…
          </div>
        ) : error ? (
          <div className="flex items-center gap-2 rounded-xl border border-red-500/25 bg-red-500/8 px-4 py-3 text-sm text-red-300">
            <AlertCircle size={16} />
            {error}
          </div>
        ) : (
          <LeaderboardTable
            entries={entries}
            sort={sort}
            onSortChange={setSort}
            highlightUserId={me?.id}
          />
        )}
      </section>
    </AppShell>
  );
}
