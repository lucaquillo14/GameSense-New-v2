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
import { AlertCircle, Globe, Loader2, Users } from "lucide-react";
import { useEffect, useState } from "react";

export default function LeaderboardPage() {
  const [sort, setSort] = useState<SortKey>("points");
  const [scope, setScope] = useState<LeaderboardScope>("global");
  const [entries, setEntries] = useState<LeaderboardEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Read after mount only — reading localStorage during render breaks SSR hydration.
  const [me, setMe] = useState<User | null>(null);
  useEffect(() => setMe(getStoredUser()), []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getLeaderboard(sort, scope)
      .then((data) => {
        if (!cancelled) {
          setEntries(data);
          setError(null);
        }
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load leaderboard.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [sort, scope]);

  return (
    <AppShell>
      <section className="mx-auto max-w-4xl px-5 py-12 fade-in">
        <div className="mb-8">
          <h1 className="text-3xl font-semibold tracking-tight text-[#f1f5f9]">Global leaderboard</h1>
          <p className="mt-2 text-sm text-[#64748b]">
            Earn points for every clip you upload, plus bonuses for your speed, shot power, and technique.
          </p>
        </div>

        {me && (
          <div className="mb-5 inline-flex rounded-lg border border-[#ffffff14] bg-[#111118] p-1 text-sm">
            <button
              type="button"
              onClick={() => setScope("global")}
              className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 font-medium transition-colors ${
                scope === "global" ? "bg-[#3b82f6] text-white" : "text-[#64748b]"
              }`}
            >
              <Globe size={14} /> Global
            </button>
            <button
              type="button"
              onClick={() => setScope("following")}
              className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 font-medium transition-colors ${
                scope === "following" ? "bg-[#3b82f6] text-white" : "text-[#64748b]"
              }`}
            >
              <Users size={14} /> Following
            </button>
          </div>
        )}

        {loading ? (
          <div className="flex items-center justify-center gap-2 py-20 text-[#64748b]">
            <Loader2 size={20} className="animate-spin" /> Loading rankings…
          </div>
        ) : error ? (
          <div className="flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
            <AlertCircle size={16} />
            {error}
          </div>
        ) : (
          <LeaderboardTable entries={entries} sort={sort} onSortChange={setSort} highlightUserId={me?.id} />
        )}
      </section>
    </AppShell>
  );
}
