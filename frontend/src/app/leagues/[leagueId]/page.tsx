"use client";

import { AppShell } from "@/components/AppShell";
import { LeaderboardTable } from "@/components/LeaderboardTable";
import {
  getLeagueDetail,
  getStoredUser,
  leaveLeague,
  type League,
  type LeaderboardEntry,
  type SortKey,
  type User,
} from "@/lib/socialApi";
import { AlertCircle, ArrowLeft, Check, Copy, Link2, Loader2, LogOut } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";

export default function LeagueDetailPage() {
  const params = useParams<{ leagueId: string }>();
  const router = useRouter();
  const leagueId = params.leagueId;
  // Read after mount only — reading localStorage during render breaks SSR hydration.
  const [me, setMe] = useState<User | null>(null);
  useEffect(() => setMe(getStoredUser()), []);

  const [league, setLeague] = useState<League | null>(null);
  const [entries, setEntries] = useState<LeaderboardEntry[]>([]);
  const [sort, setSort] = useState<SortKey>("points");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [linkCopied, setLinkCopied] = useState(false);

  function shareLink(): string {
    if (typeof window === "undefined" || !league) return "";
    return `${window.location.origin}/join/${league.invite_code}`;
  }

  function copyLink() {
    if (!league) return;
    navigator.clipboard.writeText(shareLink()).then(() => {
      setLinkCopied(true);
      setTimeout(() => setLinkCopied(false), 1500);
    });
  }

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getLeagueDetail(leagueId, sort)
      .then((data) => {
        if (!cancelled) {
          setLeague(data.league);
          setEntries(data.entries);
          setError(null);
        }
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load league.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [leagueId, sort]);

  function copyCode() {
    if (!league) return;
    navigator.clipboard.writeText(league.invite_code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  async function onLeave() {
    if (!confirm("Leave this league?")) return;
    try {
      await leaveLeague(leagueId);
      router.push("/leagues");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not leave league.");
    }
  }

  return (
    <AppShell>
      <section className="mx-auto max-w-4xl px-5 py-12 fade-in">
        <Link href="/leagues" className="mb-6 inline-flex items-center gap-1.5 text-sm text-[#64748b] hover:text-[#f1f5f9]">
          <ArrowLeft size={15} /> All leagues
        </Link>

        {loading ? (
          <div className="flex items-center justify-center gap-2 py-20 text-[#64748b]">
            <Loader2 size={20} className="animate-spin" /> Loading league…
          </div>
        ) : error ? (
          <div className="flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
            <AlertCircle size={16} />
            {error}
          </div>
        ) : league ? (
          <>
            <div className="mb-8 flex flex-wrap items-start justify-between gap-4">
              <div>
                <h1 className="text-3xl font-semibold tracking-tight text-[#f1f5f9]">{league.name}</h1>
                <p className="mt-1 text-sm text-[#64748b]">
                  {league.member_count} member{league.member_count === 1 ? "" : "s"} competing
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={copyCode}
                  title="Copy invite code"
                  className="flex items-center gap-2 rounded-lg border border-[#ffffff14] bg-[#111118] px-3 py-2 text-sm transition-colors hover:border-[#3b82f6]/40"
                >
                  {copied ? <Check size={15} className="text-green-400" /> : <Copy size={15} className="text-[#3b82f6]" />}
                  <span className="font-mono tracking-widest text-[#f1f5f9]">{league.invite_code}</span>
                </button>
                <button
                  type="button"
                  onClick={copyLink}
                  className="flex items-center gap-2 rounded-lg bg-[#3b82f6] px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-[#2563eb]"
                >
                  {linkCopied ? <Check size={15} /> : <Link2 size={15} />}
                  {linkCopied ? "Link copied" : "Share link"}
                </button>
                <button
                  type="button"
                  onClick={onLeave}
                  title="Leave league"
                  className="rounded-lg border border-[#ffffff14] bg-[#111118] p-2 text-[#94a3b8] transition-colors hover:text-red-300"
                >
                  <LogOut size={16} />
                </button>
              </div>
            </div>

            <p className="mb-6 rounded-lg border border-[#ffffff14] bg-[#111118] px-4 py-3 text-sm text-[#94a3b8]">
              Share the link or code <span className="font-mono font-semibold text-[#3b82f6]">{league.invite_code}</span>{" "}
              with friends — anyone who opens the link can join and compete.
            </p>

            <LeaderboardTable entries={entries} sort={sort} onSortChange={setSort} highlightUserId={me?.id} />
          </>
        ) : null}
      </section>
    </AppShell>
  );
}
