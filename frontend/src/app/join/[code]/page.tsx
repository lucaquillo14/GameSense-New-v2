"use client";

import { AppShell } from "@/components/AppShell";
import {
  getLeaguePreview,
  getStoredUser,
  joinLeague,
  type LeaguePreview,
} from "@/lib/socialApi";
import { AlertCircle, Loader2, LogIn, Trophy, Users } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";

export default function JoinLeaguePage() {
  const params = useParams<{ code: string }>();
  const router = useRouter();
  const code = (params.code || "").toUpperCase();

  const [preview, setPreview] = useState<LeaguePreview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [joining, setJoining] = useState(false);
  // Read after mount only — reading localStorage during render breaks SSR hydration.
  const [signedIn, setSignedIn] = useState(false);
  useEffect(() => setSignedIn(!!getStoredUser()), []);

  useEffect(() => {
    getLeaguePreview(code)
      .then((p) => {
        setPreview(p);
        setError(null);
      })
      .catch((err) => setError(err instanceof Error ? err.message : "League not found."))
      .finally(() => setLoading(false));
  }, [code]);

  async function join() {
    setJoining(true);
    setError(null);
    try {
      const league = await joinLeague(code);
      router.push(`/leagues/${league.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not join the league.");
      setJoining(false);
    }
  }

  return (
    <AppShell>
      <section className="mx-auto max-w-md px-5 py-20 text-center fade-in">
        {loading ? (
          <div className="flex items-center justify-center gap-2 py-10 text-[#64748b]">
            <Loader2 size={20} className="animate-spin" /> Loading invite…
          </div>
        ) : error && !preview ? (
          <>
            <AlertCircle size={36} className="mx-auto mb-4 text-red-300" />
            <h1 className="text-2xl font-semibold text-[#f1f5f9]">Invite not found</h1>
            <p className="mt-2 text-sm text-[#64748b]">{error}</p>
            <Link href="/leagues" className="btn-primary mt-6 inline-flex px-5 py-2.5">
              Go to leagues
            </Link>
          </>
        ) : preview ? (
          <>
            <span className="mx-auto mb-4 grid h-14 w-14 place-items-center rounded-xl bg-[#3b82f6] text-white">
              <Trophy size={26} />
            </span>
            <p className="text-sm text-[#64748b]">You&apos;ve been invited to join</p>
            <h1 className="mt-1 text-3xl font-semibold tracking-tight text-[#f1f5f9]">{preview.name}</h1>
            <p className="mt-2 flex items-center justify-center gap-1.5 text-sm text-[#64748b]">
              <Users size={15} /> {preview.member_count} member{preview.member_count === 1 ? "" : "s"} competing
            </p>

            {error && (
              <div className="mt-5 flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-left text-sm text-red-200">
                <AlertCircle size={16} />
                {error}
              </div>
            )}

            {signedIn ? (
              <button
                type="button"
                onClick={join}
                disabled={joining}
                className="btn-primary mt-6 inline-flex w-full items-center justify-center gap-2 px-5 py-3 disabled:opacity-60"
              >
                {joining ? <Loader2 size={18} className="animate-spin" /> : <Users size={18} />}
                {joining ? "Joining…" : "Join league"}
              </button>
            ) : (
              <>
                <p className="mt-6 text-sm text-[#64748b]">Sign in to join this league.</p>
                <Link
                  href={`/login?next=${encodeURIComponent(`/join/${code}`)}`}
                  className="btn-primary mt-3 inline-flex w-full items-center justify-center gap-2 px-5 py-3"
                >
                  <LogIn size={18} /> Sign in to join
                </Link>
              </>
            )}
          </>
        ) : null}
      </section>
    </AppShell>
  );
}
