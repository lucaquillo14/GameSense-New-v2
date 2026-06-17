"use client";

import { AppShell } from "@/components/AppShell";
import {
  createLeague,
  getMyLeagues,
  getStoredUser,
  joinLeague,
  type League,
} from "@/lib/socialApi";
import { AlertCircle, Loader2, Plus, Users } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

export default function LeaguesPage() {
  const router = useRouter();
  const [leagues, setLeagues] = useState<League[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const [joinCode, setJoinCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [signedOut, setSignedOut] = useState(false);

  function load() {
    setLoading(true);
    getMyLeagues()
      .then((data) => {
        setLeagues(data);
        setError(null);
      })
      .catch((err) => {
        const msg = err instanceof Error ? err.message : "Failed to load leagues.";
        if (msg.toLowerCase().includes("auth")) setSignedOut(true);
        else setError(msg);
      })
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    if (!getStoredUser()) {
      setSignedOut(true);
      setLoading(false);
      return;
    }
    load();
  }, []);

  async function onCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!newName.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const league = await createLeague(newName.trim());
      setNewName("");
      router.push(`/leagues/${league.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not create league.");
    } finally {
      setBusy(false);
    }
  }

  async function onJoin(e: React.FormEvent) {
    e.preventDefault();
    if (!joinCode.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const league = await joinLeague(joinCode.trim());
      setJoinCode("");
      router.push(`/leagues/${league.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not join league.");
    } finally {
      setBusy(false);
    }
  }

  if (signedOut) {
    return (
      <AppShell>
        <section className="analytics-grid hero-glow mx-auto max-w-2xl px-5 py-20 text-center fade-in">
          <div className="mx-auto mb-5 grid h-16 w-16 place-items-center rounded-2xl bg-gradient-to-br from-cyan-500 to-blue-600 text-[#04121f] shadow-[0_0_28px_rgba(6,182,212,0.45)]">
            <Users size={28} />
          </div>
          <h1 className="display text-3xl text-[#eef2ff]">Sign in to join leagues</h1>
          <p className="mx-auto mt-3 max-w-sm text-sm text-[#6b7a99]">
            Leagues let you and your friends compete on a private leaderboard.
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
            <Users size={11} />
            Private competition
          </div>
          <h1 className="display text-4xl text-[#eef2ff]">Your leagues</h1>
          <p className="mt-3 text-sm text-[#6b7a99]">
            Create a league, share the invite code, and compete with friends.
          </p>
        </div>

        <div className="mb-8 grid gap-4 sm:grid-cols-2">
          <form onSubmit={onCreate} className="card space-y-3 p-5">
            <h2 className="flex items-center gap-2 font-display font-semibold text-[#eef2ff]">
              <Plus size={16} className="text-cyan-400" /> Create a league
            </h2>
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="League name"
              className="w-full rounded-xl border border-white/[0.07] bg-[#09090f] px-4 py-2.5 text-[#eef2ff] placeholder:text-[#3a4560] transition-colors focus:border-cyan-500/50 focus:outline-none focus:ring-1 focus:ring-cyan-500/20"
            />
            <button type="submit" disabled={busy} className="btn-primary w-full px-4 py-2.5 text-sm disabled:opacity-60">
              Create
            </button>
          </form>

          <form onSubmit={onJoin} className="card space-y-3 p-5">
            <h2 className="flex items-center gap-2 font-display font-semibold text-[#eef2ff]">
              <Users size={16} className="text-cyan-400" /> Join with a code
            </h2>
            <input
              value={joinCode}
              onChange={(e) => setJoinCode(e.target.value.toUpperCase())}
              placeholder="e.g. 7XK2QP"
              maxLength={12}
              className="w-full rounded-xl border border-white/[0.07] bg-[#09090f] px-4 py-2.5 font-mono uppercase tracking-widest text-[#eef2ff] placeholder:text-[#3a4560] transition-colors focus:border-cyan-500/50 focus:outline-none focus:ring-1 focus:ring-cyan-500/20"
            />
            <button
              type="submit"
              disabled={busy}
              className="w-full rounded-xl border border-white/[0.07] bg-[#09090f] px-4 py-2.5 text-sm font-semibold text-[#eef2ff] transition-colors hover:border-cyan-500/40 disabled:opacity-60"
            >
              Join
            </button>
          </form>
        </div>

        {error && (
          <div className="mb-4 flex items-center gap-2 rounded-xl border border-red-500/25 bg-red-500/8 px-4 py-3 text-sm text-red-300">
            <AlertCircle size={16} />
            {error}
          </div>
        )}

        {loading ? (
          <div className="flex items-center justify-center gap-2 py-16 text-[#6b7a99]">
            <Loader2 size={20} className="animate-spin text-cyan-500" /> Loading…
          </div>
        ) : leagues.length === 0 ? (
          <div className="card p-10 text-center text-[#6b7a99]">
            You&apos;re not in any leagues yet. Create one or join with a friend&apos;s code.
          </div>
        ) : (
          <div className="space-y-3">
            {leagues.map((l) => (
              <Link
                key={l.id}
                href={`/leagues/${l.id}`}
                className="card card-hover flex items-center justify-between p-5"
              >
                <div className="flex items-center gap-3">
                  <span className="grid h-10 w-10 place-items-center rounded-xl bg-cyan-500/12 text-cyan-300">
                    <Users size={18} />
                  </span>
                  <div>
                    <div className="font-semibold text-[#eef2ff]">{l.name}</div>
                    <div className="text-xs text-[#6b7a99]">
                      {l.member_count} member{l.member_count === 1 ? "" : "s"}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="data-label">Invite code</div>
                  <div className="stat-value tracking-widest text-cyan-300">{l.invite_code}</div>
                </div>
              </Link>
            ))}
          </div>
        )}
      </section>
    </AppShell>
  );
}
