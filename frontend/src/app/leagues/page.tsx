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
        <section className="mx-auto max-w-2xl px-5 py-20 text-center fade-in">
          <Users size={40} className="mx-auto mb-4 text-[#3b82f6]" />
          <h1 className="text-2xl font-semibold text-[#f1f5f9]">Sign in to join leagues</h1>
          <p className="mt-2 text-sm text-[#64748b]">
            Leagues let you and your friends compete on a private leaderboard.
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
          <h1 className="text-3xl font-semibold tracking-tight text-[#f1f5f9]">Your leagues</h1>
          <p className="mt-2 text-sm text-[#64748b]">
            Create a league, share the invite code, and compete with friends.
          </p>
        </div>

        <div className="mb-8 grid gap-4 sm:grid-cols-2">
          <form onSubmit={onCreate} className="card space-y-3 p-5">
            <h2 className="flex items-center gap-2 font-medium text-[#f1f5f9]">
              <Plus size={16} className="text-[#3b82f6]" /> Create a league
            </h2>
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="League name"
              className="w-full rounded-lg border border-[#ffffff14] bg-[#0a0a0f] px-3 py-2.5 text-[#f1f5f9] placeholder:text-[#475569] focus:border-[#3b82f6] focus:outline-none"
            />
            <button type="submit" disabled={busy} className="btn-primary w-full px-4 py-2.5 disabled:opacity-60">
              Create
            </button>
          </form>

          <form onSubmit={onJoin} className="card space-y-3 p-5">
            <h2 className="flex items-center gap-2 font-medium text-[#f1f5f9]">
              <Users size={16} className="text-[#3b82f6]" /> Join with a code
            </h2>
            <input
              value={joinCode}
              onChange={(e) => setJoinCode(e.target.value.toUpperCase())}
              placeholder="e.g. 7XK2QP"
              maxLength={12}
              className="w-full rounded-lg border border-[#ffffff14] bg-[#0a0a0f] px-3 py-2.5 font-mono uppercase tracking-widest text-[#f1f5f9] placeholder:text-[#475569] focus:border-[#3b82f6] focus:outline-none"
            />
            <button
              type="submit"
              disabled={busy}
              className="w-full rounded-lg border border-[#ffffff14] bg-[#111118] px-4 py-2.5 font-medium text-[#f1f5f9] transition-colors hover:border-[#3b82f6]/40 disabled:opacity-60"
            >
              Join
            </button>
          </form>
        </div>

        {error && (
          <div className="mb-4 flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
            <AlertCircle size={16} />
            {error}
          </div>
        )}

        {loading ? (
          <div className="flex items-center justify-center gap-2 py-16 text-[#64748b]">
            <Loader2 size={20} className="animate-spin" /> Loading…
          </div>
        ) : leagues.length === 0 ? (
          <div className="card p-10 text-center text-[#64748b]">
            You&apos;re not in any leagues yet. Create one or join with a friend&apos;s code.
          </div>
        ) : (
          <div className="space-y-3">
            {leagues.map((l) => (
              <Link
                key={l.id}
                href={`/leagues/${l.id}`}
                className="card flex items-center justify-between p-5 transition-colors hover:border-[#3b82f6]/40"
              >
                <div>
                  <div className="font-medium text-[#f1f5f9]">{l.name}</div>
                  <div className="text-xs text-[#64748b]">
                    {l.member_count} member{l.member_count === 1 ? "" : "s"}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-xs text-[#64748b]">Invite code</div>
                  <div className="font-mono tracking-widest text-[#3b82f6]">{l.invite_code}</div>
                </div>
              </Link>
            ))}
          </div>
        )}
      </section>
    </AppShell>
  );
}
