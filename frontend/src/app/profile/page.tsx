"use client";

import { AppShell } from "@/components/AppShell";
import { Avatar } from "@/components/Avatar";
import {
  followUser,
  getFollowing,
  getProfile,
  getStoredUser,
  searchUsers,
  unfollowUser,
  updateStoredUser,
  uploadAvatar,
  type Badge,
  type Profile,
  type PublicUser,
} from "@/lib/socialApi";
import {
  Activity,
  AlertCircle,
  Award,
  CalendarCheck,
  Camera,
  Crosshair,
  Crown,
  Flame,
  Footprints,
  Gauge,
  Loader2,
  Lock,
  Medal,
  Search,
  Sparkles,
  Star,
  Target,
  Trophy,
  Upload,
  UserCircle,
  UserMinus,
  UserPlus,
  Zap,
} from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

const ICONS: Record<string, React.ComponentType<{ size?: number; className?: string }>> = {
  Footprints, Upload, CalendarCheck, Flame, Crown, Gauge, Zap, Target, Crosshair, Sparkles, Award, Star, Trophy, Medal,
};

const TIER_STYLES: Record<string, { ring: string; bg: string; text: string }> = {
  bronze: { ring: "border-amber-700/50", bg: "bg-amber-700/15", text: "text-amber-500" },
  silver: { ring: "border-slate-400/50", bg: "bg-slate-400/15", text: "text-slate-300" },
  gold: { ring: "border-yellow-400/50", bg: "bg-yellow-400/15", text: "text-yellow-400" },
};

const MODE_LABELS: Record<string, string> = {
  max_speed: "Max Speed",
  max_shot_power: "Max Shot Power",
  shooting_technique: "Shooting Technique",
};

export default function ProfilePage() {
  const [profile, setProfile] = useState<Profile | null>(null);
  const [following, setFollowing] = useState<PublicUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [signedOut, setSignedOut] = useState(false);

  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const [query, setQuery] = useState("");
  const [results, setResults] = useState<PublicUser[]>([]);
  const [searching, setSearching] = useState(false);

  useEffect(() => {
    if (!getStoredUser()) {
      setSignedOut(true);
      setLoading(false);
      return;
    }
    Promise.all([getProfile(), getFollowing()])
      .then(([p, f]) => {
        setProfile(p);
        setFollowing(f.following);
        setError(null);
      })
      .catch((err) => {
        const msg = err instanceof Error ? err.message : "Failed to load profile.";
        if (msg.toLowerCase().includes("auth")) setSignedOut(true);
        else setError(msg);
      })
      .finally(() => setLoading(false));
  }, []);

  // Debounced user search.
  useEffect(() => {
    if (query.trim().length < 2) {
      setResults([]);
      return;
    }
    setSearching(true);
    const id = setTimeout(() => {
      searchUsers(query)
        .then(setResults)
        .catch(() => setResults([]))
        .finally(() => setSearching(false));
    }, 300);
    return () => clearTimeout(id);
  }, [query]);

  async function onPickAvatar(file: File | null) {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const url = await uploadAvatar(file);
      updateStoredUser({ avatar_url: url });
      setProfile((p) => (p ? { ...p, user: { ...p.user, avatar_url: url } } : p));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not upload image.");
    } finally {
      setUploading(false);
    }
  }

  async function toggleFollow(target: PublicUser) {
    const next = !target.is_following;
    // optimistic update in both lists
    const apply = (u: PublicUser) => (u.id === target.id ? { ...u, is_following: next } : u);
    setResults((r) => r.map(apply));
    try {
      if (next) await followUser(target.id);
      else await unfollowUser(target.id);
      const f = await getFollowing();
      setFollowing(f.following);
      setProfile((p) => (p ? { ...p, follow_counts: f.counts } : p));
    } catch {
      // revert on error
      setResults((r) => r.map((u) => (u.id === target.id ? { ...u, is_following: target.is_following } : u)));
    }
  }

  if (signedOut) {
    return (
      <AppShell>
        <section className="mx-auto max-w-2xl px-5 py-20 text-center fade-in">
          <UserCircle size={40} className="mx-auto mb-4 text-[#3b82f6]" />
          <h1 className="text-2xl font-semibold text-[#f1f5f9]">Sign in to see your profile</h1>
          <p className="mt-2 text-sm text-[#64748b]">Track your stats, unlock badges, and follow friends.</p>
          <Link href="/login" className="btn-primary mt-6 inline-flex px-5 py-2.5">
            Sign in
          </Link>
        </section>
      </AppShell>
    );
  }

  const categories = profile ? Array.from(new Set(profile.badges.map((b) => b.category))) : [];

  return (
    <AppShell>
      <section className="mx-auto max-w-4xl px-5 py-12 fade-in">
        {loading ? (
          <div className="flex items-center justify-center gap-2 py-20 text-[#64748b]">
            <Loader2 size={20} className="animate-spin" /> Loading profile…
          </div>
        ) : error && !profile ? (
          <div className="flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
            <AlertCircle size={16} />
            {error}
          </div>
        ) : profile ? (
          <>
            <div className="mb-8 flex items-center gap-4">
              <div className="relative">
                <Avatar name={profile.user.display_name} url={profile.user.avatar_url} size={64} />
                <button
                  type="button"
                  onClick={() => fileRef.current?.click()}
                  disabled={uploading}
                  title="Change photo"
                  className="absolute -bottom-1 -right-1 grid h-7 w-7 place-items-center rounded-full border border-[#0a0a0f] bg-[#3b82f6] text-white hover:bg-[#2563eb]"
                >
                  {uploading ? <Loader2 size={13} className="animate-spin" /> : <Camera size={13} />}
                </button>
                <input
                  ref={fileRef}
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  className="sr-only"
                  onChange={(e) => void onPickAvatar(e.target.files?.[0] ?? null)}
                />
              </div>
              <div>
                <h1 className="text-3xl font-semibold tracking-tight text-[#f1f5f9]">{profile.user.display_name}</h1>
                <p className="mt-0.5 text-sm text-[#64748b]">
                  {profile.earned_count} badges · {profile.follow_counts.following} following ·{" "}
                  {profile.follow_counts.followers} followers
                </p>
              </div>
            </div>

            {error && (
              <div className="mb-4 flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
                <AlertCircle size={16} />
                {error}
              </div>
            )}

            <div className="mb-10 grid grid-cols-2 gap-3 sm:grid-cols-5">
              <StatCard icon={<Upload size={16} />} label="Uploads" value={`${profile.stats.uploads}`} />
              <StatCard icon={<Gauge size={16} />} label="Top speed" value={`${profile.stats.best_speed_kmh} km/h`} />
              <StatCard icon={<Target size={16} />} label="Shot power" value={`${profile.stats.best_power_kmh} km/h`} />
              <StatCard icon={<Sparkles size={16} />} label="Technique" value={`${profile.stats.best_technique || "—"}`} />
              <StatCard icon={<Trophy size={16} />} label="Points" value={`${profile.stats.total_points}`} />
            </div>

            {/* People: search + following */}
            <div className="mb-10">
              <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-[#64748b]">Find people</h2>
              <div className="relative mb-3">
                <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#475569]" />
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search by name or email"
                  className="w-full rounded-lg border border-[#ffffff14] bg-[#0a0a0f] py-2.5 pl-9 pr-3 text-[#f1f5f9] placeholder:text-[#475569] focus:border-[#3b82f6] focus:outline-none"
                />
              </div>
              {searching ? (
                <p className="text-sm text-[#64748b]">Searching…</p>
              ) : results.length > 0 ? (
                <div className="space-y-2">
                  {results.map((u) => (
                    <PersonRow key={u.id} person={u} onToggle={() => toggleFollow(u)} />
                  ))}
                </div>
              ) : query.trim().length >= 2 ? (
                <p className="text-sm text-[#64748b]">No users found.</p>
              ) : (
                <>
                  <h3 className="mb-2 mt-6 text-xs font-semibold uppercase tracking-wide text-[#475569]">
                    Following ({following.length})
                  </h3>
                  {following.length === 0 ? (
                    <p className="text-sm text-[#64748b]">You&apos;re not following anyone yet. Search above to find players.</p>
                  ) : (
                    <div className="space-y-2">
                      {following.map((u) => (
                        <PersonRow key={u.id} person={u} onToggle={() => toggleFollow(u)} />
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>

            {categories.map((cat) => (
              <div key={cat} className="mb-8">
                <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-[#64748b]">{cat}</h2>
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {profile.badges.filter((b) => b.category === cat).map((b) => (
                    <BadgeCard key={b.id} badge={b} />
                  ))}
                </div>
              </div>
            ))}

            {profile.recent_sessions.length > 0 && (
              <div className="mt-10">
                <div className="mb-3 flex items-center justify-between">
                  <h2 className="text-sm font-semibold uppercase tracking-wide text-[#64748b]">Recent sessions</h2>
                  <Link href="/history" className="text-xs text-[#3b82f6] hover:underline">
                    View all
                  </Link>
                </div>
                <div className="space-y-2">
                  {profile.recent_sessions.map((s) => (
                    <Link
                      key={s.video_id}
                      href={`/results/${s.video_id}`}
                      className="card flex items-center justify-between p-4 transition-colors hover:border-[#3b82f6]/40"
                    >
                      <div className="flex items-center gap-2 text-sm text-[#f1f5f9]">
                        <Activity size={15} className="text-[#3b82f6]" />
                        {MODE_LABELS[s.mode ?? ""] ?? "Analysis"}
                        {s.max_speed_kmh > 0 && (
                          <span className="text-xs text-[#64748b]">· {s.max_speed_kmh.toFixed(1)} km/h</span>
                        )}
                      </div>
                      <span className="text-sm font-bold tabular-nums text-[#3b82f6]">{s.points} pts</span>
                    </Link>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : null}
      </section>
    </AppShell>
  );
}

function PersonRow({ person, onToggle }: { person: PublicUser; onToggle: () => void }) {
  return (
    <div className="card flex items-center justify-between p-3">
      <div className="flex items-center gap-3">
        <Avatar name={person.display_name} url={person.avatar_url} size={36} />
        <span className="font-medium text-[#f1f5f9]">{person.display_name}</span>
      </div>
      {!person.is_self && (
        <button
          type="button"
          onClick={onToggle}
          className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
            person.is_following
              ? "border border-[#ffffff14] bg-[#111118] text-[#94a3b8] hover:text-[#f1f5f9]"
              : "bg-[#3b82f6] text-white hover:bg-[#2563eb]"
          }`}
        >
          {person.is_following ? <UserMinus size={14} /> : <UserPlus size={14} />}
          {person.is_following ? "Following" : "Follow"}
        </button>
      )}
    </div>
  );
}

function StatCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="card p-4">
      <div className="flex items-center gap-1.5 text-[#64748b]">
        {icon}
        <span className="text-xs">{label}</span>
      </div>
      <div className="mt-1 text-lg font-bold tabular-nums text-[#f1f5f9]">{value}</div>
    </div>
  );
}

function BadgeCard({ badge }: { badge: Badge }) {
  const Icon = ICONS[badge.icon] ?? Award;
  const tier = TIER_STYLES[badge.tier] ?? TIER_STYLES.bronze;
  const pct = Math.round(badge.progress * 100);
  const unitSuffix = badge.unit ? ` ${badge.unit}` : "";

  return (
    <div className={`card flex gap-3 p-4 ${badge.earned ? `${tier.ring} ${tier.bg}` : "border-[#ffffff14] opacity-80"}`}>
      <span
        className={`relative grid h-11 w-11 shrink-0 place-items-center rounded-lg ${
          badge.earned ? `${tier.bg} ${tier.text}` : "bg-[#0a0a0f] text-[#475569]"
        }`}
      >
        <Icon size={22} />
        {!badge.earned && (
          <span className="absolute -bottom-1 -right-1 grid h-5 w-5 place-items-center rounded-full border border-[#ffffff14] bg-[#0a0a0f] text-[#64748b]">
            <Lock size={10} />
          </span>
        )}
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-center justify-between gap-2">
          <p className={`font-semibold ${badge.earned ? "text-[#f1f5f9]" : "text-[#94a3b8]"}`}>{badge.name}</p>
          <span className={`text-[10px] font-semibold uppercase ${badge.earned ? tier.text : "text-[#475569]"}`}>
            {badge.tier}
          </span>
        </div>
        <p className="mt-0.5 text-xs text-[#64748b]">{badge.description}</p>
        {badge.earned ? (
          <p className={`mt-2 text-xs font-medium ${tier.text}`}>Unlocked</p>
        ) : (
          <div className="mt-2">
            <div className="h-1.5 overflow-hidden rounded-full bg-[#ffffff14]">
              <div className="h-full rounded-full bg-[#3b82f6]" style={{ width: `${Math.max(pct, 3)}%` }} />
            </div>
            <p className="mt-1 text-[11px] text-[#64748b]">
              {badge.current}
              {unitSuffix} / {badge.target}
              {unitSuffix}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
