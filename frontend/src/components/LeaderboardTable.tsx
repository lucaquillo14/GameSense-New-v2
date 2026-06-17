"use client";

import type { LeaderboardEntry, SortKey } from "@/lib/socialApi";
import { Avatar } from "@/components/Avatar";
import { TierBadge } from "@/components/TierBadge";
import { Gauge, Sparkles, Target, Trophy, Upload } from "lucide-react";

const SORTS: { key: SortKey; label: string; icon: React.ReactNode }[] = [
  { key: "points", label: "Points", icon: <Trophy size={13} /> },
  { key: "speed", label: "Speed", icon: <Gauge size={13} /> },
  { key: "power", label: "Shot power", icon: <Target size={13} /> },
  { key: "technique", label: "Technique", icon: <Sparkles size={13} /> },
  { key: "uploads", label: "Uploads", icon: <Upload size={13} /> },
];

const RANK_STYLES = [
  "bg-yellow-400 text-yellow-900 shadow-[0_0_10px_rgba(250,204,21,0.55)]",
  "bg-slate-300 text-slate-900 shadow-[0_0_8px_rgba(148,163,184,0.4)]",
  "bg-amber-600 text-amber-100 shadow-[0_0_8px_rgba(217,119,6,0.45)]",
];

export function LeaderboardTable({
  entries,
  sort,
  onSortChange,
  highlightUserId,
}: {
  entries: LeaderboardEntry[];
  sort: SortKey;
  onSortChange: (s: SortKey) => void;
  highlightUserId?: string | null;
}) {
  return (
    <div>
      {/* Sort filters */}
      <div className="mb-5 flex flex-wrap gap-2">
        {SORTS.map((s) => (
          <button
            key={s.key}
            type="button"
            onClick={() => onSortChange(s.key)}
            className={`flex items-center gap-1.5 rounded-full border px-3.5 py-1.5 text-xs font-semibold transition-all ${
              sort === s.key
                ? "border-cyan-500/35 bg-cyan-500/10 text-cyan-400 shadow-[0_0_10px_rgba(6,182,212,0.15)]"
                : "border-white/[0.07] bg-[#09090f] text-[#6b7a99] hover:border-white/[0.12] hover:text-[#eef2ff]"
            }`}
          >
            {s.icon}
            {s.label}
          </button>
        ))}
      </div>

      {entries.length === 0 ? (
        <div className="card p-10 text-center text-[#6b7a99]">
          No players ranked yet. Upload a clip while signed in to get on the board.
        </div>
      ) : (
        <div className="card overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/[0.06]">
                <th className="data-label px-4 py-3 text-left">#</th>
                <th className="data-label px-4 py-3 text-left">Player</th>
                <th className="data-label px-4 py-3 text-right">Points</th>
                <th className="data-label hidden px-4 py-3 text-right sm:table-cell">Speed</th>
                <th className="data-label hidden px-4 py-3 text-right sm:table-cell">Power</th>
                <th className="data-label hidden px-4 py-3 text-right sm:table-cell">Technique</th>
                <th className="data-label px-4 py-3 text-right">Uploads</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((e) => {
                const isMe = highlightUserId === e.user_id;
                return (
                  <tr
                    key={e.user_id}
                    className={`border-b border-white/[0.04] last:border-0 transition-colors ${
                      isMe
                        ? "bg-cyan-500/[0.07] shadow-[inset_0_0_0_1px_rgba(6,182,212,0.12)]"
                        : "hover:bg-white/[0.025]"
                    }`}
                  >
                    <td className="px-4 py-3.5">
                      <span
                        className={`inline-grid h-7 w-7 place-items-center rounded-full text-xs font-bold ${
                          RANK_STYLES[e.rank - 1] ?? "bg-[#12121e] text-[#6b7a99]"
                        }`}
                      >
                        {e.rank}
                      </span>
                    </td>
                    <td className="px-4 py-3.5 font-medium text-[#eef2ff]">
                      <span className="flex items-center gap-2">
                        <Avatar name={e.display_name} url={e.avatar_url} size={27} />
                        {e.display_name}
                        <TierBadge badge={e.badge} size="xs" showLabel={false} />
                        {isMe && (
                          <span className="rounded-full border border-cyan-500/25 bg-cyan-500/10 px-1.5 py-0.5 text-[10px] font-semibold text-cyan-400">
                            you
                          </span>
                        )}
                      </span>
                    </td>
                    <td className="px-4 py-3.5 text-right">
                      <span className="stat-value font-bold text-[#eef2ff]">
                        {e.total_points.toLocaleString()}
                      </span>
                    </td>
                    <td className="hidden px-4 py-3.5 text-right text-[#6b7a99] sm:table-cell">
                      {e.best_speed_kmh ? (
                        <span className="stat-value">{e.best_speed_kmh.toFixed(1)} km/h</span>
                      ) : "—"}
                    </td>
                    <td className="hidden px-4 py-3.5 text-right text-[#6b7a99] sm:table-cell">
                      {e.best_power_kmh ? (
                        <span className="stat-value">{e.best_power_kmh.toFixed(1)} km/h</span>
                      ) : "—"}
                    </td>
                    <td className="hidden px-4 py-3.5 text-right text-[#6b7a99] sm:table-cell">
                      {e.best_technique ? (
                        <span className="stat-value">{e.best_technique.toFixed(0)}</span>
                      ) : "—"}
                    </td>
                    <td className="px-4 py-3.5 text-right text-[#6b7a99]">{e.uploads}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
