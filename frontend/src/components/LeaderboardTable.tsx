"use client";

import type { LeaderboardEntry, SortKey } from "@/lib/socialApi";
import { Avatar } from "@/components/Avatar";
import { Gauge, Sparkles, Target, Trophy, Upload } from "lucide-react";

const SORTS: { key: SortKey; label: string; icon: React.ReactNode }[] = [
  { key: "points", label: "Points", icon: <Trophy size={14} /> },
  { key: "speed", label: "Speed", icon: <Gauge size={14} /> },
  { key: "power", label: "Shot power", icon: <Target size={14} /> },
  { key: "technique", label: "Technique", icon: <Sparkles size={14} /> },
  { key: "uploads", label: "Uploads", icon: <Upload size={14} /> },
];

const RANK_BADGE = ["bg-[#facc15] text-black", "bg-[#cbd5e1] text-black", "bg-[#d97706] text-white"];

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
      <div className="mb-4 flex flex-wrap gap-2">
        {SORTS.map((s) => (
          <button
            key={s.key}
            type="button"
            onClick={() => onSortChange(s.key)}
            className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium transition-colors ${
              sort === s.key
                ? "border-[#3b82f6] bg-[#3b82f6]/15 text-[#f1f5f9]"
                : "border-[#ffffff14] bg-[#111118] text-[#64748b] hover:text-[#f1f5f9]"
            }`}
          >
            {s.icon}
            {s.label}
          </button>
        ))}
      </div>

      {entries.length === 0 ? (
        <div className="card p-10 text-center text-[#64748b]">
          No players ranked yet. Upload a clip while signed in to get on the board.
        </div>
      ) : (
        <div className="card overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#ffffff14] text-left text-xs uppercase tracking-wide text-[#64748b]">
                <th className="px-4 py-3">#</th>
                <th className="px-4 py-3">Player</th>
                <th className="px-4 py-3 text-right">Points</th>
                <th className="hidden px-4 py-3 text-right sm:table-cell">Speed</th>
                <th className="hidden px-4 py-3 text-right sm:table-cell">Power</th>
                <th className="hidden px-4 py-3 text-right sm:table-cell">Technique</th>
                <th className="px-4 py-3 text-right">Uploads</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((e) => {
                const isMe = highlightUserId === e.user_id;
                return (
                  <tr
                    key={e.user_id}
                    className={`border-b border-[#ffffff08] last:border-0 ${
                      isMe ? "bg-[#3b82f6]/10" : "hover:bg-[#ffffff05]"
                    }`}
                  >
                    <td className="px-4 py-3">
                      <span
                        className={`inline-grid h-7 w-7 place-items-center rounded-full text-xs font-bold ${
                          RANK_BADGE[e.rank - 1] ?? "bg-[#1e293b] text-[#94a3b8]"
                        }`}
                      >
                        {e.rank}
                      </span>
                    </td>
                    <td className="px-4 py-3 font-medium text-[#f1f5f9]">
                      <span className="flex items-center gap-2">
                        <Avatar name={e.display_name} url={e.avatar_url} size={28} />
                        {e.display_name}
                        {isMe && <span className="text-xs text-[#3b82f6]">you</span>}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right font-semibold text-[#f1f5f9]">
                      {e.total_points.toLocaleString()}
                    </td>
                    <td className="hidden px-4 py-3 text-right text-[#94a3b8] sm:table-cell">
                      {e.best_speed_kmh ? `${e.best_speed_kmh.toFixed(1)} km/h` : "—"}
                    </td>
                    <td className="hidden px-4 py-3 text-right text-[#94a3b8] sm:table-cell">
                      {e.best_power_kmh ? `${e.best_power_kmh.toFixed(1)} km/h` : "—"}
                    </td>
                    <td className="hidden px-4 py-3 text-right text-[#94a3b8] sm:table-cell">
                      {e.best_technique ? e.best_technique.toFixed(0) : "—"}
                    </td>
                    <td className="px-4 py-3 text-right text-[#94a3b8]">{e.uploads}</td>
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
