"use client";

import { Avatar } from "@/components/Avatar";
import { TierBadge } from "@/components/TierBadge";
import type { LeaderboardEntry, SortKey } from "@/lib/socialApi";
import { Crown } from "lucide-react";

function metricFor(entry: LeaderboardEntry, sort: SortKey): string {
  switch (sort) {
    case "speed":
      return entry.best_speed_kmh ? `${entry.best_speed_kmh.toFixed(1)} km/h` : "—";
    case "power":
      return entry.best_power_kmh ? `${entry.best_power_kmh.toFixed(1)} km/h` : "—";
    case "technique":
      return entry.best_technique ? entry.best_technique.toFixed(0) : "—";
    case "uploads":
      return `${entry.uploads}`;
    default:
      return entry.total_points.toLocaleString();
  }
}

const STEPS = {
  1: {
    order: "order-2",
    pad: "pt-0",
    avatar: 84,
    ring: "ring-2 ring-yellow-400/70 shadow-[0_0_36px_-6px_rgba(250,204,21,0.65)]",
    base: "h-28 bg-gradient-to-b from-yellow-400/15 to-transparent border-yellow-400/30",
    badge: "bg-yellow-400 text-yellow-950",
    glow: "bg-yellow-400/15",
  },
  2: {
    order: "order-1",
    pad: "pt-8",
    avatar: 64,
    ring: "ring-2 ring-slate-300/60 shadow-[0_0_24px_-8px_rgba(203,213,225,0.5)]",
    base: "h-20 bg-gradient-to-b from-slate-300/12 to-transparent border-slate-300/25",
    badge: "bg-slate-300 text-slate-900",
    glow: "bg-slate-300/10",
  },
  3: {
    order: "order-3",
    pad: "pt-8",
    avatar: 64,
    ring: "ring-2 ring-amber-600/60 shadow-[0_0_24px_-8px_rgba(217,119,6,0.5)]",
    base: "h-16 bg-gradient-to-b from-amber-600/12 to-transparent border-amber-600/25",
    badge: "bg-amber-600 text-amber-50",
    glow: "bg-amber-600/10",
  },
} as const;

export function Podium({
  entries,
  sort,
  highlightUserId,
}: {
  entries: LeaderboardEntry[];
  sort: SortKey;
  highlightUserId?: string | null;
}) {
  const top = entries.slice(0, 3);
  if (top.length < 3) return null; // podium only when there's a full top 3

  return (
    <div className="mb-8 grid grid-cols-3 items-end gap-3 sm:gap-5">
      {top.map((e) => {
        const step = STEPS[e.rank as 1 | 2 | 3] ?? STEPS[3];
        const isMe = highlightUserId === e.user_id;
        return (
          <div key={e.user_id} className={`flex flex-col items-center ${step.order} ${step.pad}`}>
            <div className="relative">
              {e.rank === 1 && (
                <Crown
                  size={22}
                  className="absolute -top-6 left-1/2 -translate-x-1/2 text-yellow-400 drop-shadow-[0_0_8px_rgba(250,204,21,0.7)]"
                />
              )}
              <div className={`rounded-full ${step.ring}`}>
                <Avatar name={e.display_name} url={e.avatar_url} size={step.avatar} />
              </div>
              <span
                className={`absolute -bottom-1.5 left-1/2 grid h-6 w-6 -translate-x-1/2 place-items-center rounded-full text-xs font-bold ${step.badge}`}
              >
                {e.rank}
              </span>
            </div>

            <div className="mt-3 flex max-w-full items-center gap-1.5">
              <span className="truncate text-sm font-semibold text-[#eef2ff]">{e.display_name}</span>
              <TierBadge badge={e.badge} size="xs" showLabel={false} />
            </div>
            {isMe && <span className="text-[10px] font-semibold text-cyan-400">you</span>}
            <span className="stat-value mt-0.5 text-sm text-cyan-300">{metricFor(e, sort)}</span>

            {/* pedestal */}
            <div className={`mt-3 w-full rounded-t-xl border-t border-x ${step.base}`}>
              <div className={`mx-auto mt-2 h-8 w-8 rounded-full blur-xl ${step.glow}`} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
