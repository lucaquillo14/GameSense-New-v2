"use client";

import { Crown, Gem } from "lucide-react";
import type { TierBadge as TierBadgeValue } from "@/lib/socialApi";

const STYLES = {
  pro: {
    label: "Pro",
    icon: Gem,
    className:
      "border-cyan-400/40 bg-cyan-500/12 text-cyan-300 shadow-[0_0_12px_rgba(6,182,212,0.25)]",
  },
  elite: {
    label: "Elite",
    icon: Crown,
    className:
      "border-amber-300/45 bg-gradient-to-r from-amber-400/15 to-yellow-300/15 text-amber-300 shadow-[0_0_14px_rgba(251,191,36,0.3)]",
  },
} as const;

/** Small membership badge. Renders nothing for free / null. */
export function TierBadge({
  badge,
  size = "sm",
  showLabel = true,
}: {
  badge: TierBadgeValue | undefined;
  size?: "xs" | "sm" | "md";
  showLabel?: boolean;
}) {
  if (badge !== "pro" && badge !== "elite") return null;
  const s = STYLES[badge];
  const Icon = s.icon;
  const sizing =
    size === "xs"
      ? "px-1.5 py-0.5 text-[9px] gap-0.5"
      : size === "md"
        ? "px-2.5 py-1 text-xs gap-1.5"
        : "px-2 py-0.5 text-[10px] gap-1";
  const iconSize = size === "md" ? 13 : size === "xs" ? 9 : 11;

  return (
    <span
      className={`inline-flex items-center rounded-full border font-bold uppercase tracking-wide ${sizing} ${s.className}`}
      title={`${s.label} member`}
    >
      <Icon size={iconSize} />
      {showLabel && s.label}
    </span>
  );
}

/** A tiny crown/gem dot for tight spaces (e.g. next to an avatar). */
export function TierGlyph({ badge }: { badge: TierBadgeValue | undefined }) {
  if (badge !== "pro" && badge !== "elite") return null;
  const s = STYLES[badge];
  const Icon = s.icon;
  return (
    <span
      className={`grid h-4 w-4 place-items-center rounded-full border ${s.className}`}
      title={`${s.label} member`}
    >
      <Icon size={9} />
    </span>
  );
}
