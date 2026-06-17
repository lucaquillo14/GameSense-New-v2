"use client";

import { AlertTriangle, X } from "lucide-react";
import { useState } from "react";

// Internal calibration diagnostics that shouldn't be surfaced to users.
const HIDDEN_PATTERNS = [
  "calibration scale looks wrong",
  "Scale calibrated from the goal frame",
];

function isHidden(warning: string): boolean {
  return HIDDEN_PATTERNS.some((p) => warning.toLowerCase().includes(p.toLowerCase()));
}

export function WarningBanners({ warnings }: { warnings: string[] }) {
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const visible = warnings.filter((warning) => !dismissed.has(warning) && !isHidden(warning));
  if (!visible.length) return null;

  return (
    <div className="mb-4 space-y-2">
      {visible.map((warning) => (
        <div
          key={warning}
          className="fade-in flex items-start justify-between gap-3 rounded-lg border border-[#f59e0b]/30 bg-[#f59e0b]/10 px-4 py-3 text-sm text-amber-100"
        >
          <div className="flex items-start gap-2">
            <AlertTriangle size={16} className="mt-0.5 shrink-0 text-[#f59e0b]" />
            <span>{warning}</span>
          </div>
          <button
            type="button"
            onClick={() => setDismissed((current) => new Set(current).add(warning))}
            className="shrink-0 rounded p-1 text-amber-200/80 hover:bg-[#f59e0b]/20 hover:shadow-none"
            aria-label="Dismiss warning"
          >
            <X size={14} />
          </button>
        </div>
      ))}
    </div>
  );
}
