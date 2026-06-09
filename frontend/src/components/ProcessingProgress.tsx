"use client";

const STAGES = ["calibration", "tracking", "filtering", "shot_detection", "metrics", "complete"] as const;

const STAGE_LABELS: Record<string, string> = {
  queued: "Queued",
  calibration: "Calibration",
  tracking: "Tracking",
  filtering: "Filtering",
  shot_detection: "Shot detection",
  metrics: "Metrics",
  complete: "Complete",
  failed: "Failed",
};

type Props = {
  stage?: string;
  percent: number;
  message?: string;
  frameLabel?: string;
};

export function ProcessingProgress({ stage, percent, message, frameLabel }: Props) {
  const activeIndex = STAGES.findIndex((item) => item === stage);
  const resolvedIndex = activeIndex >= 0 ? activeIndex : stage === "queued" ? 0 : 1;

  return (
    <div className="card fade-in p-5">
      <div className="mb-2 flex items-center justify-between text-sm">
        <span className="font-medium text-[#f1f5f9]">{message ?? "Processing video"}</span>
        <span className="tabular-nums text-[#64748b]">{percent}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-[#ffffff14]">
        <div
          className="h-full rounded-full bg-[#3b82f6] transition-all duration-500 ease-out"
          style={{ width: `${Math.min(Math.max(percent, 2), 100)}%` }}
        />
      </div>
      <div className="mt-4 flex flex-wrap gap-3 text-xs">
        {STAGES.slice(0, 5).map((item, index) => {
          const active = index === resolvedIndex;
          const done = index < resolvedIndex;
          return (
            <span
              key={item}
              className={`rounded-full px-2.5 py-1 ${
                active
                  ? "stage-pulse bg-[#3b82f6]/20 font-semibold text-[#3b82f6]"
                  : done
                    ? "bg-[#10b981]/15 text-[#10b981]"
                    : "bg-[#ffffff08] text-[#64748b]"
              }`}
            >
              {STAGE_LABELS[item]}
            </span>
          );
        })}
      </div>
      {frameLabel && <p className="mt-3 text-xs text-[#64748b]">{frameLabel}</p>}
    </div>
  );
}
