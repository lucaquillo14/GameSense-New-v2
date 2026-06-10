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
  trackedSoFar?: number;
  predictedSoFar?: number;
  lostSoFar?: number;
};

export function ProcessingProgress({
  stage,
  percent,
  message,
  frameLabel,
  trackedSoFar,
  predictedSoFar,
  lostSoFar,
}: Props) {
  const activeIndex = STAGES.findIndex((item) => item === stage);
  const resolvedIndex = activeIndex >= 0 ? activeIndex : stage === "queued" ? 0 : 1;
  const showTrackerStats =
    stage === "tracking" &&
    (trackedSoFar !== undefined || predictedSoFar !== undefined || lostSoFar !== undefined);

  return (
    <div className="card fade-in p-5">
      <div className="mb-2 flex items-center justify-between text-sm">
        <span className="font-medium text-[#f1f5f9]">{message ?? "Processing video"}</span>
        <span className="tabular-nums text-[#64748b]">{percent}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-[#ffffff14]">
        <div
          className="h-full rounded-full bg-[#3b82f6]"
          style={{ width: `${Math.min(Math.max(percent, 2), 100)}%`, transition: "width 400ms ease" }}
        />
      </div>
      {showTrackerStats ? (
        <div className="mt-3 flex flex-wrap gap-2 text-xs">
          <span className="rounded-full bg-[#10b981]/15 px-2.5 py-1 text-[#10b981]">
            Tracked {trackedSoFar ?? 0} frames
          </span>
          <span className="rounded-full bg-[#f59e0b]/15 px-2.5 py-1 text-[#f59e0b]">
            Predicted {predictedSoFar ?? 0} frames
          </span>
          <span className="rounded-full bg-[#64748b]/20 px-2.5 py-1 text-[#94a3b8]">
            Lost {lostSoFar ?? 0} frames
          </span>
        </div>
      ) : null}
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
