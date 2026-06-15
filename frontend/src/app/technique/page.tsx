"use client";

import { useCallback, useEffect, useState } from "react";
import { AppShell } from "@/components/AppShell";
import {
  getResults,
  isImageMediaUrl,
  mediaUrl,
  processVideo,
  ShootingFeedback,
  techniqueAngleDeg,
  uploadVideo,
} from "@/lib/api";
import { formatDuration, MAX_UPLOAD_MB, validateFileSize } from "@/lib/uploadLimits";
import {
  AlertCircle,
  ArrowLeft,
  Loader2,
  Target,
  UploadCloud,
  Zap,
} from "lucide-react";
import Link from "next/link";

const MAX_TECHNIQUE_DURATION_S = 30;

type LocalMeta = {
  name: string;
  sizeMb: string;
  durationS: number;
  width: number;
  height: number;
};

type ViewState = "upload" | "processing" | "results";

function readTechniqueMeta(file: File): Promise<LocalMeta> {
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.preload = "metadata";
    video.muted = true;
    video.playsInline = true;

    const cleanup = () => {
      video.removeAttribute("src");
      video.load();
      URL.revokeObjectURL(objectUrl);
    };

    video.onloadedmetadata = () => {
      const durationS = video.duration;
      const width = video.videoWidth;
      const height = video.videoHeight;
      cleanup();
      if (!Number.isFinite(durationS) || durationS <= 0) {
        reject(new Error("Could not read video duration."));
        return;
      }
      if (durationS > MAX_TECHNIQUE_DURATION_S) {
        reject(
          new Error(
            `This clip is ${formatDuration(durationS)} long. Technique analysis supports clips up to ${MAX_TECHNIQUE_DURATION_S} seconds.`,
          ),
        );
        return;
      }
      resolve({
        name: file.name,
        sizeMb: (file.size / (1024 * 1024)).toFixed(1),
        durationS,
        width,
        height,
      });
    };

    video.onerror = () => {
      cleanup();
      reject(new Error("Could not read video metadata."));
    };

    video.src = objectUrl;
  });
}

const IDEAL_RANGES: Record<string, string> = {
  "Backswing Knee Flexion": "75–115°",
  "Knee at Contact": "140–170°",
  "Ankle Lock": "< 12° variation",
  "Plant Foot Distance": "5–30 cm",
  "Approach Angle": "25–50°",
  "Hip Rotation": "25–70°",
  "Trunk Lean": "5–25°",
  "Follow-through Height": "≥ 0.55× leg",
  "Shot Distance": "—",
};

function isWithinIdeal(label: string, value: number): boolean {
  if (label === "Backswing Knee Flexion") return value >= 75 && value <= 115;
  if (label === "Knee at Contact") return value >= 140 && value <= 170;
  if (label === "Ankle Lock") return value >= 0 && value <= 12;
  if (label === "Plant Foot Distance") return value >= 5 && value <= 30;
  if (label === "Approach Angle") return value >= 25 && value <= 50;
  if (label === "Hip Rotation") return value >= 25 && value <= 70;
  if (label === "Trunk Lean") return value >= 5 && value <= 25;
  if (label === "Follow-through Height") return value >= 0.55;
  return true;
}

function feedbackBorderClass(text: string): string {
  const lower = text.toLowerCase();
  if (
    lower.includes("ideal window") ||
    lower.includes("textbook") ||
    lower.includes("excellent") ||
    lower.includes("strong") ||
    lower.includes("good body position")
  ) {
    return "border-l-[#10b981]";
  }
  const critical = ["ankle", "knee", "plant foot", "balloon", "could not", "not detected"];
  if (critical.some((term) => lower.includes(term))) {
    return "border-l-red-500";
  }
  return "border-l-amber-500";
}

function feedbackTitle(text: string): string {
  const lower = text.toLowerCase();
  if (lower.includes("backswing")) return "Backswing loading";
  if (lower.includes("plant foot")) return "Plant foot spacing";
  if (lower.includes("trunk")) return "Trunk lean";
  if (lower.includes("hip rotation") || lower.includes("pelvis")) return "Hip rotation";
  if (lower.includes("knee")) return "Knee angle";
  if (lower.includes("ankle")) return "Ankle lock";
  if (lower.includes("follow-through")) return "Follow-through";
  if (lower.includes("approach")) return "Approach angle";
  if (lower.includes("side-on") || lower.includes("footed strike")) return "Filming tip";
  return "Technique note";
}

function formatMetricValue(label: string, value: number | null): string {
  if (value == null) return "—";
  if (label === "Shot Distance") return `${value.toFixed(1)} m`;
  if (label.includes("Distance")) return `${value.toFixed(0)} cm`;
  if (label.includes("Follow-through")) return `${value.toFixed(2)}× leg`;
  if (label === "Ankle Lock") return `${value.toFixed(0)}° var`;
  return `${value.toFixed(0)}°`;
}

function ScoreRing({ score }: { score: number }) {
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const progress = Math.min(Math.max(score / 10, 0), 1);
  const offset = circumference * (1 - progress);
  const color = score >= 7 ? "#10b981" : score >= 4 ? "#f59e0b" : "#ef4444";

  return (
    <div className="relative mx-auto h-36 w-36">
      <svg className="h-full w-full -rotate-90" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r={radius} fill="none" stroke="#ffffff14" strokeWidth="10" />
        <circle
          cx="60"
          cy="60"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="absolute inset-0 grid place-items-center">
        <span className="text-3xl font-semibold text-[#f1f5f9]">{score.toFixed(1)}</span>
        <span className="mt-10 text-xs text-[#64748b]">/ 10</span>
      </div>
    </div>
  );
}

export default function TechniquePage() {
  const [view, setView] = useState<ViewState>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [localMeta, setLocalMeta] = useState<LocalMeta | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [checking, setChecking] = useState(false);
  const [videoId, setVideoId] = useState<string | null>(null);
  const [progress, setProgress] = useState({ percent: 0, message: "Starting analysis" });
  const [feedback, setFeedback] = useState<ShootingFeedback | null>(null);
  const [resultsTab, setResultsTab] = useState<"video" | "contact" | "feedback">("video");

  async function onFileSelected(next: File | null) {
    setError(null);
    setLocalMeta(null);
    if (!next) {
      setFile(null);
      return;
    }
    const sizeError = validateFileSize(next);
    if (sizeError) {
      setFile(null);
      setError(sizeError);
      return;
    }
    setChecking(true);
    try {
      const meta = await readTechniqueMeta(next);
      setFile(next);
      setLocalMeta(meta);
    } catch (err) {
      setFile(null);
      setError(err instanceof Error ? err.message : "Could not validate this video.");
    } finally {
      setChecking(false);
    }
  }

  const pollResults = useCallback(async (id: string) => {
    const record = await getResults(id);
    if (record.progress) {
      setProgress({
        percent: record.progress.percent ?? 0,
        message: record.progress.message ?? "Processing",
      });
    }
    if (record.status === "complete" && record.shooting_result) {
      setFeedback(record.shooting_result);
      setView("results");
      return true;
    }
    if (record.status === "failed") {
      setError(record.warnings?.join(" ") || "Processing failed.");
      setView("upload");
      return true;
    }
    return false;
  }, []);

  useEffect(() => {
    if (view !== "processing" || !videoId) return;
    let cancelled = false;
    const tick = async () => {
      if (cancelled) return;
      try {
        const done = await pollResults(videoId);
        if (!done && !cancelled) {
          window.setTimeout(tick, 1500);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Could not fetch results.");
          setView("upload");
        }
      }
    };
    void tick();
    return () => {
      cancelled = true;
    };
  }, [view, videoId, pollResults]);

  async function submit() {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const upload = await uploadVideo(file);
      setVideoId(upload.video_id);
      setView("processing");
      setProgress({ percent: 5, message: "Upload complete — starting analysis" });
      await processVideo(upload.video_id, "shooting_technique");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
      setView("upload");
    } finally {
      setBusy(false);
    }
  }

  const metricsTable = feedback
    ? [
        {
          label: "Backswing Knee Flexion",
          value:
            feedback.backswing_knee_flexion_deg && feedback.backswing_knee_flexion_deg > 0
              ? feedback.backswing_knee_flexion_deg
              : techniqueAngleDeg(feedback, "backswing_knee_flexion"),
        },
        {
          label: "Knee at Contact",
          value:
            feedback.knee_bend_at_contact_deg > 0
              ? feedback.knee_bend_at_contact_deg
              : techniqueAngleDeg(feedback, "knee_angle"),
        },
        {
          label: "Ankle Lock",
          value:
            feedback.ankle_lock_variation_deg && feedback.ankle_lock_variation_deg > 0
              ? feedback.ankle_lock_variation_deg
              : techniqueAngleDeg(feedback, "ankle_lock", "ankle_angle"),
        },
        {
          label: "Plant Foot Distance",
          value: feedback.plant_foot_distance_cm > 0 ? feedback.plant_foot_distance_cm : null,
        },
        {
          label: "Approach Angle",
          value:
            feedback.approach_angle_deg > 0
              ? feedback.approach_angle_deg
              : techniqueAngleDeg(feedback, "approach_angle"),
        },
        {
          label: "Hip Rotation",
          value:
            feedback.hip_rotation_deg > 0
              ? feedback.hip_rotation_deg
              : techniqueAngleDeg(feedback, "hip_rotation"),
        },
        {
          label: "Trunk Lean",
          value: techniqueAngleDeg(feedback, "trunk_lean"),
        },
        {
          label: "Follow-through Height",
          value:
            feedback.follow_through_height_ratio && feedback.follow_through_height_ratio > 0
              ? feedback.follow_through_height_ratio
              : null,
        },
        {
          label: "Shot Distance",
          value: feedback.shot_distance_m && feedback.shot_distance_m > 0 ? feedback.shot_distance_m : null,
        },
      ]
    : [];

  const annotatedUrl = mediaUrl(feedback?.annotated_video_url);
  const annotatedIsImage = isImageMediaUrl(feedback?.annotated_video_url);

  return (
    <AppShell>
      <section className="mx-auto max-w-6xl px-5 py-10 fade-in">
        <div className="mb-8">
          <Link href="/" className="inline-flex items-center gap-2 text-sm text-[#64748b] hover:text-[#f1f5f9]">
            <ArrowLeft size={16} />
            Back to match analysis
          </Link>
          <h1 className="mt-4 text-3xl font-semibold text-[#f1f5f9] sm:text-4xl">Shooting technique analysis</h1>
          <p className="mt-2 max-w-2xl text-[#64748b]">
            Upload a clip of you shooting at goal — side or 45° angle works best. RF-DETR detects player and ball,
            MediaPipe tracks pose, and you get an annotated video with biomechanics feedback. Up to{" "}
            {MAX_TECHNIQUE_DURATION_S} seconds.
          </p>
        </div>

        {view === "upload" && (
          <div className="card p-6">
            <label className="dropzone-hover group flex min-h-64 w-full cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-[#ffffff14] bg-[#0a0a0f] px-6 py-10 text-center">
              <span className="grid h-14 w-14 place-items-center rounded-xl bg-[#2563eb] text-white shadow-[0_0_28px_rgba(37,99,235,0.35)]">
                {checking ? <Loader2 size={28} className="animate-spin" /> : <UploadCloud size={28} />}
              </span>
              <span className="mt-4 text-lg font-semibold text-[#f1f5f9]">
                {checking ? "Checking video…" : file ? file.name : "Upload your shooting clip"}
              </span>
              <span className="mt-2 text-sm text-[#64748b]">
                MP4 or MOV · up to {MAX_UPLOAD_MB} MB · max {MAX_TECHNIQUE_DURATION_S}s
              </span>
              <input
                type="file"
                accept="video/mp4,video/quicktime"
                className="sr-only"
                onChange={(event) => void onFileSelected(event.target.files?.[0] ?? null)}
              />
            </label>

            {localMeta && (
              <p className="mt-3 text-sm text-[#64748b]">
                {localMeta.sizeMb} MB · {formatDuration(localMeta.durationS)} · {localMeta.width}×{localMeta.height}
              </p>
            )}

            {error && (
              <div className="mt-4 flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
                <AlertCircle size={16} />
                {error}
              </div>
            )}

            <button
              type="button"
              onClick={() => void submit()}
              disabled={!file || !localMeta || busy || checking}
              className="btn-primary mt-5 flex w-full items-center justify-center gap-2 px-4 py-3 disabled:cursor-not-allowed"
            >
              {busy ? <Loader2 size={18} className="animate-spin" /> : <Target size={18} />}
              {busy ? "Uploading…" : "Analyse technique"}
            </button>
          </div>
        )}

        {view === "processing" && (
          <div className="card p-6">
            <div className="flex items-center gap-3">
              <Loader2 className="animate-spin text-[#3b82f6]" size={22} />
              <div>
                <p className="font-medium text-[#f1f5f9]">{progress.message}</p>
                <p className="text-sm text-[#64748b]">Detecting objects, tracking pose, and rendering annotated output…</p>
              </div>
            </div>
            <div className="mt-4 h-2 overflow-hidden rounded-full bg-[#ffffff14]">
              <div
                className="h-full rounded-full bg-[#3b82f6] transition-all duration-500"
                style={{ width: `${Math.max(progress.percent, 4)}%` }}
              />
            </div>
          </div>
        )}

        {view === "results" && feedback && (
          <div className="space-y-8">
            <div className="card flex flex-col items-center gap-6 p-6 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-6">
                <div className="text-center">
                  <p className="mb-2 text-sm font-medium text-[#64748b]">Technique score</p>
                  <ScoreRing score={feedback.technique_score} />
                </div>
                <div className="flex items-center gap-3 rounded-xl border border-[#ffffff14] bg-[#111118] px-6 py-4">
                  <Zap className="text-[#3b82f6]" size={28} />
                  <div>
                    <p className="text-xs uppercase tracking-wide text-[#64748b]">Shot power</p>
                    {feedback.shot_power_kmh > 0 ? (
                      <>
                        <p className="text-4xl font-semibold tabular-nums text-[#f1f5f9]">
                          {feedback.shot_power_kmh.toFixed(0)}
                          <span className="ml-1 text-lg font-normal text-[#64748b]">km/h</span>
                        </p>
                        {feedback.power_rating ? (
                          <p className="text-sm text-[#64748b]">{feedback.power_rating}</p>
                        ) : null}
                      </>
                    ) : (
                      <>
                        <p className="text-4xl font-semibold text-[#64748b]">—</p>
                        <p className="max-w-44 text-xs text-[#64748b]">
                          Couldn&apos;t track the ball cleanly after impact — a side-on angle and 60 fps give the most
                          reliable speed
                        </p>
                      </>
                    )}
                  </div>
                </div>
              </div>
              <div className="flex flex-wrap justify-center gap-2 text-xs">
                {feedback.kicking_foot ? (
                  <span className="rounded-full border border-[#ffffff14] bg-[#111118] px-3 py-1.5 text-[#94a3b8]">
                    {feedback.kicking_foot === "left" ? "Left" : "Right"}-footed strike
                  </span>
                ) : null}
                {feedback.shot_distance_m && feedback.shot_distance_m > 0 ? (
                  <span className="rounded-full border border-[#ffffff14] bg-[#111118] px-3 py-1.5 text-[#94a3b8]">
                    ~{feedback.shot_distance_m.toFixed(1)} m from goal
                  </span>
                ) : null}
                {feedback.on_target != null ? (
                  <span
                    className={`rounded-full border px-3 py-1.5 ${
                      feedback.on_target
                        ? "border-[#10b98140] bg-[#10b98115] text-[#10b981]"
                        : "border-[#ef444440] bg-[#ef444415] text-[#ef4444]"
                    }`}
                  >
                    {feedback.on_target ? "On target" : "Off target"}
                    {feedback.goal_crossing_height_m
                      ? ` — crossed ${feedback.goal_crossing_height_m.toFixed(1)} m high, ${Math.abs(
                          feedback.goal_crossing_offset_m ?? 0,
                        ).toFixed(1)} m ${(feedback.goal_crossing_offset_m ?? 0) >= 0 ? "right" : "left"} of centre`
                      : ""}
                  </span>
                ) : null}
                {feedback.contact_frame_id != null ? (
                  <span className="rounded-full border border-[#ffffff14] bg-[#111118] px-3 py-1.5 text-[#94a3b8]">
                    Contact at frame {feedback.contact_frame_id}
                  </span>
                ) : null}
                {feedback.confidence > 0 ? (
                  <span className="rounded-full border border-[#ffffff14] bg-[#111118] px-3 py-1.5 text-[#94a3b8]">
                    Confidence {(feedback.confidence * 100).toFixed(0)}%
                  </span>
                ) : null}
                {feedback.scale_source ? (
                  <span className="rounded-full border border-[#ffffff14] bg-[#111118] px-3 py-1.5 text-[#94a3b8]">
                    Scale: {feedback.scale_source}
                  </span>
                ) : null}
              </div>
            </div>

            <div className="flex gap-2 border-b border-[#ffffff14]">
              {(
                [
                  ["video", "Annotated video"],
                  ["contact", "Contact frame"],
                  ["feedback", "Feedback & metrics"],
                ] as const
              ).map(([tab, label]) => (
                <button
                  key={tab}
                  type="button"
                  onClick={() => setResultsTab(tab)}
                  className={`-mb-px rounded-t-lg border-b-2 px-4 py-2.5 text-sm font-medium transition-colors ${
                    resultsTab === tab
                      ? "border-[#3b82f6] text-[#f1f5f9]"
                      : "border-transparent text-[#64748b] hover:text-[#94a3b8]"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>

            {resultsTab === "video" && (
              <div className="mx-auto max-w-3xl space-y-3">
                {annotatedUrl ? (
                  annotatedIsImage ? (
                    <img
                      src={annotatedUrl}
                      alt="Shooting technique contact frame overlay"
                      className="w-full rounded-xl border border-[#ffffff14] bg-black object-contain"
                    />
                  ) : (
                    <video
                      src={annotatedUrl}
                      className="w-full rounded-xl border border-[#ffffff14] bg-black"
                      autoPlay
                      muted
                      loop
                      playsInline
                      controls
                    />
                  )
                ) : (
                  <div className="flex h-64 items-center justify-center rounded-xl border border-[#ffffff14] bg-[#111118] text-[#64748b]">
                    Annotated output not available
                  </div>
                )}
                {!annotatedIsImage && annotatedUrl && (
                  <p className="text-center text-xs text-[#64748b]">
                    Skeleton, phase bar and shot power overlays, then a 1/3x slow-motion replay of the strike and a
                    summary card. Full feedback lives in the Feedback &amp; metrics tab.
                  </p>
                )}
              </div>
            )}

            {resultsTab === "contact" && (
              <div className="mx-auto max-w-3xl space-y-4">
                {feedback.contact_frame_url ? (
                  <img
                    src={mediaUrl(feedback.contact_frame_url) ?? undefined}
                    alt="Point of contact with pose skeleton and joint angles"
                    className="w-full rounded-xl border border-[#ffffff14] bg-black object-contain"
                  />
                ) : (
                  <div className="flex h-64 items-center justify-center rounded-xl border border-[#ffffff14] bg-[#111118] text-[#64748b]">
                    Contact frame image not available — re-run the analysis to generate it
                  </div>
                )}
                <p className="text-center text-xs text-[#64748b]">
                  The exact frame of ball contact{feedback.contact_frame_id != null ? ` (frame ${feedback.contact_frame_id})` : ""} with
                  skeleton and joint angles
                </p>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                  {metricsTable
                    .filter((row) =>
                      ["Knee at Contact", "Ankle Lock", "Trunk Lean", "Hip Rotation", "Plant Foot Distance", "Approach Angle"].includes(
                        row.label,
                      ),
                    )
                    .map((row) => {
                      const ok = row.value != null && isWithinIdeal(row.label, row.value);
                      return (
                        <div key={row.label} className="rounded-xl border border-[#ffffff14] bg-[#111118] px-4 py-3">
                          <p className="text-xs text-[#64748b]">{row.label}</p>
                          <p
                            className={`mt-1 text-xl font-semibold tabular-nums ${
                              row.value == null ? "text-[#64748b]" : ok ? "text-[#10b981]" : "text-red-400"
                            }`}
                          >
                            {formatMetricValue(row.label, row.value)}
                          </p>
                          <p className="text-xs text-[#64748b]">ideal {IDEAL_RANGES[row.label]}</p>
                        </div>
                      );
                    })}
                </div>
              </div>
            )}

            {resultsTab === "feedback" && (
              <div className="grid gap-8 lg:grid-cols-2">
                <div className="space-y-3">
                  <h2 className="text-xl font-semibold text-[#f1f5f9]">Feedback</h2>
                  <ol className="space-y-3">
                    {feedback.feedback_points.map((point, index) => (
                      <li
                        key={`${index}-${point.slice(0, 24)}`}
                        className={`rounded-lg border border-[#ffffff14] border-l-4 bg-[#111118] p-4 ${feedbackBorderClass(point)}`}
                      >
                        <p className="font-medium text-[#f1f5f9]">{feedbackTitle(point)}</p>
                        <p className="mt-1 text-sm leading-relaxed text-[#94a3b8]">{point}</p>
                      </li>
                    ))}
                  </ol>
                </div>

                <div className="space-y-3">
                  <h2 className="text-xl font-semibold text-[#f1f5f9]">Measurements</h2>
                  <div className="overflow-hidden rounded-xl border border-[#ffffff14]">
                    <table className="w-full text-sm">
                      <thead className="bg-[#111118] text-left text-[#64748b]">
                        <tr>
                          <th className="px-4 py-3 font-medium">Metric</th>
                          <th className="px-4 py-3 font-medium">Measured</th>
                          <th className="px-4 py-3 font-medium">Ideal</th>
                        </tr>
                      </thead>
                      <tbody>
                        {metricsTable.map((row) => {
                          const ok = row.value != null && isWithinIdeal(row.label, row.value);
                          return (
                            <tr key={row.label} className="border-t border-[#ffffff14]">
                              <td className="px-4 py-3 text-[#f1f5f9]">{row.label}</td>
                              <td
                                className={`px-4 py-3 tabular-nums ${row.value == null ? "text-[#64748b]" : ok ? "text-[#10b981]" : "text-red-400"}`}
                              >
                                {formatMetricValue(row.label, row.value)}
                              </td>
                              <td className="px-4 py-3 text-[#64748b]">{IDEAL_RANGES[row.label]}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </section>
    </AppShell>
  );
}
