"use client";

import { useEffect, useMemo, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { ProcessingPreview } from "@/components/ProcessingPreview";
import { ProcessingProgress } from "@/components/ProcessingProgress";
import { ConfidenceDot } from "@/components/SpeedChart";
import { WarningBanners } from "@/components/WarningBanners";
import { VideoPlaybackOverlay } from "@/components/VideoPlaybackOverlay";
import { getDetectionsOverlay, getFrame, getResults, mediaUrl, Metrics, ShotMetrics, VideoResult } from "@/lib/api";
import type { DetectionsOverlay } from "@/lib/overlay";
import { Activity, Crosshair, Crown, Footprints, Loader2, Lock, Route, Target, Trophy, Zap } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";

type ResultsTab = "overview" | "heatmaps" | "playback";

function isShotMetrics(results: Metrics | ShotMetrics): results is ShotMetrics {
  return "peak_shot_speed_kmh" in results;
}

export default function ResultsPage() {
  const params = useParams<{ videoId: string }>();
  const videoId = params.videoId;
  const [result, setResult] = useState<VideoResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [bestShotFrameUrl, setBestShotFrameUrl] = useState<string | null>(null);
  const [overlay, setOverlay] = useState<DetectionsOverlay | null>(null);
  const [overlayLoading, setOverlayLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<ResultsTab>("overview");
  const [pollMs, setPollMs] = useState(1500);

  useEffect(() => {
    function onVisibilityChange() {
      setPollMs(document.hidden ? 5000 : 1500);
    }
    onVisibilityChange();
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => document.removeEventListener("visibilitychange", onVisibilityChange);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const refresh = async () => {
      try {
        const next = await getResults(videoId);
        if (!cancelled) setResult(next);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Could not load results.");
      }
    };
    refresh();
    const interval = window.setInterval(refresh, pollMs);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [videoId, pollMs]);

  const results = result?.results;
  const analysisMode = result?.mode ?? "max_speed";
  const progress = result?.progress;
  const isProcessing =
    !result ||
    result.status === "processing" ||
    (result.status !== "complete" && result.status !== "failed" && (progress?.percent ?? 0) < 100);
  const progressPercent = Math.max(progress?.percent ?? (isProcessing ? 2 : 100), isProcessing ? 2 : 0);
  const target = result?.target_player as { player_id?: string; team_label?: string } | null | undefined;

  const speedMetrics = results && !isShotMetrics(results) ? results : null;
  const shotMetrics = results && isShotMetrics(results) ? results : null;
  const confidenceScore = results?.confidence_score ?? 0;

  const playerLabel =
    speedMetrics?.player_label ??
    shotMetrics?.player_label ??
    (target?.player_id as string | undefined) ??
    "Player";
  const teamLabel = speedMetrics?.team_label ?? shotMetrics?.team_label ?? (target?.team_label as string | undefined);

  useEffect(() => {
    if (activeTab !== "playback" || result?.status !== "complete") return;
    const detectionsPath = result.assets?.detections_json ?? `/media/${videoId}/detections.json`;
    let cancelled = false;
    setOverlayLoading(true);
    getDetectionsOverlay(detectionsPath)
      .then((payload) => { if (!cancelled) setOverlay(payload); })
      .catch(() => { if (!cancelled) setOverlay(null); })
      .finally(() => { if (!cancelled) setOverlayLoading(false); });
    return () => { cancelled = true; };
  }, [activeTab, result?.status, result?.assets?.detections_json, videoId]);

  useEffect(() => {
    if (!shotMetrics?.best_shot) { setBestShotFrameUrl(null); return; }
    let cancelled = false;
    getFrame(videoId, shotMetrics.best_shot.frame_id)
      .then((frame) => { if (!cancelled) setBestShotFrameUrl(mediaUrl(frame.frame_url)); })
      .catch(() => { if (!cancelled) setBestShotFrameUrl(null); });
    return () => { cancelled = true; };
  }, [shotMetrics?.best_shot, videoId]);

  const sortedShots = useMemo(
    () => (shotMetrics ? [...shotMetrics.shots].sort((a, b) => b.ball_speed_kmh - a.ball_speed_kmh) : []),
    [shotMetrics],
  );

  return (
    <AppShell>
      <section className="analytics-grid mx-auto max-w-7xl px-5 py-8 fade-in">

        {/* ── Page header ─────────────────────────────────────── */}
        <div className="mb-8">
          <div className="chip mb-4 w-fit">
            <Trophy size={11} />
            Results
          </div>
          <h1 className="display text-4xl text-[#eef2ff]">Performance dashboard</h1>
          <p className="mt-2 text-sm text-[#6b7a99]">
            {result?.status ?? "loading"}
            {teamLabel ? ` · ${playerLabel} — ${teamLabel}` : playerLabel ? ` · ${playerLabel}` : ""}
          </p>
        </div>

        {error && (
          <div className="mb-4 rounded-xl border border-red-500/25 bg-red-500/8 px-4 py-3 text-sm text-red-300">
            {error}
          </div>
        )}

        <WarningBanners warnings={result?.warnings ?? []} />

        {isProcessing && (
          <>
            <ProcessingProgress
              stage={progress?.stage}
              percent={progressPercent}
              message={progress?.message}
              trackedSoFar={progress?.tracked_so_far}
              predictedSoFar={progress?.predicted_so_far}
              lostSoFar={progress?.lost_so_far}
            />
            <ProcessingPreview videoId={videoId} active />
          </>
        )}

        {result?.status === "complete" && (
          <div className="mb-6 flex flex-wrap gap-2">
            <TabButton active={activeTab === "overview"} onClick={() => setActiveTab("overview")} label="Overview" />
            <TabButton active={activeTab === "heatmaps"} onClick={() => setActiveTab("heatmaps")} label="Heatmaps" />
            <TabButton active={activeTab === "playback"} onClick={() => setActiveTab("playback")} label="Playback" />
          </div>
        )}

        {result?.status === "complete" && activeTab === "playback" && result.source_url && result.video_metadata ? (
          <div className="mb-6">
            {overlayLoading ? (
              <div className="card grid aspect-video place-items-center text-sm text-[#6b7a99]">
                <Loader2 className="mb-2 animate-spin text-cyan-500" size={22} />
                Loading detection overlay…
              </div>
            ) : (
              <VideoPlaybackOverlay
                videoUrl={mediaUrl(result.source_url) ?? ""}
                fps={result.video_metadata.fps}
                frameCount={Math.max(result.video_metadata.frame_count - 1, 0)}
                durationS={result.video_metadata.duration_s}
                overlay={overlay}
                targetPlayerId={(result.target_player as { player_id?: string } | null)?.player_id ?? overlay?.target_id}
                speedSeries={speedMetrics?.speed_series}
              />
            )}
          </div>
        ) : null}

        {result?.status === "complete" && activeTab === "heatmaps" ? (
          result.locked_features?.includes("heatmaps") ? (
            <UpgradeLock
              title="Heatmaps are a Pro feature"
              body="Upgrade to Pro to unlock movement & touch heatmaps, advanced analytics, and downloadable reports."
            />
          ) : (
            <HeatmapsPanel
              movementUrl={result.assets?.movement_heatmap}
              touchUrl={result.assets?.touch_heatmap}
              touchCount={results?.touch_count}
              passCount={results?.pass_count}
            />
          )
        ) : null}

        {/* ── Speed metrics ─────────────────────────────────────── */}
        {speedMetrics && analysisMode !== "max_shot_power" && result?.status === "complete" && activeTab === "overview" ? (
          <div className="mt-2 space-y-4">
            <TrackingResultWarnings metrics={speedMetrics} />

            {/* Hero stat card */}
            <div className="card relative overflow-hidden p-8">
              <div className="pointer-events-none absolute -right-12 -top-12 h-48 w-48 rounded-full bg-cyan-500/8 blur-3xl" />
              <div className="pointer-events-none absolute -bottom-8 left-1/3 h-32 w-32 rounded-full bg-blue-500/6 blur-2xl" />
              <div className="grid gap-8 sm:grid-cols-2">
                <div>
                  <p className="data-label">Top speed</p>
                  <div className="mt-3 flex items-end gap-2">
                    <span className="stat-value text-7xl font-bold leading-none neon-text-cyan">
                      {speedMetrics.units === "pixels"
                        ? speedMetrics.top_speed_px_per_s
                        : (speedMetrics.max_speed_kmh ?? speedMetrics.top_speed_kmh)}
                    </span>
                    <span className="mb-1.5 text-xl text-[#6b7a99]">
                      {speedMetrics.units === "pixels" ? "px/s" : "km/h"}
                    </span>
                  </div>
                </div>
                <div>
                  <p className="data-label">Distance covered</p>
                  <div className="mt-3 flex items-end gap-2">
                    <span className="stat-value text-7xl font-bold leading-none text-[#eef2ff]">
                      {speedMetrics.units === "pixels"
                        ? (speedMetrics.total_distance_px ?? speedMetrics.total_distance_m)
                        : (speedMetrics.distance_m ?? speedMetrics.total_distance_m)}
                    </span>
                    <span className="mb-1.5 text-xl text-[#6b7a99]">
                      {speedMetrics.units === "pixels" ? "px" : "m"}
                    </span>
                  </div>
                </div>
              </div>
              <div className="mt-6 flex items-center justify-between border-t border-white/[0.06] pt-5">
                <p className="text-sm font-medium text-[#6b7a99]">
                  {playerLabel}
                  {teamLabel ? ` · ${teamLabel}` : ""}
                </p>
                <ConfidenceDot score={confidenceScore} />
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              <MetricCard
                icon={<Activity size={16} />}
                label="Average speed"
                value={
                  speedMetrics.units === "pixels"
                    ? `${speedMetrics.avg_speed_px_per_s} px/s`
                    : `${speedMetrics.avg_speed_kmh} km/h`
                }
              />
              <MetricCard
                icon={<Footprints size={16} />}
                label="Sprint distance"
                value={
                  speedMetrics.units === "pixels"
                    ? `${speedMetrics.sprint_distance_px ?? speedMetrics.sprint_distance_m ?? 0} px`
                    : `${speedMetrics.sprint_distance_m ?? 0} m`
                }
              />
              <MetricCard
                icon={<Zap size={16} />}
                label="Sprints"
                value={`${speedMetrics.sprint_count ?? 0}`}
              />
            </div>
          </div>

        /* ── Shot metrics ─────────────────────────────────────── */
        ) : shotMetrics && analysisMode === "max_shot_power" && result?.status === "complete" && activeTab === "overview" ? (
          <div className="mt-2 space-y-4">
            <div className="card relative overflow-hidden p-8">
              <div className="pointer-events-none absolute -right-12 -top-12 h-48 w-48 rounded-full bg-cyan-500/8 blur-3xl" />
              <p className="data-label">Peak shot speed</p>
              <div className="mt-3 flex flex-wrap items-end justify-between gap-4">
                <div className="flex items-end gap-2">
                  <span className="stat-value text-7xl font-bold leading-none neon-text-cyan">
                    {shotMetrics.peak_shot_speed_kmh}
                  </span>
                  <span className="mb-1.5 text-xl text-[#6b7a99]">km/h</span>
                </div>
                <ConfidenceDot score={confidenceScore} />
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <MetricCard icon={<Target size={16} />} label="Shots detected" value={`${shotMetrics.shot_count}`} />
              <MetricCard
                icon={<Crosshair size={16} />}
                label="Best shot"
                value={
                  shotMetrics.best_shot
                    ? `${shotMetrics.best_shot.ball_speed_kmh} km/h @ ${shotMetrics.best_shot.timestamp_s.toFixed(1)}s`
                    : "—"
                }
              />
            </div>

            {shotMetrics.best_shot && bestShotFrameUrl && (
              <img
                src={bestShotFrameUrl}
                alt="Best shot frame"
                className="card aspect-video w-full object-cover"
                loading="lazy"
              />
            )}

            <div className="card p-5">
              <h2 className="data-label mb-4">All shots — ranked by speed</h2>
              {sortedShots.length ? (
                <div className="space-y-2">
                  {sortedShots.map((shot, index) => (
                    <div
                      key={`${shot.frame_id}-${index}`}
                      className="flex items-center justify-between rounded-xl border border-white/[0.06] bg-[#09090f] px-4 py-3"
                    >
                      <div>
                        <p className="font-semibold text-[#eef2ff]">
                          #{index + 1} · Frame {shot.frame_id}
                        </p>
                        <p className="text-xs text-[#6b7a99]">{shot.timestamp_s.toFixed(2)}s</p>
                      </div>
                      <span className="stat-value text-lg font-bold neon-text-cyan">
                        {shot.ball_speed_kmh} km/h
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="rounded-xl border border-white/[0.06] bg-[#09090f] p-6 text-center text-sm text-[#6b7a99]">
                  <p className="font-semibold text-[#eef2ff]">No shots detected</p>
                  <p className="mt-2">
                    Check that you selected the correct player and that the clip includes visible
                    shots with enough length.
                  </p>
                </div>
              )}
            </div>
          </div>

        ) : result?.status === "failed" ? (
          <div className="card mt-6 grid min-h-48 place-items-center p-8 text-[#6b7a99]">
            Processing failed. Try re-uploading or selecting a clearer player frame.
          </div>
        ) : null}
      </section>
    </AppShell>
  );
}

function TabButton({ active, onClick, label }: { active: boolean; onClick: () => void; label: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-lg px-5 py-2 text-sm font-semibold transition-all ${
        active
          ? "bg-cyan-500/10 text-cyan-400 shadow-[inset_0_0_0_1px_rgba(6,182,212,0.3)]"
          : "bg-white/[0.04] text-[#6b7a99] hover:text-[#eef2ff]"
      }`}
    >
      {label}
    </button>
  );
}

function HeatmapsPanel({
  movementUrl,
  touchUrl,
  touchCount,
  passCount,
}: {
  movementUrl?: string | null;
  touchUrl?: string | null;
  touchCount?: number;
  passCount?: number;
}) {
  if (!movementUrl && !touchUrl) {
    return (
      <div className="card grid min-h-48 place-items-center p-8 text-sm text-[#6b7a99]">
        Heatmaps are not available for this analysis. Pitch calibration is required for field
        heatmaps.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {(touchCount !== undefined || passCount !== undefined) && (
        <div className="flex flex-wrap gap-2">
          {touchCount !== undefined && (
            <span className="rounded-full border border-white/[0.08] bg-white/[0.05] px-3 py-1 text-xs font-semibold text-[#eef2ff]">
              {touchCount} touches
            </span>
          )}
          {passCount !== undefined && (
            <span className="rounded-full border border-cyan-500/20 bg-cyan-500/10 px-3 py-1 text-xs font-semibold text-cyan-400">
              {passCount} passes
            </span>
          )}
        </div>
      )}
      <div className="grid gap-4 md:grid-cols-2">
        {movementUrl && (
          <div className="card overflow-hidden p-3">
            <p className="data-label mb-3">Movement heatmap</p>
            <img
              src={mediaUrl(movementUrl) ?? ""}
              alt="Movement heatmap"
              className="w-full rounded-lg"
              loading="lazy"
            />
          </div>
        )}
        {touchUrl && (
          <div className="card overflow-hidden p-3">
            <p className="data-label mb-3">Touch &amp; pass heatmap</p>
            <img
              src={mediaUrl(touchUrl) ?? ""}
              alt="Touch heatmap"
              className="w-full rounded-lg"
              loading="lazy"
            />
          </div>
        )}
      </div>
    </div>
  );
}

function TrackingResultWarnings({ metrics }: { metrics: Metrics }) {
  const tracked = metrics.tracked_frames ?? metrics.usable_track_points ?? 0;
  const predicted = metrics.predicted_frames ?? 0;
  const lost = metrics.lost_frames ?? 0;
  const total = tracked + predicted + lost;
  const warnings: string[] = [];

  if (total > 0 && lost / total > 0.2) {
    warnings.push(
      "The selected player was out of view for a significant portion of the clip. Speed and distance may be understated.",
    );
  }
  if ((metrics.confidence_score ?? 0) < 0.5) {
    warnings.push("Tracking confidence is low. Results should be treated as estimates.");
  }
  if (tracked > 0 && predicted / tracked > 0.3) {
    warnings.push(
      "Significant portions of the track were predicted rather than directly observed.",
    );
  }

  if (!warnings.length) return null;

  return (
    <div className="space-y-1.5">
      {warnings.map((warning) => (
        <p key={warning} className="text-sm text-amber-400">
          {warning}
        </p>
      ))}
    </div>
  );
}

function UpgradeLock({ title, body }: { title: string; body: string }) {
  return (
    <div className="card relative overflow-hidden p-8 text-center">
      <div className="pointer-events-none absolute -right-12 -top-12 h-48 w-48 rounded-full bg-cyan-500/8 blur-3xl" />
      <span className="mx-auto mb-4 grid h-14 w-14 place-items-center rounded-2xl bg-cyan-500/12 text-cyan-300">
        <Lock size={24} />
      </span>
      <h3 className="font-display text-xl font-bold text-[#eef2ff]">{title}</h3>
      <p className="mx-auto mt-2 max-w-md text-sm text-[#6b7a99]">{body}</p>
      <Link
        href="/pricing"
        className="btn-primary mt-6 inline-flex items-center gap-2 px-6 py-3 text-sm"
      >
        <Crown size={16} /> View plans
      </Link>
    </div>
  );
}

function MetricCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="card p-5">
      <div className="data-label flex items-center gap-1.5">
        <span className="text-cyan-500">{icon}</span>
        {label}
      </div>
      <p className="stat-value mt-3 text-2xl text-[#eef2ff]">{value}</p>
    </div>
  );
}
