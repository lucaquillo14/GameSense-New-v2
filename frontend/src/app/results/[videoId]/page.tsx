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
import { Activity, Crosshair, Footprints, Loader2, Route, Target, Trophy, Zap } from "lucide-react";
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
      .then((payload) => {
        if (!cancelled) setOverlay(payload);
      })
      .catch(() => {
        if (!cancelled) setOverlay(null);
      })
      .finally(() => {
        if (!cancelled) setOverlayLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [activeTab, result?.status, result?.assets?.detections_json, videoId]);

  useEffect(() => {
    if (!shotMetrics?.best_shot) {
      setBestShotFrameUrl(null);
      return;
    }
    let cancelled = false;
    getFrame(videoId, shotMetrics.best_shot.frame_id)
      .then((frame) => {
        if (!cancelled) setBestShotFrameUrl(mediaUrl(frame.frame_url));
      })
      .catch(() => {
        if (!cancelled) setBestShotFrameUrl(null);
      });
    return () => {
      cancelled = true;
    };
  }, [shotMetrics?.best_shot, videoId]);

  const sortedShots = useMemo(
    () => (shotMetrics ? [...shotMetrics.shots].sort((a, b) => b.ball_speed_kmh - a.ball_speed_kmh) : []),
    [shotMetrics],
  );

  return (
    <AppShell>
      <section className="analytics-grid mx-auto max-w-7xl px-5 py-8 fade-in">
        <div className="mb-6">
          <div className="inline-flex items-center gap-2 rounded-full border border-[#ffffff14] bg-[#111118] px-3 py-1.5 text-sm text-[#64748b]">
            <Trophy size={15} className="text-[#3b82f6]" />
            Results
          </div>
          <h1 className="mt-3 text-3xl font-semibold text-[#f1f5f9]">Performance dashboard</h1>
          <p className="mt-1 text-sm text-[#64748b]">
            {result?.status ?? "loading"}
            {teamLabel ? ` · ${playerLabel} — ${teamLabel}` : playerLabel ? ` · ${playerLabel}` : ""}
          </p>
        </div>

        {error && (
          <div className="mb-4 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
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
              <div className="card grid aspect-video place-items-center text-sm text-[#64748b]">
                <Loader2 className="mb-2 animate-spin text-[#3b82f6]" size={22} />
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
          <HeatmapsPanel
            movementUrl={result.assets?.movement_heatmap}
            touchUrl={result.assets?.touch_heatmap}
            touchCount={results?.touch_count}
            passCount={results?.pass_count}
          />
        ) : null}

        {speedMetrics && analysisMode !== "max_shot_power" && result?.status === "complete" && activeTab === "overview" ? (
          <div className="mt-2 space-y-5">
            <TrackingResultWarnings metrics={speedMetrics} />

            <div className="card p-6">
              <div className="grid gap-6 sm:grid-cols-2">
                <div>
                  <p className="text-sm text-[#64748b]">Top speed</p>
                  <div className="mt-2 flex items-end gap-2">
                    <span className="text-5xl font-bold tabular-nums text-[#3b82f6]">
                      {speedMetrics.units === "pixels"
                        ? speedMetrics.top_speed_px_per_s
                        : (speedMetrics.max_speed_kmh ?? speedMetrics.top_speed_kmh)}
                    </span>
                    <span className="text-xl text-[#64748b]">
                      {speedMetrics.units === "pixels" ? "px/s" : "km/h"}
                    </span>
                  </div>
                </div>
                <div>
                  <p className="text-sm text-[#64748b]">Distance covered</p>
                  <div className="mt-2 flex items-end gap-2">
                    <span className="text-5xl font-bold tabular-nums text-[#f1f5f9]">
                      {speedMetrics.units === "pixels"
                        ? (speedMetrics.total_distance_px ?? speedMetrics.total_distance_m)
                        : (speedMetrics.distance_m ?? speedMetrics.total_distance_m)}
                    </span>
                    <span className="text-xl text-[#64748b]">
                      {speedMetrics.units === "pixels" ? "px" : "m"}
                    </span>
                  </div>
                </div>
              </div>
              <div className="mt-4 flex items-center justify-between">
                <p className="text-sm text-[#64748b]">
                  {playerLabel}
                  {teamLabel ? ` · ${teamLabel}` : ""}
                </p>
                <ConfidenceDot score={confidenceScore} />
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              <MetricCard
                icon={<Activity size={18} />}
                label="Average speed"
                value={
                  speedMetrics.units === "pixels"
                    ? `${speedMetrics.avg_speed_px_per_s} px/s`
                    : `${speedMetrics.avg_speed_kmh} km/h`
                }
              />
              <MetricCard
                icon={<Footprints size={18} />}
                label="Sprint distance"
                value={
                  speedMetrics.units === "pixels"
                    ? `${speedMetrics.sprint_distance_px ?? speedMetrics.sprint_distance_m ?? 0} px`
                    : `${speedMetrics.sprint_distance_m ?? 0} m`
                }
              />
              <MetricCard
                icon={<Zap size={18} />}
                label="Sprints"
                value={`${speedMetrics.sprint_count ?? 0}`}
              />
            </div>
          </div>
        ) : shotMetrics && analysisMode === "max_shot_power" && result?.status === "complete" && activeTab === "overview" ? (
          <div className="mt-2 space-y-5">
            <div className="card p-6">
              <p className="text-sm text-[#64748b]">Peak shot speed</p>
              <div className="mt-2 flex flex-wrap items-end justify-between gap-4">
                <div>
                  <span className="text-5xl font-bold tabular-nums text-[#3b82f6]">{shotMetrics.peak_shot_speed_kmh}</span>
                  <span className="ml-2 text-xl text-[#64748b]">km/h</span>
                </div>
                <ConfidenceDot score={confidenceScore} />
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <MetricCard icon={<Target size={18} />} label="Shots detected" value={`${shotMetrics.shot_count}`} />
              <MetricCard
                icon={<Crosshair size={18} />}
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
              <h2 className="mb-4 text-sm font-semibold text-[#f1f5f9]">All shots (by speed)</h2>
              {sortedShots.length ? (
                <div className="space-y-2">
                  {sortedShots.map((shot, index) => (
                    <div
                      key={`${shot.frame_id}-${index}`}
                      className="flex items-center justify-between rounded-lg border border-[#ffffff14] bg-[#0a0a0f] px-4 py-3"
                    >
                      <div>
                        <p className="font-semibold text-[#f1f5f9]">#{index + 1} · Frame {shot.frame_id}</p>
                        <p className="text-xs text-[#64748b]">{shot.timestamp_s.toFixed(2)}s</p>
                      </div>
                      <span className="text-lg font-bold tabular-nums text-[#3b82f6]">{shot.ball_speed_kmh} km/h</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="rounded-lg border border-[#ffffff14] bg-[#0a0a0f] p-6 text-center text-sm text-[#64748b]">
                  <p className="font-medium text-[#f1f5f9]">No shots detected</p>
                  <p className="mt-2">
                    Check that you selected the correct player and that the clip includes visible shots with enough length.
                  </p>
                </div>
              )}
            </div>
          </div>
        ) : result?.status === "failed" ? (
          <div className="card mt-6 grid min-h-48 place-items-center p-8 text-[#64748b]">
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
      className={`rounded-full px-4 py-2 text-sm font-medium ${
        active ? "bg-[#3b82f6] text-white" : "bg-[#ffffff08] text-[#64748b] hover:text-[#f1f5f9]"
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
      <div className="card grid min-h-48 place-items-center p-8 text-sm text-[#64748b]">
        Heatmaps are not available for this analysis. Pitch calibration is required for field heatmaps.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {(touchCount !== undefined || passCount !== undefined) && (
        <div className="flex flex-wrap gap-2">
          {touchCount !== undefined ? (
            <span className="rounded-full bg-[#ffffff10] px-3 py-1 text-xs font-medium text-[#f1f5f9]">
              {touchCount} touches
            </span>
          ) : null}
          {passCount !== undefined ? (
            <span className="rounded-full bg-[#3b82f6]/15 px-3 py-1 text-xs font-medium text-[#3b82f6]">
              {passCount} passes
            </span>
          ) : null}
        </div>
      )}
      <div className="grid gap-4 md:grid-cols-2">
        {movementUrl ? (
          <div className="card overflow-hidden p-3">
            <p className="mb-2 text-sm text-[#64748b]">Movement heatmap (pitch)</p>
            <img src={mediaUrl(movementUrl) ?? ""} alt="Movement heatmap" className="w-full rounded-lg" loading="lazy" />
          </div>
        ) : null}
        {touchUrl ? (
          <div className="card overflow-hidden p-3">
            <p className="mb-2 text-sm text-[#64748b]">Touch &amp; pass heatmap</p>
            <img src={mediaUrl(touchUrl) ?? ""} alt="Touch heatmap" className="w-full rounded-lg" loading="lazy" />
          </div>
        ) : null}
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
    warnings.push("Significant portions of the track were predicted rather than directly observed.");
  }

  if (!warnings.length) return null;

  return (
    <div className="space-y-1">
      {warnings.map((warning) => (
        <p key={warning} className="text-sm text-amber-400">
          {warning}
        </p>
      ))}
    </div>
  );
}

function MetricCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="card p-4">
      <div className="flex items-center gap-2 text-sm text-[#64748b]">
        <span className="text-[#3b82f6]">{icon}</span>
        {label}
      </div>
      <p className="mt-2 text-2xl font-semibold tabular-nums text-[#f1f5f9]">{value}</p>
    </div>
  );
}
