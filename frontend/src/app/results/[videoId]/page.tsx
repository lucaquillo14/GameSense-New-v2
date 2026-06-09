"use client";

import { useEffect, useMemo, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { ProcessingProgress } from "@/components/ProcessingProgress";
import { ConfidenceDot, SpeedChart } from "@/components/SpeedChart";
import { WarningBanners } from "@/components/WarningBanners";
import { getFrame, getResults, mediaUrl, Metrics, ShotMetrics, VideoResult } from "@/lib/api";
import { Activity, Crosshair, Gauge, Loader2, Route, Target, Timer, Trophy, Zap } from "lucide-react";
import { useParams } from "next/navigation";

function isShotMetrics(results: Metrics | ShotMetrics): results is ShotMetrics {
  return "peak_shot_speed_kmh" in results;
}

export default function ResultsPage() {
  const params = useParams<{ videoId: string }>();
  const videoId = params.videoId;
  const [result, setResult] = useState<VideoResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [bestShotFrameUrl, setBestShotFrameUrl] = useState<string | null>(null);

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
    const interval = window.setInterval(refresh, 1500);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [videoId]);

  const results = result?.results;
  const analysisMode = result?.mode ?? "max_speed";
  const progress = result?.progress;
  const progressPercent = progress?.percent ?? (result?.status === "complete" ? 100 : 0);
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

  const frameLabel =
    progress?.message && progress.message.includes("frame")
      ? progress.message
      : progress?.stage === "tracking"
        ? progress.message
        : undefined;

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

        {(result?.status === "processing" || !result) && (
          <ProcessingProgress
            stage={progress?.stage}
            percent={progressPercent}
            message={progress?.message}
            frameLabel={frameLabel}
          />
        )}

        {speedMetrics && analysisMode !== "max_shot_power" ? (
          <div className="mt-6 space-y-5">
            <div className="card p-6">
              <p className="text-sm text-[#64748b]">Top speed</p>
              <div className="mt-2 flex flex-wrap items-end justify-between gap-4">
                <div>
                  <span className="text-5xl font-bold tabular-nums text-[#3b82f6]">{speedMetrics.top_speed_kmh}</span>
                  <span className="ml-2 text-xl text-[#64748b]">km/h</span>
                </div>
                <ConfidenceDot score={confidenceScore} />
              </div>
              <p className="mt-2 text-sm text-[#64748b]">
                {playerLabel}
                {teamLabel ? ` · ${teamLabel}` : ""}
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCard icon={<Activity size={18} />} label="Average speed" value={`${speedMetrics.avg_speed_kmh} km/h`} />
              <MetricCard icon={<Route size={18} />} label="Total distance" value={`${speedMetrics.total_distance_m} m`} />
              <MetricCard icon={<Zap size={18} />} label="Sprint count" value={`${speedMetrics.sprint_count}`} />
              <MetricCard icon={<Route size={18} />} label="Sprint distance" value={`${speedMetrics.sprint_distance_m} m`} />
            </div>

            <div className="card p-5">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-sm font-semibold text-[#f1f5f9]">Speed over time</h2>
                <span className="text-xs text-[#64748b]">{speedMetrics.usable_track_points} track points</span>
              </div>
              <SpeedChart data={speedMetrics.speed_series ?? []} />
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <MetricCard icon={<Timer size={18} />} label="Peak acceleration" value={`${speedMetrics.peak_acceleration_mps2} m/s²`} />
              <MetricCard icon={<Gauge size={18} />} label="Active distance" value={`${speedMetrics.active_distance_m} m`} />
            </div>
          </div>
        ) : shotMetrics && analysisMode === "max_shot_power" ? (
          <div className="mt-6 space-y-5">
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
        ) : result?.status !== "processing" ? (
          <div className="card mt-6 grid min-h-48 place-items-center p-8 text-[#64748b]">
            <Loader2 className="mb-3 animate-spin text-[#3b82f6]" size={24} />
            Metrics will appear when processing completes.
          </div>
        ) : null}
      </section>
    </AppShell>
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
