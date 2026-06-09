"use client";

import { useEffect, useMemo, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { getFrame, getResults, mediaUrl, Metrics, ShotMetrics, VideoResult } from "@/lib/api";
import { Activity, AlertTriangle, Crosshair, Gauge, Loader2, Route, ShieldCheck, Target, Timer, Trophy, Zap } from "lucide-react";
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

  const speedMetrics = results && !isShotMetrics(results) ? results : null;
  const shotMetrics = results && isShotMetrics(results) ? results : null;
  const confidencePercent = Math.round((results?.confidence_score ?? 0) * 100);

  const speedSeries = useMemo(
    () => [
      { label: "Top", value: speedMetrics?.top_speed_kmh ?? 0, max: 38 },
      { label: "Avg", value: speedMetrics?.avg_speed_kmh ?? 0, max: 20 },
      { label: "Sprint", value: speedMetrics?.sprint_distance_m ?? 0, max: Math.max(speedMetrics?.total_distance_m ?? 1, 1) },
    ],
    [speedMetrics],
  );

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

  return (
    <AppShell>
      <section className="mx-auto max-w-7xl px-5 py-7">
        <div className="mb-6 flex flex-col justify-between gap-3 lg:flex-row lg:items-end">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/8 px-3 py-1.5 text-sm font-medium text-slate-300">
              <Trophy size={15} className="text-cyan-300" />
              Results
            </div>
            <h1 className="mt-3 text-3xl font-semibold tracking-normal text-white">Performance dashboard</h1>
            <p className="mt-2 text-sm text-slate-400">
              Status: {result?.status ?? "loading"}
              {result?.status === "complete" && (
                <span className="text-slate-500"> · {analysisMode === "max_shot_power" ? "Max Shot Power" : "Max Speed"}</span>
              )}
            </p>
          </div>
          {(result?.status === "processing" || !result) && (
            <div className="rounded-lg border border-white/10 bg-white/[0.04] px-4 py-3 text-sm text-slate-300 shadow-sm">
              <div className="flex items-center gap-2">
                <Loader2 size={16} className="animate-spin text-cyan-300" />
                {progress?.message ?? "Processing video"}
              </div>
              <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/10">
                <div className="h-full rounded-full bg-cyan-300 transition-all duration-500" style={{ width: `${progressPercent}%` }} />
              </div>
            </div>
          )}
        </div>

        {error && <Notice tone="error" message={error} />}
        {result?.warnings?.map((warning) => <Notice key={warning} tone="warn" message={warning} />)}

        {speedMetrics && analysisMode !== "max_shot_power" ? (
          <>
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              <MetricCard icon={<Gauge size={19} />} label="Top speed" value={`${speedMetrics.top_speed_kmh} km/h`} emphasis="cyan" />
              <MetricCard icon={<Activity size={19} />} label="Average speed" value={`${speedMetrics.avg_speed_kmh} km/h`} />
              <MetricCard icon={<Route size={19} />} label="Total distance" value={`${speedMetrics.total_distance_m} m`} />
              <MetricCard icon={<ShieldCheck size={19} />} label="Confidence" value={`${confidencePercent}%`} emphasis="green" />
              <MetricCard icon={<Timer size={19} />} label="Peak acceleration" value={`${speedMetrics.peak_acceleration_mps2} m/s^2`} />
              <MetricCard icon={<Timer size={19} />} label="Avg acceleration" value={`${speedMetrics.avg_acceleration_mps2} m/s^2`} />
              <MetricCard icon={<Route size={19} />} label="Active distance" value={`${speedMetrics.active_distance_m} m`} />
              <MetricCard icon={<Zap size={19} />} label="Sprint distance" value={`${speedMetrics.sprint_distance_m} m`} />
            </div>

            <div className="mt-6 grid gap-5 xl:grid-cols-[0.85fr_1.15fr]">
              <section className="rounded-lg border border-white/10 bg-white/[0.04] p-4 shadow-2xl shadow-black/20">
                <div className="mb-4 flex items-center justify-between">
                  <div className="flex items-center gap-2 text-sm font-semibold text-slate-200">
                    <Zap size={18} className="text-cyan-300" />
                    Movement profile
                  </div>
                  <span className="rounded-full bg-white/8 px-2.5 py-1 text-xs text-slate-400">{speedMetrics.usable_track_points} points</span>
                </div>
                <div className="space-y-4">
                  {speedSeries.map((item) => (
                    <BarRow key={item.label} label={item.label} value={item.value} max={item.max} />
                  ))}
                </div>
                <div className="mt-5 grid grid-cols-2 gap-3 text-sm">
                  <MiniStat label="Sprints" value={speedMetrics.sprint_count} />
                  <MiniStat label="Rejected jumps" value={speedMetrics.rejected_jump_count} />
                </div>
              </section>

              <section className="rounded-lg border border-white/10 bg-white/[0.04] p-4 shadow-2xl shadow-black/20">
                <div className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-200">
                  <Activity size={18} className="text-cyan-300" />
                  Tracking quality
                </div>
                <div className="grid gap-3 md:grid-cols-3">
                  <QualityTile label="Confidence" value={`${confidencePercent}%`} tone={confidencePercent >= 70 ? "good" : confidencePercent >= 45 ? "warn" : "bad"} />
                  <QualityTile label="Track points" value={speedMetrics.usable_track_points.toString()} tone="neutral" />
                  <QualityTile label="ID stability" value={speedMetrics.rejected_jump_count === 0 ? "Clean" : "Guarded"} tone={speedMetrics.rejected_jump_count === 0 ? "good" : "warn"} />
                </div>
                <div className="mt-5 rounded-lg border border-white/10 bg-slate-950/45 p-4 text-sm leading-6 text-slate-400">
                  Tracking uses YOLO person detections with ByteTrack IDs and an appearance-based recovery guard for post-occlusion ID switches.
                </div>
              </section>
            </div>
          </>
        ) : shotMetrics && analysisMode === "max_shot_power" ? (
          <>
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              <MetricCard icon={<Gauge size={19} />} label="Peak shot speed" value={`${shotMetrics.peak_shot_speed_kmh} km/h`} emphasis="cyan" />
              <MetricCard icon={<Activity size={19} />} label="Average shot speed" value={`${shotMetrics.avg_shot_speed_kmh} km/h`} />
              <MetricCard icon={<Target size={19} />} label="Shots detected" value={`${shotMetrics.shot_count}`} />
              <MetricCard icon={<ShieldCheck size={19} />} label="Confidence" value={`${confidencePercent}%`} emphasis="green" />
            </div>

            <div className="mt-6 grid gap-5 xl:grid-cols-[0.9fr_1.1fr]">
              <section className="rounded-lg border border-white/10 bg-white/[0.04] p-4 shadow-2xl shadow-black/20">
                <div className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-200">
                  <Crosshair size={18} className="text-cyan-300" />
                  Best shot
                </div>
                {shotMetrics.best_shot ? (
                  <div>
                    {bestShotFrameUrl ? (
                      <img
                        src={bestShotFrameUrl}
                        alt="Best shot contact frame"
                        className="aspect-video w-full rounded-lg border border-white/10 object-cover"
                      />
                    ) : (
                      <div className="grid aspect-video place-items-center rounded-lg border border-white/10 bg-slate-950/45 text-sm text-slate-500">
                        Loading frame…
                      </div>
                    )}
                    <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                      <MiniStat label="Exit speed" value={shotMetrics.best_shot.ball_speed_kmh} suffix=" km/h" />
                      <MiniStat label="Timestamp" value={shotMetrics.best_shot.timestamp_s} suffix=" s" decimals={2} />
                    </div>
                  </div>
                ) : (
                  <div className="rounded-lg border border-white/10 bg-slate-950/45 p-4 text-sm text-slate-500">
                    No shots were detected in this clip.
                  </div>
                )}
              </section>

              <section className="rounded-lg border border-white/10 bg-white/[0.04] p-4 shadow-2xl shadow-black/20">
                <div className="mb-4 flex items-center justify-between">
                  <div className="flex items-center gap-2 text-sm font-semibold text-slate-200">
                    <Target size={18} className="text-cyan-300" />
                    All detected shots
                  </div>
                  <span className="rounded-full bg-white/8 px-2.5 py-1 text-xs text-slate-400">{shotMetrics.usable_track_points} ball points</span>
                </div>
                {shotMetrics.shots.length > 0 ? (
                  <div className="max-h-96 space-y-2 overflow-y-auto pr-1">
                    {shotMetrics.shots.map((shot, index) => (
                      <div
                        key={`${shot.frame_id}-${index}`}
                        className="flex items-center justify-between rounded-lg border border-white/10 bg-slate-950/40 px-3 py-2.5 text-sm"
                      >
                        <div>
                          <div className="font-semibold text-white">Shot {index + 1}</div>
                          <div className="text-xs text-slate-500">
                            Frame {shot.frame_id} · {shot.timestamp_s.toFixed(2)}s
                          </div>
                        </div>
                        <div className="text-lg font-semibold tabular-nums text-cyan-300">{shot.ball_speed_kmh} km/h</div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="rounded-lg border border-white/10 bg-slate-950/45 p-4 text-sm text-slate-500">
                    Shot list will populate when ball contacts are found near the selected player.
                  </div>
                )}
                <div className="mt-5 grid grid-cols-2 gap-3 text-sm">
                  <MiniStat label="Ball track points" value={shotMetrics.usable_track_points} />
                  <MiniStat label="Low-confidence frames" value={shotMetrics.rejected_track_points} />
                </div>
              </section>
            </div>
          </>
        ) : (
          <div className="grid min-h-80 place-items-center rounded-lg border border-white/10 bg-white/[0.04] text-slate-500">
            {result?.status === "failed" ? "Processing failed." : "Metrics will appear here when processing completes."}
          </div>
        )}
      </section>
    </AppShell>
  );
}

function MetricCard({
  icon,
  label,
  value,
  emphasis = "default",
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  emphasis?: "default" | "cyan" | "green";
}) {
  const color = emphasis === "cyan" ? "text-cyan-300" : emphasis === "green" ? "text-emerald-300" : "text-slate-400";
  return (
    <div className="rounded-lg border border-white/10 bg-white/[0.04] p-4 shadow-sm">
      <div className={`flex items-center gap-2 text-sm font-medium ${color}`}>
        {icon}
        {label}
      </div>
      <div className="mt-3 text-2xl font-semibold tracking-normal text-white">{value}</div>
    </div>
  );
}

function BarRow({ label, value, max }: { label: string; value: number; max: number }) {
  const width = `${Math.min(Math.max((value / max) * 100, 2), 100)}%`;
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-sm">
        <span className="text-slate-400">{label}</span>
        <span className="font-semibold tabular-nums text-white">{value.toFixed(value % 1 ? 1 : 0)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-white/10">
        <div className="h-full rounded-full bg-cyan-300 transition-all duration-700" style={{ width }} />
      </div>
    </div>
  );
}

function MiniStat({
  label,
  value,
  suffix = "",
  decimals = 0,
}: {
  label: string;
  value: number;
  suffix?: string;
  decimals?: number;
}) {
  const formatted = decimals > 0 ? value.toFixed(decimals) : value.toString();
  return (
    <div className="rounded-lg border border-white/10 bg-slate-950/40 p-3">
      <div className="text-xl font-semibold tabular-nums text-white">
        {formatted}
        {suffix}
      </div>
      <div className="text-xs text-slate-500">{label}</div>
    </div>
  );
}

function QualityTile({ label, value, tone }: { label: string; value: string; tone: "good" | "warn" | "bad" | "neutral" }) {
  const toneClass =
    tone === "good"
      ? "text-emerald-300"
      : tone === "warn"
        ? "text-amber-200"
        : tone === "bad"
          ? "text-red-200"
          : "text-cyan-300";
  return (
    <div className="rounded-lg border border-white/10 bg-slate-950/40 p-3">
      <div className={`text-xl font-semibold tabular-nums ${toneClass}`}>{value}</div>
      <div className="text-xs text-slate-500">{label}</div>
    </div>
  );
}

function Notice({ tone, message }: { tone: "warn" | "error"; message: string }) {
  return (
    <div
      className={`mb-3 flex items-center gap-2 rounded-lg border px-3 py-2 text-sm ${
        tone === "error" ? "border-red-400/25 bg-red-500/10 text-red-200" : "border-amber-300/25 bg-amber-300/10 text-amber-100"
      }`}
    >
      <AlertTriangle size={16} />
      {message}
    </div>
  );
}
