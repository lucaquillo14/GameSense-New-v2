"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { AppShell } from "@/components/AppShell";
import { SetupCanvas } from "@/components/SetupCanvas";
import { getFrameDetections, getResults, mediaUrl, processVideo, selectPlayer, setPitchPolygon } from "@/lib/api";
import type { AnalysisMode, Detection, Point, TeamClassificationInfo, VideoMetadata } from "@/lib/api";
import {
  CheckCircle2,
  Flag,
  Gauge,
  Loader2,
  MousePointer2,
  Play,
  RotateCcw,
  ShieldAlert,
  Target,
  Waypoints,
} from "lucide-react";

type Mode = "player" | "pitch" | "goal-posts";

export default function SetupPage() {
  const router = useRouter();
  const params = useParams<{ videoId: string }>();
  const videoId = params.videoId;
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<VideoMetadata | null>(null);
  const [frameId, setFrameId] = useState(0);
  const [mode, setMode] = useState<Mode>("player");
  const [detections, setDetections] = useState<Detection[]>([]);
  const [selectedDetection, setSelectedDetection] = useState<Detection | null>(null);
  const [playerPoint, setPlayerPoint] = useState<Point | null>(null);
  const [pitchPolygon, setPitchPolygonState] = useState<Point[]>([]);
  const [goalPosts, setGoalPosts] = useState<Point[]>([]);
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>("max_speed");
  const [playerHeightCm, setPlayerHeightCm] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [teamClassification, setTeamClassification] = useState<TeamClassificationInfo | null>(null);
  const [preparing, setPreparing] = useState(true);
  const [prepareMessage, setPrepareMessage] = useState("Preparing setup frame…");
  const [detectingPlayers, setDetectingPlayers] = useState(false);
  const [setupProgress, setSetupProgress] = useState<{ percent: number; message: string } | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      try {
        const result = await getResults(videoId);
        if (cancelled) return;

        setSourceUrl(mediaUrl(result.source_url));
        setFrameId(result.setup_frame_id ?? 0);
        setMetadata(result.video_metadata ?? null);
        if (result.team_classification) setTeamClassification(result.team_classification);

        const prog = result.progress as { stage?: string; message?: string; setup_percent?: number } | null;
        if (result.team_classification || prog?.stage === "team_ready") {
          setSetupProgress(null);
        } else if (prog?.stage === "team_calibration") {
          setSetupProgress({
            percent: prog.setup_percent ?? 0,
            message: prog.message ?? "Detecting team colours…",
          });
        }
        if (prog?.message && (prog.stage === "team_calibration" || prog.stage === "uploaded")) {
          setPrepareMessage(prog.message);
        }

        const target = result.target_player as { click?: Point; bbox?: Detection["bbox"]; detection_id?: string } | null;
        if (target?.click) setPlayerPoint(target.click);
        if (target?.bbox) {
          setSelectedDetection({
            id: target.detection_id ?? "persisted-target",
            label: "player",
            confidence: 0.9,
            bbox: target.bbox,
          });
        }

        if (result.setup_frame) {
          setPreparing(false);
          return;
        }

        setPrepareMessage("Extracting first frame…");
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Could not load video.");
          setPreparing(false);
        }
      }
    }

    bootstrap();
    const interval = window.setInterval(bootstrap, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [videoId]);

  useEffect(() => {
    if (preparing) return;

    let cancelled = false;
    let attempts = 0;

    // Old boxes belong to the previous frame — clear them immediately so the
    // overlay never shows players in the wrong positions while refetching.
    setDetections([]);
    setSelectedDetection(null);

    async function loadDetections() {
      if (!cancelled) setDetectingPlayers(true);
      try {
        const response = await getFrameDetections(videoId, frameId);
        if (cancelled) return;
        setDetections(response.detections);
        setDetectingPlayers(false);
        if (!response.detections.length && attempts < 60) {
          attempts += 1;
          window.setTimeout(loadDetections, 1500);
        }
      } catch (err) {
        if (!cancelled) {
          if (attempts < 60) {
            attempts += 1;
            window.setTimeout(loadDetections, 1500);
            return;
          }
          setDetectingPlayers(false);
          setError(err instanceof Error ? err.message : "Could not load detections.");
        }
      }
    }

    const timer = window.setTimeout(loadDetections, 260);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [frameId, videoId, preparing]);

  const ready = useMemo(() => Boolean(playerPoint), [playerPoint]);

  async function confirmSetup() {
    if (!playerPoint) return;
    setBusy(true);
    setError(null);
    try {
      await selectPlayer(videoId, playerPoint, frameId, selectedDetection);
      await setPitchPolygon(videoId, pitchPolygon, frameId, null, null, goalPosts.length === 4 ? goalPosts : null);
      const heightCm = Number(playerHeightCm);
      await processVideo(
        videoId,
        analysisMode,
        Number.isFinite(heightCm) && heightCm >= 100 && heightCm <= 250 ? heightCm : null,
      );
      router.push(`/results/${videoId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Setup failed.");
    } finally {
      setBusy(false);
    }
  }

  function choosePlayer(point: Point, detection: Detection | null) {
    setPlayerPoint(point);
    setSelectedDetection(detection);
  }

  return (
    <AppShell>
      <section className="mx-auto max-w-7xl px-5 py-7">
        <div className="mb-6 flex flex-col justify-between gap-4 lg:flex-row lg:items-end">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-[#ffffff14] bg-[#0d0d17] px-3 py-1.5 text-sm text-[#6b7a99]">
              <Target size={15} className="text-[#06b6d4]" />
              Player selection
            </div>
            <h1 className="mt-3 text-3xl font-semibold text-[#eef2ff]">Select your target player</h1>
            <p className="mt-2 max-w-3xl text-sm text-[#6b7a99]">
              Boxes are coloured by team. Click a player to highlight them, then mark the 4 corners of the goal
              frame — its known size (7.32 × 2.44 m) gives an accurate speed scale even when the whole pitch
              is not in view.
            </p>
            {metadata && (
              <div className="mt-4 flex flex-wrap gap-2">
                <MetaPill label={`${metadata.width}×${metadata.height}`} />
                <MetaPill label={`${metadata.fps} fps`} />
                <MetaPill label={`${metadata.duration_s.toFixed(1)}s`} />
              </div>
            )}
          </div>
          <button
            type="button"
            disabled={!ready || busy}
            onClick={confirmSetup}
            className="btn-primary flex items-center justify-center gap-2 px-5 py-3 disabled:cursor-not-allowed"
          >
            {busy ? <Loader2 size={18} className="animate-spin" /> : <Play size={18} />}
            Start analysis
          </button>
        </div>

        <div className="grid gap-5 xl:grid-cols-[320px_1fr]">
          <aside className="card p-4">
            <div className="text-xs font-semibold uppercase tracking-normal text-slate-500">Workflow</div>
            <div className="mt-3 grid gap-2">
              <ModeButton active={mode === "player"} onClick={() => setMode("player")} icon={<MousePointer2 size={17} />} label="Select player" />
              <ModeButton active={mode === "goal-posts"} onClick={() => setMode("goal-posts")} icon={<Flag size={17} />} label="Goal frame (scale)" />
              <ModeButton active={mode === "pitch"} onClick={() => setMode("pitch")} icon={<Waypoints size={17} />} label="Pitch boundary" />
            </div>

            <div className="mt-5 space-y-3 text-sm text-slate-400">
              <StatusRow label="Frame" done value={`${frameId}`} />
              <StatusRow label="Player" done={Boolean(playerPoint)} value={selectedDetection ? "box assigned" : playerPoint ? "point selected" : "required"} />
              <StatusRow
                label="Goal scale"
                done={goalPosts.length === 4}
                value={goalPosts.length === 4 ? "scale locked" : goalPosts.length > 0 ? `${goalPosts.length}/4 points` : "recommended"}
              />
              <StatusRow
                label="Pitch"
                done={pitchPolygon.length >= 4}
                value={pitchPolygon.length >= 4 ? `${pitchPolygon.length} points` : "auto-detect or pixel mode"}
              />
            </div>

            {goalPosts.length === 4 && (
              <p className="mt-3 rounded-lg border border-amber-400/25 bg-amber-500/10 px-3 py-2 text-xs leading-5 text-amber-200">
                Speeds and distances will be scaled from the goal frame (7.32 m × 2.44 m) — the most reliable
                reference when the whole pitch is not in frame.
              </p>
            )}

            {ready && (
              <div className="mt-5 rounded-lg border border-white/10 bg-slate-950/50 p-3">
                <div className="text-xs font-semibold uppercase tracking-normal text-slate-500">Analysis mode</div>
                <p className="mt-2 text-xs leading-5 text-slate-500">This clip will be analysed for sprint performance.</p>
                <div className="mt-3 grid gap-2">
                  <AnalysisModeButton
                    active={analysisMode === "max_speed"}
                    onClick={() => setAnalysisMode("max_speed")}
                    icon={<Gauge size={17} />}
                    label="Max Speed"
                    description="Peak running speed, total distance covered, sprint distance, and sprint count."
                  />
                </div>
                <div className="mt-3">
                  <label className="text-xs font-medium text-slate-400">
                    Player height (cm) — optional, improves speed accuracy
                  </label>
                  <input
                    type="number"
                    inputMode="numeric"
                    min={100}
                    max={250}
                    value={playerHeightCm}
                    onChange={(event) => setPlayerHeightCm(event.target.value)}
                    placeholder="e.g. 178"
                    className="mt-1 w-full rounded-lg border border-white/10 bg-slate-950/50 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-600 focus:border-[#06b6d4] focus:outline-none"
                  />
                  <p className="mt-1 text-[11px] leading-4 text-slate-500">
                    Used only if the goal/pitch isn&apos;t marked — turns the scale from a
                    1.75&nbsp;m guess into your real measurement.
                  </p>
                </div>
              </div>
            )}

            <button
              type="button"
              onClick={() => setPitchPolygonState((points) => points.slice(0, -1))}
              className="mt-5 flex w-full items-center justify-center gap-2 rounded-lg border border-white/10 bg-white/6 px-3 py-2 text-sm font-medium text-slate-200 hover:bg-white/10"
            >
              <RotateCcw size={16} />
              Undo pitch point
            </button>
            {pitchPolygon.length > 0 && (
              <button
                type="button"
                onClick={() => setPitchPolygonState([])}
                className="mt-2 w-full rounded-lg border border-white/10 bg-white/6 px-3 py-2 text-sm font-medium text-slate-200 hover:bg-white/10"
              >
                Clear pitch boundary
              </button>
            )}
            {goalPosts.length > 0 && (
              <button
                type="button"
                onClick={() => setGoalPosts([])}
                className="mt-2 w-full rounded-lg border border-white/10 bg-white/6 px-3 py-2 text-sm font-medium text-slate-200 hover:bg-white/10"
              >
                Clear goal frame
              </button>
            )}
          </aside>

          <div>
            {preparing ? (
              <div className="card aspect-video overflow-hidden p-0">
                <div className="grid h-full place-items-center p-8 text-center">
                  <div className="mb-4 h-40 w-full max-w-xl animate-pulse rounded-lg bg-[#ffffff08]" />
                  <Loader2 className="mb-3 animate-spin text-[#06b6d4]" size={28} />
                  <p className="text-sm text-[#eef2ff]">{prepareMessage}</p>
                  {setupProgress && (
                    <div className="mt-3 w-full max-w-xl">
                      <div className="h-2 overflow-hidden rounded-full bg-[#ffffff14]">
                        <div
                          className="h-full rounded-full bg-[#06b6d4] transition-all duration-500"
                          style={{ width: `${Math.max(setupProgress.percent, 4)}%` }}
                        />
                      </div>
                      <p className="mt-1.5 text-xs text-[#6b7a99]">{setupProgress.percent}%</p>
                    </div>
                  )}
                </div>
              </div>
            ) : sourceUrl ? (
              <>
              {detectingPlayers && detections.length === 0 && (
                <div className="mb-3 rounded-lg border border-[#ffffff14] bg-[#0d0d17] px-4 py-3 text-sm text-[#6b7a99]">
                  <div className="flex items-center gap-2">
                    <Loader2 size={16} className="animate-spin text-[#06b6d4]" />
                    {setupProgress ? setupProgress.message : "Detecting players…"}
                  </div>
                  {setupProgress && (
                    <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-[#ffffff14]">
                      <div
                        className="h-full rounded-full bg-[#06b6d4] transition-all duration-500"
                        style={{ width: `${Math.max(setupProgress.percent, 4)}%` }}
                      />
                    </div>
                  )}
                </div>
              )}
              <SetupCanvas
                videoUrl={sourceUrl}
                mode={mode}
                metadata={metadata}
                frameId={frameId}
                detections={detections}
                selectedDetection={selectedDetection}
                playerPoint={playerPoint}
                pitchPolygon={pitchPolygon}
                goalPosts={goalPosts}
                onFrameChange={setFrameId}
                onPlayerPoint={choosePlayer}
                onPitchPoint={(point) => setPitchPolygonState((points) => [...points, point])}
                onGoalPostPoint={(point) => setGoalPosts((points) => (points.length < 4 ? [...points, point] : points))}
                onRemovePitchPoint={(index) => setPitchPolygonState((points) => points.filter((_, pointIndex) => pointIndex !== index))}
                onRemoveGoalPostPoint={(index) => setGoalPosts((points) => points.filter((_, pointIndex) => pointIndex !== index))}
                teamClassification={teamClassification}
              />
              </>
            ) : (
              <div className="card grid aspect-video place-items-center">
                <Loader2 className="animate-spin text-[#06b6d4]" />
              </div>
            )}
            {error && (
              <div className="mt-4 flex items-center gap-2 rounded-lg border border-red-400/25 bg-red-500/10 px-3 py-2 text-sm text-red-200">
                <ShieldAlert size={16} />
                {error}
              </div>
            )}
          </div>
        </div>
      </section>
    </AppShell>
  );
}

function AnalysisModeButton({
  active,
  onClick,
  icon,
  label,
  description,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  description: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-lg px-3 py-2.5 text-left ${
        active ? "bg-cyan-400 text-slate-950 shadow-lg shadow-cyan-500/15" : "bg-white/6 text-slate-300 hover:bg-white/10"
      }`}
    >
      <div className="flex items-center gap-2 text-sm font-semibold">
        {icon}
        {label}
      </div>
      <p className={`mt-1 text-xs leading-5 ${active ? "text-slate-800" : "text-slate-500"}`}>{description}</p>
    </button>
  );
}

function ModeButton({ active, onClick, icon, label }: { active: boolean; onClick: () => void; icon: React.ReactNode; label: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex items-center gap-2 rounded-lg px-3 py-2.5 text-left text-sm font-medium ${
        active ? "bg-cyan-400 text-slate-950 shadow-lg shadow-cyan-500/15" : "bg-white/6 text-slate-300 hover:bg-white/10"
      }`}
    >
      {icon}
      {label}
    </button>
  );
}

function StatusRow({ label, value, done }: { label: string; value: string; done: boolean }) {
  return (
    <div className="flex items-center justify-between border-b border-white/10 pb-2">
      <span>{label}</span>
      <span className={`flex items-center gap-1.5 ${done ? "font-semibold text-emerald-300" : "text-slate-500"}`}>
        {done && <CheckCircle2 size={14} />}
        {value}
      </span>
    </div>
  );
}

function MetaPill({ label }: { label: string }) {
  return (
    <span className="rounded-full border border-[#ffffff14] bg-[#0d0d17] px-3 py-1 text-xs text-[#eef2ff]">
      {label}
    </span>
  );
}
