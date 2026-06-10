"use client";

import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Detection, Point, TeamClassificationInfo, VideoMetadata } from "@/lib/api";
import { isPlayerDetection, teamColorCss } from "@/lib/api";
import { drawSetupDetections } from "@/lib/overlay";

type Mode = "player" | "pitch" | "goal-left" | "goal-right";

type Props = {
  videoUrl: string;
  mode: Mode;
  metadata: VideoMetadata | null;
  frameId: number;
  detections: Detection[];
  selectedDetection: Detection | null;
  playerPoint: Point | null;
  pitchPolygon: Point[];
  goalLeft: Point | null;
  goalRight: Point | null;
  teamClassification?: TeamClassificationInfo | null;
  onFrameChange: (frameId: number) => void;
  onPlayerPoint: (point: Point, detection: Detection | null) => void;
  onPitchPoint: (point: Point) => void;
  onGoalLeft: (point: Point) => void;
  onGoalRight: (point: Point) => void;
  onRemovePitchPoint: (index: number) => void;
};

export function SetupCanvas({
  videoUrl,
  mode,
  metadata,
  frameId,
  detections,
  selectedDetection,
  playerPoint,
  pitchPolygon,
  goalLeft,
  goalRight,
  teamClassification,
  onFrameChange,
  onPlayerPoint,
  onPitchPoint,
  onGoalLeft,
  onGoalRight,
  onRemovePitchPoint,
}: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const fps = metadata?.fps || 30;
  const frameCount = Math.max((metadata?.frame_count ?? 1) - 1, 0);
  const duration = metadata?.duration_s || 0;
  const width = metadata?.width || 1;
  const height = metadata?.height || 1;

  const playerDetections = useMemo(() => detections.filter(isPlayerDetection), [detections]);

  const redraw = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    drawSetupDetections(
      canvas,
      video,
      playerDetections,
      width,
      height,
      selectedDetection?.player_id ?? selectedDetection?.id ?? null,
    );
  }, [height, playerDetections, selectedDetection, width]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const observer = new ResizeObserver(() => redraw());
    observer.observe(video);
    video.addEventListener("timeupdate", redraw);
    video.addEventListener("loadedmetadata", redraw);
    video.addEventListener("loadeddata", redraw);
    video.addEventListener("seeked", redraw);
    return () => {
      observer.disconnect();
      video.removeEventListener("timeupdate", redraw);
      video.removeEventListener("loadedmetadata", redraw);
      video.removeEventListener("loadeddata", redraw);
      video.removeEventListener("seeked", redraw);
    };
  }, [redraw]);

  useEffect(() => {
    redraw();
  }, [detections, frameId, redraw]);

  function syncFrameFromVideo() {
    const video = videoRef.current;
    if (!video) return;
    onFrameChange(Math.min(Math.round(video.currentTime * fps), frameCount));
    redraw();
  }

  function setFrame(nextFrame: number) {
    const video = videoRef.current;
    if (!video) return;
    const bounded = Math.min(Math.max(nextFrame, 0), frameCount);
    video.pause();
    setPlaying(false);
    video.currentTime = bounded / fps;
    onFrameChange(bounded);
    redraw();
  }

  function togglePlayback() {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      void video.play();
      setPlaying(true);
    } else {
      video.pause();
      setPlaying(false);
      syncFrameFromVideo();
    }
  }

  function toVideoPoint(clientX: number, clientY: number): Point | null {
    const overlay = overlayRef.current;
    if (!overlay) return null;
    const rect = overlay.getBoundingClientRect();
    if (clientX < rect.left || clientX > rect.right || clientY < rect.top || clientY > rect.bottom) return null;
    return {
      x: ((clientX - rect.left) / rect.width) * width,
      y: ((clientY - rect.top) / rect.height) * height,
    };
  }

  function handleOverlayClick(event: React.MouseEvent<HTMLDivElement>) {
    const point = toVideoPoint(event.clientX, event.clientY);
    if (!point) return;
    const detection = playerDetections.find((candidate) => containsPoint(candidate, point)) ?? null;
    if (mode === "player") onPlayerPoint(point, detection);
    if (mode === "pitch") onPitchPoint(point);
    if (mode === "goal-left") onGoalLeft(point);
    if (mode === "goal-right") onGoalRight(point);
  }

  const teamALegend = teamClassification?.team_a.display_color;
  const teamBLegend = teamClassification?.team_b.display_color;

  return (
    <div className="card overflow-hidden shadow-[0_24px_60px_rgba(0,0,0,0.45)]">
      <div className="relative aspect-video bg-black">
        <video
          key={videoUrl}
          ref={videoRef}
          src={videoUrl}
          className="h-full w-full object-contain"
          preload="metadata"
          playsInline
          onTimeUpdate={syncFrameFromVideo}
          onPause={() => {
            setPlaying(false);
            syncFrameFromVideo();
          }}
          onPlay={() => setPlaying(true)}
          onLoadedMetadata={syncFrameFromVideo}
        />
        <div ref={overlayRef} className="absolute inset-0 z-10 cursor-crosshair" onClick={handleOverlayClick}>
          <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
            {pitchPolygon.length > 1 && (
              <polyline
                points={pitchPolygon.map((point) => `${point.x},${point.y}`).join(" ")}
                fill={pitchPolygon.length >= 4 ? "rgba(16,185,129,0.12)" : "none"}
                stroke="#10b981"
                strokeWidth={Math.max(width / 360, 2)}
                strokeLinejoin="round"
              />
            )}
            {[goalLeft, goalRight].map((goal, index) =>
              goal ? (
                <circle
                  key={index}
                  cx={goal.x}
                  cy={goal.y}
                  r={Math.max(width / 120, 8)}
                  fill={index === 0 ? "#f59e0b" : "#10b981"}
                />
              ) : null,
            )}
          </svg>

          {pitchPolygon.map((point, index) => (
            <button
              key={`${point.x}-${point.y}-${index}`}
              type="button"
              title="Remove calibration point"
              onClick={(event) => {
                event.stopPropagation();
                onRemovePitchPoint(index);
              }}
              className="absolute h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-[#10b981] shadow-lg"
              style={{ left: `${(point.x / width) * 100}%`, top: `${(point.y / height) * 100}%` }}
            />
          ))}
          {playerPoint && (
            <span
              className="pointer-events-none absolute h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-[#3b82f6] shadow-[0_0_16px_rgba(59,130,246,0.8)]"
              style={{ left: `${(playerPoint.x / width) * 100}%`, top: `${(playerPoint.y / height) * 100}%` }}
            />
          )}
        </div>
        <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 z-20 h-full w-full" />

        {(teamALegend || teamBLegend) && (
          <div className="pointer-events-none absolute right-3 top-3 flex gap-2 rounded-lg border border-[#ffffff14] bg-[#111118]/90 px-3 py-2 text-xs text-[#f1f5f9]">
            {teamALegend && (
              <span className="inline-flex items-center gap-1.5">
                <span className="h-3 w-3 rounded-sm" style={{ background: teamColorCss(teamALegend) }} />
                Team A
              </span>
            )}
            {teamBLegend && (
              <span className="inline-flex items-center gap-1.5">
                <span className="h-3 w-3 rounded-sm" style={{ background: teamColorCss(teamBLegend) }} />
                Team B
              </span>
            )}
          </div>
        )}
      </div>

      <div className="border-t border-[#ffffff14] bg-[#111118] p-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-center">
          <div className="flex items-center gap-2">
            <button type="button" onClick={() => setFrame(frameId - 1)} className="grid h-10 w-10 place-items-center rounded-lg border border-[#ffffff14] bg-[#0a0a0f] text-[#f1f5f9] hover:bg-[#ffffff08]">
              <SkipBack size={18} />
            </button>
            <button type="button" onClick={togglePlayback} className="grid h-10 w-10 place-items-center rounded-lg bg-[#3b82f6] text-white">
              {playing ? <Pause size={18} /> : <Play size={18} />}
            </button>
            <button type="button" onClick={() => setFrame(frameId + 1)} className="grid h-10 w-10 place-items-center rounded-lg border border-[#ffffff14] bg-[#0a0a0f] text-[#f1f5f9] hover:bg-[#ffffff08]">
              <SkipForward size={18} />
            </button>
          </div>
          <input
            type="range"
            min={0}
            max={frameCount}
            value={frameId}
            onChange={(event) => setFrame(Number(event.target.value))}
            className="min-w-0 flex-1"
          />
          <div className="min-w-36 text-right text-sm tabular-nums text-[#64748b]">
            Frame {frameId} / {frameCount}
          </div>
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-[#64748b]">
          <span>{duration.toFixed(1)}s</span>
          <span className="h-1 w-1 rounded-full bg-[#ffffff14]" />
          <span>{fps} fps</span>
          <span className="h-1 w-1 rounded-full bg-[#ffffff14]" />
          <span>{playerDetections.length} players</span>
        </div>
      </div>
    </div>
  );
}

function containsPoint(detection: Detection, point: Point) {
  const { bbox } = detection;
  return point.x >= bbox.x && point.x <= bbox.x + bbox.width && point.y >= bbox.y && point.y <= bbox.y + bbox.height;
}
