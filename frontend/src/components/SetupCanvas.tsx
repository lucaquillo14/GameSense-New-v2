"use client";

import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import type { Detection, Point, VideoMetadata } from "@/lib/api";

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
  onFrameChange,
  onPlayerPoint,
  onPitchPoint,
  onGoalLeft,
  onGoalRight,
  onRemovePitchPoint,
}: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const fps = metadata?.fps || 30;
  const frameCount = Math.max((metadata?.frame_count ?? 1) - 1, 0);
  const duration = metadata?.duration_s || 0;
  const width = metadata?.width || 1;
  const height = metadata?.height || 1;

  const playerDetections = useMemo(() => detections.filter((detection) => detection.label === "player"), [detections]);
  const ballDetections = useMemo(() => detections.filter((detection) => detection.label === "ball"), [detections]);

  function syncFrameFromVideo() {
    const video = videoRef.current;
    if (!video) return;
    onFrameChange(Math.min(Math.round(video.currentTime * fps), frameCount));
  }

  function setFrame(nextFrame: number) {
    const video = videoRef.current;
    if (!video) return;
    const bounded = Math.min(Math.max(nextFrame, 0), frameCount);
    video.pause();
    setPlaying(false);
    video.currentTime = bounded / fps;
    onFrameChange(bounded);
  }

  function togglePlayback() {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      video.play();
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

  return (
    <div className="overflow-hidden rounded-lg border border-white/10 bg-[#06080d] shadow-2xl shadow-black/40">
      <div className="relative aspect-video bg-black">
        <video
          ref={videoRef}
          src={videoUrl}
          className="h-full w-full object-contain"
          preload="metadata"
          onTimeUpdate={syncFrameFromVideo}
          onPause={() => {
            setPlaying(false);
            syncFrameFromVideo();
          }}
          onPlay={() => setPlaying(true)}
          onLoadedMetadata={syncFrameFromVideo}
        />
        <div ref={overlayRef} className="absolute inset-0 cursor-crosshair" onClick={handleOverlayClick}>
          <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
            {pitchPolygon.length > 1 && (
              <polyline
                points={pitchPolygon.map((point) => `${point.x},${point.y}`).join(" ")}
                fill={pitchPolygon.length >= 4 ? "rgba(20,184,166,0.15)" : "none"}
                stroke="#14b8a6"
                strokeWidth={Math.max(width / 360, 2)}
                strokeLinejoin="round"
              />
            )}
            {[goalLeft, goalRight].map((goal, index) =>
              goal ? <circle key={index} cx={goal.x} cy={goal.y} r={Math.max(width / 120, 8)} fill={index === 0 ? "#facc15" : "#22c55e"} /> : null,
            )}
          </svg>

          {playerDetections.map((detection) => (
            <DetectionBox
              key={detection.id}
              detection={detection}
              selected={selectedDetection?.id === detection.id}
              frameWidth={width}
              frameHeight={height}
            />
          ))}
          {ballDetections.map((detection) => (
            <DetectionBox key={detection.id} detection={detection} selected={false} frameWidth={width} frameHeight={height} compact />
          ))}
          {pitchPolygon.map((point, index) => (
            <button
              key={`${point.x}-${point.y}-${index}`}
              type="button"
              title="Remove calibration point"
              onClick={(event) => {
                event.stopPropagation();
                onRemovePitchPoint(index);
              }}
              className="absolute h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-teal-400 shadow-lg shadow-black/40"
              style={{ left: `${(point.x / width) * 100}%`, top: `${(point.y / height) * 100}%` }}
            />
          ))}
          {playerPoint && (
            <span
              className="absolute h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-blue-400 shadow-lg shadow-blue-500/40"
              style={{ left: `${(playerPoint.x / width) * 100}%`, top: `${(playerPoint.y / height) * 100}%` }}
            />
          )}
        </div>
      </div>

      <div className="border-t border-white/10 bg-[#0c111c] p-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-center">
          <div className="flex items-center gap-2">
            <button type="button" onClick={() => setFrame(frameId - 1)} className="grid h-10 w-10 place-items-center rounded-lg bg-white/8 text-white hover:bg-white/14">
              <SkipBack size={18} />
            </button>
            <button type="button" onClick={togglePlayback} className="grid h-10 w-10 place-items-center rounded-lg bg-blue-500 text-white hover:bg-blue-400">
              {playing ? <Pause size={18} /> : <Play size={18} />}
            </button>
            <button type="button" onClick={() => setFrame(frameId + 1)} className="grid h-10 w-10 place-items-center rounded-lg bg-white/8 text-white hover:bg-white/14">
              <SkipForward size={18} />
            </button>
          </div>
          <input
            type="range"
            min={0}
            max={frameCount}
            value={frameId}
            onChange={(event) => setFrame(Number(event.target.value))}
            className="min-w-0 flex-1 accent-blue-500"
          />
          <div className="min-w-36 text-right text-sm tabular-nums text-slate-300">
            Frame {frameId} / {frameCount}
          </div>
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-slate-400">
          <span>{duration.toFixed(1)}s</span>
          <span className="h-1 w-1 rounded-full bg-slate-600" />
          <span>{fps} fps</span>
          <span className="h-1 w-1 rounded-full bg-slate-600" />
          <span>{playerDetections.length} players detected</span>
          <span className="h-1 w-1 rounded-full bg-slate-600" />
          <span>{ballDetections.length} ball candidates</span>
        </div>
      </div>
    </div>
  );
}

function DetectionBox({
  detection,
  selected,
  frameWidth,
  frameHeight,
  compact = false,
}: {
  detection: Detection;
  selected: boolean;
  frameWidth: number;
  frameHeight: number;
  compact?: boolean;
}) {
  const { bbox } = detection;
  return (
    <div
      className={`pointer-events-none absolute border ${
        selected
          ? "border-blue-300 bg-blue-400/14 shadow-[0_0_0_2px_rgba(96,165,250,0.45)]"
          : compact
            ? "border-amber-300 bg-amber-300/12"
            : "border-emerald-300 bg-emerald-300/10"
      }`}
      style={{
        left: `${(bbox.x / frameWidth) * 100}%`,
        top: `${(bbox.y / frameHeight) * 100}%`,
        width: `${(bbox.width / frameWidth) * 100}%`,
        height: `${(bbox.height / frameHeight) * 100}%`,
      }}
    >
      <span className="absolute left-0 top-0 -translate-y-full rounded-t bg-black/75 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-white">
        {detection.label} {(detection.confidence * 100).toFixed(0)}%
      </span>
    </div>
  );
}

function containsPoint(detection: Detection, point: Point) {
  const { bbox } = detection;
  return point.x >= bbox.x && point.x <= bbox.x + bbox.width && point.y >= bbox.y && point.y <= bbox.y + bbox.height;
}
