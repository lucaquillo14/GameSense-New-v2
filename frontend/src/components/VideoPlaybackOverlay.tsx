"use client";

import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  DetectionsOverlay,
  drawOverlayDetections,
  frameIdFromTime,
  lookupOverlayFrame,
} from "@/lib/overlay";

type Props = {
  videoUrl: string;
  fps: number;
  frameCount: number;
  durationS: number;
  overlay: DetectionsOverlay | null;
  targetPlayerId?: string | null;
};

export function VideoPlaybackOverlay({
  videoUrl,
  fps,
  frameCount,
  durationS,
  overlay,
  targetPlayerId,
}: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [frameId, setFrameId] = useState(0);
  const targetId = targetPlayerId ?? overlay?.target_id ?? null;

  const redraw = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !overlay) {
      if (canvas) {
        const context = canvas.getContext("2d");
        context?.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }
    const currentFrame = frameIdFromTime(video.currentTime, overlay.fps || fps);
    const detections = lookupOverlayFrame(overlay.frames, currentFrame, overlay.interval || 1);
    drawOverlayDetections(canvas, video, detections, targetId);
  }, [overlay, fps, targetId]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const observer = new ResizeObserver(() => redraw());
    observer.observe(video);

    const onTimeUpdate = () => {
      setFrameId(frameIdFromTime(video.currentTime, overlay?.fps || fps));
      redraw();
    };

    video.addEventListener("timeupdate", onTimeUpdate);
    video.addEventListener("loadedmetadata", onTimeUpdate);
    video.addEventListener("seeked", onTimeUpdate);

    return () => {
      observer.disconnect();
      video.removeEventListener("timeupdate", onTimeUpdate);
      video.removeEventListener("loadedmetadata", onTimeUpdate);
      video.removeEventListener("seeked", onTimeUpdate);
    };
  }, [fps, overlay, redraw]);

  useEffect(() => {
    redraw();
  }, [overlay, frameId, redraw]);

  function setFrame(nextFrame: number) {
    const video = videoRef.current;
    if (!video) return;
    const bounded = Math.min(Math.max(nextFrame, 0), frameCount);
    video.pause();
    setPlaying(false);
    video.currentTime = bounded / Math.max(fps, 1);
    setFrameId(bounded);
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
    }
  }

  return (
    <div className="card overflow-hidden shadow-[0_24px_60px_rgba(0,0,0,0.45)]">
      <div className="relative aspect-video bg-black">
        <video ref={videoRef} src={videoUrl} className="h-full w-full object-contain" preload="metadata" playsInline />
        <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 z-20 h-full w-full" />
      </div>

      <div className="border-t border-[#ffffff14] bg-[#111118] p-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-center">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setFrame(frameId - 1)}
              className="grid h-10 w-10 place-items-center rounded-lg border border-[#ffffff14] bg-[#0a0a0f] text-[#f1f5f9] hover:bg-[#ffffff08]"
            >
              <SkipBack size={18} />
            </button>
            <button type="button" onClick={togglePlayback} className="grid h-10 w-10 place-items-center rounded-lg bg-[#3b82f6] text-white">
              {playing ? <Pause size={18} /> : <Play size={18} />}
            </button>
            <button
              type="button"
              onClick={() => setFrame(frameId + 1)}
              className="grid h-10 w-10 place-items-center rounded-lg border border-[#ffffff14] bg-[#0a0a0f] text-[#f1f5f9] hover:bg-[#ffffff08]"
            >
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
        <div className="mt-2 text-xs text-[#64748b]">
          {durationS.toFixed(1)}s · {fps} fps
          {!overlay
            ? " · Loading detections…"
            : Object.keys(overlay.frames).length === 0
              ? " · No overlay data — re-run analysis to generate boxes"
              : ` · ${Object.keys(overlay.frames).length} sampled frames`}
        </div>
      </div>
    </div>
  );
}
