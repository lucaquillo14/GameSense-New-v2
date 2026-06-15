"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { AppShell } from "@/components/AppShell";
import { API_BASE, uploadVideo } from "@/lib/api";

const API = process.env.NEXT_PUBLIC_API_URL ?? API_BASE;
import {
  formatDuration,
  LocalVideoMeta,
  MAX_UPLOAD_MB,
  MAX_VIDEO_DURATION_S,
  readLocalVideoMeta,
  validateFileSize,
} from "@/lib/uploadLimits";
import { AlertCircle, ArrowRight, Film, Loader2, Maximize2, Target, Timer, UploadCloud } from "lucide-react";
import Link from "next/link";

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [localMeta, setLocalMeta] = useState<LocalVideoMeta | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);

  async function onFileSelected(next: File | null) {
    setError(null);
    setLocalMeta(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);

    if (!next) {
      setFile(null);
      setPreviewUrl(null);
      return;
    }

    const sizeError = validateFileSize(next);
    if (sizeError) {
      setFile(null);
      setPreviewUrl(null);
      setError(sizeError);
      return;
    }

    setChecking(true);
    try {
      const meta = await readLocalVideoMeta(next);
      setFile(next);
      setLocalMeta(meta);
      setPreviewUrl(URL.createObjectURL(next));
    } catch (err) {
      setFile(null);
      setPreviewUrl(null);
      setError(err instanceof Error ? err.message : "Could not validate this video.");
    } finally {
      setChecking(false);
    }
  }

  async function submit() {
    if (!file || !localMeta) return;
    setBusy(true);
    setError(null);
    try {
      console.info("Uploading to", API);
      const upload = await uploadVideo(file);
      router.push(`/setup/${upload.video_id}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Upload failed. Try another clip.";
      setError(message);
      console.error("Upload failed:", err);
    } finally {
      setBusy(false);
    }
  }

  return (
    <AppShell>
      <section className="analytics-grid hero-glow mx-auto max-w-4xl px-5 py-16 fade-in">

        {/* ── Hero ────────────────────────────────────────────── */}
        <div className="mb-12 text-center">
          <div className="chip mx-auto mb-6 w-fit">
            <span className="h-1.5 w-1.5 rounded-full bg-cyan-400 glow-pulse" />
            AI sprint &amp; technique analysis
          </div>
          <h1 className="display text-5xl text-[#eef2ff] sm:text-[3.75rem]">
            Measure your game
            <br />
            in <span className="gradient-text">real numbers</span>
          </h1>
          <p className="mx-auto mt-5 max-w-xl text-lg leading-relaxed text-[#6b7a99]">
            Drop a clip to track players and measure sprint speed, distance, and technique — then
            climb the leaderboard.
          </p>
        </div>

        {/* ── Upload card ──────────────────────────────────────── */}
        <div className="card p-6">
          <label className="dropzone-hover group relative flex min-h-72 w-full cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-cyan-500/20 bg-cyan-500/[0.02] px-6 py-10 text-center transition-all duration-300 hover:bg-cyan-500/[0.04]">
            {/* Corner brackets */}
            <span className="absolute left-3 top-3 h-5 w-5 rounded-tl-sm border-l-2 border-t-2 border-cyan-500/40" />
            <span className="absolute right-3 top-3 h-5 w-5 rounded-tr-sm border-r-2 border-t-2 border-cyan-500/40" />
            <span className="absolute bottom-3 left-3 h-5 w-5 rounded-bl-sm border-b-2 border-l-2 border-cyan-500/40" />
            <span className="absolute bottom-3 right-3 h-5 w-5 rounded-br-sm border-b-2 border-r-2 border-cyan-500/40" />

            <span className="mb-5 grid h-16 w-16 place-items-center rounded-2xl bg-gradient-to-br from-cyan-500 to-blue-600 text-white shadow-[0_0_32px_rgba(6,182,212,0.45)]">
              {checking ? (
                <Loader2 size={28} className="animate-spin" />
              ) : (
                <UploadCloud size={28} />
              )}
            </span>

            <span className="font-display text-xl font-semibold text-[#eef2ff]">
              {checking ? "Checking video…" : file ? file.name : "Drag & drop your video clip"}
            </span>
            <span className="mt-2 text-sm text-[#6b7a99]">
              MP4 or MOV · up to {MAX_UPLOAD_MB} MB · max {MAX_VIDEO_DURATION_S}s
            </span>

            <input
              type="file"
              accept="video/mp4,video/quicktime"
              className="sr-only"
              onChange={(event) => void onFileSelected(event.target.files?.[0] ?? null)}
            />
          </label>

          {localMeta && (
            <div className="mt-4 flex flex-wrap gap-2">
              <MetaPill icon={<Film size={13} />} label={`${localMeta.sizeMb} MB`} />
              <MetaPill icon={<Timer size={13} />} label={formatDuration(localMeta.durationS)} />
              <MetaPill
                icon={<Maximize2 size={13} />}
                label={`${localMeta.width}×${localMeta.height}`}
              />
            </div>
          )}

          {previewUrl && (
            <video
              src={previewUrl}
              className="mt-4 max-h-48 w-full rounded-xl border border-white/[0.07] object-contain"
              muted
            />
          )}

          {error && (
            <div className="mt-4 flex items-center gap-2 rounded-xl border border-red-500/25 bg-red-500/8 px-4 py-3 text-sm text-red-300">
              <AlertCircle size={16} className="shrink-0" />
              {error}
            </div>
          )}

          <button
            type="button"
            onClick={submit}
            disabled={!file || !localMeta || busy || checking}
            className="btn-primary mt-5 flex w-full items-center justify-center gap-2 px-4 py-3.5 text-sm disabled:cursor-not-allowed"
          >
            {busy ? <Loader2 size={18} className="animate-spin" /> : <ArrowRight size={18} />}
            {busy ? "Uploading video…" : "Continue to player selection"}
          </button>
        </div>

        {/* ── Secondary CTA ────────────────────────────────────── */}
        <div className="mt-6 text-center">
          <Link
            href="/technique"
            className="inline-flex items-center gap-2 rounded-xl border border-white/[0.07] bg-[#09090f] px-5 py-3 text-sm font-medium text-[#eef2ff] transition-all hover:border-cyan-500/25 hover:bg-cyan-500/5"
          >
            <Target size={16} className="text-cyan-400" />
            Technique Analysis — upload a shooting clip
          </Link>
        </div>

        <p className="mt-5 text-center text-xs text-[#3a4560]">
          FPS is detected after upload. Clips longer than {MAX_VIDEO_DURATION_S} seconds are
          rejected before upload.
        </p>
      </section>
    </AppShell>
  );
}

function MetaPill({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-white/[0.07] bg-[#09090f] px-3 py-1 text-xs font-medium text-[#eef2ff]">
      {icon}
      {label}
    </span>
  );
}
