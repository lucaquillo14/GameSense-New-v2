"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { AppShell } from "@/components/AppShell";
import { uploadVideo } from "@/lib/api";
import {
  formatDuration,
  LocalVideoMeta,
  MAX_UPLOAD_MB,
  MAX_VIDEO_DURATION_S,
  readLocalVideoMeta,
  validateFileSize,
} from "@/lib/uploadLimits";
import { AlertCircle, ArrowRight, Film, Gauge, Loader2, Maximize2, Timer, UploadCloud } from "lucide-react";

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
      const upload = await uploadVideo(file);
      router.push(`/setup/${upload.video_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed. Try another clip.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <AppShell>
      <section className="analytics-grid mx-auto max-w-5xl px-5 py-12 fade-in">
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-semibold tracking-tight text-[#f1f5f9] sm:text-5xl">
            Upload match footage
          </h1>
          <p className="mx-auto mt-3 max-w-2xl text-[#64748b]">
            Drop a clip to detect teams, track players, and measure speed or shot power.
          </p>
        </div>

        <div className="card p-6">
          <label className="dropzone-hover group flex min-h-72 w-full cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-[#ffffff14] bg-[#0a0a0f] px-6 py-10 text-center transition-colors hover:bg-[#3b82f6]/5">
            <span className="grid h-16 w-16 place-items-center rounded-xl bg-[#3b82f6] text-white shadow-[0_0_28px_rgba(59,130,246,0.35)]">
              {checking ? <Loader2 size={30} className="animate-spin" /> : <UploadCloud size={30} />}
            </span>
            <span className="mt-5 text-xl font-semibold text-[#f1f5f9]">
              {checking ? "Checking video…" : file ? file.name : "Drag and drop your video here"}
            </span>
            <span className="mt-2 text-sm text-[#64748b]">
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
              <MetaPill icon={<Film size={14} />} label={`${localMeta.sizeMb} MB`} />
              <MetaPill icon={<Timer size={14} />} label={formatDuration(localMeta.durationS)} />
              <MetaPill icon={<Maximize2 size={14} />} label={`${localMeta.width}×${localMeta.height}`} />
            </div>
          )}

          {previewUrl && (
            <video src={previewUrl} className="mt-4 max-h-48 w-full rounded-lg border border-[#ffffff14] object-contain" muted />
          )}

          {error && (
            <div className="mt-4 flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
              <AlertCircle size={16} />
              {error}
            </div>
          )}

          <button
            type="button"
            onClick={submit}
            disabled={!file || !localMeta || busy || checking}
            className="btn-primary mt-5 flex w-full items-center justify-center gap-2 px-4 py-3 disabled:cursor-not-allowed"
          >
            {busy ? <Loader2 size={18} className="animate-spin" /> : <ArrowRight size={18} />}
            {busy ? "Uploading video…" : "Continue to player selection"}
          </button>
        </div>

        <p className="mt-6 text-center text-xs text-[#64748b]">
          FPS is detected after upload. Clips longer than {MAX_VIDEO_DURATION_S} seconds are rejected before upload.
        </p>
      </section>
    </AppShell>
  );
}

function MetaPill({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-[#ffffff14] bg-[#111118] px-3 py-1 text-xs text-[#f1f5f9]">
      {icon}
      {label}
    </span>
  );
}
