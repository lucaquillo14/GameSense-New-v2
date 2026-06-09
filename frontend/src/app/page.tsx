"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { AppShell } from "@/components/AppShell";
import { uploadVideo } from "@/lib/api";
import { AlertCircle, ArrowRight, Film, Gauge, Loader2, Timer, UploadCloud } from "lucide-react";

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const localMeta = useMemo(() => {
    if (!file || !previewUrl) return null;
    return { name: file.name, sizeMb: (file.size / (1024 * 1024)).toFixed(1) };
  }, [file, previewUrl]);

  function onFileSelected(next: File | null) {
    setFile(next);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(next ? URL.createObjectURL(next) : null);
  }

  async function submit() {
    if (!file) return;
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
              <UploadCloud size={30} />
            </span>
            <span className="mt-5 text-xl font-semibold text-[#f1f5f9]">
              {file ? file.name : "Drag and drop your video here"}
            </span>
            <span className="mt-2 text-sm text-[#64748b]">MP4 or MOV · sideline or broadcast angle works best</span>
            <input
              type="file"
              accept="video/mp4,video/quicktime"
              className="sr-only"
              onChange={(event) => onFileSelected(event.target.files?.[0] ?? null)}
            />
          </label>

          {localMeta && (
            <div className="mt-4 flex flex-wrap gap-2">
              <MetaPill icon={<Film size={14} />} label={`${localMeta.sizeMb} MB`} />
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
            disabled={!file || busy}
            className="btn-primary mt-5 flex w-full items-center justify-center gap-2 px-4 py-3 disabled:cursor-not-allowed"
          >
            {busy ? <Loader2 size={18} className="animate-spin" /> : <ArrowRight size={18} />}
            {busy ? "Uploading video…" : "Continue to player selection"}
          </button>
        </div>

        <p className="mt-6 text-center text-xs text-[#64748b]">
          After upload, resolution, fps, and duration appear as badges on the setup screen.
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
