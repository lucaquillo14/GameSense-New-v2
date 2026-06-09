"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { AppShell } from "@/components/AppShell";
import { uploadVideo } from "@/lib/api";
import { AlertCircle, ArrowRight, BarChart3, CheckCircle2, Loader2, Radar, UploadCloud, Video } from "lucide-react";

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit() {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const upload = await uploadVideo(file);
      router.push(`/setup/${upload.video_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <AppShell>
      <section className="analytics-grid mx-auto grid max-w-7xl gap-8 px-5 py-10 lg:min-h-[calc(100vh-137px)] lg:grid-cols-[0.9fr_1.1fr] lg:items-center">
        <div className="max-w-2xl">
          <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/8 px-3 py-1.5 text-sm font-medium text-slate-300">
            <Radar size={15} className="text-cyan-300" />
            Sports performance intelligence
          </div>
          <h1 className="mt-5 max-w-2xl text-5xl font-semibold leading-[1.04] tracking-normal text-white sm:text-6xl">
            Turn raw match video into player tracking data.
          </h1>
          <p className="mt-5 max-w-xl text-lg leading-8 text-slate-400">
            Upload footage, select one player, calibrate the pitch when useful, then generate speed, distance, sprint, and confidence outputs.
          </p>
          <div className="mt-8 grid gap-3 text-sm text-slate-300 sm:grid-cols-3">
            <Feature icon={<Video size={16} />} label="Video workstation" />
            <Feature icon={<BarChart3 size={16} />} label="Analytics dashboard" />
            <Feature icon={<CheckCircle2 size={16} />} label="Confidence scoring" />
          </div>
        </div>

        <div className="rounded-lg border border-white/10 bg-white/[0.04] p-4 shadow-2xl shadow-black/30">
          <label className="group flex min-h-80 cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-white/15 bg-slate-950/50 px-6 py-10 text-center hover:border-cyan-300/70 hover:bg-cyan-300/5">
            <span className="grid h-16 w-16 place-items-center rounded-lg bg-cyan-400 text-slate-950 shadow-lg shadow-cyan-500/20 group-hover:bg-cyan-300">
              <UploadCloud size={30} />
            </span>
            <span className="mt-5 max-w-full break-words text-xl font-semibold text-white">{file ? file.name : "Upload match footage"}</span>
            <span className="mt-2 text-sm text-slate-500">MP4 or MOV. Broadcast or elevated sideline footage works best.</span>
            <input
              type="file"
              accept="video/mp4,video/quicktime"
              className="sr-only"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
          </label>
          {error && (
            <div className="mt-4 flex items-center gap-2 rounded-lg border border-red-400/25 bg-red-500/10 px-3 py-2 text-sm text-red-200">
              <AlertCircle size={16} />
              {error}
            </div>
          )}
          <button
            type="button"
            onClick={submit}
            disabled={!file || busy}
            className="mt-5 flex w-full items-center justify-center gap-2 rounded-lg bg-cyan-400 px-4 py-3 font-semibold text-slate-950 shadow-lg shadow-cyan-500/20 hover:bg-cyan-300 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400 disabled:shadow-none"
          >
            {busy ? <Loader2 size={18} className="animate-spin" /> : <ArrowRight size={18} />}
            Continue to setup
          </button>
          <div className="mt-4 flex items-center justify-center gap-2 text-xs text-slate-500">
            <CheckCircle2 size={14} className="text-emerald-300" />
            Local storage, background processing, progress polling.
          </div>
        </div>
      </section>
    </AppShell>
  );
}

function Feature({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.04] px-3 py-2 shadow-sm">
      <span className="text-cyan-300">{icon}</span>
      <span>{label}</span>
    </div>
  );
}
