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
import { Activity, AlertCircle, ArrowRight, Film, Gauge, Loader2, Maximize2, Route, Target, Timer, UploadCloud, Zap } from "lucide-react";
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
          <h1 className="display text-6xl text-[#eef2ff] sm:text-[4.5rem]">
            Measure your game
            <br />
            in <span className="gradient-text">real numbers</span>
          </h1>
          <p className="mx-auto mt-6 max-w-xl text-lg leading-relaxed text-[#8b95a7]">
            Drop a clip to track players and measure sprint speed, distance, and technique — then
            climb the leaderboard.
          </p>

          {/* Trust strip */}
          <div className="mt-7 flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-xs font-medium text-[#6b7a99]">
            <span className="inline-flex items-center gap-1.5">
              <Gauge size={13} className="text-cyan-400" /> Sprint speed &amp; distance
            </span>
            <span className="inline-flex items-center gap-1.5">
              <Target size={13} className="text-cyan-400" /> Shot power
            </span>
            <span className="inline-flex items-center gap-1.5">
              <Activity size={13} className="text-cyan-400" /> Technique scoring
            </span>
          </div>

          {/* Live stats showcase */}
          <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
            <ShowcaseStat
              className="float-slow"
              icon={<Gauge size={16} />}
              label="Top speed"
              value="32.4"
              unit="km/h"
              tone="cyan"
            />
            <ShowcaseStat
              className="float-slower -mt-4"
              icon={<Zap size={16} />}
              label="Shot power"
              value="98"
              unit="km/h"
              tone="violet"
            />
            <ShowcaseStat
              className="float-slow"
              icon={<Activity size={16} />}
              label="Technique"
              value="8.6"
              unit="/ 10"
              tone="energy"
            />
          </div>
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

        {/* ── How it works ─────────────────────────────────────── */}
        <div className="mt-16 grid gap-4 sm:grid-cols-3">
          <StepCard
            step="01"
            icon={<UploadCloud size={20} />}
            title="Upload a clip"
            body="Drop an MP4 or MOV of your match or shooting session — no setup required."
          />
          <StepCard
            step="02"
            icon={<Route size={20} />}
            title="AI tracks the play"
            body="Players and the ball are detected and tracked frame-by-frame, fully automatically."
          />
          <StepCard
            step="03"
            icon={<Zap size={20} />}
            title="Get real metrics"
            body="Sprint speed, distance, shot power and technique — then climb the leaderboard."
          />
        </div>
      </section>
    </AppShell>
  );
}

function ShowcaseStat({
  icon,
  label,
  value,
  unit,
  tone,
  className = "",
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  unit: string;
  tone: "cyan" | "violet" | "energy";
  className?: string;
}) {
  const tones = {
    cyan: "text-cyan-300 shadow-[0_0_30px_-8px_rgba(6,182,212,0.5)]",
    violet: "text-violet-300 shadow-[0_0_30px_-8px_rgba(139,92,246,0.5)]",
    energy: "text-lime-300 shadow-[0_0_30px_-8px_rgba(163,230,53,0.5)]",
  } as const;
  return (
    <div className={`card flex w-40 flex-col gap-1 p-4 text-left ${tones[tone]} ${className}`}>
      <span className="data-label flex items-center gap-1.5">
        <span className={tones[tone].split(" ")[0]}>{icon}</span>
        {label}
      </span>
      <div className="flex items-end gap-1">
        <span className="stat-value text-3xl font-bold text-[#eef2ff]">{value}</span>
        <span className="mb-1 text-xs text-[#6b7a99]">{unit}</span>
      </div>
    </div>
  );
}

function StepCard({
  step,
  icon,
  title,
  body,
}: {
  step: string;
  icon: React.ReactNode;
  title: string;
  body: string;
}) {
  return (
    <div className="card card-hover p-6">
      <div className="mb-4 flex items-center justify-between">
        <span className="grid h-11 w-11 place-items-center rounded-xl bg-cyan-500/12 text-cyan-300">
          {icon}
        </span>
        <span className="stat-value text-2xl text-white/10">{step}</span>
      </div>
      <h3 className="font-display text-lg font-semibold text-[#eef2ff]">{title}</h3>
      <p className="mt-1.5 text-sm leading-relaxed text-[#6b7a99]">{body}</p>
    </div>
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
