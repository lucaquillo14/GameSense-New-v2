"use client";

import { AppShell } from "@/components/AppShell";
import {
  Activity,
  ArrowRight,
  Crown,
  FileText,
  Footprints,
  Gauge,
  Route,
  Sparkles,
  Target,
  TrendingUp,
  UploadCloud,
  Zap,
} from "lucide-react";
import Link from "next/link";

export default function HomePage() {
  return (
    <AppShell>
      <section className="hero-glow mx-auto max-w-6xl px-5 py-16 fade-in">
        {/* ── Hero ────────────────────────────────────────────── */}
        <div className="mb-14 text-center">
          <div className="chip mx-auto mb-6 w-fit">
            <span className="h-1.5 w-1.5 rounded-full bg-cyan-400 glow-pulse" />
            AI football performance analytics
          </div>
          <h1 className="display text-6xl text-[#eef2ff] sm:text-[4.75rem]">
            Measure your game
            <br />
            in <span className="gradient-text">real numbers</span>
          </h1>
          <p className="mx-auto mt-6 max-w-xl text-lg leading-relaxed text-[#8b95a7]">
            Turn match and training clips into hard data — sprint speed, shot power, and biomechanics
            — then climb the global leaderboard.
          </p>

          {/* Live stats showcase */}
          <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
            <ShowcaseStat className="float-slow" icon={<Gauge size={16} />} label="Top speed" value="32.4" unit="km/h" tone="cyan" />
            <ShowcaseStat className="float-slower -mt-4" icon={<Zap size={16} />} label="Shot power" value="98" unit="km/h" tone="violet" />
            <ShowcaseStat className="float-slow" icon={<Activity size={16} />} label="Technique" value="8.6" unit="/ 10" tone="energy" />
          </div>
        </div>

        {/* ── Mode selector ───────────────────────────────────── */}
        <div className="mb-6 text-center">
          <h2 className="font-display text-sm font-semibold uppercase tracking-[0.2em] text-[#6b7a99]">
            Choose your analysis
          </h2>
        </div>

        <div className="grid gap-5 md:grid-cols-2">
          <ModeCard
            href="/speed"
            icon={<Gauge size={26} />}
            tone="cyan"
            eyebrow="Match video"
            title="Sprint Speed"
            body="Track any player through match footage and measure top speed, distance covered, and sprint counts."
            features={[
              { icon: <TrendingUp size={14} />, text: "Top & average speed (km/h)" },
              { icon: <Footprints size={14} />, text: "Distance & sprint distance" },
              { icon: <Route size={14} />, text: "Movement heatmaps (Pro)" },
            ]}
            cta="Analyse match speed"
          />

          <ModeCard
            href="/technique"
            icon={<Target size={26} />}
            tone="violet"
            eyebrow="Shooting clip"
            title="Shooting Technique"
            body="Upload a shooting clip for full biomechanics analysis — pose, joint angles, shot power, and a technique score."
            features={[
              { icon: <Activity size={14} />, text: "Technique score & joint angles" },
              { icon: <Zap size={14} />, text: "Shot power (km/h)" },
              { icon: <FileText size={14} />, text: "Downloadable PDF report (Pro)" },
            ]}
            cta="Analyse technique"
          />
        </div>

        {/* ── How it works ─────────────────────────────────────── */}
        <div className="mt-20">
          <div className="mb-7 text-center">
            <h2 className="font-display text-2xl font-bold text-[#eef2ff]">How it works</h2>
            <p className="mt-2 text-sm text-[#6b7a99]">Three steps from clip to data.</p>
          </div>
          <div className="grid gap-4 sm:grid-cols-3">
            <StepCard step="01" icon={<UploadCloud size={20} />} title="Upload a clip" body="Drop an MP4 or MOV — match footage or a shooting session. No setup needed." />
            <StepCard step="02" icon={<Route size={20} />} title="AI tracks the play" body="Players, the ball, and body pose are detected and tracked frame by frame." />
            <StepCard step="03" icon={<Sparkles size={20} />} title="Get real metrics" body="Speed, power, and technique — saved to your profile and the leaderboard." />
          </div>
        </div>

        {/* ── Upgrade banner ──────────────────────────────────── */}
        <Link
          href="/pricing"
          className="card card-hover group mt-12 flex flex-col items-center justify-between gap-4 overflow-hidden p-6 sm:flex-row"
        >
          <div className="pointer-events-none absolute -right-10 -top-10 h-40 w-40 rounded-full bg-violet-500/10 blur-3xl" />
          <div className="flex items-center gap-4">
            <span className="grid h-12 w-12 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-amber-400/20 to-yellow-300/20 text-amber-300">
              <Crown size={24} />
            </span>
            <div>
              <p className="font-display text-lg font-semibold text-[#eef2ff]">Go further with Pro &amp; Elite</p>
              <p className="text-sm text-[#6b7a99]">Heatmaps, downloadable reports, AI insights, and unlimited analyses.</p>
            </div>
          </div>
          <span className="inline-flex shrink-0 items-center gap-1.5 rounded-xl bg-cyan-500 px-5 py-2.5 text-sm font-bold text-[#04121f] transition-transform group-hover:translate-x-0.5">
            View plans <ArrowRight size={15} />
          </span>
        </Link>
      </section>
    </AppShell>
  );
}

const TONES = {
  cyan: {
    iconBg: "bg-cyan-500/12 text-cyan-300",
    glow: "bg-cyan-500/10",
    ring: "hover:border-cyan-500/35",
    bullet: "text-cyan-400",
  },
  violet: {
    iconBg: "bg-violet-500/12 text-violet-300",
    glow: "bg-violet-500/10",
    ring: "hover:border-violet-500/35",
    bullet: "text-violet-400",
  },
} as const;

function ModeCard({
  href,
  icon,
  tone,
  eyebrow,
  title,
  body,
  features,
  cta,
}: {
  href: string;
  icon: React.ReactNode;
  tone: keyof typeof TONES;
  eyebrow: string;
  title: string;
  body: string;
  features: { icon: React.ReactNode; text: string }[];
  cta: string;
}) {
  const t = TONES[tone];
  return (
    <Link href={href} className={`card group relative overflow-hidden p-7 transition-all duration-200 hover:-translate-y-1 ${t.ring}`}>
      {/* corner brackets */}
      <span className="absolute left-3 top-3 h-4 w-4 rounded-tl-sm border-l-2 border-t-2 border-white/10 transition-colors group-hover:border-white/25" />
      <span className="absolute right-3 top-3 h-4 w-4 rounded-tr-sm border-r-2 border-t-2 border-white/10 transition-colors group-hover:border-white/25" />
      <span className="absolute bottom-3 left-3 h-4 w-4 rounded-bl-sm border-b-2 border-l-2 border-white/10 transition-colors group-hover:border-white/25" />
      <span className="absolute bottom-3 right-3 h-4 w-4 rounded-br-sm border-b-2 border-r-2 border-white/10 transition-colors group-hover:border-white/25" />
      {/* glow */}
      <div className={`pointer-events-none absolute -right-12 -top-12 h-48 w-48 rounded-full blur-3xl ${t.glow}`} />

      <div className="relative">
        <div className="mb-5 flex items-center justify-between">
          <span className={`grid h-14 w-14 place-items-center rounded-2xl ${t.iconBg}`}>{icon}</span>
          <span className="data-label">{eyebrow}</span>
        </div>

        <h3 className="font-display text-2xl font-bold text-[#eef2ff]">{title}</h3>
        <p className="mt-2 text-sm leading-relaxed text-[#8b95a7]">{body}</p>

        <ul className="mt-5 space-y-2.5">
          {features.map((f) => (
            <li key={f.text} className="flex items-center gap-2 text-sm text-[#c4d0f0]">
              <span className={t.bullet}>{f.icon}</span>
              {f.text}
            </li>
          ))}
        </ul>

        <div className="mt-7 flex items-center gap-2 text-sm font-bold text-[#eef2ff]">
          {cta}
          <ArrowRight size={16} className="transition-transform group-hover:translate-x-1" />
        </div>
      </div>
    </Link>
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
    cyan: "shadow-[0_0_30px_-8px_rgba(6,182,212,0.5)]",
    violet: "shadow-[0_0_30px_-8px_rgba(139,92,246,0.5)]",
    energy: "shadow-[0_0_30px_-8px_rgba(163,230,53,0.5)]",
  } as const;
  const iconTone = {
    cyan: "text-cyan-300",
    violet: "text-violet-300",
    energy: "text-lime-300",
  } as const;
  return (
    <div className={`card flex w-40 flex-col gap-1 p-4 text-left ${tones[tone]} ${className}`}>
      <span className="data-label flex items-center gap-1.5">
        <span className={iconTone[tone]}>{icon}</span>
        {label}
      </span>
      <div className="flex items-end gap-1">
        <span className="stat-value text-3xl font-bold text-[#eef2ff]">{value}</span>
        <span className="mb-1 text-xs text-[#6b7a99]">{unit}</span>
      </div>
    </div>
  );
}

function StepCard({ step, icon, title, body }: { step: string; icon: React.ReactNode; title: string; body: string }) {
  return (
    <div className="card card-hover p-6">
      <div className="mb-4 flex items-center justify-between">
        <span className="grid h-11 w-11 place-items-center rounded-xl bg-cyan-500/12 text-cyan-300">{icon}</span>
        <span className="stat-value text-2xl text-white/10">{step}</span>
      </div>
      <h3 className="font-display text-lg font-semibold text-[#eef2ff]">{title}</h3>
      <p className="mt-1.5 text-sm leading-relaxed text-[#6b7a99]">{body}</p>
    </div>
  );
}
