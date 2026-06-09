import { Activity, BarChart3, Gauge, Sparkles, Upload } from "lucide-react";
import Link from "next/link";

export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <main className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-white/10 bg-[#070b13]/82 backdrop-blur-2xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-3">
          <Link href="/" className="flex items-center gap-3 font-semibold text-white">
            <span className="grid h-9 w-9 place-items-center rounded-lg bg-cyan-400 text-slate-950 shadow-lg shadow-cyan-500/20">
              <Gauge size={20} />
            </span>
            <span className="text-[15px]">GameSense AI</span>
          </Link>
          <nav className="hidden items-center gap-1 rounded-lg border border-white/10 bg-white/[0.04] p-1 text-sm text-slate-300 shadow-sm sm:flex">
            <Link href="/" className="flex items-center gap-2 rounded-md px-3 py-1.5 hover:bg-white/8">
              <Upload size={15} />
              Upload
            </Link>
            <span className="flex items-center gap-2 rounded-md px-3 py-1.5 text-slate-500">
              <Activity size={15} />
              Setup
            </span>
            <span className="flex items-center gap-2 rounded-md px-3 py-1.5 text-slate-500">
              <BarChart3 size={15} />
              Results
            </span>
          </nav>
        </div>
      </header>
      {children}
      <footer className="border-t border-white/10 bg-[#070b13]/70">
        <div className="mx-auto flex max-w-7xl items-center gap-2 px-5 py-4 text-sm text-slate-500">
          <Sparkles size={16} />
          <span>Single-player physical metrics, detection overlays, calibration, and tracking confidence.</span>
        </div>
      </footer>
    </main>
  );
}
