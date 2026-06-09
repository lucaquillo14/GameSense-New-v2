import { Activity, BarChart3, Gauge, Upload } from "lucide-react";
import Link from "next/link";

export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <main className="min-h-screen bg-[#0a0a0f]">
      <header className="sticky top-0 z-20 border-b border-[#ffffff14] bg-[#0a0a0f]/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4">
          <Link href="/" className="flex items-center gap-3 font-semibold text-[#f1f5f9]">
            <span className="grid h-9 w-9 place-items-center rounded-lg bg-[#3b82f6] text-white shadow-[0_0_24px_rgba(59,130,246,0.35)]">
              <Gauge size={20} />
            </span>
            <span>GameSense AI</span>
          </Link>
          <nav className="hidden items-center gap-1 rounded-lg border border-[#ffffff14] bg-[#111118] p-1 text-sm text-[#64748b] sm:flex">
            <Link href="/" className="flex items-center gap-2 rounded-md px-3 py-1.5 text-[#f1f5f9] hover:bg-[#ffffff08]">
              <Upload size={15} />
              Upload
            </Link>
            <span className="flex items-center gap-2 rounded-md px-3 py-1.5">
              <Activity size={15} />
              Setup
            </span>
            <span className="flex items-center gap-2 rounded-md px-3 py-1.5">
              <BarChart3 size={15} />
              Results
            </span>
          </nav>
        </div>
      </header>
      {children}
    </main>
  );
}
