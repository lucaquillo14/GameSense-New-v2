"use client";

import { Clock, Gauge, LogIn, LogOut, Medal, Trophy, Upload, Users } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { clearAuth, getStoredUser, type User } from "@/lib/socialApi";
import { Avatar } from "@/components/Avatar";

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    const sync = () => setUser(getStoredUser());
    sync();
    window.addEventListener("gamesense-auth", sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener("gamesense-auth", sync);
      window.removeEventListener("storage", sync);
    };
  }, []);

  const navItem = (href: string, icon: React.ReactNode, label: string) => {
    const active = pathname === href || (href !== "/" && pathname?.startsWith(href));
    return (
      <Link
        href={href}
        className={`relative flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-all duration-200 ${
          active
            ? "bg-cyan-500/10 text-cyan-400"
            : "text-[#6b7a99] hover:bg-white/[0.04] hover:text-[#c4d0f0]"
        }`}
      >
        {icon}
        {label}
        {active && (
          <span className="absolute bottom-0.5 left-1/2 h-px w-5 -translate-x-1/2 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(6,182,212,0.9)]" />
        )}
      </Link>
    );
  };

  const mobileNavItem = (href: string, icon: React.ReactNode, label: string) => {
    const active = pathname === href || (href !== "/" && pathname?.startsWith(href));
    return (
      <Link
        href={href}
        className={`flex flex-col items-center gap-1 px-3 py-2 text-[10px] font-semibold uppercase tracking-widest transition-colors ${
          active ? "text-cyan-400" : "text-[#6b7a99]"
        }`}
      >
        <span className={active ? "drop-shadow-[0_0_6px_rgba(6,182,212,0.8)]" : ""}>{icon}</span>
        {label}
      </Link>
    );
  };

  return (
    <main className="min-h-screen bg-[#04040a] pb-20 sm:pb-0">
      {/* ── Header ──────────────────────────────────────────── */}
      <header className="sticky top-0 z-20 border-b border-white/[0.06] bg-[#04040a]/88 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-3.5">

          {/* Logo */}
          <Link href="/" className="flex items-center gap-2.5">
            <span className="relative grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 text-[#04121f] shadow-[0_0_22px_rgba(6,182,212,0.5)]">
              <Gauge size={18} strokeWidth={2.5} />
            </span>
            <span className="font-display text-base font-bold tracking-tight text-[#eef2ff]">
              Game<span className="gradient-text">Sense</span>
            </span>
          </Link>

          {/* Desktop nav */}
          <nav className="hidden items-center gap-1 rounded-xl border border-white/[0.07] bg-[#09090f]/80 p-1.5 backdrop-blur-md sm:flex">
            {navItem("/", <Upload size={14} />, "Upload")}
            {navItem("/leaderboard", <Trophy size={14} />, "Leaderboard")}
            {navItem("/leagues", <Users size={14} />, "Leagues")}
            {navItem("/history", <Clock size={14} />, "History")}
            {navItem("/profile", <Medal size={14} />, "Profile")}
          </nav>

          {/* Auth */}
          <div className="flex items-center gap-2 text-sm">
            {user ? (
              <>
                <Link
                  href="/profile"
                  className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-[#94a3b8] transition-colors hover:text-[#eef2ff]"
                >
                  <Avatar name={user.display_name} url={user.avatar_url} size={27} />
                  <span className="hidden font-medium sm:inline">{user.display_name}</span>
                </Link>
                <button
                  type="button"
                  onClick={() => {
                    clearAuth();
                    setUser(null);
                  }}
                  className="flex items-center gap-1.5 rounded-lg border border-white/[0.07] bg-[#09090f] px-3 py-1.5 text-[#6b7a99] transition-all hover:border-white/[0.13] hover:text-[#eef2ff]"
                >
                  <LogOut size={14} />
                  <span className="hidden sm:inline">Sign out</span>
                </button>
              </>
            ) : (
              <Link
                href="/login"
                className="flex items-center gap-1.5 rounded-lg bg-cyan-500 px-3.5 py-1.5 text-sm font-bold text-[#04121f] shadow-[0_0_18px_rgba(6,182,212,0.4)] transition-all hover:bg-cyan-400 hover:shadow-[0_0_28px_rgba(6,182,212,0.55)]"
              >
                <LogIn size={14} />
                Sign in
              </Link>
            )}
          </div>
        </div>
      </header>

      {/* ── Content ─────────────────────────────────────────── */}
      {children}

      {/* ── Mobile bottom dock ──────────────────────────────── */}
      <nav className="fixed bottom-0 left-0 right-0 z-20 flex items-center justify-around border-t border-white/[0.06] bg-[#04040a]/95 px-2 py-1 backdrop-blur-xl sm:hidden">
        {mobileNavItem("/", <Upload size={20} />, "Upload")}
        {mobileNavItem("/leaderboard", <Trophy size={20} />, "Board")}
        {mobileNavItem("/leagues", <Users size={20} />, "Leagues")}
        {mobileNavItem("/history", <Clock size={20} />, "History")}
        {mobileNavItem("/profile", <Medal size={20} />, "Profile")}
      </nav>
    </main>
  );
}
