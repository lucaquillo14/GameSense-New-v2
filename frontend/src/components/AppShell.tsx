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
        className={`flex items-center gap-2 rounded-md px-3 py-1.5 transition-colors hover:bg-[#ffffff08] ${
          active ? "text-[#f1f5f9]" : "text-[#64748b]"
        }`}
      >
        {icon}
        {label}
      </Link>
    );
  };

  return (
    <main className="min-h-screen bg-[#08080c]">
      <header className="sticky top-0 z-20 border-b border-[#ffffff12] bg-[#08080c]/80 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4">
          <Link href="/" className="flex items-center gap-2.5 font-semibold">
            <span className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-[#3b82f6] to-[#22d3ee] text-[#04121f] shadow-[0_0_28px_rgba(34,211,238,0.35)]">
              <Gauge size={20} />
            </span>
            <span className="text-[#f8fafc]">
              Game<span className="gradient-text">Sense</span>
            </span>
          </Link>

          <nav className="hidden items-center gap-1 rounded-lg border border-[#ffffff14] bg-[#111118] p-1 text-sm sm:flex">
            {navItem("/", <Upload size={15} />, "Upload")}
            {navItem("/leaderboard", <Trophy size={15} />, "Leaderboard")}
            {navItem("/leagues", <Users size={15} />, "Leagues")}
            {navItem("/history", <Clock size={15} />, "History")}
            {navItem("/profile", <Medal size={15} />, "Profile")}
          </nav>

          <div className="flex items-center gap-2 text-sm">
            {user ? (
              <>
                <Link href="/profile" className="flex items-center gap-2 text-[#94a3b8] hover:text-[#f1f5f9]">
                  <Avatar name={user.display_name} url={user.avatar_url} size={28} />
                  <span className="hidden sm:inline">{user.display_name}</span>
                </Link>
                <button
                  type="button"
                  onClick={() => {
                    clearAuth();
                    setUser(null);
                  }}
                  className="flex items-center gap-1.5 rounded-md border border-[#ffffff14] bg-[#111118] px-3 py-1.5 text-[#94a3b8] transition-colors hover:text-[#f1f5f9]"
                >
                  <LogOut size={15} />
                  <span className="hidden sm:inline">Sign out</span>
                </button>
              </>
            ) : (
              <Link
                href="/login"
                className="flex items-center gap-1.5 rounded-md bg-[#3b82f6] px-3 py-1.5 font-medium text-white transition-colors hover:bg-[#2563eb]"
              >
                <LogIn size={15} />
                Sign in
              </Link>
            )}
          </div>
        </div>

        <nav className="flex items-center justify-center gap-1 border-t border-[#ffffff14] px-5 py-2 text-sm sm:hidden">
          {navItem("/", <Upload size={15} />, "Upload")}
          {navItem("/leaderboard", <Trophy size={15} />, "Board")}
          {navItem("/leagues", <Users size={15} />, "Leagues")}
          {navItem("/history", <Clock size={15} />, "History")}
          {navItem("/profile", <Medal size={15} />, "Profile")}
        </nav>
      </header>
      {children}
    </main>
  );
}
