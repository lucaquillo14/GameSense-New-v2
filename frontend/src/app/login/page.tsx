"use client";

import { AppShell } from "@/components/AppShell";
import { login, signup } from "@/lib/socialApi";
import { AlertCircle, Loader2, LogIn, UserPlus } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === "signup") {
        await signup(email, displayName, password);
      } else {
        await login(email, password);
      }
      const next =
        typeof window !== "undefined"
          ? new URLSearchParams(window.location.search).get("next")
          : null;
      router.push(next && next.startsWith("/") ? next : "/leaderboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <AppShell>
      <section className="analytics-grid hero-glow mx-auto max-w-md px-5 py-16 fade-in">
        {/* Header */}
        <div className="mb-8 text-center">
          <div className="chip mx-auto mb-5 w-fit">GameSense AI</div>
          <h1 className="display text-4xl text-[#eef2ff]">
            {mode === "login" ? "Welcome back" : "Create account"}
          </h1>
          <p className="mt-3 text-sm text-[#6b7a99]">
            {mode === "login"
              ? "Sign in to climb the leaderboard and join leagues."
              : "Sign up to track your stats and compete with friends."}
          </p>
        </div>

        <div className="card p-6">
          {/* Mode toggle */}
          <div className="mb-6 flex rounded-xl border border-white/[0.07] bg-[#04040a] p-1 text-sm">
            <button
              type="button"
              onClick={() => setMode("login")}
              className={`flex-1 rounded-lg py-2.5 font-semibold transition-all ${
                mode === "login"
                  ? "bg-cyan-500 text-[#04121f] shadow-[0_0_16px_rgba(6,182,212,0.4)]"
                  : "text-[#6b7a99] hover:text-[#eef2ff]"
              }`}
            >
              Sign in
            </button>
            <button
              type="button"
              onClick={() => setMode("signup")}
              className={`flex-1 rounded-lg py-2.5 font-semibold transition-all ${
                mode === "signup"
                  ? "bg-cyan-500 text-[#04121f] shadow-[0_0_16px_rgba(6,182,212,0.4)]"
                  : "text-[#6b7a99] hover:text-[#eef2ff]"
              }`}
            >
              Sign up
            </button>
          </div>

          <form onSubmit={submit} className="space-y-4">
            {mode === "signup" && (
              <Field
                label="Display name"
                value={displayName}
                onChange={setDisplayName}
                placeholder="How others see you"
                type="text"
              />
            )}
            <Field
              label="Email"
              value={email}
              onChange={setEmail}
              placeholder="you@example.com"
              type="email"
            />
            <Field
              label="Password"
              value={password}
              onChange={setPassword}
              placeholder="At least 6 characters"
              type="password"
            />

            {error && (
              <div className="flex items-center gap-2 rounded-xl border border-red-500/25 bg-red-500/8 px-4 py-3 text-sm text-red-300">
                <AlertCircle size={16} className="shrink-0" />
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={busy}
              className="btn-primary mt-2 flex w-full items-center justify-center gap-2 px-4 py-3.5 text-sm disabled:cursor-not-allowed disabled:opacity-60"
            >
              {busy ? (
                <Loader2 size={18} className="animate-spin" />
              ) : mode === "login" ? (
                <LogIn size={18} />
              ) : (
                <UserPlus size={18} />
              )}
              {mode === "login" ? "Sign in" : "Create account"}
            </button>
          </form>
        </div>
      </section>
    </AppShell>
  );
}

function Field({
  label,
  value,
  onChange,
  placeholder,
  type,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
  type: string;
}) {
  return (
    <label className="block">
      <span className="data-label mb-1.5 block">{label}</span>
      <input
        type={type}
        value={value}
        required
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-xl border border-white/[0.07] bg-[#09090f] px-4 py-2.5 text-[#eef2ff] placeholder:text-[#3a4560] transition-colors focus:border-cyan-500/50 focus:outline-none focus:ring-1 focus:ring-cyan-500/20"
      />
    </label>
  );
}
