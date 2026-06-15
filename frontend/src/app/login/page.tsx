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
        typeof window !== "undefined" ? new URLSearchParams(window.location.search).get("next") : null;
      router.push(next && next.startsWith("/") ? next : "/leaderboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <AppShell>
      <section className="mx-auto max-w-md px-5 py-16 fade-in">
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-semibold tracking-tight text-[#f1f5f9]">
            {mode === "login" ? "Welcome back" : "Create your account"}
          </h1>
          <p className="mt-2 text-sm text-[#64748b]">
            {mode === "login"
              ? "Sign in to climb the leaderboard and join leagues."
              : "Sign up to track your stats and compete with friends."}
          </p>
        </div>

        <form onSubmit={submit} className="card space-y-4 p-6">
          <div className="flex rounded-lg border border-[#ffffff14] bg-[#0a0a0f] p-1 text-sm">
            <button
              type="button"
              onClick={() => setMode("login")}
              className={`flex-1 rounded-md py-2 font-medium transition-colors ${
                mode === "login" ? "bg-[#3b82f6] text-white" : "text-[#64748b]"
              }`}
            >
              Sign in
            </button>
            <button
              type="button"
              onClick={() => setMode("signup")}
              className={`flex-1 rounded-md py-2 font-medium transition-colors ${
                mode === "signup" ? "bg-[#3b82f6] text-white" : "text-[#64748b]"
              }`}
            >
              Sign up
            </button>
          </div>

          {mode === "signup" && (
            <Field
              label="Display name"
              value={displayName}
              onChange={setDisplayName}
              placeholder="How others see you"
              type="text"
            />
          )}
          <Field label="Email" value={email} onChange={setEmail} placeholder="you@example.com" type="email" />
          <Field
            label="Password"
            value={password}
            onChange={setPassword}
            placeholder="At least 6 characters"
            type="password"
          />

          {error && (
            <div className="flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">
              <AlertCircle size={16} />
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={busy}
            className="btn-primary flex w-full items-center justify-center gap-2 px-4 py-3 disabled:cursor-not-allowed disabled:opacity-60"
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
      <span className="mb-1.5 block text-sm font-medium text-[#94a3b8]">{label}</span>
      <input
        type={type}
        value={value}
        required
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border border-[#ffffff14] bg-[#0a0a0f] px-3 py-2.5 text-[#f1f5f9] placeholder:text-[#475569] focus:border-[#3b82f6] focus:outline-none"
      />
    </label>
  );
}
