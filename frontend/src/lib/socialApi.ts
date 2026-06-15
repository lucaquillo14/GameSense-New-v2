import { API_BASE } from "./api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export type User = {
  id: string;
  email: string;
  display_name: string;
  avatar_url?: string | null;
  created_at?: string;
};

export type AuthResponse = { token: string; user: User };

export type LeaderboardEntry = {
  rank: number;
  user_id: string;
  display_name: string;
  avatar_url?: string | null;
  uploads: number;
  total_points: number;
  best_speed_kmh: number;
  best_power_kmh: number;
  best_technique: number;
};

export type League = {
  id: string;
  name: string;
  owner_id: string;
  invite_code: string;
  created_at: string;
  member_count: number;
};

export type LeaguePreview = {
  id: string;
  name: string;
  invite_code: string;
  member_count: number;
};

export type PublicUser = {
  id: string;
  display_name: string;
  avatar_url?: string | null;
  is_following: boolean;
  is_self: boolean;
};

export type FollowCounts = { following: number; followers: number };

export type SortKey = "points" | "speed" | "power" | "technique" | "uploads";
export type LeaderboardScope = "global" | "following";

export type HistoryItem = {
  video_id: string;
  mode: string | null;
  filename: string | null;
  max_speed_kmh: number;
  shot_power_kmh: number;
  technique_score: number;
  points: number;
  created_at: string;
};

export type CareerStats = {
  uploads: number;
  total_points: number;
  best_speed_kmh: number;
  best_power_kmh: number;
  best_technique: number;
};

export type Badge = {
  id: string;
  name: string;
  description: string;
  category: string;
  tier: "bronze" | "silver" | "gold";
  icon: string;
  unit: string;
  target: number;
  current: number;
  earned: boolean;
  progress: number;
};

export type Profile = {
  user: User;
  stats: CareerStats;
  badges: Badge[];
  earned_count: number;
  total_count: number;
  recent_sessions: HistoryItem[];
  follow_counts: FollowCounts;
};

// ---------------------------------------------------------------------------
// Token storage (localStorage) + small auth event bus
// ---------------------------------------------------------------------------
const TOKEN_KEY = "gamesense_token";
const USER_KEY = "gamesense_user";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function getStoredUser(): User | null {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as User;
  } catch {
    return null;
  }
}

export function setAuth(token: string, user: User): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(TOKEN_KEY, token);
  window.localStorage.setItem(USER_KEY, JSON.stringify(user));
  window.dispatchEvent(new Event("gamesense-auth"));
}

export function clearAuth(): void {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(TOKEN_KEY);
  window.localStorage.removeItem(USER_KEY);
  window.dispatchEvent(new Event("gamesense-auth"));
}

export function updateStoredUser(patch: Partial<User>): void {
  if (typeof window === "undefined") return;
  const current = getStoredUser();
  if (!current) return;
  window.localStorage.setItem(USER_KEY, JSON.stringify({ ...current, ...patch }));
  window.dispatchEvent(new Event("gamesense-auth"));
}

export function avatarSrc(url?: string | null): string | null {
  if (!url) return null;
  return url.startsWith("http") ? url : `${API_BASE}${url}`;
}

export function authHeader(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

// ---------------------------------------------------------------------------
// Fetch helper
// ---------------------------------------------------------------------------
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      cache: "no-store",
      headers: {
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...authHeader(),
        ...(init?.headers ?? {}),
      },
    });
  } catch {
    throw new Error("Could not reach the server. Is the backend running on port 8000?");
  }
  if (!response.ok) {
    let message = `Request failed (HTTP ${response.status}).`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      const detail = body.detail;
      if (typeof detail === "string") {
        message = detail;
      } else if (Array.isArray(detail)) {
        // FastAPI 422 validation errors -> [{ loc, msg, type }, ...]
        message = detail
          .map((e) =>
            typeof e === "object" && e && "msg" in e
              ? String((e as { msg: unknown }).msg)
              : String(e),
          )
          .join("; ");
      }
    } catch {
      /* response had no JSON body */
    }
    throw new Error(message);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------
export async function signup(email: string, displayName: string, password: string): Promise<AuthResponse> {
  const res = await request<AuthResponse>("/auth/signup", {
    method: "POST",
    body: JSON.stringify({ email, display_name: displayName, password }),
  });
  setAuth(res.token, res.user);
  return res;
}

export async function login(email: string, password: string): Promise<AuthResponse> {
  const res = await request<AuthResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  setAuth(res.token, res.user);
  return res;
}

// ---------------------------------------------------------------------------
// Session history
// ---------------------------------------------------------------------------
export async function getHistory(): Promise<HistoryItem[]> {
  const res = await request<{ uploads: HistoryItem[] }>("/history");
  return res.uploads;
}

// ---------------------------------------------------------------------------
// Profile (stats + badges)
// ---------------------------------------------------------------------------
export async function getProfile(): Promise<Profile> {
  return request<Profile>("/profile");
}

// ---------------------------------------------------------------------------
// Leaderboard
// ---------------------------------------------------------------------------
export async function getLeaderboard(
  sort: SortKey = "points",
  scope: LeaderboardScope = "global",
): Promise<LeaderboardEntry[]> {
  const res = await request<{ entries: LeaderboardEntry[] }>(`/leaderboard?sort=${sort}&scope=${scope}`);
  return res.entries;
}

// ---------------------------------------------------------------------------
// Leagues
// ---------------------------------------------------------------------------
export async function getMyLeagues(): Promise<League[]> {
  const res = await request<{ leagues: League[] }>("/leagues");
  return res.leagues;
}

export async function createLeague(name: string): Promise<League> {
  return request<League>("/leagues", { method: "POST", body: JSON.stringify({ name }) });
}

export async function joinLeague(inviteCode: string): Promise<League> {
  return request<League>("/leagues/join", {
    method: "POST",
    body: JSON.stringify({ invite_code: inviteCode }),
  });
}

export async function getLeagueDetail(
  leagueId: string,
  sort: SortKey = "points",
): Promise<{ league: League; entries: LeaderboardEntry[] }> {
  return request<{ league: League; entries: LeaderboardEntry[] }>(`/leagues/${leagueId}?sort=${sort}`);
}

export async function leaveLeague(leagueId: string): Promise<void> {
  await request(`/leagues/${leagueId}/leave`, { method: "POST" });
}

export async function getLeaguePreview(code: string): Promise<LeaguePreview> {
  return request<LeaguePreview>(`/leagues/preview?code=${encodeURIComponent(code)}`);
}

// ---------------------------------------------------------------------------
// Following
// ---------------------------------------------------------------------------
export async function searchUsers(query: string): Promise<PublicUser[]> {
  const res = await request<{ users: PublicUser[] }>(`/users/search?q=${encodeURIComponent(query)}`);
  return res.users;
}

export async function followUser(userId: string): Promise<void> {
  await request(`/follow/${userId}`, { method: "POST" });
}

export async function unfollowUser(userId: string): Promise<void> {
  await request(`/unfollow/${userId}`, { method: "POST" });
}

export async function getFollowing(): Promise<{
  following: PublicUser[];
  followers: PublicUser[];
  counts: FollowCounts;
}> {
  return request("/following");
}

// ---------------------------------------------------------------------------
// Profile picture
// ---------------------------------------------------------------------------
export async function uploadAvatar(file: File): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/profile/avatar`, {
      method: "POST",
      body: form,
      headers: authHeader(), // do NOT set Content-Type; the browser sets the multipart boundary
    });
  } catch {
    throw new Error("Could not reach the server.");
  }
  if (!response.ok) {
    let message = `Upload failed (HTTP ${response.status}).`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      /* ignore */
    }
    throw new Error(message);
  }
  const data = (await response.json()) as { avatar_url: string };
  updateStoredUser({ avatar_url: data.avatar_url });
  return data.avatar_url;
}
