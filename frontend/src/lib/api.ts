export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export type Point = { x: number; y: number };
export type BoundingBox = { x: number; y: number; width: number; height: number };

export type TeamColor = { r: number; g: number; b: number };

export type Detection = {
  id: string;
  label: "player" | "ball" | "team_a" | "team_b" | string;
  confidence: number;
  bbox: BoundingBox;
  team?: "team_a" | "team_b";
  team_label?: "Team A" | "Team B";
  team_color?: TeamColor;
  player_id?: string;
};

export type TeamClassificationInfo = {
  team_a: { histogram: number[]; display_color: TeamColor };
  team_b: { histogram: number[]; display_color: TeamColor };
  calibration_frames: number;
  referee_distance_threshold: number;
};

export type SpeedSeriesPoint = { time_s: number; speed_kmh: number };

export type UploadResponse = {
  video_id: string;
  filename: string;
  setup_frame_url: string;
  source_url: string;
  metadata: VideoMetadata;
};

export type VideoMetadata = {
  fps: number;
  frame_count: number;
  duration_s: number;
  width: number;
  height: number;
};

export type AnalysisMode = "max_speed" | "max_shot_power" | "shooting_technique";

export type BodyAngle = {
  name: string;
  value_deg: number;
  frame_id: number;
  time_s: number;
};

export type TechniqueFrame = {
  frame_id: number;
  time_s: number;
  angles: BodyAngle[];
  ball_visible: boolean;
  foot_to_ball_px: number | null;
  phase: string;
};

export type ShootingFeedback = {
  shot_power_kmh: number;
  technique_score: number;
  approach_angle_deg: number;
  plant_foot_distance_cm: number;
  knee_bend_at_contact_deg: number;
  hip_rotation_deg: number;
  follow_through_height: string;
  feedback_points: string[];
  annotated_video_url: string | null;
  contact_frame_url?: string | null;
  frame_analysis: TechniqueFrame[];
  confidence: number;
  contact_frame_id?: number | null;
  backswing_knee_flexion_deg?: number;
  ankle_lock_variation_deg?: number;
  follow_through_height_ratio?: number;
  power_rating?: string;
  kicking_foot?: string;
  scale_source?: string;
  shot_distance_m?: number;
  on_target?: boolean | null;
  goal_crossing_height_m?: number;
  goal_crossing_offset_m?: number;
};

export type Metrics = {
  player_id: number;
  player_label?: string;
  team_label?: string;
  units?: "metric" | "pixels";
  speed_series?: SpeedSeriesPoint[];
  max_speed_kmh?: number;
  top_speed_kmh: number;
  avg_speed_kmh: number;
  distance_m?: number;
  tracked_frames?: number;
  predicted_frames?: number;
  lost_frames?: number;
  top_speed_px_per_s?: number;
  avg_speed_px_per_s?: number;
  peak_acceleration_mps2?: number;
  avg_acceleration_mps2?: number;
  total_distance_m: number;
  active_distance_m?: number;
  total_distance_px?: number;
  active_distance_px?: number;
  sprint_count?: number;
  sprint_distance_m?: number;
  sprint_distance_px?: number;
  calibrated_point_ratio?: number;
  usable_track_points: number;
  rejected_jump_count?: number;
  confidence_score: number;
  touch_count?: number;
  pass_count?: number;
};

export type ShotEvent = {
  frame_id: number;
  timestamp_s: number;
  ball_speed_kmh: number;
  contact_point: Point;
};

export type ShotMetrics = {
  player_id: number;
  player_label?: string;
  team_label?: string;
  peak_shot_speed_kmh: number;
  avg_shot_speed_kmh: number;
  shot_count: number;
  best_shot: ShotEvent | null;
  shots: ShotEvent[];
  confidence_score: number;
  usable_track_points: number;
  rejected_track_points: number;
  touch_count?: number;
  pass_count?: number;
};

export type VideoResult = {
  video_id: string;
  filename?: string;
  status: "uploaded" | "player_selected" | "setup_complete" | "processing" | "complete" | "failed";
  mode?: AnalysisMode;
  setup_frame?: string;
  setup_frame_id?: number | null;
  source_url?: string | null;
  video_metadata?: VideoMetadata | null;
  target_player?: Record<string, unknown> | null;
  team_classification?: TeamClassificationInfo | null;
  pitch_setup?: Record<string, unknown> | null;
  results?: Metrics | ShotMetrics | null;
  shooting_result?: ShootingFeedback | null;
  assets?: {
    sprint_highlights: string[];
    detections_json?: string;
    position_heatmap?: string;
    speed_heatmap?: string;
    movement_heatmap?: string;
    touch_heatmap?: string;
  } | null;
  warnings: string[];
  locked_features?: string[];
  progress?: {
    stage: string;
    percent: number;
    message: string;
    tracked_so_far?: number;
    predicted_so_far?: number;
    lost_so_far?: number;
  } | null;
};

function friendlyError(detail: unknown, status: number): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return "The request was invalid. Check your inputs and try again.";
  if (status === 404) return "That video could not be found. Try uploading again.";
  if (status >= 500) return "The server ran into a problem. Please try again in a moment.";
  return `Something went wrong (${status}). Please try again.`;
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(friendlyError(payload?.detail, response.status));
  }
  return response.json() as Promise<T>;
}

export function teamColorCss(color?: TeamColor | null): string {
  if (!color) return "#64748b";
  return `rgb(${color.r}, ${color.g}, ${color.b})`;
}

export function isPlayerDetection(detection: Detection): boolean {
  return Boolean(
    detection.team ||
      detection.label === "team_a" ||
      detection.label === "team_b" ||
      detection.label === "player",
  );
}

export async function uploadVideo(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  let response: Response;
  try {
    const token = typeof window !== "undefined" ? window.localStorage.getItem("gamesense_token") : null;
    response = await fetch(`${API_BASE}/upload-video`, {
      method: "POST",
      body: formData,
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    });
  } catch (error) {
    console.error("Network error — backend may not be running", error);
    throw new Error("Could not reach the server. Is the backend running on port 8000?");
  }

  if (!response.ok) {
    const text = await response.text();
    let message = "Upload failed.";
    try {
      const body = JSON.parse(text) as { detail?: unknown; message?: string };
      const detail = body.detail ?? body.message;
      if (typeof detail === "string") {
        message = detail;
      } else if (Array.isArray(detail)) {
        message = detail.map((entry) => (typeof entry === "object" && entry && "msg" in entry ? String(entry.msg) : String(entry))).join("; ");
      } else if (detail) {
        message = String(detail);
      } else {
        message = text || `HTTP ${response.status}`;
      }
    } catch {
      message = text || `HTTP ${response.status}`;
    }
    console.error("Upload error:", response.status, text);
    throw new Error(message);
  }

  return response.json() as Promise<UploadResponse>;
}

export async function getFrame(videoId: string, frameId: number): Promise<{ frame_url: string; frame_id: number; detections: Detection[] }> {
  return parseResponse<{ frame_url: string; frame_id: number; detections: Detection[] }>(
    await fetch(`${API_BASE}/frames/${videoId}/${frameId}`, { cache: "no-store" }),
  );
}

export async function getFrameDetections(videoId: string, frameId: number): Promise<{ frame_id: number; detections: Detection[] }> {
  return parseResponse<{ frame_id: number; detections: Detection[] }>(
    await fetch(`${API_BASE}/frames/${videoId}/${frameId}/detections`, { cache: "no-store" }),
  );
}

export async function selectPlayer(
  videoId: string,
  click: Point,
  frameId: number,
  detection?: Detection | null,
): Promise<void> {
  await parseResponse(
    await fetch(`${API_BASE}/select-player`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_id: videoId,
        click,
        frame_id: frameId,
        detection_id: detection?.id,
        bbox: detection?.bbox,
      }),
    }),
  );
}

export async function setPitchPolygon(
  videoId: string,
  pitchPolygon: Point[],
  frameId: number,
  goalLeft?: Point | null,
  goalRight?: Point | null,
  goalPosts?: Point[] | null,
): Promise<void> {
  await parseResponse(
    await fetch(`${API_BASE}/set-pitch-polygon`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_id: videoId,
        pitch_polygon: pitchPolygon,
        frame_id: frameId,
        goal_left: goalLeft,
        goal_right: goalRight,
        goal_posts: goalPosts && goalPosts.length === 4 ? { points: goalPosts } : null,
      }),
    }),
  );
}

export async function processVideo(
  videoId: string,
  mode: AnalysisMode = "max_speed",
  playerHeightCm?: number | null,
): Promise<void> {
  const token = typeof window !== "undefined" ? window.localStorage.getItem("gamesense_token") : null;
  await parseResponse(
    await fetch(`${API_BASE}/process-video`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({
        video_id: videoId,
        mode,
        player_height_cm: playerHeightCm ?? null,
      }),
    }),
  );
}

export async function getResults(videoId: string): Promise<VideoResult> {
  return parseResponse<VideoResult>(await fetch(`${API_BASE}/results/${videoId}`, { cache: "no-store" }));
}

export async function getShootingResult(videoId: string): Promise<ShootingFeedback> {
  return parseResponse<ShootingFeedback>(
    await fetch(`${API_BASE}/shooting-result/${videoId}`, { cache: "no-store" }),
  );
}

export async function getPreviewFrame(videoId: string): Promise<string> {
  const response = await fetch(`${API_BASE}/preview/${videoId}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Preview not ready");
  }
  const blob = await response.blob();
  return URL.createObjectURL(blob);
}

export async function getDetectionsOverlay(path: string): Promise<import("@/lib/overlay").DetectionsOverlay> {
  const url = mediaUrl(path);
  if (!url) {
    return { fps: 30, interval: 1, frames: {} };
  }
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    return { fps: 30, interval: 1, frames: {} };
  }
  const payload = (await response.json()) as import("@/lib/overlay").DetectionsOverlay;
  return {
    fps: payload.fps ?? 30,
    interval: payload.interval ?? 1,
    target_id: payload.target_id,
    frames: payload.frames ?? {},
  };
}

export function mediaUrl(path?: string | null): string | null {
  if (!path) return null;
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}

/** True when annotated output is a still frame (Roboflow workflow) rather than a video. */
export function isImageMediaUrl(path?: string | null): boolean {
  if (!path) return false;
  const normalized = path.split("?")[0]?.toLowerCase() ?? "";
  return normalized.endsWith(".jpg") || normalized.endsWith(".jpeg") || normalized.endsWith(".png") || normalized.endsWith(".webp");
}

export function contactTechniqueFrame(feedback: ShootingFeedback): TechniqueFrame | undefined {
  if (feedback.contact_frame_id != null) {
    const match = feedback.frame_analysis.find((frame) => frame.frame_id === feedback.contact_frame_id);
    if (match) return match;
  }
  return feedback.frame_analysis.find((frame) => frame.phase.toLowerCase().includes("contact"));
}

export function techniqueAngleDeg(
  feedback: ShootingFeedback,
  ...names: string[]
): number | null {
  const frame = contactTechniqueFrame(feedback);
  if (!frame) return null;
  for (const name of names) {
    const angle = frame.angles.find((entry) => entry.name === name);
    if (angle != null) return angle.value_deg;
  }
  return null;
}
