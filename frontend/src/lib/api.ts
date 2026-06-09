export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export type Point = { x: number; y: number };
export type BoundingBox = { x: number; y: number; width: number; height: number };

export type Detection = {
  id: string;
  label: "player" | "ball" | string;
  confidence: number;
  bbox: BoundingBox;
};

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

export type AnalysisMode = "max_speed" | "max_shot_power";

export type Metrics = {
  player_id: number;
  top_speed_kmh: number;
  avg_speed_kmh: number;
  peak_acceleration_mps2: number;
  avg_acceleration_mps2: number;
  total_distance_m: number;
  active_distance_m: number;
  sprint_count: number;
  sprint_distance_m: number;
  usable_track_points: number;
  rejected_jump_count: number;
  confidence_score: number;
};

export type ShotEvent = {
  frame_id: number;
  timestamp_s: number;
  ball_speed_kmh: number;
  contact_point: Point;
};

export type ShotMetrics = {
  player_id: number;
  peak_shot_speed_kmh: number;
  avg_shot_speed_kmh: number;
  shot_count: number;
  best_shot: ShotEvent | null;
  shots: ShotEvent[];
  confidence_score: number;
  usable_track_points: number;
  rejected_track_points: number;
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
  pitch_setup?: Record<string, unknown> | null;
  results?: Metrics | ShotMetrics | null;
  assets?: {
    sprint_highlights: string[];
  } | null;
  warnings: string[];
  progress?: {
    stage: string;
    percent: number;
    message: string;
  } | null;
};

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function uploadVideo(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return parseResponse<UploadResponse>(
    await fetch(`${API_BASE}/upload-video`, {
      method: "POST",
      body: formData,
    }),
  );
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
      }),
    }),
  );
}

export async function processVideo(videoId: string, mode: AnalysisMode = "max_speed"): Promise<void> {
  await parseResponse(
    await fetch(`${API_BASE}/process-video`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: videoId, mode }),
    }),
  );
}

export async function getResults(videoId: string): Promise<VideoResult> {
  return parseResponse<VideoResult>(await fetch(`${API_BASE}/results/${videoId}`, { cache: "no-store" }));
}

export function mediaUrl(path?: string | null): string | null {
  if (!path) return null;
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}
