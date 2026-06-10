export type OverlayDetection = {
  id: string;
  team: string;
  c: number;
  b: [number, number, number, number];
  color?: { r: number; g: number; b: number };
};

export type DetectionsOverlay = {
  fps: number;
  interval: number;
  target_id?: string | null;
  frames: Record<string, OverlayDetection[]>;
};

export type VideoRenderRect = {
  width: number;
  height: number;
  offsetX: number;
  offsetY: number;
};

export function getVideoRenderRect(video: HTMLVideoElement): VideoRenderRect {
  const elementWidth = video.clientWidth;
  const elementHeight = video.clientHeight;
  const videoWidth = video.videoWidth || elementWidth;
  const videoHeight = video.videoHeight || elementHeight;
  if (!elementWidth || !elementHeight || !videoWidth || !videoHeight) {
    return { width: elementWidth, height: elementHeight, offsetX: 0, offsetY: 0 };
  }

  const elementRatio = elementWidth / elementHeight;
  const videoRatio = videoWidth / videoHeight;
  if (videoRatio > elementRatio) {
    const width = elementWidth;
    const height = elementWidth / videoRatio;
    return { width, height, offsetX: 0, offsetY: (elementHeight - height) / 2 };
  }

  const height = elementHeight;
  const width = elementHeight * videoRatio;
  return { width, height, offsetX: (elementWidth - width) / 2, offsetY: 0 };
}

export function frameIdFromTime(currentTime: number, fps: number): number {
  return Math.max(0, Math.round(currentTime * fps));
}

export function lookupOverlayFrame(
  frames: Record<string, OverlayDetection[]>,
  frameId: number,
  sampleInterval = 1,
): OverlayDetection[] | null {
  const direct = frames[String(frameId)];
  if (direct) return direct;

  const tolerance = Math.max(1, Math.ceil(sampleInterval / 2) + 1);
  let best: OverlayDetection[] | null = null;
  let bestDistance = Infinity;
  for (const [key, detections] of Object.entries(frames)) {
    const distance = Math.abs(Number(key) - frameId);
    if (distance <= tolerance && distance < bestDistance) {
      bestDistance = distance;
      best = detections;
    }
  }
  return best;
}

export function teamColorCss(color?: { r: number; g: number; b: number }): string {
  if (!color) return "#94a3b8";
  return `rgb(${color.r}, ${color.g}, ${color.b})`;
}

export type SetupDetection = {
  id: string;
  label: string;
  team_label?: string;
  player_id?: string;
  team_color?: { r: number; g: number; b: number };
  bbox: { x: number; y: number; width: number; height: number };
};

export function drawSetupDetections(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
  detections: SetupDetection[],
  frameWidth: number,
  frameHeight: number,
  selectedId?: string | null,
) {
  const normalized: OverlayDetection[] = detections.map((detection) => ({
    id: detection.player_id ?? detection.team_label ?? detection.label ?? detection.id,
    team: detection.team_label ?? detection.label,
    c: 1,
    b: [
      detection.bbox.x / frameWidth,
      detection.bbox.y / frameHeight,
      detection.bbox.width / frameWidth,
      detection.bbox.height / frameHeight,
    ],
    color: detection.team_color,
  }));
  drawOverlayDetections(canvas, video, normalized, selectedId ? detections.find((d) => d.id === selectedId)?.player_id ?? selectedId : null);
}

export function drawOverlayDetections(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
  detections: OverlayDetection[] | null,
  targetId?: string | null,
) {
  const context = canvas.getContext("2d");
  if (!context) return;

  const dpr = window.devicePixelRatio || 1;
  const displayWidth = video.clientWidth;
  const displayHeight = video.clientHeight;
  canvas.width = Math.max(1, Math.floor(displayWidth * dpr));
  canvas.height = Math.max(1, Math.floor(displayHeight * dpr));
  canvas.style.width = `${displayWidth}px`;
  canvas.style.height = `${displayHeight}px`;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, displayWidth, displayHeight);

  if (!detections?.length) return;

  const rect = getVideoRenderRect(video);
  for (const detection of detections) {
    const [nx, ny, nw, nh] = detection.b;
    const x = rect.offsetX + nx * rect.width;
    const y = rect.offsetY + ny * rect.height;
    const width = nw * rect.width;
    const height = nh * rect.height;
    const isTarget = Boolean(targetId && detection.id === targetId);
    const color = isTarget ? "#3b82f6" : teamColorCss(detection.color);

    context.lineWidth = isTarget ? 3 : 2;
    context.strokeStyle = color;
    context.strokeRect(x, y, width, height);

    const label = detection.id || detection.team;
    context.font = "bold 11px Inter, system-ui, sans-serif";
    const textWidth = context.measureText(label).width + 8;
    const labelHeight = 16;
    context.fillStyle = color;
    context.fillRect(x, Math.max(y - labelHeight, 0), textWidth, labelHeight);
    context.fillStyle = "#ffffff";
    context.fillText(label, x + 4, Math.max(y - 4, 12));
  }
}
