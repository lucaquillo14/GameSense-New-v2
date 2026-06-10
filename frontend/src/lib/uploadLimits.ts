export const MAX_UPLOAD_BYTES = 250 * 1024 * 1024;
export const MAX_UPLOAD_MB = 250;
export const MAX_VIDEO_DURATION_S = 60;

export type LocalVideoMeta = {
  name: string;
  sizeMb: string;
  durationS: number;
  width: number;
  height: number;
};

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "0s";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}m ${secs}s`;
}

export function validateFileSize(file: File): string | null {
  if (file.size > MAX_UPLOAD_BYTES) {
    return `This file is ${(file.size / (1024 * 1024)).toFixed(1)} MB. Maximum upload size is ${MAX_UPLOAD_MB} MB.`;
  }
  return null;
}

export function readLocalVideoMeta(file: File): Promise<LocalVideoMeta> {
  return new Promise((resolve, reject) => {
    const objectUrl = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.preload = "metadata";
    video.muted = true;
    video.playsInline = true;

    const cleanup = () => {
      video.removeAttribute("src");
      video.load();
      URL.revokeObjectURL(objectUrl);
    };

    video.onloadedmetadata = () => {
      const durationS = video.duration;
      const width = video.videoWidth;
      const height = video.videoHeight;
      cleanup();
      if (!Number.isFinite(durationS) || durationS <= 0) {
        reject(new Error("Could not read video duration. Try a different file or re-export the clip."));
        return;
      }
      if (durationS > MAX_VIDEO_DURATION_S) {
        reject(
          new Error(
            `This clip is ${formatDuration(durationS)} long. Maximum allowed duration is ${MAX_VIDEO_DURATION_S} seconds (${formatDuration(MAX_VIDEO_DURATION_S)}).`,
          ),
        );
        return;
      }
      resolve({
        name: file.name,
        sizeMb: (file.size / (1024 * 1024)).toFixed(1),
        durationS,
        width,
        height,
      });
    };

    video.onerror = () => {
      cleanup();
      reject(new Error("Could not read video metadata. Only MP4 and MOV files are supported."));
    };

    video.src = objectUrl;
  });
}
