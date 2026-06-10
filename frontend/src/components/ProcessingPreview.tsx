"use client";

import { useEffect, useState } from "react";
import { getPreviewFrame } from "@/lib/api";

type Props = {
  videoId: string;
  active: boolean;
};

export function ProcessingPreview({ videoId, active }: Props) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    let objectUrl: string | null = null;

    async function refresh() {
      try {
        const url = await getPreviewFrame(videoId);
        if (cancelled) {
          URL.revokeObjectURL(url);
          return;
        }
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        objectUrl = url;
        setPreviewUrl(url);
        setReady(true);
      } catch {
        if (!cancelled) setReady(false);
      }
    }

    refresh();
    const interval = window.setInterval(refresh, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [active, videoId]);

  return (
    <div className="preview-pulse mt-4 overflow-hidden rounded-xl border border-[#ffffff14] bg-[#0a0a0f]">
      <div className="aspect-video w-full">
        {previewUrl && ready ? (
          <img src={previewUrl} alt="Live detection preview" className="h-full w-full object-contain" />
        ) : (
          <div className="grid h-full place-items-center px-6 text-center text-sm text-[#64748b]">
            Starting detection…
          </div>
        )}
      </div>
    </div>
  );
}
