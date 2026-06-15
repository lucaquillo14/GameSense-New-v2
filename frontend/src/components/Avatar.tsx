"use client";

import { avatarSrc } from "@/lib/socialApi";

export function Avatar({
  name,
  url,
  size = 36,
  className = "",
}: {
  name: string;
  url?: string | null;
  size?: number;
  className?: string;
}) {
  const src = avatarSrc(url);
  const style = { width: size, height: size };
  if (src) {
    return (
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={src}
        alt={name}
        style={style}
        className={`shrink-0 rounded-full border border-[#ffffff14] object-cover ${className}`}
      />
    );
  }
  return (
    <span
      style={{ ...style, fontSize: Math.round(size * 0.42) }}
      className={`grid shrink-0 place-items-center rounded-full bg-[#3b82f6] font-semibold text-white ${className}`}
    >
      {(name || "?").charAt(0).toUpperCase()}
    </span>
  );
}
