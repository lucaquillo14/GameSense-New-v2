#!/usr/bin/env python3
"""Extract a varied set of frames from GameSense source clips for labeling.

Walks the media-storage folder for `source.*` videos, samples candidate frames
at a fixed time interval, and keeps a frame only when it differs enough from the
last kept frame — so a player standing still doesn't produce 50 near-duplicates.
The output JPGs are ready to upload to a Roboflow project for labeling.

No new dependencies: uses the OpenCV + NumPy already in the backend venv.

Examples (run from the `backend` folder with your project Python):
    py -3.12 scripts/extract_training_frames.py
    py -3.12 scripts/extract_training_frames.py -i "C:\\GameSenseStorage" -o "C:\\training_frames"
    py -3.12 scripts/extract_training_frames.py --per-video 30 --max-total 500 --interval 0.4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

THUMB = 64  # grayscale thumbnail size used for the frame-difference check


def _load_backend_env() -> None:
    """Pick up GAMESENSE_MEDIA_ROOT from backend/.env if present."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def find_videos(root: Path) -> list[Path]:
    return [p for p in sorted(root.rglob("source.*")) if p.suffix.lower() in (".mp4", ".mov")]


def frame_signature(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (THUMB, THUMB), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32)


def extract_from_video(
    path: Path,
    out_dir: Path,
    per_video: int,
    interval_s: float,
    diff_threshold: float,
    max_width: int,
    taken_total: int,
    max_total: int,
) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  ! could not open {path}")
        return 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * interval_s)))
    clip_id = path.parent.name
    last_sig: np.ndarray | None = None
    saved = 0
    fid = 0
    while fid < frame_count and saved < per_video and (taken_total + saved) < max_total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok:
            break
        sig = frame_signature(frame)
        is_varied = last_sig is None or float(np.mean(np.abs(sig - last_sig))) >= diff_threshold
        if is_varied:
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / float(w)
                frame = cv2.resize(frame, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
            out_name = f"{clip_id}_f{fid:06d}.jpg"
            cv2.imwrite(str(out_dir / out_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            last_sig = sig
            saved += 1
        fid += step
    cap.release()
    return saved


def main() -> None:
    _load_backend_env()
    try:
        project_storage = Path(__file__).resolve().parents[2] / "storage"
    except IndexError:
        project_storage = Path.cwd() / "storage"
    default_root = os.environ.get("GAMESENSE_MEDIA_ROOT", "").strip() or str(project_storage)

    parser = argparse.ArgumentParser(description="Extract varied frames for Roboflow labeling.")
    parser.add_argument("-i", "--input", default=default_root,
                        help="Media storage root (default: GAMESENSE_MEDIA_ROOT or <project>/storage)")
    parser.add_argument("-o", "--output", default="training_frames", help="Output folder for JPGs")
    parser.add_argument("--per-video", type=int, default=40, help="Max frames kept per clip")
    parser.add_argument("--max-total", type=int, default=600, help="Max frames total")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between candidate frames")
    parser.add_argument("--diff-threshold", type=float, default=3.0,
                        help="Min mean pixel difference (0-255) vs last kept frame; raise it if you get "
                             "too many near-duplicates, lower it if you get too few frames")
    parser.add_argument("--max-width", type=int, default=1280, help="Downscale frames wider than this")
    args = parser.parse_args()

    root = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()
    if not root.exists():
        print(f"Input folder not found: {root}")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(root)
    if not videos:
        print(f"No source.mp4/.mov clips found under {root}")
        sys.exit(1)

    print(f"Found {len(videos)} clip(s) under {root}")
    print(f"Writing frames to {out_dir}\n")

    total = 0
    for video in videos:
        if total >= args.max_total:
            break
        n = extract_from_video(
            video, out_dir, args.per_video, args.interval,
            args.diff_threshold, args.max_width, total, args.max_total,
        )
        total += n
        print(f"  {video.parent.name}: {n} frames")

    print(f"\nDone — {total} frames in {out_dir}")
    print("Next: upload this folder to your Roboflow project (Object Detection).")


if __name__ == "__main__":
    main()
