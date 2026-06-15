from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name, "").strip()
        return float(raw) if raw else default
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.environ.get(name, "").strip()
        return int(float(raw)) if raw else default
    except (TypeError, ValueError):
        return default


# Frames analysed per second. Lower = fewer detection inferences = faster on
# CPU (at some cost to peak-speed temporal resolution). Tune via env.
ANALYSIS_SAMPLE_FPS = _env_float("GAMESENSE_ANALYSIS_FPS", 8.0)
# Frames taller than this are downscaled before inference. Detection cost
# scales with pixels, so this is the biggest CPU lever (e.g. set 720 on a
# CPU-only machine). Defaults to 1080 — clips at or below it are untouched.
MAX_PROCESSING_HEIGHT = _env_int("GAMESENSE_MAX_PROCESSING_HEIGHT", 1080)
LONG_CLIP_DURATION_S = 30.0


def compute_frame_interval(source_fps: float, target_fps: float = ANALYSIS_SAMPLE_FPS) -> int:
    return max(1, int(round(source_fps / max(target_fps, 1e-6))))


def should_downscale_to_1080p(duration_s: float, frame_height: int) -> bool:
    # Retained for backward compatibility; downscaling is now height-based.
    return frame_height > MAX_PROCESSING_HEIGHT


def prepare_frame(frame: np.ndarray, duration_s: float) -> np.ndarray:
    # Always cap height at MAX_PROCESSING_HEIGHT — on CPU, processing a clip at
    # native 1080p/4K dominates analysis time, so this is the main speed win.
    height, width = frame.shape[:2]
    if height <= MAX_PROCESSING_HEIGHT:
        return frame
    scale = MAX_PROCESSING_HEIGHT / float(height)
    new_width = max(int(round(width * scale)), 1)
    return cv2.resize(frame, (new_width, MAX_PROCESSING_HEIGHT), interpolation=cv2.INTER_AREA)


def iter_sampled_frames(
    video_path: Path,
    *,
    start_frame_id: int = 0,
    target_fps: float = ANALYSIS_SAMPLE_FPS,
) -> Iterator[tuple[int, float, np.ndarray, float, int]]:
    """
    Stream sampled frames using seek — never loads the full video into memory.
    Yields: frame_id, source_fps, frame_bgr, frame_interval, frame_count
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = frame_count / source_fps if source_fps > 0 else 0.0
    frame_interval = compute_frame_interval(source_fps, target_fps)
    start_frame_id = min(max(start_frame_id, 0), max(frame_count - 1, 0))

    frame_id = start_frame_id
    while frame_id < frame_count:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        if not ok:
            break
        yield frame_id, source_fps, prepare_frame(frame, duration_s), frame_interval, frame_count
        frame_id += frame_interval

    capture.release()


def iter_all_frames(
    video_path: Path,
    *,
    start_frame_id: int = 0,
) -> Iterator[tuple[int, float, np.ndarray, int]]:
    """
    Stream every frame sequentially from start_frame_id — one frame in memory at a time.
    Yields: frame_id, source_fps, frame_bgr, frame_count
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = frame_count / source_fps if source_fps > 0 else 0.0
    start_frame_id = min(max(start_frame_id, 0), max(frame_count - 1, 0))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)

    frame_id = start_frame_id
    while frame_id < frame_count:
        ok, frame = capture.read()
        if not ok:
            break
        yield frame_id, source_fps, prepare_frame(frame, duration_s), frame_count
        frame_id += 1

    capture.release()
