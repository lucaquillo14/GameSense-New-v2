from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

ANALYSIS_SAMPLE_FPS = 8.0
MAX_PROCESSING_HEIGHT = 1080
LONG_CLIP_DURATION_S = 30.0


def compute_frame_interval(source_fps: float, target_fps: float = ANALYSIS_SAMPLE_FPS) -> int:
    return max(1, int(round(source_fps / max(target_fps, 1e-6))))


def should_downscale_to_1080p(duration_s: float, frame_height: int) -> bool:
    return duration_s > LONG_CLIP_DURATION_S and frame_height > MAX_PROCESSING_HEIGHT


def prepare_frame(frame: np.ndarray, duration_s: float) -> np.ndarray:
    height, width = frame.shape[:2]
    if not should_downscale_to_1080p(duration_s, height):
        return frame
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
