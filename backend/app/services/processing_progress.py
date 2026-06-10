from __future__ import annotations

from app.services.storage import get_video_record, update_video_record
from app.services.video_streaming import compute_frame_interval


def compute_total_steps(frame_count: int, start_frame_id: int, source_fps: float) -> int:
    frame_interval = compute_frame_interval(source_fps)
    if frame_count <= start_frame_id:
        return 1
    remaining = frame_count - start_frame_id
    return max(1, (remaining + frame_interval - 1) // frame_interval)


def tracking_percent(steps_completed: int, total_steps: int) -> int:
    ratio = min(max(steps_completed / max(total_steps, 1), 0.0), 1.0)
    return 10 + int(ratio * 75)


def set_progress(
    video_id: str,
    *,
    stage: str,
    percent: int,
    message: str,
    extra: dict | None = None,
) -> None:
    record = get_video_record(video_id)
    if not record:
        return
    if record.get("status") == "processing" and stage not in {"failed", "complete"}:
        pass
    progress = {"stage": stage, "percent": max(0, min(percent, 100)), "message": message}
    if extra:
        progress.update(extra)
    record["progress"] = progress
    update_video_record(video_id, record)


def set_tracking_progress(
    video_id: str,
    *,
    steps_completed: int,
    total_steps: int,
    frame_id: int,
    frame_total: int,
    stats: dict | None = None,
) -> None:
    percent = tracking_percent(steps_completed, total_steps)
    extra = stats or {}
    set_progress(
        video_id,
        stage="tracking",
        percent=percent,
        message=f"Tracking player — frame {frame_id} of {frame_total}",
        extra=extra,
    )
