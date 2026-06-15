from __future__ import annotations

from pathlib import Path
from typing import Callable

from app.services.robust_player_tracker import YOLO_CADENCE
from app.services.team_classification import TeamTemplates, team_label
from app.services.video_streaming import ANALYSIS_SAMPLE_FPS, compute_frame_interval, iter_all_frames


def _interpolate_bbox(start_bbox: list[float], end_bbox: list[float], ratio: float) -> list[float]:
    return [
        round(start_bbox[0] + (end_bbox[0] - start_bbox[0]) * ratio, 4),
        round(start_bbox[1] + (end_bbox[1] - start_bbox[1]) * ratio, 4),
        round(start_bbox[2] + (end_bbox[2] - start_bbox[2]) * ratio, 4),
        round(start_bbox[3] + (end_bbox[3] - start_bbox[3]) * ratio, 4),
    ]


def _interpolate_overlay_frames(key_frames: dict[int, list[dict]]) -> dict[str, list[dict]]:
    frames: dict[str, list[dict]] = {str(frame_id): entries for frame_id, entries in key_frames.items()}
    ordered = sorted(key_frames.keys())
    for start_frame, end_frame in zip(ordered, ordered[1:]):
        gap = end_frame - start_frame
        if gap <= 1:
            continue
        start_by_id = {entry["id"]: entry for entry in key_frames[start_frame]}
        end_by_id = {entry["id"]: entry for entry in key_frames[end_frame]}
        for frame_id in range(start_frame + 1, end_frame):
            ratio = (frame_id - start_frame) / gap
            interpolated_entries: list[dict] = []
            for player_id in set(start_by_id).intersection(end_by_id):
                start_entry = start_by_id[player_id]
                end_entry = end_by_id[player_id]
                interpolated_entries.append({
                    "id": player_id,
                    "team": start_entry["team"],
                    "c": round(min(start_entry["c"], end_entry["c"]) * 0.9, 4),
                    "b": _interpolate_bbox(start_entry["b"], end_entry["b"], ratio),
                    "color": start_entry["color"],
                    "interpolated": True,
                    "is_target": start_entry.get("is_target", False),
                })
            if interpolated_entries:
                frames[str(frame_id)] = interpolated_entries
    return frames


def collect_overlay_detections(
    pipeline,
    video_path: Path,
    team_templates: TeamTemplates,
    metadata: dict,
    target_player: dict | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    source_fps = float(metadata.get("fps") or 30.0)
    if source_fps <= 0:
        source_fps = 30.0
    frame_count = int(metadata.get("frame_count") or 0)
    frame_width = int(metadata.get("width") or 0)
    frame_height = int(metadata.get("height") or 0)
    if frame_width <= 0 or frame_height <= 0 or frame_count <= 0:
        return {"fps": source_fps, "interval": 1, "frames": {}}

    frame_interval = compute_frame_interval(source_fps, ANALYSIS_SAMPLE_FPS)
    pipeline.team_service.set_templates(team_templates)
    tracker = pipeline.create_player_tracker(source_fps)
    target_id = (target_player or {}).get("player_id")

    key_frames: dict[int, list[dict]] = {}
    processed = 0
    for frame_id, fps, frame, total_frames in iter_all_frames(video_path):
        if frame_id % YOLO_CADENCE != 0:
            continue

        tracked = pipeline.track_players(frame, tracker)
        entries: list[dict] = []
        if tracked.tracker_id is not None:
            player_boxes = pipeline.tracked_to_tuples(tracked)
            for index, track_id in enumerate(tracked.tracker_id):
                if index >= len(player_boxes):
                    continue
                x, y, w, h, confidence = player_boxes[index]
                bbox_dict = {
                    "x": float(x),
                    "y": float(y),
                    "width": float(w),
                    "height": float(h),
                }
                player_key = f"track-{int(track_id)}"
                team_id, color_rgb = pipeline._classify_player_detection(
                    frame,
                    bbox_dict,
                    player_key,
                    team_templates,
                    apply_temporal=False,
                )
                if team_id not in {"team_a", "team_b"}:
                    continue
                entries.append({
                    "id": player_key,
                    "team": team_label(team_id),  # type: ignore[arg-type]
                    "c": round(float(confidence), 4),
                    "b": [
                        round(float(x) / frame_width, 4),
                        round(float(y) / frame_height, 4),
                        round(float(w) / frame_width, 4),
                        round(float(h) / frame_height, 4),
                    ],
                    "color": {"r": color_rgb[0], "g": color_rgb[1], "b": color_rgb[2]},
                    "interpolated": False,
                })

        if entries:
            key_frames[frame_id] = entries
        processed += 1
        if progress_callback and processed % 20 == 0:
            progress_callback(frame_id, total_frames)

    frames = _interpolate_overlay_frames(key_frames)
    populated = len(frames)
    coverage = populated / max(frame_count, 1)
    print(
        f"[GameSense] overlay JSON frames={populated} "
        f"coverage={coverage:.1%} keyframes={len(key_frames)}"
    )
    return {
        "fps": round(source_fps, 4),
        "interval": frame_interval,
        "target_id": target_id,
        "frames": frames,
    }
