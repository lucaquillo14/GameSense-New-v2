from __future__ import annotations

from pathlib import Path
from typing import Callable

from app.services.player_identity import PlayerIdentityManager
from app.services.team_classification import TeamTemplates, team_label
from app.services.video_streaming import ANALYSIS_SAMPLE_FPS, compute_frame_interval, iter_sampled_frames


def collect_overlay_detections(
    pipeline,
    video_path: Path,
    team_templates: TeamTemplates,
    metadata: dict,
    target_player: dict | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    source_fps = float(metadata.get("fps") or 30.0)
    frame_count = int(metadata.get("frame_count") or 0)
    frame_width = int(metadata.get("width") or 0)
    frame_height = int(metadata.get("height") or 0)
    if frame_width <= 0 or frame_height <= 0 or frame_count <= 0:
        return {"fps": source_fps, "interval": 1, "frames": {}}

    frame_interval = compute_frame_interval(source_fps, ANALYSIS_SAMPLE_FPS)
    pipeline.team_service.set_templates(team_templates)
    identity_manager = PlayerIdentityManager(team_templates)
    target_id = (target_player or {}).get("player_id")

    frames: dict[str, list[dict]] = {}
    sampled = 0
    for frame_id, _fps, frame, _interval, _frame_count in iter_sampled_frames(video_path):
        resized, scale = (frame, 1.0) if frame.shape[1] <= pipeline.max_width else pipeline._resize(frame)
        entries: list[dict] = []
        player_index = 0
        stream_to_source_x = frame_width / max(frame.shape[1], 1)
        stream_to_source_y = frame_height / max(frame.shape[0], 1)
        for x, y, w, h, confidence in pipeline._detect_people(resized):
            stream_bbox = pipeline._unscale_bbox((x, y, w, h), scale)
            original = {
                "x": stream_bbox["x"] * stream_to_source_x,
                "y": stream_bbox["y"] * stream_to_source_y,
                "width": stream_bbox["width"] * stream_to_source_x,
                "height": stream_bbox["height"] * stream_to_source_y,
            }
            bbox_dict = {
                "x": original["x"],
                "y": original["y"],
                "width": original["width"],
                "height": original["height"],
            }
            bbox_tuple = (
                float(original["x"]),
                float(original["y"]),
                float(original["width"]),
                float(original["height"]),
            )
            player_key = f"export-{frame_id}-{player_index}"
            team_id, color_rgb = pipeline._classify_player_detection(
                frame,
                bbox_dict,
                player_key,
                team_templates,
                apply_temporal=False,
            )
            player_index += 1
            if team_id not in {"team_a", "team_b"}:
                continue

            stable_id = identity_manager.assign_identity(
                frame,
                bbox_tuple,
                frame_id,
                frame_id / max(source_fps, 1e-6),
                team=team_id,
            )
            entry = {
                "id": stable_id or player_key,
                "team": team_label(team_id),  # type: ignore[arg-type]
                "c": round(float(confidence), 4),
                "b": [
                    round(float(original["x"]) / frame_width, 4),
                    round(float(original["y"]) / frame_height, 4),
                    round(float(original["width"]) / frame_width, 4),
                    round(float(original["height"]) / frame_height, 4),
                ],
                "color": {"r": color_rgb[0], "g": color_rgb[1], "b": color_rgb[2]},
            }
            entries.append(entry)

        if entries:
            frames[str(frame_id)] = entries
        sampled += 1
        if progress_callback and sampled % 20 == 0:
            progress_callback(frame_id, frame_count)
    return {
        "fps": round(source_fps, 4),
        "interval": frame_interval,
        "target_id": target_id,
        "frames": frames,
    }
