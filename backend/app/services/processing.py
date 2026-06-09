from pathlib import Path

from app.services.calibration import compute_homography
from app.services.cv_pipeline import ClassicalCvPipeline, ReIdByteTrackPipeline
from app.services.metrics import build_speed_series, compute_metrics, compute_shot_metrics, stabilize_track_points
from app.services.shot_detection import detect_shots
from app.services.storage import get_video_record, update_video_record, video_metadata
from app.services.team_classification import TeamTemplates


def process_video(video_id: str) -> None:
    record = get_video_record(video_id)
    if not record:
        return

    warnings = list(record.get("warnings") or [])
    _set_progress(video_id, record, "calibration", 14, "Preparing pitch calibration")
    try:
        pitch_setup = record.get("pitch_setup") or {}
        metadata = record.get("video_metadata") or video_metadata(Path(record["video_path"]))
        record["video_metadata"] = metadata
        matrix, calibration_warnings = compute_homography(
            pitch_setup.get("pitch_polygon"),
            int(metadata.get("width") or 0),
            int(metadata.get("height") or 0),
        )
        warnings.extend(calibration_warnings)
        if matrix is None:
            raise ValueError("Pitch calibration insufficient - metrics may be inaccurate.")

        target = record["target_player"]
        player_id = int(target["player_id_target"])
        mode = record.get("mode") or "max_speed"
        video_path = Path(record["video_path"])
        team_templates = _load_team_templates(video_id, record, video_path)
        pipeline = ReIdByteTrackPipeline()
        start_frame_id = int(target.get("frame_id") or 0)
        progress_callback = lambda frame_id, frame_count: _set_tracking_progress(video_id, frame_id, frame_count)

        if mode == "max_shot_power":
            _set_progress(video_id, record, "tracking", 22, "Tracking player and ball for shot power")
            player_points, ball_points = pipeline.track_ball_and_player(
                video_path,
                target["click"],
                matrix,
                start_frame_id,
                target.get("bbox"),
                progress_callback,
                team_templates,
            )
            raw_ball_count = int(metadata.get("frame_count") or 0) - start_frame_id
            if raw_ball_count <= 0:
                raw_ball_count = max(len(ball_points), 1)
            low_confidence_count = sum(1 for point in ball_points if point.confidence < 0.5)

            _set_progress(video_id, record, "shot_detection", 76, "Detecting shots from ball trajectory")
            shots = detect_shots(
                ball_points,
                player_points,
                float(metadata.get("fps") or 30.0),
                matrix,
            )
            if not shots:
                warnings.append("No shots were detected - try widening contact radius or lowering shot speed threshold.")
            if len(ball_points) < 30:
                warnings.append("Ball tracking confidence is low - too few stable ball positions were captured.")

            _set_progress(video_id, record, "metrics", 86, "Calculating shot power metrics")
            metrics = compute_shot_metrics(
                player_id,
                shots,
                ball_points,
                rejected_track_points=low_confidence_count,
                raw_point_count=raw_ball_count,
            )
            metrics["player_label"] = target.get("player_id") or f"Player {player_id}"
            metrics["team_label"] = target.get("team_label")
        else:
            _set_progress(video_id, record, "tracking", 22, "Tracking selected player with ByteTrack/ReID")
            points = pipeline.track_target(
                video_path,
                target["click"],
                matrix,
                start_frame_id,
                target.get("bbox"),
                progress_callback,
                team_templates,
            )

            raw_point_count = len(points)
            _set_progress(video_id, record, "filtering", 76, "Filtering noisy track points")
            points, rejected_jump_count = stabilize_track_points(points)
            if rejected_jump_count:
                warnings.append(f"Rejected {rejected_jump_count} impossible tracking jumps before metric calculation.")
            if len(points) < 10:
                warnings.append("Tracking confidence is low - too few stable points were available for reliable metrics.")
            elif rejected_jump_count > raw_point_count * 0.25:
                warnings.append("Tracking confidence is reduced - many detections were rejected as implausible jumps.")

            _set_progress(video_id, record, "metrics", 86, "Calculating speed, distance, and confidence")
            metrics = compute_metrics(player_id, points, rejected_jump_count, raw_point_count)
            metrics["speed_series"] = build_speed_series(points)
            metrics["player_label"] = target.get("player_id") or f"Player {player_id}"
            metrics["team_label"] = target.get("team_label")

        record["status"] = "complete"
        record["warnings"] = warnings
        record["results"] = metrics
        record["assets"] = {
            "sprint_highlights": [],
        }
        record["progress"] = {"stage": "complete", "percent": 100, "message": "Analysis complete"}
    except Exception as exc:
        record["status"] = "failed"
        record["warnings"] = warnings + [str(exc)]
        record["progress"] = {"stage": "failed", "percent": 100, "message": "Processing failed"}

    update_video_record(video_id, record)


def _load_team_templates(video_id: str, record: dict, video_path: Path) -> TeamTemplates:
    cached = record.get("team_classification")
    if cached:
        return TeamTemplates.from_dict(cached)

    pipeline = ClassicalCvPipeline()
    templates = pipeline.calibrate_team_templates(video_path)
    record["team_classification"] = templates.to_dict()
    update_video_record(video_id, record)
    return templates


def _set_tracking_progress(video_id: str, frame_id: int, frame_count: int) -> None:
    if frame_count <= 0:
        percent = 45
    else:
        percent = 22 + min(int((frame_id / frame_count) * 52), 52)
    record = get_video_record(video_id)
    if not record or record.get("status") != "processing":
        return
    record["progress"] = {
        "stage": "tracking",
        "percent": max(min(percent, 74), 22),
        "message": f"Tracking frame {frame_id} of {frame_count or '?'}",
    }
    update_video_record(video_id, record)


def _set_progress(video_id: str, record: dict, stage: str, percent: int, message: str) -> None:
    record["progress"] = {"stage": stage, "percent": percent, "message": message}
    update_video_record(video_id, record)
