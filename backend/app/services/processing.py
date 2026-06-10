from pathlib import Path

import cv2

from app.services.calibration import compute_calibration
from app.services.detections_export import collect_overlay_detections
from app.services.cv_pipeline import get_cv_pipeline
from app.services.heatmaps import generate_heatmaps
from app.services.metrics import compute_shot_metrics
from app.services.processing_progress import compute_total_steps, set_progress, set_tracking_progress
from app.services.profiling import StageTimer
from app.services.robust_player_tracker import TrackerRuntimeStats, track_player_robust
from app.services.shot_detection import detect_shots
from app.services.storage import get_video_record, update_video_record, video_metadata, write_json
from app.services.team_classification import TeamTemplates


def process_video(video_id: str) -> None:
    record = get_video_record(video_id)
    if not record:
        return

    timer = StageTimer()
    warnings = list(record.get("warnings") or [])
    set_progress(video_id, stage="calibration", percent=2, message="Calibrating pitch")

    try:
        timer.start("calibration")
        pitch_setup = record.get("pitch_setup") or {}
        metadata = record.get("video_metadata") or video_metadata(Path(record["video_path"]))
        record["video_metadata"] = metadata
        video_path = Path(record["video_path"])
        setup_frame_id = int(
            record.get("setup_frame_id")
            or pitch_setup.get("frame_id")
            or record.get("target_player", {}).get("frame_id")
            or 0
        )
        calibration_frame = _load_video_frame(video_path, setup_frame_id)
        pipeline = get_cv_pipeline()
        player_bboxes = _sample_player_bboxes(pipeline, video_path, setup_frame_id)
        calibration = compute_calibration(
            pitch_setup.get("pitch_polygon"),
            int(metadata.get("width") or 0),
            int(metadata.get("height") or 0),
            calibration_frame=calibration_frame,
            player_bboxes=player_bboxes,
        )
        warnings.extend(calibration.warnings)
        if calibration.errors:
            fatal_errors = [error for error in calibration.errors if "re-mark the pitch polygon" in error.lower()]
            if fatal_errors:
                raise ValueError(fatal_errors[0])
            warnings.extend(calibration.errors)
        record["calibration"] = calibration.to_dict()
        timer.stop("calibration")

        target = record["target_player"]
        player_id = int(target["player_id_target"])
        mode = record.get("mode") or "max_speed"
        source_fps = float(metadata.get("fps") or 30.0)
        frame_count = int(metadata.get("frame_count") or 0)
        start_frame_id = int(target.get("frame_id") or 0)
        total_steps = compute_total_steps(frame_count, start_frame_id, source_fps)
        record["processing"] = {"total_steps": total_steps}
        update_video_record(video_id, record)

        timer.start("team_classification")
        team_templates = _load_team_templates(video_id, record, video_path)
        timer.stop("team_classification")

        def progress_callback(
            steps_completed: int,
            steps_total: int,
            frame_id: int,
            frame_total: int,
            stats: TrackerRuntimeStats | None = None,
        ) -> None:
            extra = None
            if stats is not None:
                extra = {
                    "tracked_so_far": stats.tracked_so_far or stats.visible_frames,
                    "predicted_so_far": stats.predicted_so_far or stats.predicted_frames,
                    "lost_so_far": stats.lost_so_far or stats.lost_frames,
                }
            set_tracking_progress(
                video_id,
                steps_completed=steps_completed,
                total_steps=steps_total,
                frame_id=frame_id,
                frame_total=frame_total,
                stats=extra,
            )

        track_result = None
        if mode == "max_shot_power":
            if not calibration.scale_known:
                warnings.append(
                    "Shot speed in km/h requires pitch calibration. Mark the pitch polygon or ensure visible field markings."
                )
            timer.start("tracking")
            def shot_progress(frame_id: int, frame_total: int) -> None:
                from app.services.video_streaming import compute_frame_interval

                frame_interval = compute_frame_interval(source_fps)
                step_index = max(1, (frame_id - start_frame_id) // frame_interval)
                set_tracking_progress(
                    video_id,
                    steps_completed=min(step_index, total_steps),
                    total_steps=total_steps,
                    frame_id=frame_id,
                    frame_total=frame_total,
                )

            player_points, ball_points = pipeline.track_ball_and_player(
                video_path,
                target["click"],
                calibration,
                start_frame_id,
                target.get("bbox"),
                shot_progress,
                team_templates,
            )
            timer.stop("tracking")
            raw_ball_count = max(frame_count - start_frame_id, max(len(ball_points), 1))
            low_confidence_count = sum(1 for point in ball_points if point.confidence < 0.5)

            timer.start("metrics")
            shots = []
            if calibration.scale_known and calibration.matrix is not None:
                shots = detect_shots(
                    ball_points,
                    player_points,
                    source_fps,
                    calibration.matrix,
                )
            if not shots:
                warnings.append("No shots were detected - try widening contact radius or lowering shot speed threshold.")
            if len(ball_points) < 30:
                warnings.append("Ball tracking confidence is low - too few stable ball positions were captured.")
            metrics = compute_shot_metrics(
                player_id,
                shots,
                ball_points,
                rejected_track_points=low_confidence_count,
                raw_point_count=raw_ball_count,
            )
            metrics["player_label"] = target.get("player_id") or f"Player {player_id}"
            metrics["team_label"] = target.get("team_label")
            metrics["units"] = calibration.units
            timer.stop("metrics")
        else:
            timer.start("tracking")
            track_result = track_player_robust(
                pipeline,
                video_path,
                target["click"],
                calibration,
                start_frame_id,
                target.get("bbox"),
                progress_callback,
                video_id=video_id,
                team_templates=team_templates,
                total_steps=total_steps,
            )
            timer.stop("tracking")
            record["sampling"] = {
                "source_fps": track_result.source_fps,
                "sampling_fps": track_result.sampling_fps,
                "frame_interval": track_result.frame_interval,
            }
            if calibration.units == "pixels":
                warnings.append(
                    "Metric accuracy requires pitch marking. Speeds and distances are currently reported in pixels."
                )
            if track_result.stats.visible_frames < 10:
                warnings.append("Tracking confidence is low - too few directly observed frames were available.")

            set_progress(video_id, stage="metrics", percent=86, message="Calculating metrics")
            timer.start("metrics")
            metrics = {
                "player_id": player_id,
                "units": calibration.units,
                "max_speed_kmh": track_result.max_speed_kmh,
                "top_speed_kmh": track_result.max_speed_kmh,
                "avg_speed_kmh": track_result.avg_speed_kmh,
                "distance_m": track_result.distance_m,
                "total_distance_m": track_result.distance_m,
                "tracked_frames": track_result.stats.visible_frames,
                "predicted_frames": track_result.stats.predicted_frames,
                "lost_frames": track_result.stats.lost_frames,
                "confidence_score": track_result.confidence_score,
                "speed_series": track_result.speed_series,
                "usable_track_points": track_result.stats.visible_frames + track_result.stats.predicted_frames,
                "rejected_jump_count": 0,
                "top_speed_px_per_s": 0.0,
                "avg_speed_px_per_s": 0.0,
                "peak_acceleration_mps2": 0.0,
                "avg_acceleration_mps2": 0.0,
                "active_distance_m": track_result.distance_m,
                "sprint_count": 0,
                "sprint_distance_m": 0.0,
                "calibrated_point_ratio": 1.0 if calibration.scale_known else 0.0,
                "player_label": target.get("player_id") or f"Player {player_id}",
                "team_label": target.get("team_label"),
            }
            timer.stop("metrics")

        set_progress(video_id, stage="heatmaps", percent=93, message="Generating heatmaps")
        timer.start("heatmap_generation")
        heatmap_urls: dict[str, str] = {}
        if track_result is not None:
            heatmap_urls = generate_heatmaps(
                video_id,
                track_result.position_samples,
                track_result.speed_samples,
                int(metadata.get("width") or 0),
                int(metadata.get("height") or 0),
            )
        timer.stop("heatmap_generation")

        set_progress(video_id, stage="saving", percent=97, message="Saving results")
        timer.start("writing_output_files")
        overlay_payload = collect_overlay_detections(
            pipeline,
            video_path,
            team_templates,
            metadata,
            target_player=target,
        )
        detections_url = write_json(video_id, "detections.json", overlay_payload, compact=True)
        timer.stop("writing_output_files")

        record = get_video_record(video_id) or record
        record["status"] = "complete"
        record["warnings"] = warnings
        record["results"] = metrics
        record["assets"] = {
            "sprint_highlights": [],
            "detections_json": detections_url,
            "position_heatmap": heatmap_urls.get("position_heatmap"),
            "speed_heatmap": heatmap_urls.get("speed_heatmap"),
        }
        record["progress"] = {"stage": "complete", "percent": 100, "message": "Complete"}
        update_video_record(video_id, record)
        timer.log_summary(video_id)
    except Exception as exc:
        record = get_video_record(video_id) or record
        record["status"] = "failed"
        record["warnings"] = warnings + [str(exc)]
        record["progress"] = {"stage": "failed", "percent": 100, "message": "Processing failed"}
        update_video_record(video_id, record)
        timer.log_summary(video_id)


def _load_video_frame(video_path: Path, frame_id: int) -> cv2.typing.MatLike | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return None
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count > 0:
        frame_id = min(max(frame_id, 0), frame_count - 1)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def _sample_player_bboxes(
    pipeline,
    video_path: Path,
    frame_id: int,
) -> list[tuple[float, float, float, float]]:
    frame = _load_video_frame(video_path, frame_id)
    if frame is None:
        return []
    resized, scale = pipeline._resize(frame)
    bboxes: list[tuple[float, float, float, float]] = []
    for x, y, w, h, _confidence in pipeline._detect_people(resized):
        original = pipeline._unscale_bbox((x, y, w, h), scale)
        bboxes.append((float(original["x"]), float(original["y"]), float(original["width"]), float(original["height"])))
    return bboxes


def _load_team_templates(video_id: str, record: dict, video_path: Path) -> TeamTemplates:
    cached = record.get("team_classification")
    if cached:
        return TeamTemplates.from_dict(cached)

    pipeline = get_cv_pipeline()
    templates = pipeline.calibrate_team_templates(video_path)
    record["team_classification"] = templates.to_dict()
    update_video_record(video_id, record)
    return templates
