import traceback
from pathlib import Path

import cv2
import numpy as np

from app.services.calibration import CalibrationResult, compute_calibration
from app.services.detections_export import collect_overlay_detections
from app.services.cv_pipeline import get_cv_pipeline
from app.services.heatmaps import generate_heatmaps
from app.services.metrics import TrackPoint, compute_shot_metrics
from app.services.pitch_heatmaps import generate_movement_heatmap, generate_touch_heatmap
from app.services.processing_progress import compute_total_steps, set_progress, set_tracking_progress
from app.services.profiling import StageTimer
from app.services.robust_player_tracker import RobustTrackResult, TrackerRuntimeStats, track_player_robust
from app.services.shot_detection import FieldTouchEvent, detect_shots, detect_touches_and_passes
from app.services.storage import get_video_record, update_video_record, video_dir, video_metadata, write_json
from app.services.team_classification import TeamTemplates

MAX_TECHNIQUE_DURATION_S = 30.0


def process_video(video_id: str) -> None:
    record = get_video_record(video_id)
    if not record:
        return

    mode = record.get("mode") or "max_speed"
    if mode == "shooting_technique":
        _process_shooting_technique(video_id, record)
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
        pitch_polygon = pitch_setup.get("pitch_polygon") or []
        had_manual_polygon = len(pitch_polygon) >= 4
        _height_cm = record.get("player_height_cm")
        player_height_m = (float(_height_cm) / 100.0) if _height_cm else None
        calibration = compute_calibration(
            pitch_polygon or None,
            int(metadata.get("width") or 0),
            int(metadata.get("height") or 0),
            calibration_frame=calibration_frame,
            player_bboxes=player_bboxes,
            goal_posts=pitch_setup.get("goal_posts"),
            player_height_m=player_height_m,
        )
        _log_calibration_diagnostics(calibration, had_manual_polygon, calibration_frame is not None)
        warnings.extend(calibration.warnings)
        if calibration.errors:
            warnings.extend(calibration.errors)

        if calibration.matrix is None:
            raise ValueError(_calibration_failure_message(had_manual_polygon, calibration_frame is not None, calibration))

        if not np.isfinite(calibration.matrix).all():
            raise ValueError(
                "Pitch calibration failed: the homography matrix contains invalid values (NaN/Inf). "
                "Please redraw the pitch boundary on the setup screen."
            )

        record["calibration"] = calibration.to_dict()
        timer.stop("calibration")

        target = record["target_player"]
        player_id = int(target["player_id_target"])
        mode = record.get("mode") or "max_speed"
        source_fps = float(metadata.get("fps") or 30.0)
        if source_fps <= 0:
            source_fps = 30.0
            warnings.append("Video FPS was missing or zero; defaulting to 30 fps for speed calculations.")
        print(f"[GameSense] source_fps={source_fps} frame_count={metadata.get('frame_count')} mode={mode}")
        frame_count = int(metadata.get("frame_count") or 0)
        start_frame_id = int(target.get("frame_id") or 0)
        total_steps = compute_total_steps(frame_count, start_frame_id, source_fps)
        record["processing"] = {"total_steps": total_steps}
        update_video_record(video_id, record)

        set_progress(video_id, stage="calibration", percent=6, message="Calibrating team colours")
        timer.start("team_classification")
        team_templates = _load_team_templates(video_id, record, video_path)
        timer.stop("team_classification")
        set_progress(video_id, stage="tracking", percent=10, message="Starting player tracking")

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

        track_result: RobustTrackResult | None = None
        player_points_for_pitch: list[TrackPoint] = []
        touch_events: list[FieldTouchEvent] = []
        touch_count = 0
        pass_count = 0
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

            player_points_for_pitch = player_points
            timer.start("metrics")
            shots = []
            if calibration.scale_known and calibration.matrix is not None:
                shots = detect_shots(
                    ball_points,
                    player_points,
                    source_fps,
                    calibration.matrix,
                )
                touch_events, touch_count, pass_count = detect_touches_and_passes(
                    player_points,
                    ball_points,
                    source_fps,
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
                touch_count=touch_count,
                pass_count=pass_count,
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
                # Max-speed mode only tracks the player — skip ball inference to
                # save CPU (no GPU). Touch/pass counts aren't used for speed.
                detect_ball=False,
            )
            timer.stop("tracking")
            _log_track_speed_diagnostics(track_result)
            record["sampling"] = {
                "source_fps": track_result.source_fps,
                "sampling_fps": track_result.sampling_fps,
                "frame_interval": track_result.frame_interval,
            }
            if track_result.stats.visible_frames < 10:
                warnings.append("Tracking confidence is low - too few directly observed frames were available.")

            set_progress(video_id, stage="metrics", percent=86, message="Calculating metrics")
            timer.start("metrics")
            # Touch / pass events from the ball detections collected during
            # tracking — powers the touch heatmap in max-speed mode too.
            if calibration.scale_known and track_result.ball_points:
                touch_events, touch_count, pass_count = detect_touches_and_passes(
                    track_result.player_points,
                    track_result.ball_points,
                    source_fps,
                )
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
                "rejected_jump_count": track_result.rejected_outliers,
                "top_speed_px_per_s": 0.0,
                "avg_speed_px_per_s": 0.0,
                "peak_acceleration_mps2": 0.0,
                "avg_acceleration_mps2": 0.0,
                "active_distance_m": track_result.distance_m,
                "sprint_count": track_result.sprint_count,
                "sprint_distance_m": track_result.sprint_distance_m,
                "touch_count": touch_count,
                "pass_count": pass_count,
                "calibrated_point_ratio": (
                    track_result.calibrated_ratio
                    if track_result.calibrated_ratio > 0
                    else (1.0 if calibration.scale_known else 0.0)
                ),
                "player_label": target.get("player_id") or f"Player {player_id}",
                "team_label": target.get("team_label"),
            }
            timer.stop("metrics")

        set_progress(video_id, stage="heatmaps", percent=93, message="Generating heatmaps")
        timer.start("heatmap_generation")
        heatmap_urls: dict[str, str | None] = {}
        # Max-speed mode only needs pitch movement/touch heatmaps — skip the
        # pixel-space position and speed heatmaps the UI no longer shows.
        if track_result is not None and mode != "max_speed":
            heatmap_urls = generate_heatmaps(
                video_id,
                track_result.position_samples,
                track_result.speed_samples,
                int(metadata.get("width") or 0),
                int(metadata.get("height") or 0),
            )
        if calibration.scale_known:
            pitch_track_dicts = _pitch_track_dicts(track_result, player_points_for_pitch)
            pitch_urls = _generate_pitch_heatmaps(
                video_id,
                pitch_track_dicts,
                touch_events,
                touch_count,
                pass_count,
                warnings,
            )
            heatmap_urls.update(pitch_urls)
        timer.stop("heatmap_generation")

        set_progress(video_id, stage="saving", percent=97, message="Saving results")
        timer.start("writing_output_files")
        if track_result is not None and track_result.overlay_frames:
            # Overlay was collected DURING tracking (per-frame, all players) —
            # no need to re-run detection over the whole clip.
            from app.services.detections_export import _interpolate_overlay_frames

            overlay_payload = {
                "fps": round(source_fps, 4),
                "interval": track_result.frame_interval,
                "target_id": target.get("player_id"),
                "frames": _interpolate_overlay_frames(track_result.overlay_frames),
            }
        else:
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
            "movement_heatmap": heatmap_urls.get("movement_heatmap"),
            "touch_heatmap": heatmap_urls.get("touch_heatmap"),
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


def _log_calibration_diagnostics(
    calibration,
    had_manual_polygon: bool,
    had_calibration_frame: bool,
) -> None:
    if calibration.matrix is None:
        print("[GameSense] homography matrix: None")
        return
    print(f"[GameSense] homography matrix:\n{calibration.matrix}")
    print(
        f"[GameSense] calibration scale_known={calibration.scale_known} "
        f"auto_detected={calibration.auto_detected} method={calibration.detection_method} "
        f"manual_polygon={had_manual_polygon} calibration_frame={had_calibration_frame}"
    )


def _calibration_failure_message(
    had_manual_polygon: bool,
    had_calibration_frame: bool,
    calibration,
) -> str:
    if had_manual_polygon:
        if calibration.errors:
            return (
                "Pitch calibration failed: the polygon you drew could not be mapped to field coordinates. "
                f"{calibration.errors[0]}"
            )
        return (
            "Pitch calibration failed: the polygon you drew could not be mapped to field coordinates. "
            "Please redraw the pitch boundary with at least four corners aligned to visible pitch lines."
        )
    if had_calibration_frame:
        return (
            "Pitch calibration failed: automatic pitch marking detection did not find usable lines. "
            "Draw the pitch boundary on the setup screen before starting analysis."
        )
    return (
        "Pitch calibration failed: no pitch polygon was provided and the setup frame was unavailable. "
        "Draw the pitch boundary on the setup screen before starting analysis."
    )


def _log_track_speed_diagnostics(track_result: RobustTrackResult | None) -> None:
    if track_result is None:
        return
    sample = list(track_result.recent_points)[:5]
    print(
        "[GameSense] track field coords (first 5): "
        + ", ".join(
            f"({point.x_m}, {point.y_m}) cal={point.calibrated}"
            for point in sample
        )
        if sample
        else "(no points)"
    )
    top_speeds = sorted(
        (entry["speed_kmh"] for entry in track_result.speed_series),
        reverse=True,
    )[:5]
    print(f"[GameSense] top 5 computed speeds (km/h): {top_speeds}")


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
    bboxes: list[tuple[float, float, float, float]] = []
    for x, y, w, h, _confidence in pipeline._detect_people(frame, high_resolution=True):
        bboxes.append((float(x), float(y), float(w), float(h)))
    return bboxes


def _pitch_track_dicts(
    track_result: RobustTrackResult | None,
    player_points: list[TrackPoint],
) -> list[dict[str, float]]:
    if track_result is not None and track_result.field_position_samples:
        return [{"x_m": x, "y_m": y} for x, y in track_result.field_position_samples]
    return [
        {"x_m": float(point.x_m), "y_m": float(point.y_m)}
        for point in player_points
        if point.calibrated and point.x_m is not None and point.y_m is not None and point.track_state == "visible"
    ]


def _generate_pitch_heatmaps(
    video_id: str,
    track_dicts: list[dict[str, float]],
    touch_events: list[FieldTouchEvent],
    touch_count: int,
    pass_count: int,
    warnings: list[str],
) -> dict[str, str | None]:
    urls: dict[str, str | None] = {
        "movement_heatmap": None,
        "touch_heatmap": None,
    }
    output_dir = video_dir(video_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[GameSense] pitch heatmap track points: {len(track_dicts)}")
    if len(track_dicts) >= 4:
        movement_path = output_dir / "movement-heatmap.png"
        print(f"[GameSense] writing movement heatmap to {movement_path.resolve()}")
        try:
            generate_movement_heatmap(track_dicts, movement_path)
            urls["movement_heatmap"] = f"/media/{video_id}/movement-heatmap.png"
        except Exception:
            traceback.print_exc()
            warnings.append("Movement heatmap generation failed — see server logs for traceback.")

    touch_dicts = [{"x_m": event.contact_x_m, "y_m": event.contact_y_m} for event in touch_events]
    pass_dicts = [
        {"x_m": event.contact_x_m, "y_m": event.contact_y_m, "angle_deg": 0}
        for event in touch_events
        if event.is_pass
    ]
    touch_path = output_dir / "touch-heatmap.png"
    print(f"[GameSense] writing touch heatmap to {touch_path.resolve()}")
    try:
        generate_touch_heatmap(touch_dicts, pass_dicts, touch_count, pass_count, touch_path)
        urls["touch_heatmap"] = f"/media/{video_id}/touch-heatmap.png"
    except Exception:
        traceback.print_exc()
        warnings.append("Touch heatmap generation failed — see server logs for traceback.")

    return urls


def _process_shooting_technique(video_id: str, record: dict) -> None:
    from app.services.processing_progress import set_progress as _set_progress
    from app.services.roboflow_inference import RoboflowConfigError
    from app.services.shooting_technique_pipeline import (
        ShootingTechniquePipelineError,
        run_shooting_technique_analysis,
    )
    from app.services.storage import MEDIA_ROOT

    warnings = list(record.get("warnings") or [])
    timer = StageTimer()
    try:
        metadata = record.get("video_metadata") or video_metadata(Path(record["video_path"]))
        record["video_metadata"] = metadata
        video_path = Path(record["video_path"])
        duration_s = float(metadata.get("duration_s") or 0.0)
        if duration_s > MAX_TECHNIQUE_DURATION_S:
            raise ValueError(
                f"This clip is {duration_s:.1f}s long. Technique analysis supports clips up to "
                f"{int(MAX_TECHNIQUE_DURATION_S)} seconds."
            )

        output_dir = MEDIA_ROOT / video_id / "workflow"
        _set_progress(video_id, stage="detection", percent=10, message="Starting shooting technique analysis")
        timer.start("shooting_technique")

        def on_progress(stage: str, percent: int, message: str) -> None:
            _set_progress(video_id, stage=stage, percent=percent, message=message)

        feedback = run_shooting_technique_analysis(
            video_path,
            output_dir,
            video_id=video_id,
            progress=on_progress,
        )
        timer.stop("shooting_technique")

        if feedback.scale_source and "player height" in feedback.scale_source.lower():
            warnings.append(
                "No goal detected in this clip — shot power and distances are scaled from player height. "
                "Include the goal in frame for goal-based scaling."
            )
        if feedback.shot_power_kmh > 0 and (feedback.shot_power_kmh < 20 or feedback.shot_power_kmh > 120):
            warnings.append(
                f"Shot power ({feedback.shot_power_kmh:.1f} km/h) is outside the typical range (20–120 km/h). "
                f"Scale source: {feedback.scale_source or 'unknown'}."
            )

        record = get_video_record(video_id) or record
        record["status"] = "complete"
        record["warnings"] = warnings
        record["shooting_result"] = feedback.model_dump()
        record["progress"] = {"stage": "complete", "percent": 100, "message": "Complete"}
        update_video_record(video_id, record)
        timer.log_summary(video_id)
    except (RoboflowConfigError, ShootingTechniquePipelineError) as exc:
        record = get_video_record(video_id) or record
        record["status"] = "failed"
        record["warnings"] = warnings + [str(exc)]
        record["progress"] = {"stage": "failed", "percent": 100, "message": "Processing failed"}
        update_video_record(video_id, record)
        timer.log_summary(video_id)
    except Exception as exc:
        record = get_video_record(video_id) or record
        record["status"] = "failed"
        record["warnings"] = warnings + [str(exc)]
        record["progress"] = {"stage": "failed", "percent": 100, "message": "Processing failed"}
        update_video_record(video_id, record)
        timer.log_summary(video_id)


def _load_team_templates(video_id: str, record: dict, video_path: Path) -> TeamTemplates:
    cached = record.get("team_classification")
    if cached:
        return TeamTemplates.from_dict(cached)

    pipeline = get_cv_pipeline()
    templates = pipeline.calibrate_team_templates(video_path)
    record["team_classification"] = templates.to_dict()
    update_video_record(video_id, record)
    return templates
