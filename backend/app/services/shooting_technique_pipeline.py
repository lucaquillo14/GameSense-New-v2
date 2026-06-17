"""Integrate the shooting_analysis pipeline with GameSense upload/process flow.

Improvements over the original:
  * Streams the clip in passes instead of loading every frame into RAM
    (a 30 s 1080p60 clip previously needed ~11 GB resident).
  * Hosted RF-DETR calls run in parallel (I/O bound) — both the stride pass
    and the ball backfill, which previously issued one sequential network
    call per frame and could take 10+ minutes.
  * Ball backfill is targeted: pose data localises the kick first, then only
    a ±2.5 s window around it is backfilled at full per-frame density. Same
    accuracy where it matters (contact + power fit), a fraction of the calls.
  * Pose runs on a padded ROI around the detected player when the player is
    small in frame — markedly better landmarks on wide framings.
  * Goal box is the median over several sampled frames (stable px->m scale).
  * Annotated video tries the browser-playable avc1 codec before mp4v.
  * Granular progress updates throughout, so the UI never looks frozen.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import cv2
import numpy as np

from app.models import BodyAngle, ShootingFeedback, TechniqueFrame
from app.services.roboflow_inference import RoboflowConfigError
from app.services.shooting_analysis.config import PipelineConfig

ProgressFn = Callable[[str, int, str], None]

_METRIC_ANGLE_NAMES = (
    ("knee_angle", "contact_knee_angle"),
    ("ankle_lock", "ankle_lock_variation"),
    ("approach_angle", "approach_angle"),
    ("hip_rotation", "hip_rotation"),
    ("trunk_lean", "trunk_lean"),
    ("backswing_knee_flexion", "backswing_knee_flexion"),
)

_DETECT_CHUNK = 12          # frames per parallel inference batch


class ShootingTechniquePipelineError(RuntimeError):
    """Raised when shooting technique analysis cannot complete."""


def build_pipeline_config(source_fps: float,
                          player_height_m: float | None = None) -> PipelineConfig:
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise RoboflowConfigError(
            "ROBOFLOW_API_KEY is not set. Get a free API key at https://app.roboflow.com "
            "and add it to backend/.env as ROBOFLOW_API_KEY=your_key_here"
        )

    cfg = PipelineConfig()
    cfg.detector.api_key = api_key
    cfg.detector.backend = os.environ.get("SHOOTING_TECHNIQUE_BACKEND", "hosted").strip() or "hosted"
    # Self-hosted inference server support (same env var as max-speed mode).
    inference_url = os.environ.get("ROBOFLOW_INFERENCE_URL", "").strip()
    if inference_url:
        cfg.detector.api_url = inference_url
    # Default goal model comes from config (ball-and-goalpost-detection-2). The env var
    # overrides it; set ROBOFLOW_GOAL_MODEL= (empty) to disable goal detection entirely.
    if "ROBOFLOW_GOAL_MODEL" in os.environ:
        cfg.detector.goal_model_id = os.environ["ROBOFLOW_GOAL_MODEL"].strip()
    cfg.detector.detect_stride = max(1, int(round(source_fps / 2.0)))
    cfg.kicking_foot = os.environ.get("SHOOTING_TECHNIQUE_KICKING_FOOT", "auto").strip() or "auto"

    ball_conf = os.environ.get("SHOOTING_TECHNIQUE_BALL_CONF", "").strip()
    if ball_conf:
        cfg.detector.conf_ball = float(ball_conf)

    player_height = os.environ.get("SHOOTING_TECHNIQUE_PLAYER_HEIGHT_M", "").strip()
    if player_height:
        cfg.player_height_m = float(player_height)
    # A per-request height (from the UI) beats the env default: every distance
    # and speed scales off it when no goal anchors the scene.
    if player_height_m and 1.2 <= player_height_m <= 2.3:
        cfg.player_height_m = float(player_height_m)

    goal_width = os.environ.get("SHOOTING_TECHNIQUE_GOAL_WIDTH_M", "").strip()
    if goal_width:
        cfg.goal_width_m = float(goal_width)

    workers = os.environ.get("SHOOTING_TECHNIQUE_PARALLEL_REQUESTS", "").strip()
    if workers:
        cfg.detector.parallel_requests = max(1, int(workers))

    return cfg


def _notify(progress: ProgressFn | None, stage: str, percent: int, message: str) -> None:
    if progress is not None:
        progress(stage, percent, message)


def _follow_through_label(ratio: float | None) -> str:
    if ratio is None:
        return "medium"
    if ratio < 0.55:
        return "low"
    if ratio >= 0.85:
        return "high"
    return "medium"


def _contact_angles(contact_frame_id: int, metrics: dict) -> list[BodyAngle]:
    angles: list[BodyAngle] = []
    for angle_name, metric_key in _METRIC_ANGLE_NAMES:
        value = metrics.get(metric_key)
        if isinstance(value, (int, float)):
            angles.append(
                BodyAngle(
                    name=angle_name,
                    value_deg=float(value),
                    frame_id=contact_frame_id,
                    time_s=0.0,
                )
            )
    return angles


def _build_frame_analysis(
    *,
    fps: float,
    poses: list[Any],
    ev: Any,
    metrics: dict,
) -> list[TechniqueFrame]:
    key_frames = sorted({ev.plant, ev.backswing_peak, ev.contact, min(ev.contact + 6, len(poses) - 1)})
    frames: list[TechniqueFrame] = []
    for frame_id in key_frames:
        if frame_id < 0 or frame_id >= len(poses):
            continue
        angles: list[BodyAngle] = []
        if frame_id == ev.contact:
            for angle in _contact_angles(ev.contact, metrics):
                angles.append(
                    BodyAngle(
                        name=angle.name,
                        value_deg=angle.value_deg,
                        frame_id=frame_id,
                        time_s=frame_id / max(fps, 1e-6),
                    )
                )
        frames.append(
            TechniqueFrame(
                frame_id=frame_id,
                time_s=frame_id / max(fps, 1e-6),
                angles=angles,
                ball_visible=True,
                phase=ev.phase_at(frame_id, len(poses)).lower(),
            )
        )
    return frames


def analysis_to_shooting_feedback(
    *,
    analysis: Any,
    ev: Any,
    poses: list[Any],
    fps: float,
    annotated_video_url: str | None,
    contact_frame_url: str | None = None,
) -> ShootingFeedback:
    metrics = analysis.metrics
    plant_m = metrics.get("plant_foot_distance_m")
    plant_cm = round(float(plant_m) * 100.0, 1) if isinstance(plant_m, (int, float)) else 0.0
    follow_ratio = metrics.get("follow_through_height")
    follow_ratio_f = float(follow_ratio) if isinstance(follow_ratio, (int, float)) else 0.0

    return ShootingFeedback(
        shot_power_kmh=round(float(analysis.shot_speed_kmh or 0.0), 1),
        technique_score=round(float(analysis.score), 1),
        approach_angle_deg=_float_or_zero(metrics.get("approach_angle")),
        plant_foot_distance_cm=plant_cm,
        knee_bend_at_contact_deg=_float_or_zero(metrics.get("contact_knee_angle")),
        hip_rotation_deg=_float_or_zero(metrics.get("hip_rotation")),
        follow_through_height=_follow_through_label(follow_ratio if isinstance(follow_ratio, (int, float)) else None),
        feedback_points=list(analysis.feedback),
        annotated_video_url=annotated_video_url,
        contact_frame_url=contact_frame_url,
        frame_analysis=_build_frame_analysis(fps=fps, poses=poses, ev=ev, metrics=metrics),
        confidence=_analysis_confidence(analysis, poses),
        contact_frame_id=ev.contact,
        backswing_knee_flexion_deg=_float_or_zero(metrics.get("backswing_knee_flexion")),
        ankle_lock_variation_deg=_float_or_zero(metrics.get("ankle_lock_variation")),
        follow_through_height_ratio=follow_ratio_f,
        power_rating=analysis.power_label or "",
        kicking_foot=ev.kicking_foot,
        scale_source=analysis.scale_source or "",
        shot_distance_m=round(_float_or_zero(metrics.get("shot_distance_m")), 1),
        on_target=getattr(analysis, "on_target", None),
        goal_crossing_height_m=round(_float_or_zero(metrics.get("goal_crossing_height_m")), 2),
        goal_crossing_offset_m=round(_float_or_zero(metrics.get("goal_crossing_offset_m")), 2),
    )


def _float_or_zero(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _analysis_confidence(analysis: Any, poses: list[Any]) -> float:
    measured = sum(1 for value in analysis.metrics.values() if value is not None)
    total = max(len(analysis.metrics), 1)
    pose_ratio = sum(1 for pose in poses if getattr(pose, "ok", False)) / max(len(poses), 1)
    return round(min(0.95, 0.35 + 0.45 * (measured / total) + 0.2 * pose_ratio), 3)


# --------------------------------------------------------------------- video IO
class _FrameReader:
    """Sequential + random access frame reader that never holds the whole clip."""

    def __init__(self, video_path: Path):
        self.path = str(video_path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise ShootingTechniquePipelineError(f"Could not open video: {video_path}")
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._next_id = 0

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        self._seek(0)
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield self._next_id, frame
            self._next_id += 1

    def read_frame(self, frame_id: int) -> Optional[np.ndarray]:
        self._seek(frame_id)
        ok, frame = self._cap.read()
        if not ok:
            return None
        self._next_id = frame_id + 1
        return frame

    def _seek(self, frame_id: int) -> None:
        if self._next_id != frame_id:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            self._next_id = frame_id

    def count_frames(self) -> int:
        n = 0
        self._seek(0)
        while True:
            ok = self._cap.grab()
            if not ok:
                break
            n += 1
        self._next_id = n
        return n

    def close(self) -> None:
        self._cap.release()


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Prefer browser-playable H.264 (avc1); fall back to mp4v."""
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise ShootingTechniquePipelineError("Could not create annotated output video.")


def _ensure_browser_playable(path: Path) -> None:
    """Re-encode the annotated clip to H.264 + faststart so browsers can play it.

    The pip OpenCV build usually lacks an H.264 encoder and falls back to mp4v,
    which most browsers won't decode. ffmpeg (present in the Docker image)
    transcodes to libx264. No-op if ffmpeg is unavailable or the file is missing.
    """
    import shutil
    import subprocess

    if not path.exists() or shutil.which("ffmpeg") is None:
        return
    tmp = path.with_name(path.stem + "_h264.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(path),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-an", str(tmp),
            ],
            check=True,
            timeout=180,
        )
        if tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(path)
    except Exception as exc:  # keep the original file on any failure
        print(f"[GameSense] annotated video transcode skipped: {exc}")
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# ------------------------------------------------------------- parallel infer
def _detect_batch(detector, items: list[tuple[int, np.ndarray]],
                  model_id: Optional[str], workers: int) -> dict[int, list]:
    """Run detection over (frame_id, frame) pairs. Hosted backend fans out over
    a thread pool (calls are network-bound); local backend stays sequential
    (the local model is not thread-safe)."""
    results: dict[int, list] = {}
    if not items:
        return results

    def infer(frame):
        if model_id is not None:
            return detector.detect_with_model(frame, model_id)
        return detector.detect(frame)

    if detector.cfg.backend != "hosted" or workers <= 1:
        for frame_id, frame in items:
            results[frame_id] = infer(frame) or []
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(infer, frame): frame_id for frame_id, frame in items}
        for future in as_completed(futures):
            try:
                results[futures[future]] = future.result() or []
            except Exception:
                results[futures[future]] = []
    return results


# ------------------------------------------------------------------- pipeline
def run_shooting_technique_analysis(
    video_path: Path,
    output_dir: Path,
    *,
    video_id: str,
    progress: ProgressFn | None = None,
    player_height_m: float | None = None,
) -> ShootingFeedback:
    """Run the full shooting technique pipeline and return API-ready feedback."""
    from app.services.shooting_analysis.annotate import Annotator
    from app.services.shooting_analysis.detect import Detector, center, pick_ball, pick_player
    from app.services.shooting_analysis.events import detect_events, estimate_kick_frame
    from app.services.shooting_analysis.metrics import compute_metrics
    from app.services.shooting_analysis.pose import (
        PoseTracker,
        interpolate_boxes,
        interpolate_missing,
    )

    print(f"[GameSense] shooting technique v2 pipeline starting for {video_id}")
    _notify(progress, "detection", 11, "Starting shooting technique analysis (v2 engine)")

    reader = _FrameReader(video_path)
    fps, width, height = reader.fps, reader.width, reader.height
    frame_diag = float(np.hypot(width, height))

    n = reader.count_frames()
    if n < 10:
        reader.close()
        raise ShootingTechniquePipelineError("Clip too short — need at least ~0.5 s of video.")

    cfg = build_pipeline_config(fps, player_height_m)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / "annotated.mp4"
    workers = cfg.detector.parallel_requests

    try:
        detector = Detector(cfg.detector)
    except RuntimeError as exc:
        reader.close()
        raise RoboflowConfigError(str(exc)) from exc

    try:
        # ------------------------------------------------ pass 1: detection
        _notify(progress, "detection", 15, "Detecting player and ball with RF-DETR")
        ball_track: list[Optional[np.ndarray]] = [None] * n
        player_boxes: list[Optional[dict]] = [None] * n
        prev_player_center: tuple[float, float] | None = None
        prev_ball_center: tuple[float, float] | None = None
        stride = max(1, cfg.detector.detect_stride)
        goal_samples: list[np.ndarray] = []
        goal_sample_ids = set(
            int(round(t)) for t in np.linspace(0, n - 1, cfg.detector.goal_sample_count)
        )
        total_stride_frames = max(1, (n + stride - 1) // stride)
        processed = 0

        ball_diams_px: list[tuple[int, float]] = []

        def assign_detections(dets_by_frame: dict[int, list]) -> None:
            nonlocal prev_player_center, prev_ball_center
            for frame_id in sorted(dets_by_frame):
                detections = dets_by_frame[frame_id]
                player = pick_player(detections, cfg.detector, prev_player_center)
                ball = pick_ball(detections, cfg.detector, prev_ball_center, frame_diag)
                if player is not None:
                    player_boxes[frame_id] = player
                    prev_player_center = center(player["xyxy"])
                if ball is not None:
                    ball_track[frame_id] = np.array(center(ball["xyxy"]))
                    prev_ball_center = tuple(ball_track[frame_id])
                    bx1, by1, bx2, by2 = ball["xyxy"]
                    ball_diams_px.append((
                        frame_id,
                        float(max(bx2 - bx1, by2 - by1)),
                        float(min(bx2 - bx1, by2 - by1)),
                    ))

        chunk: list[tuple[int, np.ndarray]] = []
        for frame_id, frame in reader.iter_frames():
            if frame_id in goal_sample_ids and cfg.detector.goal_model_id:
                goal_samples.append(frame.copy())
            if frame_id % stride != 0:
                continue
            chunk.append((frame_id, frame))
            if len(chunk) >= _DETECT_CHUNK:
                assign_detections(_detect_batch(detector, chunk, None, workers))
                processed += len(chunk)
                chunk = []
                percent = 15 + int((processed / total_stride_frames) * 23)
                _notify(progress, "detection", min(percent, 38),
                        f"Detecting objects — {processed} of {total_stride_frames} sampled frames")
        if chunk:
            assign_detections(_detect_batch(detector, chunk, None, workers))
            chunk = []

        ball_track = interpolate_missing(ball_track, cfg.events.max_interp_gap)
        player_boxes = interpolate_boxes(player_boxes, max_gap=max(stride * 3, 12))
        ball_coverage = sum(item is not None for item in ball_track) / n

        goal_box = detector.detect_goal_robust(goal_samples) if goal_samples else None
        goal_samples.clear()

        # ------------------------------------------------ pass 2: pose (local, cheap)
        _notify(progress, "pose", 40, "Tracking body pose with MediaPipe")
        tracker = PoseTracker()
        poses = []
        try:
            for frame_id, frame in reader.iter_frames():
                box = player_boxes[frame_id]
                poses.append(tracker.process(frame, box["xyxy"] if box else None))
                if frame_id % 20 == 0:
                    percent = 40 + int((frame_id / n) * 18)
                    _notify(progress, "pose", percent, f"Tracking pose — frame {frame_id} of {n}")
        finally:
            tracker.close()

        if sum(pose.ok for pose in poses) < max(5, n // 10):
            raise ShootingTechniquePipelineError(
                "Could not track body pose reliably. Use a side-on view with your full body in frame."
            )

        # ------------------------------------------------ events (coarse attempt)
        _notify(progress, "events", 60, "Detecting contact frame and kick phases")
        events = detect_events(ball_track, poses, cfg.events, cfg.kicking_foot, frame_h=height)

        # --------------------------- targeted parallel backfill around the kick
        needs_backfill = (
            (events is None or ball_coverage < cfg.events.min_ball_coverage)
            and cfg.detector.backend == "hosted"
        )
        if needs_backfill:
            kick_guess = events.contact if events is not None else estimate_kick_frame(ball_track, poses)
            if kick_guess is None:
                kick_guess = n // 2
            half = int(cfg.detector.backfill_window_s * fps)
            lo, hi = max(0, kick_guess - half), min(n, kick_guess + half + 1)
            missing = [i for i in range(lo, hi) if ball_track[i] is None]
            if len(missing) > cfg.detector.backfill_budget:
                # keep the frames closest to the kick estimate
                missing.sort(key=lambda i: abs(i - kick_guess))
                missing = sorted(missing[:cfg.detector.backfill_budget])
            missing_set = set(missing)

            backfill_model = cfg.detector.goal_model_id or None
            label = "goalpost model" if backfill_model else "core model"
            _notify(progress, "events", 62,
                    f"Refining ball track near the kick ({len(missing)} frames, {label})")

            done = 0
            chunk = []
            for frame_id, frame in reader.iter_frames():
                if frame_id >= hi:
                    break
                if frame_id not in missing_set:
                    continue
                chunk.append((frame_id, frame))
                if len(chunk) >= _DETECT_CHUNK:
                    dets = _detect_batch(detector, chunk, backfill_model, workers)
                    _assign_balls(dets, ball_track, cfg, frame_diag, ball_diams_px)
                    done += len(chunk)
                    chunk = []
                    percent = 62 + int((done / max(len(missing), 1)) * 10)
                    _notify(progress, "events", min(percent, 72),
                            f"Refining ball track — {done} of {len(missing)} frames")
            if chunk:
                dets = _detect_batch(detector, chunk, backfill_model, workers)
                _assign_balls(dets, ball_track, cfg, frame_diag, ball_diams_px)

            ball_track = interpolate_missing(ball_track, cfg.events.max_interp_gap)
            ball_coverage = sum(item is not None for item in ball_track) / n
            refined = detect_events(ball_track, poses, cfg.events, cfg.kicking_foot, frame_h=height)
            if refined is not None:
                events = refined

        if events is None:
            pose_coverage = sum(pose.ok for pose in poses) / n
            raise ShootingTechniquePipelineError(
                "Could not locate the contact frame. "
                f"Diagnostics: ball coverage {ball_coverage:.0%}, pose coverage {pose_coverage:.0%}. "
                "Try trimming the clip so the kick is the main action, filming closer and side-on "
                "so the ball is bigger in frame, or using 60 fps to reduce motion blur."
            )

        # ------------------- second-chance goal detection after the shot
        # The goal often only enters frame (or fills it) once the ball is
        # struck and the camera pans. Without a goal box there is no shot
        # distance and no time-of-flight speed, so it's worth a retry.
        if goal_box is None and cfg.detector.goal_model_id and cfg.detector.backend == "hosted":
            retry_ids = sorted({
                min(n - 1, events.contact + int(0.4 * fps)),
                min(n - 1, events.contact + int(0.9 * fps)),
                min(n - 1, events.contact + int(1.5 * fps)),
                n - 1,
            })
            retry_frames = []
            for frame_id in retry_ids:
                frame = reader.read_frame(frame_id)
                if frame is not None:
                    retry_frames.append(frame)
            if retry_frames:
                goal_box = detector.detect_goal_robust(retry_frames)
                if goal_box is not None:
                    print(f"[GameSense] goal found on post-shot retry ({goal_box.get('samples', 1)} samples)")

        # ----------------------- densify contact window with the core model
        lo = max(0, events.contact - 8)
        hi = min(n, events.contact + cfg.events.post_contact_fit_frames + 6)
        missing = [i for i in range(lo, hi) if ball_track[i] is None]
        if missing and cfg.detector.backend == "hosted":
            _notify(progress, "events", 73, "Refining ball track around contact")
            items = []
            for frame_id in missing:
                frame = reader.read_frame(frame_id)
                if frame is not None:
                    items.append((frame_id, frame))
            # Query BOTH models and merge: the COCO checkpoint and the
            # football-specific model miss different frames; their union has
            # markedly better recall on blurred, just-struck balls.
            dets = _detect_batch(detector, items, None, workers)
            if cfg.detector.goal_model_id:
                dets_goal = _detect_batch(detector, items, cfg.detector.goal_model_id, workers)
                for frame_id in list(dets.keys()):
                    dets[frame_id] = (dets.get(frame_id) or []) + (dets_goal.get(frame_id) or [])
            # A struck ball is motion-blurred and scores low confidence; in
            # this small, event-critical window accept weaker detections.
            saved_conf = cfg.detector.conf_ball
            cfg.detector.conf_ball = min(saved_conf, 0.15)
            try:
                _assign_balls(dets, ball_track, cfg, frame_diag, ball_diams_px)
                # Zoom fallback: re-detect on a crop around the predicted ball
                # position for frames that both full-frame passes missed.
                still_missing = [i for i in range(lo, hi) if ball_track[i] is None]
                for frame_id in still_missing:
                    anchor = _nearest_ball(ball_track, frame_id, radius=10)
                    if anchor is None:
                        continue
                    frame = reader.read_frame(frame_id)
                    if frame is None:
                        continue
                    crop_dets = _crop_detect(detector, frame, anchor,
                                             int(0.4 * min(width, height)), cfg)
                    _assign_balls({frame_id: crop_dets}, ball_track, cfg, frame_diag, ball_diams_px)
            finally:
                cfg.detector.conf_ball = saved_conf
            ball_track = interpolate_missing(ball_track, cfg.events.max_interp_gap)
            refined = detect_events(ball_track, poses, cfg.events, cfg.kicking_foot, frame_h=height)
            if refined is not None:
                events = refined

        # ------------------------------------------------ metrics + feedback
        _notify(progress, "feedback", 76, "Computing technique metrics and feedback")
        analysis = compute_metrics(poses, ball_track, events, cfg, fps, goal_box, player_boxes,
                                   frame_w=width, ball_diams_px=ball_diams_px)

        # Power diagnostics: post-contact step sizes in px/frame. If speed
        # looks wrong, this line tells us exactly what the tracker measured.
        c = events.contact
        dbg_steps = []
        for i in range(c, min(n - 1, c + 12)):
            a, b = ball_track[i], ball_track[i + 1]
            dbg_steps.append(round(float(np.linalg.norm(b - a)), 1)
                             if a is not None and b is not None else None)
        print(
            f"[GameSense] power debug: kmh={analysis.shot_speed_kmh} "
            f"scale={analysis.meters_per_px:.6f} m/px ({analysis.scale_source}) "
            f"contact={c} dist_m={analysis.metrics.get('shot_distance_m')} "
            f"goal={'yes' if goal_box is not None else 'NO'} "
            f"post_contact_steps_px={dbg_steps}"
        )

        # ------------------------------------------------ pass 3: render
        _notify(progress, "render", 85, "Rendering annotated video")
        writer = _open_writer(annotated_path, fps, (width, height))
        try:
            annotator = Annotator(cfg, events, analysis, n, fps, (width, height))
            for frame_id, frame in reader.iter_frames():
                writer.write(
                    annotator.draw(frame, frame_id, poses[frame_id], ball_track[frame_id], goal_box)
                )
                if frame_id % 40 == 0:
                    percent = 85 + int((frame_id / n) * 12)
                    _notify(progress, "render", percent, f"Rendering — frame {frame_id} of {n}")
            # slow-motion replay of plant -> just past contact
            for frame_id in range(max(0, events.plant - 3), min(n, events.contact + 12)):
                frame = reader.read_frame(frame_id)
                if frame is None:
                    continue
                replay = annotator.draw(
                    frame, frame_id, poses[frame_id], ball_track[frame_id], goal_box
                )
                cv2.putText(
                    replay,
                    "REPLAY 1/3x",
                    (12, height - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (60, 180, 255),
                    2,
                    cv2.LINE_AA,
                )
                for _ in range(3):
                    writer.write(replay)
        finally:
            writer.release()

        # Transcode to H.264 so the <video> tag actually plays in the browser.
        _notify(progress, "render", 97, "Finalising annotated video")
        _ensure_browser_playable(annotated_path)

        # annotated contact-frame still (skeleton + angle labels at impact)
        contact_url = None
        contact_img = reader.read_frame(events.contact)
        if contact_img is not None:
            still = annotator.draw(
                contact_img, events.contact, poses[events.contact],
                ball_track[events.contact], goal_box,
            )
            if cv2.imwrite(str(output_dir / "contact.jpg"), still):
                contact_url = f"/media/{video_id}/workflow/contact.jpg"
    finally:
        reader.close()

    annotated_url = f"/media/{video_id}/workflow/annotated.mp4"
    feedback = analysis_to_shooting_feedback(
        analysis=analysis,
        ev=events,
        poses=poses,
        fps=fps,
        annotated_video_url=annotated_url,
        contact_frame_url=contact_url,
    )
    _notify(progress, "complete", 100, "Complete")
    return feedback


def _nearest_ball(ball_track, frame_id: int, radius: int = 10) -> Optional[np.ndarray]:
    """Nearest known ball position within +-radius frames (for crop anchoring)."""
    for d in range(1, radius + 1):
        for j in (frame_id - d, frame_id + d):
            if 0 <= j < len(ball_track) and ball_track[j] is not None:
                return ball_track[j]
    return None


def _crop_detect(detector, frame: np.ndarray, center_xy: np.ndarray,
                 crop_size: int, cfg: PipelineConfig) -> list:
    """Detect on a zoomed crop around `center_xy`, mapping boxes back to
    full-frame coordinates. Small/blurred balls that full-frame inference
    misses are usually recovered at crop scale."""
    h, w = frame.shape[:2]
    half = max(crop_size // 2, 64)
    cx, cy = int(center_xy[0]), int(center_xy[1])
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)
    if x2 - x1 < 64 or y2 - y1 < 64:
        return []
    crop = frame[y1:y2, x1:x2]
    dets = detector.detect(crop) or []
    if cfg.detector.goal_model_id:
        dets = dets + (detector.detect_with_model(crop, cfg.detector.goal_model_id) or [])
    out = []
    for d in dets:
        bx1, by1, bx2, by2 = d["xyxy"]
        out.append({**d, "xyxy": (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)})
    return out


def _assign_balls(dets_by_frame: dict[int, list], ball_track, cfg: PipelineConfig,
                  frame_diag: float,
                  diam_out: list | None = None) -> None:
    """Fill ball_track from a batch of detections, walking frames in order and
    using the nearest known ball position as the continuity prior. Records
    (frame_id, max_side, min_side) box sizes when `diam_out` is given — the
    min side survives motion blur and anchors the depth model."""
    from app.services.shooting_analysis.detect import center, pick_ball

    for frame_id in sorted(dets_by_frame):
        prev = None
        for j in range(frame_id - 1, max(-1, frame_id - 6), -1):
            if ball_track[j] is not None:
                prev = tuple(ball_track[j])
                break
        ball = pick_ball(dets_by_frame[frame_id], cfg.detector, prev, frame_diag)
        if ball is not None:
            ball_track[frame_id] = np.array(center(ball["xyxy"]))
            if diam_out is not None:
                bx1, by1, bx2, by2 = ball["xyxy"]
                diam_out.append((
                    frame_id,
                    float(max(bx2 - bx1, by2 - by1)),
                    float(min(bx2 - bx1, by2 - by1)),
                ))
