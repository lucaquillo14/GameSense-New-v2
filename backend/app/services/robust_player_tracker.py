from __future__ import annotations

import gc
from collections import deque
from dataclasses import dataclass, field
from math import hypot
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np

from app.services.calibration import CalibrationResult, build_track_point
from app.services.metrics import TrackPoint
from app.services.player_identity import PlayerIdentityManager
from app.services.preview_frame import save_preview_frame
from app.services.team_classification import TeamTemplates
from app.services.video_streaming import ANALYSIS_SAMPLE_FPS, compute_frame_interval, iter_all_frames

TrackState = Literal["visible", "predicted", "lost"]
YOLO_CADENCE = 8
PREDICTED_MAX_CONSECUTIVE = 45
REID_COSINE_THRESHOLD = 0.65
SHORT_LOSS_SPATIAL_GATE_S = 1.0
KALMAN_PROCESS_NOISE = 20.0
KALMAN_MEASUREMENT_NOISE = 8.0
BACKGROUND_FLOW_POINTS = 30
TRACK_POINT_BUFFER = 50
PREVIEW_EVERY_N_FRAMES = 30


@dataclass
class TrackerRuntimeStats:
    visible_frames: int = 0
    predicted_frames: int = 0
    lost_frames: int = 0
    tracked_so_far: int = 0
    predicted_so_far: int = 0
    lost_so_far: int = 0


@dataclass
class RobustTrackResult:
    speed_series: list[dict]
    recent_points: list[TrackPoint]
    stats: TrackerRuntimeStats
    max_speed_kmh: float = 0.0
    avg_speed_kmh: float = 0.0
    distance_m: float = 0.0
    confidence_score: float = 0.0
    sampling_fps: float = ANALYSIS_SAMPLE_FPS
    source_fps: float = 0.0
    frame_interval: int = 1
    position_samples: list[tuple[float, float]] = field(default_factory=list)
    speed_samples: list[tuple[float, float, float]] = field(default_factory=list)
    yolo_frames: int = 0
    flow_frames: int = 0


@dataclass
class _PlayerKalman:
    filter: cv2.KalmanFilter
    initialized: bool = False

    @classmethod
    def create(cls) -> _PlayerKalman:
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        return cls(filter=kf)

    def predict(self, camera_dx: float = 0.0, camera_dy: float = 0.0) -> tuple[float, float]:
        state = self.filter.predict()
        x = float(state[0, 0]) - camera_dx
        y = float(state[1, 0]) - camera_dy
        self.filter.statePre[0, 0] = x
        self.filter.statePre[1, 0] = y
        return x, y

    def correct(self, x: float, y: float) -> None:
        measurement = np.array([[x], [y]], dtype=np.float32)
        if not self.initialized:
            self.filter.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.initialized = True
        self.filter.correct(measurement)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h


def _point_in_bbox(px: float, py: float, bbox: tuple[float, float, float, float]) -> bool:
    x, y, w, h = bbox
    return x <= px <= x + w and y <= py <= y + h


def _estimate_camera_motion(
    previous_gray: np.ndarray | None,
    current_gray: np.ndarray,
    player_bboxes: list[tuple[float, float, float, float]],
) -> tuple[float, float]:
    if previous_gray is None:
        return 0.0, 0.0

    height, width = current_gray.shape[:2]
    rng = np.random.default_rng(42)
    candidates: list[list[float]] = []
    attempts = 0
    while len(candidates) < BACKGROUND_FLOW_POINTS and attempts < BACKGROUND_FLOW_POINTS * 20:
        attempts += 1
        px = float(rng.integers(0, max(width - 1, 1)))
        py = float(rng.integers(0, max(height - 1, 1)))
        if any(_point_in_bbox(px, py, bbox) for bbox in player_bboxes):
            continue
        candidates.append([px, py])

    if not candidates:
        return 0.0, 0.0

    previous_points = np.array(candidates, dtype=np.float32).reshape(-1, 1, 2)
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, previous_points, None)
    if next_points is None or status is None:
        return 0.0, 0.0

    displacements = []
    for index, ok in enumerate(status.flatten()):
        if ok != 1:
            continue
        dx = float(next_points[index, 0, 0] - previous_points[index, 0, 0])
        dy = float(next_points[index, 0, 1] - previous_points[index, 0, 1])
        displacements.append((dx, dy))

    if not displacements:
        return 0.0, 0.0
    dx_values = [item[0] for item in displacements]
    dy_values = [item[1] for item in displacements]
    return float(np.median(dx_values)), float(np.median(dy_values))


def track_player_robust(
    pipeline,
    video_path: Path,
    click: dict,
    calibration: CalibrationResult,
    start_frame_id: int = 0,
    initial_bbox: dict | None = None,
    progress_callback: Callable[[int, int, int, int, TrackerRuntimeStats | None], None] | None = None,
    *,
    video_id: str | None = None,
    team_templates: TeamTemplates | None = None,
    total_steps: int | None = None,
) -> RobustTrackResult:
    stats = TrackerRuntimeStats()
    kalman = _PlayerKalman.create()
    appearance_gallery: np.ndarray | None = None
    consecutive_predicted = 0
    in_lost_state = False
    lost_since_time_s: float | None = None
    previous_gray: np.ndarray | None = None
    previous_metric_point: TrackPoint | None = None
    recent_points: deque[TrackPoint] = deque(maxlen=TRACK_POINT_BUFFER)
    speed_series: list[dict] = []
    visible_speeds: list[float] = []
    max_speed_kmh = 0.0
    distance_m = 0.0
    source_fps = 0.0
    frame_interval = 1
    locked = False
    last_frame_id = start_frame_id
    last_frame_count = 0
    processing_scale = 1.0
    previous_state: TrackState | None = None
    steps_completed = 0
    yolo_frames = 0
    flow_frames = 0
    position_samples: list[tuple[float, float]] = []
    speed_samples: list[tuple[float, float, float]] = []
    identity_manager = PlayerIdentityManager(team_templates) if team_templates else None
    last_detections: list[tuple[float, float, float, float, float]] = []
    last_ball_box: tuple[float, float, float, float] | None = None
    resolved_total_steps = total_steps

    def to_frame_coords(px: float, py: float) -> tuple[float, float]:
        return px * processing_scale, py * processing_scale

    def to_source_coords(px: float, py: float) -> tuple[float, float]:
        if processing_scale <= 0:
            return px, py
        return px / processing_scale, py / processing_scale

    def scaled_bbox(bbox: dict | tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        if isinstance(bbox, dict):
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        else:
            x, y, w, h = bbox
        fx, fy = to_frame_coords(float(x), float(y))
        return fx, fy, float(w) * processing_scale, float(h) * processing_scale

    def maybe_record_progress(frame_id: int, frame_count: int, *, force: bool = False) -> None:
        if not progress_callback:
            return
        offset = max(frame_id - start_frame_id, 0)
        span = max(frame_count - start_frame_id, 1)
        frame_steps = min(resolved_total_steps or 1, max(1, int((offset / span) * (resolved_total_steps or 1))))
        report_steps = max(steps_completed, frame_steps)
        should_report = force or frame_id % 15 == 0 or (steps_completed > 0 and steps_completed % 10 == 0)
        if not should_report:
            return
        stats.tracked_so_far = stats.visible_frames
        stats.predicted_so_far = stats.predicted_frames
        stats.lost_so_far = stats.lost_frames
        progress_callback(report_steps, resolved_total_steps or 1, frame_id, frame_count, stats)

    def build_preview_boxes(frame, detections: list[tuple[float, float, float, float, float]], frame_id: int, time_s: float) -> list[dict]:
        boxes: list[dict] = []
        for index, (x, y, w, h, _confidence) in enumerate(detections):
            bbox_dict = {"x": x, "y": y, "width": w, "height": h}
            team_id, color_rgb = ("team_a", (59, 130, 246))
            if team_templates is not None:
                team_id, color_rgb = pipeline._classify_player_detection(
                    frame,
                    bbox_dict,
                    f"preview-{frame_id}-{index}",
                    team_templates,
                    apply_temporal=False,
                )
            label = f"P{index + 1}"
            if identity_manager is not None:
                stable_id = identity_manager.assign_identity(
                    frame,
                    (x, y, w, h),
                    frame_id,
                    time_s,
                    team=team_id if team_id in {"team_a", "team_b"} else None,
                )
                if stable_id:
                    label = stable_id
            boxes.append({
                "bbox": (x, y, w, h),
                "label": label,
                "color_bgr": (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0])),
            })
        return boxes

    def append_metric_point(point: TrackPoint, state: TrackState) -> None:
        nonlocal previous_metric_point, previous_state, distance_m, max_speed_kmh
        point.track_state = state
        recent_points.append(point)
        if state == "visible":
            stats.visible_frames += 1
        elif state == "predicted":
            stats.predicted_frames += 1

        if previous_metric_point is None:
            previous_metric_point = point
            previous_state = state
            return

        if not point.calibrated or not previous_metric_point.calibrated:
            previous_metric_point = point
            previous_state = state
            return

        dt = max((point.frame_id - previous_metric_point.frame_id) / max(source_fps, 1e-6), 1e-6)
        displacement = hypot(
            float(point.x_m) - float(previous_metric_point.x_m),
            float(point.y_m) - float(previous_metric_point.y_m),
        )
        distance_m += displacement
        speed_kmh = (displacement / dt) * 3.6
        include_speed = state == "visible" and previous_state == "visible"
        if include_speed:
            visible_speeds.append(speed_kmh)
            max_speed_kmh = max(max_speed_kmh, speed_kmh)
            speed_series.append({"time_s": round(point.time_s, 2), "speed_kmh": round(speed_kmh, 2)})
            if point.x_px and point.y_px:
                speed_samples.append((point.x_px, point.y_px, speed_kmh))
        if point.x_px and point.y_px:
            position_samples.append((point.x_px, point.y_px))
        previous_metric_point = point
        previous_state = state

    try:
        for frame_id, fps, frame, frame_count in iter_all_frames(video_path, start_frame_id=start_frame_id):
            source_fps = fps
            last_frame_id = frame_id
            last_frame_count = frame_count
            if frame_interval == 1:
                frame_interval = compute_frame_interval(source_fps)
            if resolved_total_steps is None:
                resolved_total_steps = max(1, (frame_count - start_frame_id + frame_interval - 1) // frame_interval)
            if calibration.frame_width > 0:
                processing_scale = frame.shape[1] / float(calibration.frame_width)
            time_s = frame_id / max(source_fps, 1e-6)
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            offset = frame_id - start_frame_id
            is_yolo_frame = offset % YOLO_CADENCE == 0
            is_sample_frame = offset % frame_interval == 0

            detections: list[tuple[float, float, float, float, float]] = []
            if is_yolo_frame:
                yolo_frames += 1
                for x, y, w, h, confidence in pipeline._detect_people(frame):
                    detections.append((float(x), float(y), float(w), float(h), float(confidence)))
                last_detections = detections
                for ball in pipeline._detect_ball(frame):
                    last_ball_box = tuple(float(v) for v in ball[:4])
            else:
                flow_frames += 1

            if not locked:
                target_bbox = scaled_bbox(initial_bbox) if initial_bbox else None
                if target_bbox is None:
                    click_x, click_y = to_frame_coords(float(click["x"]), float(click["y"]))
                    best = None
                    best_dist = float("inf")
                    for bbox in detections:
                        x, y, w, h, _confidence = bbox
                        if x <= click_x <= x + w and y <= click_y <= y + h:
                            cx, cy = _bbox_center((x, y, w, h))
                            dist = (cx - click_x) ** 2 + (cy - click_y) ** 2
                            if dist < best_dist:
                                best_dist = dist
                                best = bbox
                    target_bbox = best[:4] if best else None
                if target_bbox is None:
                    previous_gray = current_gray
                    continue
                if not is_yolo_frame:
                    previous_gray = current_gray
                    continue
                appearance_gallery = pipeline._appearance_feature(frame, target_bbox)
                cx, cy = _bbox_center(target_bbox)
                kalman.correct(cx, cy)
                locked = True
                in_lost_state = False
                consecutive_predicted = 0
                if is_sample_frame:
                    foot = to_source_coords(cx, cy)
                    point = build_track_point(calibration, frame_id, time_s, foot, 0.95)
                    append_metric_point(point, "visible")
                    steps_completed += 1
                    maybe_record_progress(frame_id, frame_count)
                previous_gray = current_gray
                if video_id and frame_id % PREVIEW_EVERY_N_FRAMES == 0 and detections:
                    save_preview_frame(
                        video_id,
                        frame,
                        build_preview_boxes(frame, detections, frame_id, time_s),
                        last_ball_box,
                    )
                continue

            player_bboxes = [item[:4] for item in (detections or last_detections)]
            camera_dx, camera_dy = _estimate_camera_motion(previous_gray, current_gray, player_bboxes)
            previous_gray = current_gray

            matched_bbox: tuple[float, float, float, float] | None = None
            matched_confidence = 0.0
            best_match_score = -1.0
            if is_yolo_frame and appearance_gallery is not None:
                for x, y, w, h, confidence in detections:
                    feature = pipeline._appearance_feature(frame, (x, y, w, h))
                    if feature is None:
                        continue
                    similarity = _cosine_similarity(appearance_gallery, feature)
                    cx, cy = _bbox_center((x, y, w, h))
                    if in_lost_state:
                        if similarity >= REID_COSINE_THRESHOLD and similarity > best_match_score:
                            best_match_score = similarity
                            matched_bbox = (x, y, w, h)
                            matched_confidence = confidence
                    else:
                        predicted_x, predicted_y = kalman.predict(camera_dx, camera_dy)
                        if lost_since_time_s is not None and (time_s - lost_since_time_s) <= SHORT_LOSS_SPATIAL_GATE_S:
                            max_jump = max(h * 1.5, 60.0)
                            if hypot(cx - predicted_x, cy - predicted_y) > max_jump:
                                continue
                        score = similarity - hypot(cx - predicted_x, cy - predicted_y) / max(h, 1.0) * 0.05
                        if score > best_match_score:
                            best_match_score = score
                            matched_bbox = (x, y, w, h)
                            matched_confidence = confidence

            if matched_bbox is not None:
                cx, cy = _bbox_center(matched_bbox)
                kalman.correct(cx, cy)
                feature = pipeline._appearance_feature(frame, matched_bbox)
                if feature is not None and appearance_gallery is not None:
                    appearance_gallery = pipeline._blend_feature(appearance_gallery, feature, alpha=0.7)
                elif feature is not None:
                    appearance_gallery = feature
                consecutive_predicted = 0
                in_lost_state = False
                lost_since_time_s = None
                if is_sample_frame:
                    foot = to_source_coords(cx, cy)
                    point = build_track_point(calibration, frame_id, time_s, foot, matched_confidence)
                    append_metric_point(point, "visible")
                    steps_completed += 1
                    maybe_record_progress(frame_id, frame_count)
            elif in_lost_state:
                if is_sample_frame:
                    stats.lost_frames += 1
                    steps_completed += 1
                    maybe_record_progress(frame_id, frame_count)
            else:
                consecutive_predicted += 1
                if consecutive_predicted > PREDICTED_MAX_CONSECUTIVE:
                    in_lost_state = True
                    lost_since_time_s = time_s
                    if is_sample_frame:
                        stats.lost_frames += 1
                        steps_completed += 1
                        maybe_record_progress(frame_id, frame_count)
                elif is_sample_frame:
                    predicted_x, predicted_y = kalman.predict(camera_dx, camera_dy)
                    foot = to_source_coords(predicted_x, predicted_y)
                    point = build_track_point(calibration, frame_id, time_s, foot, 0.4)
                    append_metric_point(point, "predicted")
                    steps_completed += 1
                    maybe_record_progress(frame_id, frame_count)
                elif not is_yolo_frame:
                    kalman.predict(camera_dx, camera_dy)

            if video_id and frame_id % PREVIEW_EVERY_N_FRAMES == 0 and is_yolo_frame and last_detections:
                save_preview_frame(
                    video_id,
                    frame,
                    build_preview_boxes(frame, last_detections, frame_id, time_s),
                    last_ball_box,
                )

            maybe_record_progress(frame_id, frame_count)

    finally:
        gc.collect()

    print(f"[profile] tracking yolo_frames={yolo_frames} flow_frames={flow_frames}")

    if progress_callback:
        stats.tracked_so_far = stats.visible_frames
        stats.predicted_so_far = stats.predicted_frames
        stats.lost_so_far = stats.lost_frames
        progress_callback(steps_completed, resolved_total_steps or 1, last_frame_id, last_frame_count, stats)

    total_frames = stats.visible_frames + stats.predicted_frames + stats.lost_frames
    confidence = stats.visible_frames / max(total_frames, 1)
    avg_speed = float(np.mean(visible_speeds)) if visible_speeds else 0.0

    return RobustTrackResult(
        speed_series=speed_series,
        recent_points=list(recent_points),
        stats=stats,
        max_speed_kmh=round(max_speed_kmh, 2),
        avg_speed_kmh=round(avg_speed, 2),
        distance_m=round(distance_m, 2),
        confidence_score=round(confidence, 3),
        sampling_fps=ANALYSIS_SAMPLE_FPS,
        source_fps=source_fps,
        frame_interval=frame_interval,
        position_samples=position_samples,
        speed_samples=speed_samples,
        yolo_frames=yolo_frames,
        flow_frames=flow_frames,
    )
