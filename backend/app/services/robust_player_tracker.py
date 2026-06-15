from __future__ import annotations

import gc
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from math import hypot
from pathlib import Path
from typing import Callable, Iterator, Literal

import cv2
import numpy as np

from app.services.calibration import CalibrationResult, build_track_point
from app.services.metrics import BallTrackPoint, TrackPoint
from app.services.player_identity import PlayerIdentityManager
from app.services.preview_frame import save_preview_frame
from app.services.team_classification import (
    TeamTemplates,
    extract_shirt_lab_sample,
    lab_distance,
    team_label,
)
from app.services.video_streaming import ANALYSIS_SAMPLE_FPS, compute_frame_interval, iter_all_frames

TrackState = Literal["visible", "predicted", "lost"]
YOLO_CADENCE = 8
PREDICTED_MAX_CONSECUTIVE = 45
SHORT_LOSS_SPATIAL_GATE_S = 1.0
KALMAN_PROCESS_NOISE = 20.0
KALMAN_MEASUREMENT_NOISE = 8.0
BACKGROUND_FLOW_POINTS = 30
TRACK_POINT_BUFFER = 50
PREVIEW_EVERY_N_FRAMES = 30

# Sprint definition (FIFA/EPTS convention): sustained running above 25.2 km/h
# (7 m/s). Hysteresis avoids double-counting when speed dips briefly.
SPRINT_ENTER_KMH = 25.2
SPRINT_EXIT_KMH = 20.0
SPRINT_MIN_DURATION_S = 0.7
MAX_SPEED_MEDIAN_WINDOW = 3   # spike-resistant top speed

# Detection prefetch: hosted inference is network-bound, ByteTrack is
# microseconds. Submitting detection for upcoming frames in parallel while
# the tracker consumes results IN ORDER gives a large wall-clock speedup
# with byte-identical tracking output.
DETECT_PARALLEL_WORKERS = 8
PREFETCH_LOOKAHEAD = 24

# Player detection at 15 Hz: ByteTrack keeps IDs perfectly continuous at this
# rate (its lost-track buffer spans seconds) while halving hosted-inference
# calls vs 30 Hz. Set TRACK_DETECT_CADENCE in .env to force a frame cadence,
# or GAMESENSE_DETECT_HZ to change the target detection rate (lower = faster).
try:
    TARGET_DETECT_HZ = float(os.environ.get("GAMESENSE_DETECT_HZ", "").strip() or 15.0)
except (TypeError, ValueError):
    TARGET_DETECT_HZ = 15.0

# Ball detection runs every Nth detection frame; the ball Kalman filter
# predicts through the gaps. Ball positions only feed touch/pass counting in
# max-speed mode, so ~7.5 Hz corrections are plenty. Raise via env to run the
# ball detector less often (fewer inferences) on slower hardware.
try:
    BALL_EVERY_N_DETECTIONS = max(1, int(float(os.environ.get("GAMESENSE_BALL_EVERY_N", "").strip() or 2)))
except (TypeError, ValueError):
    BALL_EVERY_N_DETECTIONS = 2

# Speed/distance integrity gates. No footballer reaches 41 km/h — anything
# above is an ID switch or a calibration jump, never a real sample.
MAX_PLAUSIBLE_SPEED_KMH = 41.0
DISTANCE_DEADBAND_M = 0.05        # ignore sub-noise displacements
DISTANCE_SMOOTH_WINDOW = 3        # moving-average window for distance integration
PITCH_BOUNDS_MARGIN_M = 5.0       # dynamic-calibration points must land near the pitch

# Optical flow (camera-motion estimate + homography chaining) runs on grays
# downscaled to this height: ~4x less CPU per frame at negligible accuracy
# cost, with results scaled back to full-frame coordinates exactly.
FLOW_MAX_HEIGHT = 720

# Appearance re-identification of the target after ByteTrack loses the ID
# (player left frame / long occlusion beyond the track buffer).
REID_MIN_MISSED_DETECTIONS = 3
REID_MAX_LAB_DISTANCE = 30.0
REID_SPATIAL_GATE_HEIGHTS = 3.0
TEMPLATE_EMA = 0.12

# Dynamic pitch calibration: absolute keypoint fixes every KEYPOINT_INTERVAL
# frames, flow-chained in between. Speeds only count between points whose
# calibration is fresher than FRESH_SPEED_MAX_S.
KEYPOINT_INTERVAL = 24
FRESH_SPEED_MAX_S = 1.5


def _detection_cadence(fps: float) -> int:
    env = os.environ.get("TRACK_DETECT_CADENCE", "").strip()
    if env:
        try:
            forced = int(env)
            if forced >= 1:
                return forced
        except ValueError:
            pass
    return max(1, int(round(max(fps, 1.0) / TARGET_DETECT_HZ)))


def _prefetched_frames(
    source_iter,
    pipeline,
    executor: ThreadPoolExecutor,
    lookahead: int = PREFETCH_LOOKAHEAD,
    kp_detector=None,
    detect_ball: bool = True,
) -> Iterator[tuple]:
    """Wrap the frame iterator, submitting player+ball (and periodic pitch
    keypoint) inference futures ahead of consumption.

    `detect_ball=False` skips ball inference entirely — used in max-speed mode,
    which only tracks the player, to save CPU on machines without a GPU."""
    from app.services.video_streaming import compute_frame_interval as _cfi

    state: dict = {"start": None, "cadence": None}

    def maybe_submit(item):
        frame_id, fps, frame, _frame_count = item
        if state["start"] is None:
            state["start"] = frame_id
        if state["cadence"] is None:
            state["cadence"] = min(_detection_cadence(fps), _cfi(fps))
        offset = frame_id - state["start"]
        if offset % state["cadence"] == 0:
            ball_future = None
            if detect_ball and offset % (state["cadence"] * BALL_EVERY_N_DETECTIONS) == 0:
                ball_future = executor.submit(pipeline._detect_ball, frame)
            kp_future = None
            if (
                kp_detector is not None
                and kp_detector.available
                and offset % KEYPOINT_INTERVAL == 0
            ):
                kp_future = executor.submit(kp_detector.homography, frame)
            return (
                executor.submit(pipeline.infer_player_detections, frame),
                ball_future,
                kp_future,
            )
        return None

    buffer: deque = deque()
    iterator = iter(source_iter)
    try:
        while len(buffer) < lookahead:
            item = next(iterator)
            buffer.append((item, maybe_submit(item)))
    except StopIteration:
        pass
    while buffer:
        item, futures = buffer.popleft()
        try:
            nxt = next(iterator)
            buffer.append((nxt, maybe_submit(nxt)))
        except StopIteration:
            pass
        yield (*item, futures)


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
    field_position_samples: list[tuple[float, float]] = field(default_factory=list)
    speed_samples: list[tuple[float, float, float]] = field(default_factory=list)
    yolo_frames: int = 0
    flow_frames: int = 0
    sprint_count: int = 0
    sprint_distance_m: float = 0.0
    player_points: list[TrackPoint] = field(default_factory=list)
    ball_points: list[BallTrackPoint] = field(default_factory=list)
    overlay_frames: dict[int, list[dict]] = field(default_factory=dict)
    calibrated_ratio: float = 0.0
    dynamic_calibration_fixes: int = 0
    rejected_outliers: int = 0


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


def _scale_bboxes(
    bboxes: list[tuple[float, float, float, float]], s: float
) -> list[tuple[float, float, float, float]]:
    if s == 1.0:
        return bboxes
    return [(x * s, y * s, w * s, h * s) for x, y, w, h in bboxes]


def _scaled_interframe_homography(
    prev_gray: np.ndarray | None,
    cur_gray: np.ndarray,
    bboxes_frame: list[tuple[float, float, float, float]],
    s: float,
) -> np.ndarray | None:
    """Inter-frame homography estimated on downscaled grays, converted back
    to full-frame pixel coordinates exactly (similarity conjugation)."""
    from app.services.pitch_keypoints import interframe_homography

    if prev_gray is None:
        return None
    h_small = interframe_homography(prev_gray, cur_gray, _scale_bboxes(bboxes_frame, s))
    if h_small is None or s == 1.0:
        return h_small
    scale_mat = np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    try:
        return np.linalg.inv(scale_mat) @ h_small @ scale_mat
    except np.linalg.LinAlgError:
        return None


def _camera_motion_scaled(
    prev_gray: np.ndarray | None,
    cur_gray: np.ndarray,
    bboxes_frame: list[tuple[float, float, float, float]],
    s: float,
) -> tuple[float, float]:
    dx, dy = _estimate_camera_motion(prev_gray, cur_gray, _scale_bboxes(bboxes_frame, s))
    if s != 1.0 and s > 0:
        return dx / s, dy / s
    return dx, dy


def _compute_sprints(speed_series: list[dict]) -> tuple[int, float]:
    """Sprint segments from the visible-speed series, with hysteresis:
    enter at >= SPRINT_ENTER_KMH, exit below SPRINT_EXIT_KMH, and only count
    segments sustained for SPRINT_MIN_DURATION_S."""
    count = 0
    distance_m = 0.0
    in_sprint = False
    seg_start_t: float | None = None
    seg_dist = 0.0
    prev_t: float | None = None

    def close_segment(end_t: float) -> None:
        nonlocal count, distance_m, seg_dist, seg_start_t
        if seg_start_t is not None and (end_t - seg_start_t) >= SPRINT_MIN_DURATION_S:
            count += 1
            distance_m += seg_dist
        seg_start_t = None
        seg_dist = 0.0

    for entry in speed_series:
        t = float(entry["time_s"])
        v = float(entry["speed_kmh"])
        dt = (t - prev_t) if prev_t is not None else 0.0
        if in_sprint:
            if v < SPRINT_EXIT_KMH:
                close_segment(t)
                in_sprint = False
            else:
                if dt > 0:
                    seg_dist += (v / 3.6) * dt
        elif v >= SPRINT_ENTER_KMH:
            in_sprint = True
            seg_start_t = t
        prev_t = t
    if in_sprint and prev_t is not None:
        close_segment(prev_t)
    return count, round(distance_m, 2)


def _smoothed_max_speed(speed_series: list[dict]) -> float:
    """Top speed as the max of a rolling median — one jittery sample can no
    longer set the headline number."""
    speeds = [float(entry["speed_kmh"]) for entry in speed_series]
    if not speeds:
        return 0.0
    if len(speeds) < MAX_SPEED_MEDIAN_WINDOW:
        return max(speeds)
    best = 0.0
    half = MAX_SPEED_MEDIAN_WINDOW // 2
    for i in range(len(speeds)):
        window = speeds[max(0, i - half):i + half + 1]
        best = max(best, float(np.median(window)))
    return best


# Robust speed from the metric trajectory (CPU-only friendly: pure numpy).
# Window (seconds) over which a local linear fit estimates instantaneous
# velocity. Wider = smoother (less noise), narrower = sharper peaks.
ROBUST_SPEED_WINDOW_S = 0.6


def _hampel_filter_positions(
    track: list[tuple[float, float, float]], n_sigmas: float = 2.5, half_window: int = 3
) -> list[tuple[float, float, float]]:
    """Drop positional outliers (ID switches / calibration jumps) using a
    median-absolute-deviation filter on x and y."""
    if len(track) < 2 * half_window + 1:
        return track
    xs = [p[1] for p in track]
    ys = [p[2] for p in track]
    kept: list[tuple[float, float, float]] = []
    for i in range(len(track)):
        lo = max(0, i - half_window)
        hi = min(len(track), i + half_window + 1)
        mx = float(np.median(xs[lo:hi]))
        my = float(np.median(ys[lo:hi]))
        madx = float(np.median([abs(v - mx) for v in xs[lo:hi]])) or 1e-6
        mady = float(np.median([abs(v - my) for v in ys[lo:hi]])) or 1e-6
        if abs(xs[i] - mx) > n_sigmas * 1.4826 * madx or abs(ys[i] - my) > n_sigmas * 1.4826 * mady:
            continue
        kept.append(track[i])
    return kept


def _robust_speed_from_track(
    track: list[tuple[float, float, float]],
    window_s: float = ROBUST_SPEED_WINDOW_S,
    max_kmh: float = MAX_PLAUSIBLE_SPEED_KMH,
) -> tuple[float, float, list[dict]] | None:
    """Estimate (top_kmh, avg_kmh, speed_series) from a (time, x_m, y_m) track
    via a Hampel-filtered sliding-window linear regression of position.

    Returns None when the track is too short/sparse to trust — callers fall
    back to the legacy 2-point method.
    """
    if not track or len(track) < 4:
        return None
    ordered = sorted(track, key=lambda p: p[0])
    deduped: list[tuple[float, float, float]] = []
    last_t: float | None = None
    for t, x, y in ordered:
        if last_t is None or t > last_t + 1e-6:
            deduped.append((t, x, y))
            last_t = t
    if len(deduped) < 4 or (deduped[-1][0] - deduped[0][0]) < window_s:
        return None
    cleaned = _hampel_filter_positions(deduped)
    if len(cleaned) < 4:
        return None

    ts = np.array([p[0] for p in cleaned])
    xs = np.array([p[1] for p in cleaned])
    ys = np.array([p[2] for p in cleaned])
    half = window_s / 2.0
    speeds: list[float] = []
    series: list[dict] = []
    for i in range(len(cleaned)):
        t0 = ts[i]
        mask = (ts >= t0 - half) & (ts <= t0 + half)
        if int(mask.sum()) < 3:
            continue
        tw = ts[mask]
        vt = tw - tw.mean()
        denom = float((vt * vt).sum())
        if denom < 1e-9:
            continue
        vx = float((vt * (xs[mask] - xs[mask].mean())).sum()) / denom
        vy = float((vt * (ys[mask] - ys[mask].mean())).sum()) / denom
        speed = hypot(vx, vy) * 3.6
        if speed > max_kmh:
            continue
        speeds.append(speed)
        series.append({"time_s": round(float(t0), 2), "speed_kmh": round(speed, 2)})
    if not speeds:
        return None
    # The regression series is already smooth; a tiny rolling median guards
    # against any single residual spike setting the headline number.
    arr = np.asarray(speeds)
    peak = 0.0
    for i in range(len(arr)):
        peak = max(peak, float(np.median(arr[max(0, i - 1):i + 2])))
    return round(peak, 2), round(float(arr.mean()), 2), series


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
    detect_ball: bool = True,
) -> RobustTrackResult:
    stats = TrackerRuntimeStats()
    kalman = _PlayerKalman.create()
    byte_tracker = None   # created on the first frame with the true detection rate
    target_tracker_id: int | None = None
    consecutive_predicted = 0
    in_lost_state = False
    lost_since_time_s: float | None = None
    previous_gray: np.ndarray | None = None
    previous_metric_point: TrackPoint | None = None
    recent_points: deque[TrackPoint] = deque(maxlen=TRACK_POINT_BUFFER)
    speed_series: list[dict] = []
    visible_speeds: list[float] = []
    metric_track: list[tuple[float, float, float]] = []  # (time_s, x_m, y_m) for robust speed
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
    field_position_samples: list[tuple[float, float]] = []
    speed_samples: list[tuple[float, float, float]] = []
    identity_manager = PlayerIdentityManager(team_templates) if team_templates else None
    last_detections: list[tuple[float, float, float, float, float]] = []
    last_ball_box: tuple[float, float, float, float] | None = None
    resolved_total_steps = total_steps
    all_player_points: list[TrackPoint] = []
    ball_points: list[BallTrackPoint] = []
    overlay_key_frames: dict[int, list[dict]] = {}
    target_template: np.ndarray | None = None
    last_target_h = 0.0
    missed_dets = 0
    from app.services.cv_pipeline import BallKalmanTracker
    from app.services.pitch_keypoints import (
        PITCH_LENGTH_M,
        PITCH_WIDTH_M,
        DynamicPitchCalibration,
        PitchKeypointDetector,
        pitch_point,
    )
    ball_kalman = BallKalmanTracker()
    # Manual polygon / goal-post scale: the user's calibration is
    # authoritative — anchor it and follow camera motion via flow chaining
    # (snapping back exactly whenever the camera returns to the setup pose).
    # All other calibration methods use absolute pitch-keypoint fixes chained
    # the same way.
    anchored_manual = (
        calibration.detection_method in ("manual_polygon", "goal_posts")
        and calibration.scale_known
        and calibration.matrix is not None
    )
    # The pitch-bounds plausibility gate only applies to homographies that map
    # into true pitch coordinates; a goal-post scale has no pitch origin.
    dynamic_bounds_gate = calibration.detection_method != "goal_posts"
    kp_detector = None if anchored_manual else PitchKeypointDetector()
    dynamic: DynamicPitchCalibration | None = None
    fresh_points = 0
    total_points = 0
    rejected_outliers = 0
    flow_scale = 1.0
    dist_smooth: deque[tuple[float, float]] = deque(maxlen=DISTANCE_SMOOTH_WINDOW)
    last_dist_pos: tuple[float, float] | None = None
    last_dist_time: float | None = None

    def make_point(point_frame_id: int, point_time_s: float,
                   foot_frame_xy: tuple[float, float], conf: float) -> TrackPoint:
        """Track point with per-frame dynamic homography when available;
        static calibration otherwise. Never guesses: stale dynamic
        calibration yields an uncalibrated point (anchored-manual mode falls
        back to the user's static calibration instead — never worse than the
        pre-dynamic behaviour)."""
        nonlocal fresh_points, total_points
        total_points += 1
        if dynamic is not None and dynamic.has_fix:
            h_now, age_s = dynamic.current()
            if h_now is not None:
                point = TrackPoint(
                    frame_id=point_frame_id,
                    time_s=point_time_s,
                    x_px=foot_frame_xy[0],
                    y_px=foot_frame_xy[1],
                    confidence=max(min(float(conf), 1.0), 0.0),
                )
                xm, ym = pitch_point(h_now, foot_frame_xy)
                if not dynamic_bounds_gate or (
                    -PITCH_BOUNDS_MARGIN_M <= xm <= PITCH_LENGTH_M + PITCH_BOUNDS_MARGIN_M
                    and -PITCH_BOUNDS_MARGIN_M <= ym <= PITCH_WIDTH_M + PITCH_BOUNDS_MARGIN_M
                ):
                    point.x_m, point.y_m, point.calibrated = xm, ym, True
                    point.cal_age_s = float(age_s or 0.0)
                    if point.cal_age_s <= FRESH_SPEED_MAX_S:
                        fresh_points += 1
                else:
                    # Off-pitch mapping = degenerate homography or a bad fix.
                    # Report uncalibrated rather than feed garbage metres in.
                    point.cal_age_s = 99.0
                return point
            if not anchored_manual:
                point = TrackPoint(
                    frame_id=point_frame_id,
                    time_s=point_time_s,
                    x_px=foot_frame_xy[0],
                    y_px=foot_frame_xy[1],
                    confidence=max(min(float(conf), 1.0), 0.0),
                )
                point.cal_age_s = float(age_s) if age_s is not None else 99.0
                return point
        point = build_track_point(
            calibration, point_frame_id, point_time_s,
            to_source_coords(foot_frame_xy[0], foot_frame_xy[1]), conf,
        )
        if point.calibrated:
            fresh_points += 1
        return point

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

    previous_visible_point: TrackPoint | None = None

    def append_metric_point(point: TrackPoint, state: TrackState) -> None:
        nonlocal previous_metric_point, previous_state, previous_visible_point
        nonlocal distance_m, max_speed_kmh, rejected_outliers, last_dist_pos, last_dist_time
        point.track_state = state
        recent_points.append(point)
        if state == "visible":
            all_player_points.append(point)
            stats.visible_frames += 1
        elif state == "predicted":
            stats.predicted_frames += 1

        # Distance: integrated between 3-point moving-average positions.
        # Raw foot positions jitter a few pixels per detection; integrating
        # raw displacements inflates distance for a near-stationary player.
        # Smoothing plus a small deadband keeps real movement and drops the
        # noise; the teleport gate (implied speed beyond human-plausible)
        # rejects ID switches and calibration jumps instead of adding
        # phantom metres. Long gaps are never bridged.
        if point.calibrated and point.x_m is not None and point.y_m is not None:
            if last_dist_time is not None and (point.time_s - last_dist_time) > SHORT_LOSS_SPATIAL_GATE_S:
                dist_smooth.clear()
                last_dist_pos = None
            dist_smooth.append((float(point.x_m), float(point.y_m)))
            smooth_x = sum(p[0] for p in dist_smooth) / len(dist_smooth)
            smooth_y = sum(p[1] for p in dist_smooth) / len(dist_smooth)
            if last_dist_pos is not None and last_dist_time is not None:
                dt = max(point.time_s - last_dist_time, 1e-6)
                displacement = hypot(smooth_x - last_dist_pos[0], smooth_y - last_dist_pos[1])
                implied_kmh = (displacement / dt) * 3.6
                if implied_kmh > MAX_PLAUSIBLE_SPEED_KMH:
                    rejected_outliers += 1
                elif displacement >= DISTANCE_DEADBAND_M:
                    distance_m += displacement
            last_dist_pos = (smooth_x, smooth_y)
            last_dist_time = point.time_s

        # Speed between consecutive VISIBLE observations, even when predicted
        # samples sit in between. (Requiring back-to-back visible samples
        # produced an empty speed series whenever the detection cadence was
        # coarser than the sampling interval — the "max speed 0" bug.)
        if state == "visible" and point.calibrated:
            # Collect freshly-calibrated metric positions for robust speed.
            if (
                point.cal_age_s <= FRESH_SPEED_MAX_S
                and point.x_m is not None
                and point.y_m is not None
            ):
                metric_track.append((float(point.time_s), float(point.x_m), float(point.y_m)))
            if (
                previous_visible_point is not None
                and previous_visible_point.calibrated
                and point.frame_id > previous_visible_point.frame_id
                # Speeds only between FRESHLY calibrated points: a top speed
                # must never come from a drifted or guessed calibration.
                and point.cal_age_s <= FRESH_SPEED_MAX_S
                and previous_visible_point.cal_age_s <= FRESH_SPEED_MAX_S
            ):
                dt_v = (point.frame_id - previous_visible_point.frame_id) / max(source_fps, 1e-6)
                gap_s = dt_v
                if gap_s <= SHORT_LOSS_SPATIAL_GATE_S:        # don't average over long losses
                    disp_v = hypot(
                        float(point.x_m) - float(previous_visible_point.x_m),
                        float(point.y_m) - float(previous_visible_point.y_m),
                    )
                    speed_kmh = (disp_v / max(dt_v, 1e-6)) * 3.6
                    if speed_kmh > MAX_PLAUSIBLE_SPEED_KMH:
                        # Beyond human sprinting — an ID switch or calibration
                        # jump; excluded from speed, average and top speed.
                        rejected_outliers += 1
                    else:
                        visible_speeds.append(speed_kmh)
                        max_speed_kmh = max(max_speed_kmh, speed_kmh)
                        speed_series.append({"time_s": round(point.time_s, 2), "speed_kmh": round(speed_kmh, 2)})
                        if point.x_px and point.y_px:
                            speed_samples.append((point.x_px, point.y_px, speed_kmh))
            previous_visible_point = point

        if point.x_px and point.y_px:
            position_samples.append((point.x_px, point.y_px))
        if state == "visible" and point.calibrated and point.x_m is not None and point.y_m is not None:
            field_position_samples.append((float(point.x_m), float(point.y_m)))
        previous_metric_point = point
        previous_state = state

    detect_executor = ThreadPoolExecutor(max_workers=DETECT_PARALLEL_WORKERS)
    try:
        for frame_id, fps, frame, frame_count, det_futures in _prefetched_frames(
            iter_all_frames(video_path, start_frame_id=start_frame_id),
            pipeline,
            detect_executor,
            kp_detector=kp_detector,
            detect_ball=detect_ball,
        ):
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
            frame_h = frame.shape[0]
            flow_scale = FLOW_MAX_HEIGHT / float(frame_h) if frame_h > FLOW_MAX_HEIGHT else 1.0
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if flow_scale < 1.0:
                current_gray = cv2.resize(
                    current_gray,
                    (max(int(round(frame.shape[1] * flow_scale)), 1), FLOW_MAX_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )
            prev_gray = previous_gray
            previous_gray = current_gray
            offset = frame_id - start_frame_id
            # Dense detection (~15 Hz, never coarser than the sampling
            # interval): keeps every player's ByteTrack ID continuous and the
            # speed series fully populated.
            yolo_cadence = min(_detection_cadence(source_fps), frame_interval)
            is_yolo_frame = offset % yolo_cadence == 0
            is_sample_frame = offset % frame_interval == 0
            if byte_tracker is None:
                byte_tracker = pipeline.create_player_tracker(source_fps / max(yolo_cadence, 1))
            if dynamic is None:
                dynamic = DynamicPitchCalibration(source_fps)
                if anchored_manual:
                    # The manual matrix maps SOURCE-resolution pixels; the
                    # dynamic chain operates on processed-frame pixels.
                    inv_ps = 1.0 / max(processing_scale, 1e-9)
                    to_source_mat = np.array(
                        [[inv_ps, 0.0, 0.0], [0.0, inv_ps, 0.0], [0.0, 0.0, 1.0]],
                        dtype=np.float64,
                    )
                    dynamic.set_anchor(
                        np.asarray(calibration.matrix, dtype=np.float64) @ to_source_mat,
                        (frame.shape[1], frame.shape[0]),
                    )

            # Per-frame homography chain: propagate the last absolute fix
            # through camera motion, then snap to a fresh keypoint fix if one
            # landed on this frame.
            if dynamic is not None and dynamic.has_fix and prev_gray is not None:
                h_inter = _scaled_interframe_homography(
                    prev_gray, current_gray,
                    [item[:4] for item in last_detections],
                    flow_scale,
                )
                dynamic.propagate(h_inter)
            if (
                dynamic is not None
                and det_futures is not None
                and len(det_futures) > 2
                and det_futures[2] is not None
            ):
                h_abs, kp_inliers = det_futures[2].result()
                if h_abs is not None:
                    dynamic.set_absolute(h_abs)
                    if dynamic.fix_count == 1 or dynamic.fix_count % 10 == 0:
                        print(f"[GameSense] pitch keypoint fix #{dynamic.fix_count} "
                              f"({kp_inliers} keypoints) at frame {frame_id}")

            detections: list[tuple[float, float, float, float, float]] = []
            tracked = None
            if is_yolo_frame:
                yolo_frames += 1
                if det_futures is not None:
                    raw_detections = det_futures[0].result()
                else:                                  # safety net; should not happen
                    raw_detections = pipeline.infer_player_detections(frame)
                if len(raw_detections) > 0:
                    tracked = byte_tracker.update_with_detections(raw_detections)
                else:
                    tracked = raw_detections
                detections = pipeline.tracked_to_tuples(tracked)
                last_detections = detections
                # Ball: Kalman-tracked so it has a position at ALL times —
                # detections correct the filter, skipped/missed frames are
                # predicted through.
                ball_dets = None
                if det_futures is not None and det_futures[1] is not None:
                    ball_dets = det_futures[1].result()
                best_det = None
                if ball_dets:
                    best_ball = max(ball_dets, key=lambda item: item[4])
                    last_ball_box = tuple(float(v) for v in best_ball[:4])
                    bx = best_ball[0] + best_ball[2] / 2.0
                    by = best_ball[1] + best_ball[3] / 2.0
                    src_bx, src_by = to_source_coords(bx, by)
                    best_det = (src_bx, src_by, float(best_ball[4]))
                ball_point = ball_kalman.step(frame_id, time_s, calibration, best_det)
                if ball_point is not None:
                    ball_points.append(ball_point)

                # Overlay keyframe: EVERY tracked player, never dropped —
                # unconfirmed teams render neutral instead of disappearing.
                fh, fw = frame.shape[:2]
                entries: list[dict] = []
                if tracked is not None and getattr(tracked, "tracker_id", None) is not None:
                    for det_index, track_id in enumerate(tracked.tracker_id):
                        if det_index >= len(detections):
                            continue
                        x, y, w, h, conf = detections[det_index]
                        team_id: str | None = None
                        color_rgb = (148, 163, 184)
                        if team_templates is not None:
                            team_id, color_rgb = pipeline._classify_player_detection(
                                frame,
                                {"x": x, "y": y, "width": w, "height": h},
                                f"track-{int(track_id)}",
                                team_templates,
                                apply_temporal=True,
                            )
                        entries.append({
                            "id": f"track-{int(track_id)}",
                            "team": team_label(team_id) if team_id in ("team_a", "team_b") else "Unknown",
                            "c": round(float(conf), 4),
                            "b": [
                                round(x / max(fw, 1), 4),
                                round(y / max(fh, 1), 4),
                                round(w / max(fw, 1), 4),
                                round(h / max(fh, 1), 4),
                            ],
                            "color": {"r": int(color_rgb[0]), "g": int(color_rgb[1]), "b": int(color_rgb[2])},
                            "interpolated": False,
                            # Flag the selected player so the UI can highlight it.
                            "is_target": target_tracker_id is not None and int(track_id) == int(target_tracker_id),
                        })
                if entries:
                    overlay_key_frames[frame_id] = entries
            else:
                flow_frames += 1

            if not locked:
                if not is_yolo_frame or tracked is None:
                    continue
                target_tracker_id = pipeline.select_tracker_id(
                    tracked,
                    {"x": float(click["x"]), "y": float(click["y"])},
                    initial_bbox,
                )
                matched = (
                    pipeline.bbox_for_tracker_id(tracked, target_tracker_id)
                    if target_tracker_id is not None
                    else None
                )
                if matched is None:
                    continue
                cx, cy = _bbox_center(matched[:4])
                kalman.correct(cx, cy)
                locked = True
                in_lost_state = False
                consecutive_predicted = 0
                if is_sample_frame:
                    point = make_point(frame_id, time_s, (cx, cy), 0.95)
                    append_metric_point(point, "visible")
                    steps_completed += 1
                    maybe_record_progress(frame_id, frame_count)
                if video_id and frame_id % PREVIEW_EVERY_N_FRAMES == 0 and detections:
                    save_preview_frame(
                        video_id,
                        frame,
                        build_preview_boxes(frame, detections, frame_id, time_s),
                        last_ball_box,
                    )
                continue

            player_bboxes = [item[:4] for item in (detections or last_detections)]

            matched_bbox: tuple[float, float, float, float] | None = None
            matched_confidence = 0.0
            if is_yolo_frame and target_tracker_id is not None and tracked is not None:
                tracked_match = pipeline.bbox_for_tracker_id(tracked, target_tracker_id)
                if tracked_match is not None:
                    matched_bbox = tracked_match[:4]
                    matched_confidence = tracked_match[4]

            # Appearance re-identification: when ByteTrack has lost the target's
            # ID (left frame / long occlusion), find them again by shirt colour
            # + motion-gated position, and adopt the new tracker ID.
            if (
                matched_bbox is None
                and is_yolo_frame
                and target_template is not None
                and tracked is not None
                and getattr(tracked, "tracker_id", None) is not None
                and len(detections) > 0
            ):
                missed_dets += 1
                if missed_dets >= REID_MIN_MISSED_DETECTIONS:
                    pred_x = float(kalman.filter.statePost[0, 0])
                    pred_y = float(kalman.filter.statePost[1, 0])
                    gate = REID_SPATIAL_GATE_HEIGHTS * max(last_target_h, 40.0)
                    best_id, best_cost, best_box = None, float("inf"), None
                    for det_index, track_id in enumerate(tracked.tracker_id):
                        if det_index >= len(detections):
                            continue
                        bx, by, bw, bh, bconf = detections[det_index]
                        spatial = hypot(bx + bw / 2.0 - pred_x, by + bh - pred_y)
                        if spatial > gate:
                            continue
                        sample = extract_shirt_lab_sample(
                            frame, {"x": bx, "y": by, "width": bw, "height": bh}
                        )
                        if sample is None:
                            continue
                        colour_dist = lab_distance(sample, target_template)
                        if colour_dist > REID_MAX_LAB_DISTANCE:
                            continue
                        cost = colour_dist + (spatial / max(gate, 1.0)) * 10.0
                        if cost < best_cost:
                            best_cost, best_id, best_box = cost, int(track_id), (bx, by, bw, bh, bconf)
                    if best_id is not None and best_box is not None:
                        print(f"[GameSense] target re-identified as track {best_id} "
                              f"after {missed_dets} missed detections (cost {best_cost:.1f})")
                        target_tracker_id = best_id
                        matched_bbox = best_box[:4]
                        matched_confidence = best_box[4]

            if matched_bbox is not None:
                cx, cy = _bbox_center(matched_bbox)
                kalman.correct(cx, cy)
                consecutive_predicted = 0
                in_lost_state = False
                lost_since_time_s = None
                missed_dets = 0
                last_target_h = matched_bbox[3]
                sample = extract_shirt_lab_sample(
                    frame,
                    {"x": matched_bbox[0], "y": matched_bbox[1],
                     "width": matched_bbox[2], "height": matched_bbox[3]},
                )
                if sample is not None:
                    if target_template is None:
                        target_template = sample
                    else:
                        target_template = (1.0 - TEMPLATE_EMA) * target_template + TEMPLATE_EMA * sample
                if is_sample_frame:
                    point = make_point(frame_id, time_s, (cx, cy), matched_confidence)
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
                else:
                    # Camera-motion-compensated Kalman prediction. The optical
                    # flow estimate is only computed on these unmatched frames
                    # (it was previously run on every frame even when the
                    # target was directly observed).
                    camera_dx, camera_dy = _camera_motion_scaled(
                        prev_gray, current_gray, player_bboxes, flow_scale
                    )
                    if is_sample_frame:
                        predicted_x, predicted_y = kalman.predict(camera_dx, camera_dy)
                        point = make_point(frame_id, time_s, (predicted_x, predicted_y), 0.4)
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
        detect_executor.shutdown(wait=False, cancel_futures=True)
        gc.collect()

    print(f"[profile] tracking yolo_frames={yolo_frames} flow_frames={flow_frames}")

    if progress_callback:
        stats.tracked_so_far = stats.visible_frames
        stats.predicted_so_far = stats.predicted_frames
        stats.lost_so_far = stats.lost_frames
        progress_callback(steps_completed, resolved_total_steps or 1, last_frame_id, last_frame_count, stats)

    total_frames = stats.visible_frames + stats.predicted_frames + stats.lost_frames
    confidence = stats.visible_frames / max(total_frames, 1)
    # Prefer the robust local-regression estimator; fall back to the legacy
    # 2-point + rolling-median method when the track is too short/sparse.
    robust = _robust_speed_from_track(metric_track)
    if robust is not None:
        headline_max, headline_avg, output_series = robust
    else:
        output_series = speed_series
        headline_max = _smoothed_max_speed(speed_series) if speed_series else max_speed_kmh
        headline_avg = float(np.mean(visible_speeds)) if visible_speeds else 0.0
    sprint_count, sprint_distance_m = _compute_sprints(output_series)

    return RobustTrackResult(
        speed_series=output_series,
        recent_points=list(recent_points),
        stats=stats,
        max_speed_kmh=round(headline_max, 2),
        avg_speed_kmh=round(headline_avg, 2),
        distance_m=round(distance_m, 2),
        confidence_score=round(confidence, 3),
        sampling_fps=ANALYSIS_SAMPLE_FPS,
        source_fps=source_fps,
        frame_interval=frame_interval,
        position_samples=position_samples,
        field_position_samples=field_position_samples,
        speed_samples=speed_samples,
        yolo_frames=yolo_frames,
        flow_frames=flow_frames,
        sprint_count=sprint_count,
        sprint_distance_m=sprint_distance_m,
        player_points=all_player_points,
        ball_points=ball_points,
        overlay_frames=overlay_key_frames,
        calibrated_ratio=round(fresh_points / max(total_points, 1), 3),
        dynamic_calibration_fixes=dynamic.fix_count if dynamic is not None else 0,
        rejected_outliers=rejected_outliers,
    )
