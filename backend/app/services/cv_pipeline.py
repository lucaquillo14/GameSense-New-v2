from __future__ import annotations

# ── cv_pipeline.py ────────────────────────────────────────────────────────────
# ReIdByteTrackPipeline — improved ID-switch resistance
#
# Root causes of ID switches we fixed:
#
# 1. INITIAL FEATURE TOO WEAK
#    The original code called _selected_feature() once from the raw bbox dict
#    before tracking started. If the bbox was slightly off the player, the
#    feature histogram was wrong from frame 0 and never recovered.
#    FIX: collect appearance samples from the first N frames after the target
#    is confirmed and build a richer baseline feature.
#
# 2. RE-ID THRESHOLD TOO PERMISSIVE
#    _reidentify_candidate() accepted any candidate scoring below 0.58.
#    With crowded scenes, a nearby opponent often scored under that threshold.
#    FIX: tighten to 0.42 and add a motion gate — if the candidate is further
#    than 2× the player's own height from the last known position, reject it.
#
# 3. FEATURE BLEND ALPHA TOO HIGH (0.85)
#    At 0.85 the feature barely updates. After 60 frames of occlusion or
#    kit-colour variation the stored feature diverged from reality.
#    FIX: use 0.72 normally, but when confidence is high (clear detection)
#    lower to 0.55 so the feature stays fresh.
#
# 4. NO GRACE PERIOD AFTER PAUSE
#    When the user pauses on a clean frame and then clicks, the tracker was
#    seeded from that exact paused frame. ByteTrack had not yet seen any
#    motion so IDs were assigned arbitrarily.
#    FIX: run a short warm-up pass (WARMUP_FRAMES) before frame start_frame_id
#    to let ByteTrack settle its ID assignments before we lock onto the target.
#
# 5. _choose_detection_near MAX_DISTANCE TOO GENEROUS
#    max_distance = max(w*2.5, h*1.5, 90) let the tracker jump to a player
#    1.5 bboxes away. In crowded midfield that's often someone else.
#    FIX: tighten to max(w*1.5, h*1.2, 60).
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path
from threading import Lock
from typing import Callable

import cv2
import numpy as np

_pipeline_lock = Lock()
_shared_pipeline = None
_detector = None
_detector_lock = Lock()

from app.services.calibration import CalibrationResult, build_track_point, is_pixel_in_calibrated_region, pixel_to_field
from app.services.metrics import BallTrackPoint, TrackPoint
from app.services.player_identity import PlayerIdentityManager
from app.services.team_classification import (
    TeamClassificationService,
    TeamTemplates,
    build_team_templates,
    detect_team_conflict,
    team_label,
)

NEUTRAL_TEAM_RGB = (100, 116, 139)

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# How many frames to run before start_frame_id to warm up ByteTrack IDs
WARMUP_FRAMES = 45

# Re-ID acceptance threshold (lower = stricter)
REID_THRESHOLD = 0.42

# Minimum frames to collect for initial feature baseline
FEATURE_WARMUP_SAMPLES = 6

# Ball detection and Kalman smoothing
BALL_YOLO_CONFIDENCE = 0.18
BALL_KALMAN_PROCESS_NOISE = 25.0
BALL_KALMAN_MEASUREMENT_NOISE = 10.0
BALL_INTERPOLATED_CONFIDENCE = 0.25
BALL_MAX_CONSECUTIVE_MISSES = 15
FRAME_EDGE_MARGIN_RATIO = 0.04
EDGE_REENTRY_WINDOW_S = 2.0


def _as_calibration(
    calibration: CalibrationResult | np.ndarray,
    frame_width: int = 0,
    frame_height: int = 0,
) -> CalibrationResult:
    if isinstance(calibration, CalibrationResult):
        return calibration
    return CalibrationResult(
        matrix=calibration,
        scale_known=True,
        frame_width=frame_width,
        frame_height=frame_height,
        units="metric",
    )


class BallKalmanTracker:
    """OpenCV Kalman smoother for continuous ball trajectory (state: x, y, dx, dy)."""

    def __init__(self) -> None:
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.float32,
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * BALL_KALMAN_PROCESS_NOISE
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * BALL_KALMAN_MEASUREMENT_NOISE
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False
        self.consecutive_misses = 0

    def predicted_position(self) -> tuple[float, float] | None:
        if not self.initialized:
            return None
        state = self.kf.statePost
        return (
            _kalman_state_value(state, 0) + _kalman_state_value(state, 2),
            _kalman_state_value(state, 1) + _kalman_state_value(state, 3),
        )

    def reset(self) -> None:
        self.initialized = False
        self.consecutive_misses = 0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    def step(
        self,
        frame_id: int,
        time_s: float,
        calibration: CalibrationResult,
        detection: tuple[float, float, float] | None,
    ) -> BallTrackPoint | None:
        calibration = _as_calibration(calibration)
        interpolated = False
        if detection is not None:
            cx, cy, det_conf = detection
            measurement = np.array([[cx], [cy]], dtype=np.float32)
            if not self.initialized:
                self.kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                self.initialized = True
            self.kf.predict()
            self.kf.correct(measurement)
            confidence = max(min(float(det_conf), 1.0), 0.0)
            self.consecutive_misses = 0
        else:
            if not self.initialized:
                return None
            self.consecutive_misses += 1
            if self.consecutive_misses > BALL_MAX_CONSECUTIVE_MISSES:
                self.reset()
                return None
            self.kf.predict()
            confidence = BALL_INTERPOLATED_CONFIDENCE
            interpolated = True

        x_px = _kalman_state_value(self.kf.statePost, 0)
        y_px = _kalman_state_value(self.kf.statePost, 1)
        calibrated = False
        x_m = 0.0
        y_m = 0.0
        if calibration.scale_known and calibration.matrix is not None:
            if is_pixel_in_calibrated_region(
                x_px,
                y_px,
                calibration.region_polygon_px,
                calibration.frame_width,
                calibration.frame_height,
            ):
                x_m, y_m = pixel_to_field(calibration.matrix, (x_px, y_px))
                calibrated = True
        return BallTrackPoint(
            frame_id=frame_id,
            time_s=time_s,
            x_px=x_px,
            y_px=y_px,
            x_m=x_m,
            y_m=y_m,
            calibrated=calibrated,
            confidence=confidence,
            interpolated=interpolated,
        )


def get_detector():
    global _detector
    if _detector is None and YOLO:
        with _detector_lock:
            if _detector is None:
                _detector = YOLO("yolov8n.pt")
    return _detector


class ClassicalCvPipeline:
    """Lightweight fallback pipeline — CSRT tracker + YOLO detections."""

    def __init__(self, max_width: int = 1280, sample_fps: float = 8.0):
        self.max_width  = max_width
        self.sample_fps = sample_fps
        self.detector   = get_detector()
        self.team_service = TeamClassificationService()
        self._last_video_path: Path | None = None

    def track_target(
        self,
        video_path: Path,
        click: dict,
        calibration: CalibrationResult | np.ndarray,
        output_video_path: Path,
        start_frame_id: int = 0,
        initial_bbox: dict | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[TrackPoint]:
        capture    = cv2.VideoCapture(str(video_path))
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(int(round(source_fps / self.sample_fps)), 1)
        frame_count    = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 0:
            start_frame_id = min(max(start_frame_id, 0), frame_count - 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
        ok, selected_frame_raw = capture.read()
        capture.release()
        if not ok:
            return []

        selected_frame, scale = self._resize(selected_frame_raw)
        click_x    = int(click["x"] * scale)
        click_y    = int(click["y"] * scale)
        initial_box = self._scale_bbox(initial_bbox, scale) if initial_bbox else None
        if not initial_box:
            initial_box = (
                self._choose_detection_for_click(selected_frame, click_x, click_y)
                or self._initial_box(selected_frame, click_x, click_y)
            )

        calibration_result = _as_calibration(calibration, selected_frame_raw.shape[1], selected_frame_raw.shape[0])
        samples: dict[int, tuple[TrackPoint, tuple[float, float, float, float]]] = {}
        self._track_forward(
            video_path, start_frame_id, selected_frame, initial_box,
            frame_interval, source_fps, calibration_result, scale,
            samples, frame_count, progress_callback,
        )
        self._track_backward(
            video_path, start_frame_id, selected_frame, initial_box,
            frame_interval, source_fps, calibration_result, scale, samples,
        )
        self._write_overlay_video(video_path, output_video_path, samples, frame_interval, source_fps)
        return [sample[0] for _, sample in sorted(samples.items())]

    def calibrate_team_templates(self, video_path: Path) -> TeamTemplates:
        self._last_video_path = video_path
        templates = build_team_templates(
            video_path,
            self._detect_people,
            self._resize,
            self._unscale_bbox,
        )
        self.team_service.set_templates(templates)
        return templates

    def _classify_player_detection(
        self,
        frame: np.ndarray,
        bbox: dict,
        player_key: str,
        team_templates: TeamTemplates,
        *,
        apply_temporal: bool,
    ) -> tuple[str | None, tuple[int, int, int]]:
        confirmed, _sample = self.team_service.classify_player(
            frame,
            bbox,
            player_key,
            apply_temporal=apply_temporal,
        )
        if confirmed == "referee" or confirmed == "unconfirmed":
            return None, NEUTRAL_TEAM_RGB
        color_rgb = (
            team_templates.team_a_color_rgb
            if confirmed == "team_a"
            else team_templates.team_b_color_rgb
        )
        return confirmed, color_rgb

    def _maybe_rebuild_team_templates(self, frame_counts: dict[str, int]) -> TeamTemplates | None:
        if not detect_team_conflict({"team_a": frame_counts.get("team_a", 0), "team_b": frame_counts.get("team_b", 0)}):
            return None
        if not self._last_video_path:
            return None
        rebuilt = self.calibrate_team_templates(self._last_video_path)
        rebuilt.warnings.append("Team templates were rebuilt after a same-team conflict was detected.")
        return rebuilt

    def detect_frame_objects(
        self,
        video_path: Path,
        frame_id: int,
        team_templates: TeamTemplates | None = None,
        assign_player_ids: bool = False,
    ) -> list[dict]:
        if assign_player_ids and team_templates is not None and self.detector:
            return self._detect_frame_with_identities(video_path, frame_id, team_templates)

        capture = cv2.VideoCapture(str(video_path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 0:
            frame_id = min(max(frame_id, 0), frame_count - 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            return []

        resized, scale = self._resize(frame)
        if team_templates is not None:
            self._last_video_path = video_path
            self.team_service.set_templates(team_templates)

        pending_players: list[tuple[dict, float, str]] = []
        player_index = 0
        for bbox in self._detect_people(resized):
            x, y, w, h, confidence = bbox
            original_bbox = self._unscale_bbox((x, y, w, h), scale)
            player_key = f"frame-{frame_id}-player-{player_index}"
            pending_players.append((original_bbox, float(confidence), player_key))
            player_index += 1

        if team_templates is not None and pending_players:
            confirmed_teams: list[str] = []
            for original_bbox, _confidence, player_key in pending_players:
                team_id, _color = self._classify_player_detection(
                    frame,
                    original_bbox,
                    player_key,
                    team_templates,
                    apply_temporal=False,
                )
                if team_id in {"team_a", "team_b"}:
                    confirmed_teams.append(team_id)
            counts = {"team_a": confirmed_teams.count("team_a"), "team_b": confirmed_teams.count("team_b")}
            rebuilt = self._maybe_rebuild_team_templates(counts)
            if rebuilt is not None:
                team_templates = rebuilt
                self.team_service.set_templates(team_templates)

        detections: list[dict] = []
        output_index = 0
        for original_bbox, confidence, player_key in pending_players:
            if team_templates is not None:
                team_id, color_rgb = self._classify_player_detection(
                    frame,
                    original_bbox,
                    player_key,
                    team_templates,
                    apply_temporal=False,
                )
                if team_id is None:
                    detections.append({
                        "id": f"player-{frame_id}-{output_index}",
                        "label": "player",
                        "confidence": round(confidence, 3),
                        "bbox": original_bbox,
                        "team_color": {"r": 148, "g": 163, "b": 184},
                    })
                else:
                    detections.append({
                        "id": f"player-{frame_id}-{output_index}",
                        "label": team_id,
                        "team": team_id,
                        "team_label": team_label(team_id),  # type: ignore[arg-type]
                        "team_color": {"r": color_rgb[0], "g": color_rgb[1], "b": color_rgb[2]},
                        "confidence": round(confidence, 3),
                        "bbox": original_bbox,
                    })
            else:
                detections.append({
                    "id": f"player-{frame_id}-{output_index}",
                    "label": "player",
                    "confidence": round(confidence, 3),
                    "bbox": original_bbox,
                })
            output_index += 1

        for index, bbox in enumerate(self._detect_ball(resized)):
            x, y, w, h, confidence = bbox
            detections.append({
                "id": f"ball-{frame_id}-{index}",
                "label": "ball",
                "confidence": round(float(confidence), 3),
                "bbox": self._unscale_bbox((x, y, w, h), scale),
            })
        return detections

    def _detect_frame_with_identities(
        self,
        video_path: Path,
        frame_id: int,
        team_templates: TeamTemplates,
    ) -> list[dict]:
        cap = cv2.VideoCapture(str(video_path))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if frame_count > 0:
            frame_id = min(max(frame_id, 0), frame_count - 1)

        identity_manager = PlayerIdentityManager(team_templates)
        frame_assignments: dict[int, list[dict]] = {}

        stream = self.detector.track(
            source=str(video_path),
            stream=True,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=0.25,
            iou=0.45,
            imgsz=960,
            verbose=False,
        )

        for current_frame_id, result in enumerate(stream):
            if current_frame_id > frame_id:
                break

            frame = result.orig_img
            candidates = self._filter_team_players(frame, self._tracked_candidates(result), team_templates)
            frame_players: list[dict] = []
            for index, candidate in enumerate(candidates):
                bbox = candidate["bbox"]
                stable_id = identity_manager.assign_identity(
                    frame,
                    bbox,
                    current_frame_id,
                    current_frame_id / source_fps,
                    int(candidate["track_id"]),
                    candidate.get("team"),
                )
                if stable_id is None:
                    continue
                team_id = candidate.get("team") or identity_manager.gallery[stable_id].team
                color_rgb = (
                    team_templates.team_a_color_rgb
                    if team_id == "team_a"
                    else team_templates.team_b_color_rgb
                )
                frame_players.append({
                    "id": f"{stable_id}-{current_frame_id}-{index}",
                    "label": team_id,
                    "team": team_id,
                    "team_label": team_label(team_id),
                    "team_color": {"r": color_rgb[0], "g": color_rgb[1], "b": color_rgb[2]},
                    "player_id": stable_id,
                    "confidence": round(float(candidate["confidence"]), 3),
                    "bbox": {
                        "x": round(bbox[0], 2),
                        "y": round(bbox[1], 2),
                        "width": round(bbox[2], 2),
                        "height": round(bbox[3], 2),
                    },
                })
            frame_assignments[current_frame_id] = frame_players

        if frame_id in frame_assignments:
            return frame_assignments[frame_id] + self._detect_ball_on_frame(video_path, frame_id)
        return self.detect_frame_objects(video_path, frame_id, team_templates, assign_player_ids=False)

    def _detect_ball_on_frame(self, video_path: Path, frame_id: int) -> list[dict]:
        capture = cv2.VideoCapture(str(video_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            return []
        resized, scale = self._resize(frame)
        detections: list[dict] = []
        for index, bbox in enumerate(self._detect_ball(resized)):
            x, y, w, h, confidence = bbox
            detections.append({
                "id": f"ball-{frame_id}-{index}",
                "label": "ball",
                "confidence": round(float(confidence), 3),
                "bbox": self._unscale_bbox((x, y, w, h), scale),
            })
        return detections

    # ── internal tracking helpers ────────────────────────────────────────────

    def _track_forward(self, video_path, start_frame_id, selected_frame, initial_box,
                       frame_interval, source_fps, calibration, scale,
                       samples, frame_count, progress_callback):
        tracker = self._create_tracker()
        tracker.init(selected_frame, initial_box)
        capture = cv2.VideoCapture(str(video_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
        frame_id = start_frame_id
        ok, frame = capture.read()
        current_bbox = tuple(float(v) for v in initial_box)
        while ok:
            if (frame_id - start_frame_id) % frame_interval == 0:
                resized, _ = self._resize(frame)
                detected = self._choose_detection_near(resized, current_bbox)
                if detected:
                    bbox, confidence = detected[:4], float(detected[4]) if len(detected) > 4 else 0.8
                    tracker = self._create_tracker()
                    tracker.init(resized, _int_bbox(bbox))
                    tracked = True
                else:
                    tracked, bbox = tracker.update(resized)
                    confidence = 0.45
                if tracked:
                    current_bbox = tuple(float(v) for v in bbox)
                    self._record_sample(frame_id, bbox, source_fps, calibration, scale, samples, confidence)
                    if progress_callback and len(samples) % 20 == 0:
                        progress_callback(frame_id, frame_count)
            ok, frame = capture.read()
            frame_id += 1
        capture.release()

    def _track_backward(self, video_path, start_frame_id, selected_frame, initial_box,
                        frame_interval, source_fps, calibration, scale, samples):
        tracker = self._create_tracker()
        tracker.init(selected_frame, initial_box)
        capture = cv2.VideoCapture(str(video_path))
        current_bbox = tuple(float(v) for v in initial_box)
        for frame_id in range(start_frame_id, -1, -frame_interval):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = capture.read()
            if not ok:
                continue
            resized, _ = self._resize(frame)
            detected = self._choose_detection_near(resized, current_bbox)
            if detected:
                bbox, confidence = detected[:4], float(detected[4]) if len(detected) > 4 else 0.8
                tracker = self._create_tracker()
                tracker.init(resized, _int_bbox(bbox))
                tracked = True
            else:
                tracked, bbox = tracker.update(resized)
                confidence = 0.4
            if tracked:
                current_bbox = tuple(float(v) for v in bbox)
                self._record_sample(frame_id, bbox, source_fps, calibration, scale, samples, confidence)
        capture.release()

    def _record_sample(self, frame_id, bbox, source_fps, calibration, scale, samples, confidence):
        x, y, w, h = [float(v) for v in bbox]
        foot_px = ((x + w / 2.0) / scale, (y + h) / scale)
        point = build_track_point(
            calibration,
            frame_id,
            frame_id / source_fps,
            foot_px,
            confidence,
        )
        samples[frame_id] = (point, (x, y, w, h))

    def _write_overlay_video(self, video_path, output_video_path, samples, frame_interval, source_fps):
        capture = cv2.VideoCapture(str(video_path))
        ok, frame = capture.read()
        if not ok:
            capture.release(); return
        frame, _ = self._resize(frame)
        writer = self._writer(output_video_path, frame, min(source_fps, self.sample_fps))
        frame_id = 0
        sampled = set(samples)
        while ok:
            if frame_id in sampled:
                frame, _ = self._resize(frame)
                s = samples.get(frame_id)
                if s:
                    self._draw_target(frame, s[1])
                writer.write(frame)
            ok, frame = capture.read()
            frame_id += 1
        capture.release(); writer.release()

    def _draw_target(self, frame, bbox):
        x, y, w, h = bbox
        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (27,164,255), 2)
        cv2.circle(frame, (int(x+w/2), int(y+h)), 5, (0,255,160), -1)
        cv2.putText(frame, "target", (int(x), max(int(y)-8,16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (27,164,255), 2)

    def _resize(self, frame):
        h, w = frame.shape[:2]
        if w <= self.max_width:
            return frame, 1.0
        scale  = self.max_width / w
        return cv2.resize(frame, (self.max_width, int(h * scale))), scale

    def _initial_box(self, frame, x, y):
        h, w = frame.shape[:2]
        bw = max(w // 32, 32); bh = max(h // 8, 80)
        left = min(max(x - bw//2, 0), w - bw)
        top  = min(max(y - bh//2, 0), h - bh)
        return left, top, bw, bh

    def _detect_people(self, frame):
        if not self.detector:
            return self._detect_people_hog(frame)
        results = self.detector.predict(frame, classes=[0], conf=0.25, imgsz=640, verbose=False)
        out = []
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                x1,y1,x2,y2 = [float(v) for v in box.xyxy[0].tolist()]
                out.append((x1, y1, x2-x1, y2-y1, float(box.conf[0])))
        return out

    def _detect_ball(self, frame):
        out = []
        if self.detector:
            results = self.detector.predict(
                frame,
                classes=[32],
                conf=BALL_YOLO_CONFIDENCE,
                imgsz=640,
                verbose=False,
            )
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    out.append((x1, y1, x2 - x1, y2 - y1, float(box.conf[0])))
            if out:
                return out
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        frame_height = frame.shape[0]
        max_radius = max(6, frame_height // 40)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            1.2,
            24,
            param1=80,
            param2=20,
            minRadius=6,
            maxRadius=max_radius,
        )
        if circles is None:
            return []
        for c in np.round(circles[0, :8]).astype("int"):
            x, y, r = c
            out.append((float(x - r), float(y - r), float(r * 2), float(r * 2), 0.32))
        return out

    def _best_ball_detection(
        self,
        frame: np.ndarray,
        predicted_position: tuple[float, float] | None = None,
    ) -> tuple[float, float, float] | None:
        detections = self._detect_ball(frame)
        if not detections:
            return None
        if predicted_position is None:
            best = max(detections, key=lambda item: item[4])
        else:
            pred_x, pred_y = predicted_position
            best = min(
                detections,
                key=lambda item: _center_distance_sq(item, (pred_x, pred_y)) - item[4] * 20.0,
            )
        x, y, w, h, confidence = best
        return (x + w / 2.0, y + h / 2.0, float(confidence))

    def _detect_people_hog(self, frame):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, weights = hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.05)
        return [(float(x),float(y),float(w),float(h), float(max(min(wt,1.0),0.2)))
                for (x,y,w,h),wt in zip(boxes, weights)]

    def _choose_detection_for_click(self, frame, x, y):
        dets = self._detect_people(frame)
        if not dets: return None
        containing = [d for d in dets if d[0]<=x<=d[0]+d[2] and d[1]<=y<=d[1]+d[3]]
        candidates = containing or dets
        best = min(candidates, key=lambda d: _center_distance_sq(d,(x,y)) - d[4]*500.0)
        return _int_bbox(best)

    def _choose_detection_near(self, frame, prev_bbox):
        dets = self._detect_people(frame)
        if not dets: return None
        pc = (prev_bbox[0]+prev_bbox[2]/2.0, prev_bbox[1]+prev_bbox[3]/2.0)
        # FIX 5: tightened from max(w*2.5, h*1.5, 90) → max(w*1.5, h*1.2, 60)
        max_dist = max(prev_bbox[2]*1.5, prev_bbox[3]*1.2, 60.0)
        ranked = sorted(dets, key=lambda d: _center_distance_sq(d, pc) - d[4]*300.0)
        best = ranked[0]
        if _center_distance_sq(best, pc)**0.5 > max_dist:
            return None
        return best

    def _scale_bbox(self, bbox, scale):
        if not bbox: return None
        return _int_bbox((bbox["x"]*scale, bbox["y"]*scale,
                          bbox["width"]*scale, bbox["height"]*scale))

    def _unscale_bbox(self, bbox, scale):
        safe = scale if scale > 0 else 1.0
        x,y,w,h = bbox
        return {"x":round(x/safe,2),"y":round(y/safe,2),
                "width":round(w/safe,2),"height":round(h/safe,2)}

    def _create_tracker(self):
        for name in ("legacy.TrackerCSRT_create","TrackerCSRT_create",
                     "TrackerKCF_create","TrackerMIL_create"):
            parts = name.split(".")
            obj = cv2
            for p in parts:
                obj = getattr(obj, p, None)
                if obj is None: break
            if obj is not None:
                return obj()
        return TemplateMatchingTracker()

    def _writer(self, output_video_path, frame, fps):
        h, w = frame.shape[:2]
        return cv2.VideoWriter(str(output_video_path),
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               max(fps,1.0), (w,h))

    def _filter_team_players(
        self,
        frame: np.ndarray,
        candidates: list[dict],
        team_templates: TeamTemplates | None,
    ) -> list[dict]:
        if team_templates is None:
            return candidates

        self.team_service.set_templates(team_templates)

        def classify_candidates() -> list[tuple[dict, str | None]]:
            classified: list[tuple[dict, str | None]] = []
            for candidate in candidates:
                bbox = candidate["bbox"]
                bbox_dict = {
                    "x": bbox[0],
                    "y": bbox[1],
                    "width": bbox[2],
                    "height": bbox[3],
                }
                player_key = str(candidate.get("track_id", candidate.get("player_id", "unknown")))
                team_id, _color = self._classify_player_detection(
                    frame,
                    bbox_dict,
                    player_key,
                    team_templates,
                    apply_temporal=True,
                )
                classified.append((candidate, team_id))
            return classified

        classified = classify_candidates()
        team_counts = {
            "team_a": sum(1 for _, team in classified if team == "team_a"),
            "team_b": sum(1 for _, team in classified if team == "team_b"),
        }
        rebuilt = self._maybe_rebuild_team_templates(team_counts)
        if rebuilt is not None:
            team_templates = rebuilt
            self.team_service.set_templates(team_templates)
            classified = classify_candidates()

        filtered: list[dict] = []
        for candidate, team_id in classified:
            if team_id not in {"team_a", "team_b"}:
                continue
            filtered.append({**candidate, "team": team_id})
        return filtered


# ── ReIdByteTrackPipeline ────────────────────────────────────────────────────

class ReIdByteTrackPipeline(ClassicalCvPipeline):
    """
    YOLO + ByteTrack with appearance-based Re-ID and improved ID-switch resistance.
    See module docstring for the list of fixes applied.
    """

    def __init__(self, max_width: int = 1280, sample_fps: float = 8.0):
        super().__init__(max_width=max_width, sample_fps=sample_fps)

    def track_target(
        self,
        video_path: Path,
        click: dict,
        calibration: CalibrationResult | np.ndarray,
        start_frame_id: int = 0,
        initial_bbox: dict | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        team_templates: TeamTemplates | None = None,
    ) -> list[TrackPoint]:
        if not self.detector:
            fallback = video_path.with_name("classical-fallback-overlay.mp4")
            return super().track_target(video_path, click, calibration,
                                        fallback, start_frame_id, initial_bbox,
                                        progress_callback)

        cap = cv2.VideoCapture(str(video_path))
        source_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if frame_count > 0:
            start_frame_id = min(max(start_frame_id, 0), frame_count - 1)
        calibration_result = _as_calibration(calibration, frame_width, frame_height)

        # FIX 1: build a richer initial feature from multiple frames
        target_feature = self._build_initial_feature(video_path, start_frame_id, initial_bbox)
        target_track_id: int | None = None
        last_bbox: tuple[float,float,float,float] | None = self._bbox_tuple(initial_bbox)

        # FIX 4: warm-up start — let ByteTrack settle before start_frame_id
        warmup_start = max(0, start_frame_id - WARMUP_FRAMES)

        pending_samples: dict[int, list[TrackPoint]] = {}
        target_samples:  dict[int, TrackPoint]       = {}
        id_switch_recoveries = 0
        identity_manager = PlayerIdentityManager(team_templates) if team_templates else None
        target_stable_id: str | None = None

        stream = self.detector.track(
            source=str(video_path),
            stream=True,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=0.25,      # slightly higher than original (was 0.2)
            iou=0.45,       # tightened (was 0.5)
            imgsz=960,
            verbose=False,
        )

        for frame_id, result in enumerate(stream):
            frame      = result.orig_img
            candidates = self._filter_team_players(
                frame,
                self._tracked_candidates(result),
                team_templates,
            )
            if progress_callback and frame_id % 30 == 0:
                progress_callback(frame_id, frame_count)

            # Accumulate all candidate histories
            for c in candidates:
                tid = int(c["track_id"])
                pending_samples.setdefault(tid, []).append(
                    self._track_point_from_bbox(frame_id, source_fps,
                                                c["bbox"], calibration_result,
                                                c["confidence"])
                )

            if identity_manager is not None:
                for candidate in candidates:
                    stable_id = identity_manager.assign_identity(
                        frame,
                        candidate["bbox"],
                        frame_id,
                        frame_id / source_fps,
                        int(candidate["track_id"]),
                        candidate.get("team"),
                    )
                    if stable_id:
                        candidate["player_id"] = stable_id

            if not candidates:
                continue

            # ── Lock-on phase (only after warmup + start frame) ───────────
            if target_track_id is None and frame_id >= start_frame_id:
                selected = self._select_initial_candidate(
                    candidates, click, initial_bbox, target_feature, frame)
                if selected:
                    target_track_id = int(selected["track_id"])
                    if identity_manager is not None:
                        target_stable_id = identity_manager.register_immediate(
                            frame,
                            selected["bbox"],
                            frame_id,
                            frame_id / source_fps,
                            target_track_id,
                            selected.get("team"),
                        )
                    target_samples.update(
                        {s.frame_id: s for s in pending_samples.get(target_track_id, [])}
                    )
                    target_feature = self._blend_feature(
                        target_feature,
                        self._appearance_feature(frame, selected["bbox"]),
                        alpha=0.50,
                    )
                    last_bbox = selected["bbox"]
                continue

            if target_track_id is None:
                continue

            # ── Tracking phase ────────────────────────────────────────────
            selected = None
            if identity_manager is not None and target_stable_id is not None:
                selected = next(
                    (candidate for candidate in candidates if candidate.get("player_id") == target_stable_id),
                    None,
                )
            if selected is None:
                selected = next(
                    (c for c in candidates if int(c["track_id"]) == target_track_id), None
                )
            if selected is None:
                selected = self._reidentify_candidate(
                    candidates, frame, target_feature, last_bbox)
                if selected:
                    next_tid = int(selected["track_id"])
                    if next_tid != target_track_id:
                        id_switch_recoveries += 1
                        target_samples.update(
                            {s.frame_id: s for s in pending_samples.get(next_tid, [])}
                        )
                    target_track_id = next_tid
                    if identity_manager is not None:
                        target_stable_id = identity_manager.register_immediate(
                            frame,
                            selected["bbox"],
                            frame_id,
                            frame_id / source_fps,
                            target_track_id,
                            selected.get("team"),
                        ) or target_stable_id

            if selected is None:
                continue

            pt = self._track_point_from_bbox(
                frame_id, source_fps, selected["bbox"], calibration_result,
                selected["confidence"])
            target_samples[frame_id] = pt

            # FIX 3: adaptive blend alpha — fresher updates when confident
            blend_alpha = 0.55 if selected["confidence"] > 0.65 else 0.72
            target_feature = self._blend_feature(
                target_feature,
                self._appearance_feature(frame, selected["bbox"]),
                alpha=blend_alpha,
            )
            last_bbox = selected["bbox"]

        points = [target_samples[fid] for fid in sorted(target_samples)]
        if id_switch_recoveries:
            # small confidence penalty — warn metrics layer
            penalty = min(id_switch_recoveries * 0.03, 0.18)
            for p in points:
                p.confidence = max(p.confidence - penalty, 0.2)

        if not points:
            fallback = video_path.with_name("classical-fallback-overlay.mp4")
            return super().track_target(video_path, click, calibration_result,
                                        fallback, start_frame_id, initial_bbox,
                                        progress_callback)
        return points

    def track_ball_and_player(
        self,
        video_path: Path,
        click: dict,
        calibration: CalibrationResult | np.ndarray,
        start_frame_id: int = 0,
        initial_bbox: dict | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        team_templates: TeamTemplates | None = None,
    ) -> tuple[list[TrackPoint], list[BallTrackPoint]]:
        """Track target player and ball at full frame rate for shot-power analysis."""
        if not self.detector:
            player_points = super().track_target(
                video_path, click, calibration,
                video_path.with_name("classical-fallback-overlay.mp4"),
                start_frame_id, initial_bbox, progress_callback,
            )
            return player_points, []

        cap = cv2.VideoCapture(str(video_path))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if frame_count > 0:
            start_frame_id = min(max(start_frame_id, 0), frame_count - 1)
        calibration_result = _as_calibration(calibration, frame_width, frame_height)

        target_feature = self._build_initial_feature(video_path, start_frame_id, initial_bbox)
        target_track_id: int | None = None
        last_bbox: tuple[float, float, float, float] | None = self._bbox_tuple(initial_bbox)

        pending_samples: dict[int, list[TrackPoint]] = {}
        target_samples: dict[int, TrackPoint] = {}
        ball_tracker = BallKalmanTracker()
        ball_samples: list[BallTrackPoint] = []
        id_switch_recoveries = 0

        stream = self.detector.track(
            source=str(video_path),
            stream=True,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=0.25,
            iou=0.45,
            imgsz=960,
            verbose=False,
        )

        for frame_id, result in enumerate(stream):
            if frame_id < start_frame_id:
                continue

            frame = result.orig_img
            candidates = self._filter_team_players(
                frame,
                self._tracked_candidates(result),
                team_templates,
            )
            if progress_callback and frame_id % 30 == 0:
                progress_callback(frame_id, frame_count)

            for candidate in candidates:
                track_id = int(candidate["track_id"])
                pending_samples.setdefault(track_id, []).append(
                    self._track_point_from_bbox_with_feet(
                        frame_id, source_fps, candidate["bbox"],
                        calibration_result, candidate["confidence"],
                    )
                )

            if target_track_id is None:
                selected = self._select_initial_candidate(
                    candidates, click, initial_bbox, target_feature, frame,
                )
                if selected:
                    target_track_id = int(selected["track_id"])
                    target_samples.update(
                        {point.frame_id: point for point in pending_samples.get(target_track_id, [])}
                    )
                    target_feature = self._blend_feature(
                        target_feature,
                        self._appearance_feature(frame, selected["bbox"]),
                        alpha=0.50,
                    )
                    last_bbox = selected["bbox"]
            elif candidates:
                selected = next(
                    (candidate for candidate in candidates
                     if int(candidate["track_id"]) == target_track_id),
                    None,
                )
                if selected is None:
                    selected = self._reidentify_candidate(
                        candidates, frame, target_feature, last_bbox,
                    )
                    if selected:
                        next_track_id = int(selected["track_id"])
                        if next_track_id != target_track_id:
                            id_switch_recoveries += 1
                            target_samples.update(
                                {point.frame_id: point
                                 for point in pending_samples.get(next_track_id, [])}
                            )
                        target_track_id = next_track_id

                if selected is not None:
                    point = self._track_point_from_bbox_with_feet(
                        frame_id, source_fps, selected["bbox"],
                        calibration_result, selected["confidence"],
                    )
                    target_samples[frame_id] = point
                    blend_alpha = 0.55 if selected["confidence"] > 0.65 else 0.72
                    target_feature = self._blend_feature(
                        target_feature,
                        self._appearance_feature(frame, selected["bbox"]),
                        alpha=blend_alpha,
                    )
                    last_bbox = selected["bbox"]

            ball_detection = self._best_ball_detection(frame, ball_tracker.predicted_position())
            ball_point = ball_tracker.step(
                frame_id,
                frame_id / source_fps,
                calibration_result,
                ball_detection,
            )
            if ball_point is not None:
                ball_samples.append(ball_point)

        player_points = [target_samples[fid] for fid in sorted(target_samples)]
        if id_switch_recoveries:
            penalty = min(id_switch_recoveries * 0.03, 0.18)
            for point in player_points:
                point.confidence = max(point.confidence - penalty, 0.2)

        if not player_points:
            player_points = super().track_target(
                video_path, click, calibration_result,
                video_path.with_name("classical-fallback-overlay.mp4"),
                start_frame_id, initial_bbox, progress_callback,
            )

        return player_points, ball_samples

    def _track_point_from_bbox_with_feet(
        self,
        frame_id: int,
        source_fps: float,
        bbox: tuple[float, float, float, float],
        calibration: CalibrationResult,
        confidence: float,
    ) -> TrackPoint:
        x, y, w, h = bbox
        foot_y = y + h
        center_x = x + w / 2.0
        left_foot = (center_x - w / 4.0, foot_y)
        right_foot = (center_x + w / 4.0, foot_y)
        point = build_track_point(
            calibration,
            frame_id,
            frame_id / source_fps,
            (center_x, foot_y),
            confidence,
        )
        point.left_foot_px = left_foot
        point.right_foot_px = right_foot
        return point

    # ── Candidate helpers ────────────────────────────────────────────────────

    def _tracked_candidates(self, result) -> list[dict]:
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        ids   = boxes.id.cpu().numpy().astype(int).tolist()
        xyxy  = boxes.xyxy.cpu().numpy().tolist()
        confs = boxes.conf.cpu().numpy().tolist()
        out   = []
        for tid, box, conf in zip(ids, xyxy, confs):
            x1,y1,x2,y2 = [float(v) for v in box]
            out.append({"track_id":int(tid), "bbox":(x1,y1,x2-x1,y2-y1),
                        "confidence":float(conf)})
        return out

    def _select_initial_candidate(self, candidates, click, initial_bbox,
                                   target_feature, frame):
        def score(c):
            bbox = c["bbox"]
            click_score = _center_distance_sq(bbox, (float(click["x"]),float(click["y"])))**0.5
            iou_penalty = 0.0
            if initial_bbox:
                iou_penalty = (1.0 - _iou(bbox, self._bbox_tuple(initial_bbox))) * 250.0
            app_penalty = 0.0
            if target_feature is not None:
                app_penalty = self._appearance_distance(
                    target_feature, self._appearance_feature(frame, bbox)) * 160.0
            return click_score + iou_penalty + app_penalty - c["confidence"] * 75.0
        return min(candidates, key=score) if candidates else None

    def _reidentify_candidate(self, candidates, frame, target_feature, last_bbox):
        """
        FIX 2: tighter threshold (0.42 vs 0.58) and motion gate.
        A candidate is rejected if it's more than 2× the player height
        away from the last known position.
        """
        if target_feature is None and last_bbox is None:
            return None

        # Motion gate: max allowable distance = 2× player height
        max_jump = (last_bbox[3] * 2.0) if last_bbox else 999.0

        def score(c):
            bbox = c["bbox"]
            # Distance gate check
            if last_bbox:
                dist = _center_distance_sq(bbox, _bbox_center(last_bbox))**0.5
                if dist > max_jump:
                    return 999.0  # hard reject — too far away
            app = (self._appearance_distance(target_feature,
                   self._appearance_feature(frame, bbox))
                   if target_feature is not None else 0.5)
            motion = (_center_distance_sq(bbox, _bbox_center(last_bbox))**0.5
                      / max(last_bbox[3], 1.0)) if last_bbox else 1.0
            return app * 0.72 + motion * 0.20 - c["confidence"] * 0.08

        best = min(candidates, key=score)
        return best if score(best) < REID_THRESHOLD else None

    # ── Feature helpers ──────────────────────────────────────────────────────

    def _build_initial_feature(self, video_path: Path,
                                start_frame_id: int,
                                bbox: dict | None) -> np.ndarray | None:
        """
        FIX 1: collect appearance samples from FEATURE_WARMUP_SAMPLES frames
        around the selected frame to build a robust initial histogram.
        """
        if not bbox:
            return None
        cap  = cv2.VideoCapture(str(video_path))
        fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        features = []
        for offset in range(FEATURE_WARMUP_SAMPLES):
            fid = min(start_frame_id + offset, max(fcount-1,0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame = cap.read()
            if not ok:
                break
            f = self._appearance_feature(frame, self._bbox_tuple(bbox))
            if f is not None:
                features.append(f)
        cap.release()
        if not features:
            return None
        stacked = np.mean(features, axis=0)
        total   = stacked.sum()
        return stacked / total if total > 0 else stacked

    def _selected_feature(self, video_path, frame_id, bbox):
        """Kept for compatibility; delegates to _build_initial_feature."""
        return self._build_initial_feature(video_path, frame_id, bbox)

    def _appearance_feature(self, frame, bbox) -> np.ndarray | None:
        if bbox is None:
            return None
        x, y, w, h = _clip_bbox(bbox, frame.shape[1], frame.shape[0])
        if w < 4 or h < 8:
            return None
        crop = frame[y:y+h, x:x+w]
        # Use only the upper 60% of the crop (shirt region, avoid legs/shadow)
        torso_h = max(int(h * 0.60), 4)
        crop = crop[:torso_h]
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1], None, [16,16], [0,180,0,256])
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        return hist.flatten()

    def _appearance_distance(self, left, right) -> float:
        if left is None or right is None:
            return 0.5
        return float(cv2.compareHist(
            left.astype(np.float32), right.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA))

    def _blend_feature(self, current, update, alpha: float = 0.72):
        if update is None:  return current
        if current is None: return update
        blended = current * alpha + update * (1.0 - alpha)
        total   = float(blended.sum())
        return blended / total if total > 0 else blended

    def _track_point_from_bbox(self, frame_id, source_fps, bbox, calibration, confidence):
        x, y, w, h = bbox
        return build_track_point(
            calibration,
            frame_id,
            frame_id / source_fps,
            (x + w / 2.0, y + h),
            confidence,
        )

    def _bbox_tuple(self, bbox):
        if not bbox: return None
        return (float(bbox["x"]), float(bbox["y"]),
                float(bbox["width"]), float(bbox["height"]))


# ── TemplateMatchingTracker ──────────────────────────────────────────────────

class TemplateMatchingTracker:
    """Dependency-free fallback tracker."""

    def __init__(self):
        self.template: np.ndarray | None = None
        self.bbox: tuple[int,int,int,int] | None = None

    def init(self, frame, bbox):
        x,y,w,h = bbox
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.template = gray[y:y+h, x:x+w].copy()
        self.bbox = bbox
        return self.template.size > 0

    def update(self, frame):
        if self.template is None or self.bbox is None:
            return False, (0,0,0,0)
        x,y,w,h = self.bbox
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape[:2]
        px, py = max(w*2,60), max(h,60)
        left = max(x-px,0); top = max(y-py,0)
        right = min(x+w+px,fw); bottom = min(y+h+py,fh)
        search = gray[top:bottom, left:right]
        if search.shape[0] < h or search.shape[1] < w:
            return False, self.bbox
        result = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, conf, _, loc = cv2.minMaxLoc(result)
        self.bbox = (left+loc[0], top+loc[1], w, h)
        return conf >= 0.18, self.bbox


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _kalman_state_value(state: np.ndarray, index: int) -> float:
    value = state[index]
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _center_distance_sq(bbox, point):
    cx = bbox[0] + bbox[2]/2.0
    cy = bbox[1] + bbox[3]/2.0
    return (cx-point[0])**2 + (cy-point[1])**2

def _int_bbox(bbox):
    return tuple(max(int(round(v)), 1 if i>=2 else 0) for i,v in enumerate(bbox[:4]))

def _bbox_center(bbox):
    if not bbox: return (0.0,0.0)
    return (bbox[0]+bbox[2]/2.0, bbox[1]+bbox[3]/2.0)

def _iou(left, right):
    if not left or not right: return 0.0
    lx1,ly1,lw,lh = left;  lx2,ly2 = lx1+lw, ly1+lh
    rx1,ry1,rw,rh = right; rx2,ry2 = rx1+rw, ry1+rh
    ix1,iy1 = max(lx1,rx1), max(ly1,ry1)
    ix2,iy2 = min(lx2,rx2), min(ly2,ry2)
    inter = max(ix2-ix1,0)*max(iy2-iy1,0)
    union = lw*lh + rw*rh - inter
    return inter/union if union > 0 else 0.0

def _clip_bbox(bbox, fw, fh):
    x,y,w,h = bbox
    left   = min(max(int(round(x)),0), max(fw-1,0))
    top    = min(max(int(round(y)),0), max(fh-1,0))
    right  = min(max(int(round(x+w)), left+1), fw)
    bottom = min(max(int(round(y+h)), top+1),  fh)
    return left, top, right-left, bottom-top


def get_cv_pipeline() -> ReIdByteTrackPipeline:
    global _shared_pipeline
    if _shared_pipeline is None:
        with _pipeline_lock:
            if _shared_pipeline is None:
                _shared_pipeline = ReIdByteTrackPipeline()
    return _shared_pipeline
