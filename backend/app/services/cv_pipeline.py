from __future__ import annotations

import os
from math import hypot
from pathlib import Path
from threading import Lock
from typing import Callable

import cv2
import numpy as np
import supervision as sv

_pipeline_lock = Lock()
_shared_pipeline = None
_player_model = None
_ball_model = None
_model_lock = Lock()

from app.services.calibration import (
    CalibrationResult,
    build_track_point,
    is_pixel_in_calibrated_region,
    pixel_to_field,
    region_polygon_for_metrics,
)
from app.services.metrics import BallTrackPoint, TrackPoint
from app.services.team_classification import (
    TeamClassificationService,
    TeamTemplates,
    build_team_templates,
    team_label,
)

from app.services.roboflow_inference import (
    RoboflowConfigError,
    get_model as inference_get_model,
    require_roboflow_api_key,
)

try:
    from supervision import TeamClassifier as SvTeamClassifier
except (ImportError, AttributeError):  # pragma: no cover
    SvTeamClassifier = None

NEUTRAL_TEAM_RGB = (148, 163, 184)
# Same detection engine as shooting-technique mode: the RF-DETR COCO
# checkpoint on Roboflow serverless. It generalises to phone/amateur footage
# far better than broadcast-trained football models. Override via env.
PLAYER_MODEL_ID = os.environ.get("ROBOFLOW_PLAYER_MODEL", "").strip() or "rfdetr-base"
BALL_MODEL_ID = os.environ.get("ROBOFLOW_BALL_MODEL", "").strip() or "rfdetr-base"

# Fallback class-id layout for football-players-detection-style models
# (alphabetical: 0=ball, 1=goalkeeper, 2=player, 3=referee). Filtering is by
# class NAME whenever the model provides one.
CLASS_BALL_IN_PLAYER_MODEL = 0
CLASS_GOALKEEPER = 1
CLASS_PLAYER = 2
CLASS_REFEREE = 3
REFEREE_CLASS_IDS = frozenset({CLASS_REFEREE})
TRACKABLE_PLAYER_CLASS_IDS = frozenset({CLASS_GOALKEEPER, CLASS_PLAYER})
TRACKABLE_CLASS_NAMES = frozenset({"player", "goalkeeper", "person"})
BALL_CLASS_NAMES = frozenset({"ball", "sports ball"})

BALL_KALMAN_PROCESS_NOISE = 15.0
BALL_KALMAN_MEASUREMENT_NOISE = 5.0
BALL_INTERPOLATED_CONFIDENCE = 0.25
BALL_MAX_CONSECUTIVE_MISSES = 45
BALL_DETECT_CADENCE = 2
def _env_conf(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name, "").strip()
        value = float(raw) if raw else default
    except (TypeError, ValueError):
        return default
    return min(max(value, 0.0), 1.0)


# Detection confidence thresholds. Tune per model via env without code edits:
#   players  -> favour recall (don't lose the tracked player between frames)
#   ball     -> small/blurry, keep it low and let the Kalman fill gaps
PLAYER_DETECT_CONFIDENCE = _env_conf("GAMESENSE_PLAYER_CONFIDENCE", 0.30)
BALL_DETECT_CONFIDENCE = _env_conf("GAMESENSE_BALL_CONFIDENCE", 0.25)
TEAM_CALIBRATION_FRAMES = 30


def get_player_model():
    global _player_model
    if _player_model is None:
        with _model_lock:
            if _player_model is None:
                api_key = require_roboflow_api_key()
                _player_model = inference_get_model(model_id=PLAYER_MODEL_ID, api_key=api_key)
                print("[GameSense] Football player detection model loaded")
    return _player_model


def get_ball_model():
    global _ball_model
    if _ball_model is None:
        with _model_lock:
            if _ball_model is None:
                api_key = require_roboflow_api_key()
                _ball_model = inference_get_model(model_id=BALL_MODEL_ID, api_key=api_key)
                print("[GameSense] Football ball detection model loaded")
    return _ball_model


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


def _kalman_state_value(state: np.ndarray, index: int) -> float:
    return float(state[index, 0])


class BallKalmanTracker:
    """Kalman smoother for ball trajectory fed by football-ball-detection."""

    def __init__(self) -> None:
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
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
                region_polygon_for_metrics(calibration),
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


def _center_distance_sq(bbox: tuple[float, float, float, float], point: tuple[float, float]) -> float:
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    return (cx - point[0]) ** 2 + (cy - point[1]) ** 2


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def _bbox_tuple(bbox: dict | tuple[float, float, float, float] | None) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        return float(bbox["x"]), float(bbox["y"]), float(bbox["width"]), float(bbox["height"])
    return tuple(float(v) for v in bbox)


def _detections_to_tuples(detections: sv.Detections) -> list[tuple[float, float, float, float, float]]:
    if detections is None or len(detections) == 0:
        return []
    confidences = (
        detections.confidence
        if detections.confidence is not None
        else np.ones(len(detections), dtype=float)
    )
    tuples: list[tuple[float, float, float, float, float]] = []
    for xyxy, confidence in zip(detections.xyxy, confidences):
        x1, y1, x2, y2 = [float(value) for value in xyxy]
        tuples.append((x1, y1, x2 - x1, y2 - y1, float(confidence)))
    return tuples


def _filter_player_detections(detections: sv.Detections) -> sv.Detections:
    """Keep players + goalkeepers; drop referees and the ball.

    Prefers class NAMES (robust to class-id ordering differences between
    model versions); falls back to the known id layout."""
    if detections is None or len(detections) == 0:
        return detections
    names = None
    data = getattr(detections, "data", None)
    if data:
        names = data.get("class_name")
    if names is not None and len(names) == len(detections):
        keep = [str(name).lower() in TRACKABLE_CLASS_NAMES for name in names]
        return detections[np.array(keep, dtype=bool)]
    if detections.class_id is None:
        return detections
    keep = [
        int(class_id) in TRACKABLE_PLAYER_CLASS_IDS and int(class_id) not in REFEREE_CLASS_IDS
        for class_id in detections.class_id
    ]
    return detections[np.array(keep, dtype=bool)]


def _is_valid_ball_bbox(x: float, y: float, w: float, h: float, frame_height: int) -> bool:
    aspect = w / max(h, 1.0)
    if aspect < 0.6 or aspect > 1.6:
        return False
    diameter = max(w, h)
    if diameter < 6 or diameter > frame_height / 20:
        return False
    if (y + h / 2.0) < frame_height * 0.15:
        return False
    return True


class FootballCvPipeline:
    """Football-specific detection and tracking via Roboflow inference + supervision."""

    def __init__(self, max_width: int = 1280, sample_fps: float = 8.0):
        self.max_width = max_width
        self.sample_fps = sample_fps
        self.team_service = TeamClassificationService()
        require_roboflow_api_key()

    def create_player_tracker(self, frame_rate: float = 30.0) -> sv.ByteTrack:
        return sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=150,          # ~5 s of track memory through occlusions
            minimum_matching_threshold=0.8,
            frame_rate=int(max(round(frame_rate), 1)),
        )

    def infer_player_detections(self, frame: np.ndarray, confidence: float = PLAYER_DETECT_CONFIDENCE) -> sv.Detections:
        result = get_player_model().infer(frame, confidence=confidence)[0]
        detections = sv.Detections.from_inference(result)
        return _filter_player_detections(detections)

    def infer_ball_detections(self, frame: np.ndarray, confidence: float = BALL_DETECT_CONFIDENCE) -> sv.Detections:
        result = get_ball_model().infer(frame, confidence=confidence)[0]
        return sv.Detections.from_inference(result)

    def track_players(self, frame: np.ndarray, tracker: sv.ByteTrack) -> sv.Detections:
        detections = self.infer_player_detections(frame)
        if len(detections) == 0:
            return detections
        return tracker.update_with_detections(detections)

    def _detect_people(self, frame: np.ndarray, *, high_resolution: bool = False) -> list[tuple[float, float, float, float, float]]:
        del high_resolution
        return _detections_to_tuples(self.infer_player_detections(frame))

    def _detect_people_lower_half_boost(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        return []

    def _merge_people_detections(
        self,
        primary: list[tuple[float, float, float, float, float]],
        secondary: list[tuple[float, float, float, float, float]],
    ) -> list[tuple[float, float, float, float, float]]:
        return list(primary) + list(secondary)

    def _detect_ball(self, frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        detections = self.infer_ball_detections(frame)
        # When the ball model is a general COCO checkpoint, keep only ball
        # classes; dedicated ball models return a single class and pass as-is.
        data = getattr(detections, "data", None)
        names = data.get("class_name") if data else None
        if names is not None and len(names) == len(detections):
            keep = [str(name).lower() in BALL_CLASS_NAMES for name in names]
            detections = detections[np.array(keep, dtype=bool)]
        tuples = _detections_to_tuples(detections)
        frame_height = frame.shape[0]
        return [
            item
            for item in tuples
            if _is_valid_ball_bbox(item[0], item[1], item[2], item[3], frame_height)
        ]

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
        return x + w / 2.0, y + h / 2.0, float(confidence)

    def tracked_to_tuples(self, tracked: sv.Detections) -> list[tuple[float, float, float, float, float]]:
        return _detections_to_tuples(tracked)

    def select_tracker_id(
        self,
        tracked: sv.Detections,
        click: dict,
        initial_bbox: dict | None = None,
    ) -> int | None:
        if tracked is None or len(tracked) == 0 or tracked.tracker_id is None:
            return None
        bbox = _bbox_tuple(initial_bbox)
        if bbox is not None:
            for index, xyxy in enumerate(tracked.xyxy):
                x1, y1, x2, y2 = [float(value) for value in xyxy]
                candidate = (x1, y1, x2 - x1, y2 - y1)
                click_x = bbox[0] + bbox[2] / 2.0
                click_y = bbox[1] + bbox[3] / 2.0
                if candidate[0] <= click_x <= candidate[0] + candidate[2] and candidate[1] <= click_y <= candidate[1] + candidate[3]:
                    return int(tracked.tracker_id[index])
        click_x = float(click["x"])
        click_y = float(click["y"])
        best_id = None
        best_dist = float("inf")
        for index, xyxy in enumerate(tracked.xyxy):
            x1, y1, x2, y2 = [float(value) for value in xyxy]
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                dist = (cx - click_x) ** 2 + (cy - click_y) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_id = int(tracked.tracker_id[index])
        return best_id

    def bbox_for_tracker_id(
        self,
        tracked: sv.Detections,
        tracker_id: int,
    ) -> tuple[float, float, float, float, float] | None:
        if tracked is None or len(tracked) == 0 or tracked.tracker_id is None:
            return None
        for index, track_id in enumerate(tracked.tracker_id):
            if int(track_id) != int(tracker_id):
                continue
            x1, y1, x2, y2 = [float(value) for value in tracked.xyxy[index]]
            confidence = 0.5
            if tracked.confidence is not None:
                confidence = float(tracked.confidence[index])
            return x1, y1, x2 - x1, y2 - y1, confidence
        return None

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

    def _fit_supervision_team_classifier(self, video_path: Path) -> tuple[dict[int, str] | None, SvTeamClassifier | None]:
        if SvTeamClassifier is None:
            return None, None
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            return None, None
        crops: list[np.ndarray] = []
        frames_seen = 0
        while frames_seen < TEAM_CALIBRATION_FRAMES:
            ok, frame = capture.read()
            if not ok:
                break
            for x, y, w, h, _confidence in self._detect_people(frame):
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                if crop.size > 0:
                    crops.append(crop)
            frames_seen += 1
        capture.release()
        if len(crops) < 8:
            return None, None
        try:
            classifier = SvTeamClassifier(device="cpu")
            classifier.fit(crops)
            return {0: "team_a", 1: "team_b"}, classifier
        except Exception as exc:
            print(f"[GameSense] supervision TeamClassifier unavailable, using Lab k-means fallback: {exc}")
            return None, None

    def calibrate_team_templates(
        self,
        video_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> TeamTemplates:
        """Single-pass team calibration.

        Previously this scanned the clip TWICE (once for the supervision
        classifier, once for Lab k-means), with one sequential hosted-inference
        call per frame — the main reason the setup page took so long to appear.
        Now: one scan, inference fanned out over a thread pool, progress
        reported per frame, and both the Lab templates and the optional
        supervision classifier fed from the same detections.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from app.services.team_classification import (
            _templates_from_samples,
            extract_shirt_lab_sample,
        )

        capture = cv2.VideoCapture(str(video_path))
        frames: list[np.ndarray] = []
        if capture.isOpened():
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            max_scan = TEAM_CALIBRATION_FRAMES * 2          # sample every 2nd frame
            limit = min(max_scan, frame_count) if frame_count > 0 else max_scan
            for frame_id in range(0, limit, 2):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frame = capture.read()
                if ok:
                    frames.append(frame)
        capture.release()

        total = max(len(frames), 1)
        detections_per_frame: list[list] = [[] for _ in frames]
        done = 0
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(self._detect_people, frame): idx
                       for idx, frame in enumerate(frames)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    detections_per_frame[idx] = future.result() or []
                except Exception:
                    detections_per_frame[idx] = []
                done += 1
                if progress_callback:
                    progress_callback(done, total)

        samples: list[np.ndarray] = []
        crops: list[np.ndarray] = []
        for frame, dets in zip(frames, detections_per_frame):
            for x, y, w, h, _confidence in dets:
                bbox = {"x": round(x, 2), "y": round(y, 2), "width": round(w, 2), "height": round(h, 2)}
                sample = extract_shirt_lab_sample(frame, bbox)
                if sample is not None:
                    samples.append(sample)
                x1, y1 = int(max(x, 0)), int(max(y, 0))
                crop = frame[y1:int(y + h), x1:int(x + w)]
                if crop.size > 0:
                    crops.append(crop)

        templates = _templates_from_samples(samples, len(frames))
        self.team_service.set_templates(templates)

        sv_classifier = None
        if SvTeamClassifier is not None and len(crops) >= 8:
            try:
                sv_classifier = SvTeamClassifier(device="cpu")
                sv_classifier.fit(crops)
            except Exception as exc:
                print(f"[GameSense] supervision TeamClassifier unavailable, using Lab k-means fallback: {exc}")
                sv_classifier = None
        self._sv_team_classifier = sv_classifier
        self._sv_team_mapping = {0: "team_a", 1: "team_b"}
        return templates

    def _predict_team_for_crop(self, crop: np.ndarray) -> str | None:
        classifier = getattr(self, "_sv_team_classifier", None)
        mapping = getattr(self, "_sv_team_mapping", {0: "team_a", 1: "team_b"})
        if classifier is None or crop.size == 0:
            return None
        try:
            team_ids = classifier.predict([crop])
            if len(team_ids) == 0:
                return None
            return mapping.get(int(team_ids[0]))
        except Exception:
            return None

    def detect_frame_objects(
        self,
        video_path: Path,
        frame_id: int,
        team_templates: TeamTemplates | None = None,
        assign_player_ids: bool = False,
    ) -> list[dict]:
        if assign_player_ids and team_templates is not None:
            return self._detect_frame_with_tracking(video_path, frame_id, team_templates)

        capture = cv2.VideoCapture(str(video_path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 0:
            frame_id = min(max(frame_id, 0), frame_count - 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            return []

        if team_templates is not None:
            self.team_service.set_templates(team_templates)

        player_boxes = self._detect_people(frame)
        print(f"[GameSense] frame {frame_id} player detections: {len(player_boxes)}")
        return self._format_detection_payload(frame, frame_id, player_boxes, team_templates)

    def _format_detection_payload(
        self,
        frame: np.ndarray,
        frame_id: int,
        player_boxes: list[tuple[float, float, float, float, float]],
        team_templates: TeamTemplates | None,
    ) -> list[dict]:
        detections: list[dict] = []
        for output_index, (x, y, w, h, confidence) in enumerate(player_boxes):
            original_bbox = {"x": round(x, 2), "y": round(y, 2), "width": round(w, 2), "height": round(h, 2)}
            player_key = f"player-{frame_id}-{output_index}"
            if team_templates is not None:
                crop = frame[int(y): int(y + h), int(x): int(x + w)]
                sv_team = self._predict_team_for_crop(crop)
                if sv_team in {"team_a", "team_b"}:
                    color_rgb = (
                        team_templates.team_a_color_rgb
                        if sv_team == "team_a"
                        else team_templates.team_b_color_rgb
                    )
                    detections.append({
                        "id": player_key,
                        "label": sv_team,
                        "team": sv_team,
                        "team_label": team_label(sv_team),  # type: ignore[arg-type]
                        "team_color": {"r": color_rgb[0], "g": color_rgb[1], "b": color_rgb[2]},
                        "confidence": round(confidence, 3),
                        "bbox": original_bbox,
                    })
                    continue
                team_id, color_rgb = self._classify_player_detection(
                    frame,
                    original_bbox,
                    player_key,
                    team_templates,
                    apply_temporal=False,
                )
                if team_id is None:
                    detections.append({
                        "id": player_key,
                        "label": "player",
                        "confidence": round(confidence, 3),
                        "bbox": original_bbox,
                        "team_color": {"r": NEUTRAL_TEAM_RGB[0], "g": NEUTRAL_TEAM_RGB[1], "b": NEUTRAL_TEAM_RGB[2]},
                    })
                else:
                    detections.append({
                        "id": player_key,
                        "label": team_id,
                        "team": team_id,
                        "team_label": team_label(team_id),  # type: ignore[arg-type]
                        "team_color": {"r": color_rgb[0], "g": color_rgb[1], "b": color_rgb[2]},
                        "confidence": round(confidence, 3),
                        "bbox": original_bbox,
                    })
            else:
                detections.append({
                    "id": player_key,
                    "label": "player",
                    "confidence": round(confidence, 3),
                    "bbox": original_bbox,
                })

        for index, bbox in enumerate(self._detect_ball(frame)):
            x, y, w, h, confidence = bbox
            detections.append({
                "id": f"ball-{frame_id}-{index}",
                "label": "ball",
                "confidence": round(float(confidence), 3),
                "bbox": {"x": round(x, 2), "y": round(y, 2), "width": round(w, 2), "height": round(h, 2)},
            })
        return detections

    def _detect_frame_with_tracking(
        self,
        video_path: Path,
        frame_id: int,
        team_templates: TeamTemplates,
    ) -> list[dict]:
        capture = cv2.VideoCapture(str(video_path))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 0:
            frame_id = min(max(frame_id, 0), frame_count - 1)
        tracker = self.create_player_tracker(source_fps)
        self.team_service.set_templates(team_templates)
        current_frame = 0
        last_payload: list[dict] = []
        while current_frame <= frame_id:
            ok, frame = capture.read()
            if not ok:
                break
            tracked = self.track_players(frame, tracker)
            player_boxes = self.tracked_to_tuples(tracked)
            payload: list[dict] = []
            if tracked.tracker_id is not None:
                for index, track_id in enumerate(tracked.tracker_id):
                    if index >= len(player_boxes):
                        continue
                    x, y, w, h, confidence = player_boxes[index]
                    original_bbox = {"x": round(x, 2), "y": round(y, 2), "width": round(w, 2), "height": round(h, 2)}
                    player_key = f"track-{int(track_id)}"
                    team_id, color_rgb = self._classify_player_detection(
                        frame,
                        original_bbox,
                        player_key,
                        team_templates,
                        apply_temporal=False,
                    )
                    if team_id is None:
                        continue
                    payload.append({
                        "id": player_key,
                        "label": team_id,
                        "team": team_id,
                        "team_label": team_label(team_id),  # type: ignore[arg-type]
                        "team_color": {"r": color_rgb[0], "g": color_rgb[1], "b": color_rgb[2]},
                        "confidence": round(confidence, 3),
                        "bbox": original_bbox,
                        "tracker_id": int(track_id),
                    })
            last_payload = payload
            current_frame += 1
        capture.release()
        return last_payload

    def _track_point_from_bbox(
        self,
        frame_id: int,
        source_fps: float,
        bbox: tuple[float, float, float, float, float],
        calibration: CalibrationResult,
        confidence: float,
    ) -> TrackPoint:
        x, y, w, h, _confidence = bbox
        foot_y = y + h
        center_x = x + w / 2.0
        left_foot = (center_x - w / 4.0, foot_y)
        right_foot = (center_x + w / 4.0, foot_y)
        point = build_track_point(
            calibration,
            frame_id,
            frame_id / max(source_fps, 1e-6),
            (center_x, foot_y),
            confidence,
        )
        point.left_foot_px = left_foot
        point.right_foot_px = right_foot
        return point

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
        capture = cv2.VideoCapture(str(video_path))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if frame_count > 0:
            start_frame_id = min(max(start_frame_id, 0), frame_count - 1)
        calibration_result = _as_calibration(calibration, frame_width, frame_height)
        if team_templates is not None:
            self.team_service.set_templates(team_templates)

        tracker = self.create_player_tracker(source_fps)
        ball_tracker = BallKalmanTracker()
        target_track_id: int | None = None
        target_samples: dict[int, TrackPoint] = {}
        ball_samples: list[BallTrackPoint] = []

        frame_id = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_id < start_frame_id:
                frame_id += 1
                continue

            tracked = self.track_players(frame, tracker)
            if target_track_id is None:
                target_track_id = self.select_tracker_id(tracked, click, initial_bbox)

            if target_track_id is not None:
                matched = self.bbox_for_tracker_id(tracked, target_track_id)
                if matched is not None:
                    target_samples[frame_id] = self._track_point_from_bbox(
                        frame_id,
                        source_fps,
                        matched,
                        calibration_result,
                        matched[4],
                    )

            if frame_id % BALL_DETECT_CADENCE == 0:
                ball_detection = self._best_ball_detection(frame, ball_tracker.predicted_position())
                ball_point = ball_tracker.step(
                    frame_id,
                    frame_id / max(source_fps, 1e-6),
                    calibration_result,
                    ball_detection,
                )
                if ball_point is not None:
                    ball_samples.append(ball_point)

            if progress_callback and frame_id % 30 == 0:
                progress_callback(frame_id, frame_count)
            frame_id += 1

        capture.release()
        player_points = [target_samples[fid] for fid in sorted(target_samples)]
        print(f"[GameSense] ball track points: {len(ball_samples)}")
        return player_points, ball_samples

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
        player_points, _ball_points = self.track_ball_and_player(
            video_path,
            click,
            calibration,
            start_frame_id,
            initial_bbox,
            progress_callback,
            team_templates,
        )
        return player_points


def get_cv_pipeline() -> FootballCvPipeline:
    global _shared_pipeline
    if _shared_pipeline is None:
        with _pipeline_lock:
            if _shared_pipeline is None:
                require_roboflow_api_key()
                get_player_model()
                get_ball_model()
                _shared_pipeline = FootballCvPipeline()
    return _shared_pipeline
