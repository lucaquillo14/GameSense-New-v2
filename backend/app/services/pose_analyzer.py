from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from math import degrees, hypot
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.models import BodyAngle, TechniqueFrame

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
POSE_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:  # pragma: no cover
    mp = None
    mp_tasks = None
    mp_vision = None

PLANT_VELOCITY_THRESHOLD = 2.5
APPROACH_LOOKBACK_FRAMES = 5

# MediaPipe Pose landmark indices.
LM = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


@dataclass
class PoseLandmarks:
    points: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    visible: bool = False


@dataclass
class FramePoseInput:
    frame_id: int
    time_s: float
    frame_width: int
    frame_height: int
    detection: dict[str, Any]
    landmarks: PoseLandmarks | None = None


def _ensure_pose_model() -> Path:
    POSE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = POSE_MODEL_DIR / "pose_landmarker_lite.task"
    if not model_path.exists():
        print("[GameSense] Downloading MediaPipe pose model...")
        urllib.request.urlretrieve(POSE_MODEL_URL, model_path)
    return model_path


def _landmark_visibility(landmark) -> float:
    for attr in ("visibility", "presence"):
        if hasattr(landmark, attr):
            return float(getattr(landmark, attr))
    return 1.0


class VideoPoseEstimator:
    """MediaPipe Tasks API pose landmarker for per-frame video analysis."""

    def __init__(self) -> None:
        if mp is None or mp_tasks is None or mp_vision is None:
            raise RuntimeError("mediapipe is not installed. Run: pip install mediapipe")
        model_path = _ensure_pose_model()
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray, frame_id: int, source_fps: float) -> PoseLandmarks:
        timestamp_ms = int(frame_id * 1000.0 / max(source_fps, 1e-6))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.pose_landmarks:
            return PoseLandmarks(visible=False)
        height, width = frame.shape[:2]
        pose_landmarks = result.pose_landmarks[0]
        points: dict[str, tuple[float, float, float]] = {}
        for name, index in LM.items():
            if index >= len(pose_landmarks):
                continue
            landmark = pose_landmarks[index]
            visibility = _landmark_visibility(landmark)
            if visibility < 0.35:
                continue
            points[name] = (landmark.x * width, landmark.y * height, visibility)
        return PoseLandmarks(points=points, visible=len(points) >= 8)

    def close(self) -> None:
        self._landmarker.close()


def _pose_estimator() -> VideoPoseEstimator:
    return VideoPoseEstimator()


def _point(landmarks: PoseLandmarks | None, name: str) -> tuple[float, float] | None:
    if landmarks is None or name not in landmarks.points:
        return None
    x, y, _visibility = landmarks.points[name]
    return x, y


def _angle_at_joint(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float | None:
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    norm_ba = hypot(ba[0], ba[1])
    norm_bc = hypot(bc[0], bc[1])
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return None
    cosine = (ba[0] * bc[0] + ba[1] * bc[1]) / (norm_ba * norm_bc)
    cosine = max(-1.0, min(1.0, cosine))
    return degrees(float(np.arccos(cosine)))


def _horizontal_angle(left: tuple[float, float], right: tuple[float, float]) -> float | None:
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    if hypot(dx, dy) < 1e-6:
        return None
    return degrees(float(np.arctan2(dy, dx)))


def _ball_center(detection: dict[str, Any]) -> tuple[float, float] | None:
    ball = detection.get("ball")
    if not ball:
        return None
    x, y, w, h, _conf = ball
    return x + w / 2.0, y + h / 2.0


def _foot_point(landmarks: PoseLandmarks, side: str) -> tuple[float, float] | None:
    toe = _point(landmarks, f"{side}_foot_index")
    ankle = _point(landmarks, f"{side}_ankle")
    if toe is not None:
        return toe
    return ankle


def _determine_kicking_side(frames: list[FramePoseInput], contact_index: int) -> str:
    contact = frames[contact_index]
    ball = _ball_center(contact.detection)
    if ball is None or contact.landmarks is None:
        return "right"
    left = _foot_point(contact.landmarks, "left")
    right = _foot_point(contact.landmarks, "right")
    if left is None and right is None:
        return "right"
    if left is None:
        return "right"
    if right is None:
        return "left"
    left_dist = hypot(left[0] - ball[0], left[1] - ball[1])
    right_dist = hypot(right[0] - ball[0], right[1] - ball[1])
    return "left" if left_dist < right_dist else "right"


def _foot_to_ball_px(landmarks: PoseLandmarks | None, ball_center: tuple[float, float] | None, kick_side: str) -> float | None:
    if landmarks is None or ball_center is None:
        return None
    foot = _foot_point(landmarks, kick_side)
    if foot is None:
        return None
    return hypot(foot[0] - ball_center[0], foot[1] - ball_center[1])


def _compute_angles(
    landmarks: PoseLandmarks | None,
    kick_side: str,
    plant_side: str,
    frame_id: int,
    time_s: float,
    hip_displacement: float | None,
) -> list[BodyAngle]:
    if landmarks is None:
        return []
    angles: list[BodyAngle] = []

    kick_hip = _point(landmarks, f"{kick_side}_hip")
    kick_knee = _point(landmarks, f"{kick_side}_knee")
    kick_ankle = _point(landmarks, f"{kick_side}_ankle")
    kick_toe = _point(landmarks, f"{kick_side}_foot_index") or kick_ankle

    plant_hip = _point(landmarks, f"{plant_side}_hip")
    plant_knee = _point(landmarks, f"{plant_side}_knee")
    plant_ankle = _point(landmarks, f"{plant_side}_ankle")

    left_shoulder = _point(landmarks, "left_shoulder")
    right_shoulder = _point(landmarks, "right_shoulder")
    left_hip = _point(landmarks, "left_hip")
    right_hip = _point(landmarks, "right_hip")

    if kick_hip and kick_knee and kick_ankle:
        knee = _angle_at_joint(kick_hip, kick_knee, kick_ankle)
        if knee is not None:
            angles.append(BodyAngle(name="knee_angle", value_deg=knee, frame_id=frame_id, time_s=time_s))

    if plant_hip and plant_knee and plant_ankle:
        plant_knee_angle = _angle_at_joint(plant_hip, plant_knee, plant_ankle)
        if plant_knee_angle is not None:
            angles.append(
                BodyAngle(name="plant_knee_angle", value_deg=plant_knee_angle, frame_id=frame_id, time_s=time_s)
            )

    if kick_hip and kick_knee:
        torso_top = None
        if left_shoulder and right_shoulder:
            torso_top = ((left_shoulder[0] + right_shoulder[0]) / 2.0, (left_shoulder[1] + right_shoulder[1]) / 2.0)
        if torso_top is not None:
            vertical_ref = (kick_hip[0], kick_hip[1] - 100.0)
            hip_angle = _angle_at_joint(vertical_ref, kick_hip, kick_knee)
            if hip_angle is not None:
                angles.append(BodyAngle(name="hip_angle", value_deg=hip_angle, frame_id=frame_id, time_s=time_s))

    if kick_knee and kick_ankle and kick_toe:
        ankle_angle = _angle_at_joint(kick_knee, kick_ankle, kick_toe)
        if ankle_angle is not None:
            angles.append(BodyAngle(name="ankle_angle", value_deg=ankle_angle, frame_id=frame_id, time_s=time_s))

    if left_shoulder and right_shoulder and left_hip and right_hip:
        shoulder_angle = _horizontal_angle(left_shoulder, right_shoulder)
        hip_angle_h = _horizontal_angle(left_hip, right_hip)
        if shoulder_angle is not None and hip_angle_h is not None:
            rotation = shoulder_angle - hip_angle_h
            angles.append(
                BodyAngle(name="shoulder_hip_rotation", value_deg=rotation, frame_id=frame_id, time_s=time_s)
            )

    if left_shoulder and right_shoulder and left_hip and right_hip:
        mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2.0, (left_shoulder[1] + right_shoulder[1]) / 2.0)
        mid_hip = ((left_hip[0] + right_hip[0]) / 2.0, (left_hip[1] + right_hip[1]) / 2.0)
        lateral_ref = (kick_hip[0] if kick_hip else mid_hip[0], mid_hip[1])
        trunk = _angle_at_joint(mid_shoulder, mid_hip, lateral_ref)
        if trunk is not None:
            angles.append(BodyAngle(name="trunk_lean", value_deg=trunk, frame_id=frame_id, time_s=time_s))

    if hip_displacement is not None:
        angles.append(
            BodyAngle(name="approach_angle", value_deg=hip_displacement, frame_id=frame_id, time_s=time_s)
        )

    return angles


def _find_contact_index(frames: list[FramePoseInput], kick_side: str) -> int:
    best_index = 0
    best_distance = float("inf")
    for index, frame in enumerate(frames):
        ball = _ball_center(frame.detection)
        distance = _foot_to_ball_px(frame.landmarks, ball, kick_side)
        if distance is not None and distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def _find_plant_index(frames: list[FramePoseInput], contact_index: int, plant_side: str) -> int:
    plant_positions: list[tuple[int, float | None]] = []
    for index in range(contact_index + 1):
        frame = frames[index]
        plant_positions.append((index, _point(frame.landmarks, f"{plant_side}_ankle")[0] if frame.landmarks else None))

    for index in range(contact_index, 0, -1):
        current = plant_positions[index][1]
        previous = plant_positions[index - 1][1]
        if current is None or previous is None:
            continue
        if abs(current - previous) < PLANT_VELOCITY_THRESHOLD:
            return index
    return max(contact_index // 3, 0)


def _approach_angle_for_frame(frames: list[FramePoseInput], index: int, kick_side: str) -> float | None:
    start = max(0, index - APPROACH_LOOKBACK_FRAMES)
    if index <= start:
        return None
    start_hip = _point(frames[start].landmarks, f"{kick_side}_hip")
    end_hip = _point(frames[index].landmarks, f"{kick_side}_hip")
    ball = _ball_center(frames[index].detection)
    if start_hip is None or end_hip is None or ball is None:
        return None
    move_dx = end_hip[0] - start_hip[0]
    move_dy = end_hip[1] - start_hip[1]
    ball_dx = ball[0] - end_hip[0]
    ball_dy = ball[1] - end_hip[1]
    move_mag = hypot(move_dx, move_dy)
    ball_mag = hypot(ball_dx, ball_dy)
    if move_mag < 1e-3 or ball_mag < 1e-3:
        return None
    move_angle = degrees(float(np.arctan2(move_dy, move_dx)))
    ball_angle = degrees(float(np.arctan2(ball_dy, ball_dx)))
    delta = abs(move_angle - ball_angle)
    if delta > 180:
        delta = 360 - delta
    return delta


def _assign_phases(frames: list[FramePoseInput], contact_index: int, plant_index: int) -> list[str]:
    phases: list[str] = []
    for index in range(len(frames)):
        if index < plant_index:
            phases.append("approach")
        elif index < contact_index:
            phases.append("backswing" if index > plant_index else "plant")
        elif index == contact_index:
            phases.append("contact")
        else:
            phases.append("follow_through")
    if contact_index > 0 and plant_index < contact_index:
        phases[plant_index] = "plant"
    return phases


def analyze_pose_sequence(frames: list[FramePoseInput]) -> tuple[list[TechniqueFrame], int, str]:
    if not frames:
        return [], 0, "right"

    contact_index = _find_contact_index(frames, "right")
    kick_side = _determine_kicking_side(frames, contact_index)
    plant_side = "left" if kick_side == "right" else "right"
    contact_index = _find_contact_index(frames, kick_side)
    plant_index = _find_plant_index(frames, contact_index, plant_side)
    phases = _assign_phases(frames, contact_index, plant_index)

    output: list[TechniqueFrame] = []
    for index, frame in enumerate(frames):
        ball_center = _ball_center(frame.detection)
        approach = _approach_angle_for_frame(frames, index, kick_side)
        angles = _compute_angles(
            frame.landmarks,
            kick_side,
            plant_side,
            frame.frame_id,
            frame.time_s,
            approach,
        )
        output.append(
            TechniqueFrame(
                frame_id=frame.frame_id,
                time_s=frame.time_s,
                angles=angles,
                ball_visible=ball_center is not None,
                foot_to_ball_px=_foot_to_ball_px(frame.landmarks, ball_center, kick_side),
                phase=phases[index],
            )
        )

    contact_frame_id = frames[contact_index].frame_id
    return output, contact_frame_id, kick_side


def process_frame_pose(
    frame: np.ndarray,
    pose: VideoPoseEstimator,
    frame_id: int,
    source_fps: float,
) -> PoseLandmarks:
    return pose.process(frame, frame_id, source_fps)
