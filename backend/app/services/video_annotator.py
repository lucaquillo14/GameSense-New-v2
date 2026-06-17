from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.models import ShootingFeedback, TechniqueFrame

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None

ACCENT_BGR = (235, 99, 37)   # #2563eb
PLANT_BGR = (129, 185, 16)   # #10b981
WHITE = (255, 255, 255)

KICK_JOINTS = {
    "left": {"left_hip", "left_knee", "left_ankle", "left_foot_index"},
    "right": {"right_hip", "right_knee", "right_ankle", "right_foot_index"},
}
PLANT_JOINTS = {
    "left": {"right_hip", "right_knee", "right_ankle", "right_foot_index"},
    "right": {"left_hip", "left_knee", "left_ankle", "left_foot_index"},
}

POSE_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_foot_index"),
]


def _landmark_point(landmarks: dict, name: str) -> tuple[int, int] | None:
    value = landmarks.get(name)
    if not value or len(value) < 2:
        return None
    return int(value[0]), int(value[1])


def _draw_text_pill(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    *,
    font_scale: float = 0.55,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    pad = 6
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - pad, y - text_h - pad),
        (x + text_w + pad, y + baseline + pad),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, WHITE, thickness, cv2.LINE_AA)


def _draw_phase_badge(frame: np.ndarray, phase: str) -> None:
    label = phase.replace("_", " ").title()
    _draw_text_pill(frame, label, (16, 36), font_scale=0.6)


def _draw_power_readout(frame: np.ndarray, value: float, frame_width: int) -> None:
    text = f"{value:.0f}"
    unit = "km/h"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.4
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = frame_width - text_w - 24
    y = 56
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 12, y - text_h - 12), (frame_width - 8, y + 28), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (x, y), font, scale, WHITE, thickness, cv2.LINE_AA)
    cv2.putText(frame, unit, (x, y + 24), font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)


def _draw_skeleton(
    frame: np.ndarray,
    landmarks: dict,
    kick_side: str,
) -> None:
    overlay = frame.copy()
    kick_set = KICK_JOINTS[kick_side]
    plant_set = PLANT_JOINTS[kick_side]

    for start_name, end_name in POSE_CONNECTIONS:
        start = _landmark_point(landmarks, start_name)
        end = _landmark_point(landmarks, end_name)
        if start is None or end is None:
            continue
        color = WHITE
        if start_name in kick_set and end_name in kick_set:
            color = ACCENT_BGR
        elif start_name in plant_set and end_name in plant_set:
            color = PLANT_BGR
        cv2.line(overlay, start, end, color, 2, cv2.LINE_AA)

    for name, point in ((key, _landmark_point(landmarks, key)) for key in landmarks):
        if point is None:
            continue
        if name in kick_set:
            color = ACCENT_BGR
        elif name in plant_set:
            color = PLANT_BGR
        else:
            color = WHITE
        cv2.circle(overlay, point, 4, color, -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)


def _draw_angle_labels(frame: np.ndarray, technique_frame: TechniqueFrame, landmarks: dict, kick_side: str) -> None:
    label_map = {
        "knee_angle": (f"{kick_side}_knee", "Knee"),
        "ankle_angle": (f"{kick_side}_ankle", "Ankle"),
        "plant_knee_angle": (f"{'left' if kick_side == 'right' else 'right'}_knee", "Plant knee"),
        "hip_angle": (f"{kick_side}_hip", "Hip"),
    }
    for angle in technique_frame.angles:
        mapping = label_map.get(angle.name)
        if mapping is None:
            continue
        joint_name, label = mapping
        point = _landmark_point(landmarks, joint_name)
        if point is None:
            continue
        _draw_text_pill(frame, f"{label}: {angle.value_deg:.0f}°", (point[0] + 8, point[1] - 8))


def _draw_ball(
    frame: np.ndarray,
    detection: dict,
    kalman_position: tuple[float, float] | None,
    interpolated: bool,
) -> None:
    if detection.get("ball"):
        x, y, w, h, _conf = detection["ball"]
        center = (int(x + w / 2), int(y + h / 2))
        radius = max(6, int(max(w, h) / 2))
        if interpolated:
            overlay = frame.copy()
            cv2.circle(overlay, center, radius, WHITE, 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        else:
            cv2.circle(frame, center, radius, WHITE, 1, cv2.LINE_AA)
    elif kalman_position is not None:
        center = (int(kalman_position[0]), int(kalman_position[1]))
        overlay = frame.copy()
        cv2.circle(overlay, center, 8, WHITE, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)


def _draw_goal(frame: np.ndarray, goal: tuple[float, float, float, float, float] | None) -> None:
    if goal is None:
        return
    x1, y1, x2, y2, _conf = goal
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), WHITE, 1, cv2.LINE_AA)


def annotate_video(
    video_path: Path,
    video_id: str,
    pose_frames: list[TechniqueFrame],
    frame_detections: list[dict],
    feedback: ShootingFeedback,
    *,
    kick_side: str = "right",
) -> str:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for annotation: {video_path}")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    if source_fps <= 0:
        source_fps = 30.0

    output_path = video_path.parent / f"annotated-{video_id}.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create annotated video writer.")

    pose_by_frame = {frame.frame_id: frame for frame in pose_frames}
    detection_by_frame = {entry["frame_id"]: entry for entry in frame_detections}
    contact_frame_id = feedback.contact_frame_id or 0
    power_frames = 20
    frame_id = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        entry = detection_by_frame.get(frame_id, {})
        detection = entry.get("detection", {})
        landmarks = entry.get("landmarks", {})
        technique_frame = pose_by_frame.get(frame_id)
        kalman = entry.get("ball_kalman")
        interpolated = bool(entry.get("ball_interpolated", False))

        if landmarks:
            _draw_skeleton(frame, landmarks, kick_side)
        _draw_goal(frame, detection.get("goal"))
        _draw_ball(frame, detection, kalman, interpolated)

        if technique_frame:
            _draw_phase_badge(frame, technique_frame.phase)
            if contact_frame_id and abs(frame_id - contact_frame_id) <= 5 and landmarks:
                _draw_angle_labels(frame, technique_frame, landmarks, kick_side)

        if contact_frame_id and frame_id >= contact_frame_id:
            elapsed = frame_id - contact_frame_id
            ratio = min(1.0, elapsed / max(power_frames, 1))
            display_power = feedback.shot_power_kmh * ratio
            _draw_power_readout(frame, display_power, frame_width)

        writer.write(frame)
        frame_id += 1

    writer.release()
    capture.release()

    # OpenCV's mp4v output isn't browser-playable; transcode to H.264 via ffmpeg.
    from app.services.shooting_technique_pipeline import _ensure_browser_playable

    _ensure_browser_playable(output_path)

    return f"/media/{video_id}/annotated-{video_id}.mp4"
