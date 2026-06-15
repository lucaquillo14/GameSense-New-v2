"""Local shooting technique metrics when the Roboflow workflow detector returns empty."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.services.rfdetr_detector import detect_objects
from app.services.roboflow_shooting_workflow import DetectionSummary


@dataclass
class BallTrackerState:
    ball_history: list[tuple[int, float, float, float]] = field(default_factory=list)
    frame_index: int = 0
    last_contact_index: int = -9999
    max_ball_speed_px: float = 0.0


def detections_summary_from_rfdetr(parsed: dict[str, Any]) -> DetectionSummary:
    names: list[str] = []
    confidences: list[float] = []
    boxes: list[tuple[float, float, float, float]] = []

    for player in parsed.get("players") or []:
        x, y, w, h, confidence = player
        names.append("person")
        confidences.append(float(confidence))
        boxes.append((float(x), float(y), float(x + w), float(y + h)))

    ball = parsed.get("ball")
    if ball is not None:
        x, y, w, h, confidence = ball
        names.append("sports ball")
        confidences.append(float(confidence))
        boxes.append((float(x), float(y), float(x + w), float(y + h)))

    return DetectionSummary(class_names=names, confidences=confidences, boxes_xyxy=boxes)


def parse_workflow_pose_keypoints(raw: dict[str, Any]) -> dict[str, tuple[float, float]]:
    pose_payload = raw.get("pose_predictions")
    if not isinstance(pose_payload, dict):
        return {}
    predictions = pose_payload.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        return {}
    person = predictions[0]
    if not isinstance(person, dict):
        return {}
    keypoints = person.get("keypoints")
    if not isinstance(keypoints, list):
        return {}

    parsed: dict[str, tuple[float, float]] = {}
    for keypoint in keypoints:
        if not isinstance(keypoint, dict):
            continue
        label = str(keypoint.get("class") or "").strip()
        if not label:
            continue
        parsed[label] = (float(keypoint["x"]), float(keypoint["y"]))
    return parsed


def _valid(point: tuple[float, float] | None) -> bool:
    if point is None:
        return False
    x, y = point
    return math.isfinite(x) and math.isfinite(y) and (x > 0.0 or y > 0.0)


def _angle(
    a: tuple[float, float] | None,
    b: tuple[float, float] | None,
    c: tuple[float, float] | None,
) -> float | None:
    if not (_valid(a) and _valid(b) and _valid(c)):
        return None
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]
    den = max(1e-6, math.hypot(ax, ay) * math.hypot(cx, cy))
    cosv = max(-1.0, min(1.0, (ax * cx + ay * cy) / den))
    return float(math.degrees(math.acos(cosv)))


def _line_angle(
    a: tuple[float, float] | None,
    b: tuple[float, float] | None,
) -> float | None:
    if not (_valid(a) and _valid(b)):
        return None
    return float(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])))


def _center_from_box(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _box_height(box: tuple[float, float, float, float]) -> float:
    return max(1.0, float(box[3] - box[1]))


def compute_shooting_technique_frame(
    *,
    detections: DetectionSummary,
    pose_keypoints: dict[str, tuple[float, float]],
    state: BallTrackerState,
) -> dict[str, Any]:
    """Compute technique score, feedback, and metrics from local detections + workflow pose."""
    import time

    state.frame_index += 1
    metrics: dict[str, Any] = {
        "knee_angle_deg": None,
        "ankle_lock_deg": None,
        "plant_foot_distance_player_heights": None,
        "approach_angle_deg": None,
        "hip_rotation_deg": None,
        "trunk_lean_deg": None,
        "follow_through_height_ratio": None,
        "goal_width_px": None,
        "goal_scale_note": (
            "RF-DETR COCO detects person and sports ball, not soccer goal. "
            "Shot power is scaled from player height until a custom goal detector is added."
        ),
    }

    person_box: tuple[float, float, float, float] | None = None
    ball_center: tuple[float, float] | None = None
    player_height: float | None = None

    for name, box in zip(detections.class_names, detections.boxes_xyxy):
        label = name.lower()
        if label == "person" and person_box is None:
            person_box = box
            player_height = _box_height(box)
        if "ball" in label and ball_center is None:
            ball_center = _center_from_box(box)

    if person_box is None and not pose_keypoints:
        return {
            "technique_score": 4.5,
            "feedback": [
                "No player or ball detected in this frame. Use a side or 45 degree view with the kicker and ball clearly visible.",
                "Ensure good lighting and minimal motion blur.",
            ],
            "metrics": metrics,
            "shot_power_kmh": 0.0,
            "phase": "approach",
            "contact_frame": False,
        }

    if ball_center is not None:
        now = time.time()
        state.ball_history.append((state.frame_index, now, ball_center[0], ball_center[1]))
        state.ball_history = state.ball_history[-12:]

    ball_speed_px = 0.0
    if len(state.ball_history) >= 2:
        _, t1, x1, y1 = state.ball_history[-2]
        _, t2, x2, y2 = state.ball_history[-1]
        dt = max(1.0 / 30.0, float(t2 - t1))
        ball_speed_px = math.hypot(x2 - x1, y2 - y1) / dt
        state.max_ball_speed_px = max(float(state.max_ball_speed_px), float(ball_speed_px))

    if player_height is None and person_box is not None:
        player_height = _box_height(person_box)

    left_hip = pose_keypoints.get("left_hip")
    right_hip = pose_keypoints.get("right_hip")
    left_knee = pose_keypoints.get("left_knee")
    right_knee = pose_keypoints.get("right_knee")
    left_ankle = pose_keypoints.get("left_ankle")
    right_ankle = pose_keypoints.get("right_ankle")
    left_shoulder = pose_keypoints.get("left_shoulder")
    right_shoulder = pose_keypoints.get("right_shoulder")

    pose_ready = _valid(left_hip) and _valid(right_hip) and (_valid(left_knee) or _valid(right_knee))
    feedback: list[str] = []
    score = 6.0
    phase = "approach"
    contact = False

    if pose_ready:
        left_ankle_dist = math.inf
        right_ankle_dist = math.inf
        if ball_center is not None and _valid(left_ankle):
            left_ankle_dist = math.hypot(left_ankle[0] - ball_center[0], left_ankle[1] - ball_center[1])
        if ball_center is not None and _valid(right_ankle):
            right_ankle_dist = math.hypot(right_ankle[0] - ball_center[0], right_ankle[1] - ball_center[1])

        if left_ankle_dist <= right_ankle_dist:
            hip, knee, ankle, plant = left_hip, left_knee, left_ankle, right_ankle
        else:
            hip, knee, ankle, plant = right_hip, right_knee, right_ankle, left_ankle

        knee_ang = _angle(hip, knee, ankle)
        ankle_ang = _angle(knee, ankle, ball_center) if ball_center is not None else None
        metrics["knee_angle_deg"] = None if knee_ang is None else round(knee_ang, 1)
        metrics["ankle_lock_deg"] = None if ankle_ang is None else round(ankle_ang, 1)

        if ball_center is not None and _valid(plant) and player_height is not None:
            dist = math.hypot(plant[0] - ball_center[0], plant[1] - ball_center[1])
            metrics["plant_foot_distance_player_heights"] = round(dist / player_height, 3)

        if len(state.ball_history) >= 3:
            _, _, x0, y0 = state.ball_history[-3]
            _, _, x1, y1 = state.ball_history[-1]
            metrics["approach_angle_deg"] = round(float(math.degrees(math.atan2(y1 - y0, x1 - x0))), 1)

        shoulder_ang = _line_angle(left_shoulder, right_shoulder)
        hip_ang = _line_angle(left_hip, right_hip)
        if shoulder_ang is not None and hip_ang is not None:
            metrics["hip_rotation_deg"] = round(float(abs((shoulder_ang - hip_ang + 180) % 360 - 180)), 1)

        if _valid(left_shoulder) and _valid(right_shoulder) and _valid(left_hip) and _valid(right_hip):
            shoulder_mid = (
                (left_shoulder[0] + right_shoulder[0]) / 2.0,
                (left_shoulder[1] + right_shoulder[1]) / 2.0,
            )
            hip_mid = (
                (left_hip[0] + right_hip[0]) / 2.0,
                (left_hip[1] + right_hip[1]) / 2.0,
            )
            lean = abs(
                math.degrees(
                    math.atan2(shoulder_mid[0] - hip_mid[0], hip_mid[1] - shoulder_mid[1])
                )
            )
            metrics["trunk_lean_deg"] = round(float(lean), 1)

        if _valid(ankle) and _valid(hip) and player_height is not None:
            metrics["follow_through_height_ratio"] = round(
                float(max(0.0, (hip[1] - ankle[1]) / player_height)),
                3,
            )

        contact_dist = min(left_ankle_dist, right_ankle_dist)
        contact = bool(
            ball_center is not None
            and player_height is not None
            and contact_dist < 0.18 * player_height
        )
        if contact and state.frame_index - state.last_contact_index > 8:
            state.last_contact_index = state.frame_index
        if state.frame_index - state.last_contact_index <= 4:
            phase = "contact / strike"
        elif ball_speed_px > 250:
            phase = "follow-through"
        else:
            phase = "approach"

        def grade(name: str, value: float | None, lo: float, hi: float, good: str, bad: str) -> None:
            nonlocal score
            if value is None:
                feedback.append(f"Keep the full body visible so I can measure {name}.")
                score -= 0.4
            elif lo <= float(value) <= hi:
                feedback.append(good)
                score += 0.3
            else:
                feedback.append(bad.format(value=float(value)))
                score -= min(1.0, abs(float(value) - (lo + hi) / 2.0) / max(1.0, hi - lo))

        grade(
            "plant foot spacing",
            metrics["plant_foot_distance_player_heights"],
            0.06,
            0.22,
            "Plant foot spacing looks close enough to the ball for a stable strike.",
            "Plant foot is {value:.2f} player-heights from the ball. Aim for roughly 0.06 to 0.22.",
        )
        grade(
            "trunk lean",
            metrics["trunk_lean_deg"],
            5,
            25,
            "Trunk lean is in a strong range for keeping the shot driven.",
            "Trunk lean is {value:.1f} degrees. Lean slightly over the ball, about 5 to 25 degrees.",
        )
        grade(
            "hip rotation",
            metrics["hip_rotation_deg"],
            10,
            45,
            "Hip and shoulder separation suggests useful rotation through the strike.",
            "Hip rotation estimate is {value:.1f} degrees. Try opening the hips more through contact.",
        )

        if metrics["knee_angle_deg"] is not None:
            if metrics["knee_angle_deg"] < 120:
                feedback.append("Knee is very bent at contact. Drive the thigh through and extend more after the strike.")
                score -= 0.7
            elif metrics["knee_angle_deg"] > 175:
                feedback.append("Leg is nearly straight. Keep a small knee bend to stay balanced.")
                score -= 0.4
            else:
                feedback.append("Knee extension is in a usable striking range.")
                score += 0.2

        if metrics["follow_through_height_ratio"] is not None:
            if metrics["follow_through_height_ratio"] < 0.05 and phase == "follow-through":
                feedback.append("Follow-through looks low. Let the kicking foot continue up toward the target.")
                score -= 0.5
            else:
                feedback.append("Follow-through height is acceptable for this frame.")
        feedback = feedback[:6]
    elif person_box is not None or ball_center is not None:
        feedback = [
            "Player and ball were detected, but pose landmarks were not confident enough for joint angles.",
            "Use a side or 45 degree view with the full kicking leg and plant foot visible.",
        ]
        score = 5.5
    else:
        feedback = [
            "No player or ball detected in this frame. Use a side or 45 degree view with the kicker and ball clearly visible.",
            "Ensure good lighting and minimal motion blur.",
        ]
        score = 4.5

    shot_power_kmh = 0.0
    if player_height is not None and state.max_ball_speed_px > 0:
        meters_per_px = 1.75 / player_height
        shot_power_kmh = round(float(state.max_ball_speed_px) * meters_per_px * 3.6, 1)

    return {
        "technique_score": max(0.0, min(10.0, round(float(score), 1))),
        "feedback": feedback,
        "metrics": metrics,
        "shot_power_kmh": float(shot_power_kmh),
        "phase": phase,
        "contact_frame": bool(contact),
    }


def draw_technique_overlay(
    frame_bgr: np.ndarray,
    detections: DetectionSummary,
    *,
    technique_score: float,
    phase: str,
    shot_power_kmh: float,
) -> np.ndarray:
    """Draw detector boxes and technique summary on a BGR frame."""
    annotated = frame_bgr.copy()
    color_map = {
        "person": (255, 128, 0),
        "sports ball": (0, 255, 255),
    }
    for name, box in zip(detections.class_names, detections.boxes_xyxy):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = color_map.get(name.lower(), (0, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            name,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    banner = (
        f"Technique: {technique_score:.1f}/10 | Phase: {phase} | "
        f"Shot power: {shot_power_kmh:.1f} km/h (prototype scale)"
    )
    cv2.rectangle(annotated, (8, 8), (min(annotated.shape[1] - 8, 980), 52), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        banner,
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def analyze_frame_with_local_detections(
    frame_bgr: np.ndarray,
    workflow_raw: dict[str, Any],
    state: BallTrackerState,
    *,
    output_image_dir: Path | None = None,
    output_image_name: str = "workflow_output.jpg",
) -> tuple[dict[str, Any], DetectionSummary, Path | None]:
    """Merge workflow pose with reliable local RF-DETR detections."""
    parsed = detect_objects(frame_bgr)
    detections = detections_summary_from_rfdetr(parsed)
    pose_keypoints = parse_workflow_pose_keypoints(workflow_raw)
    computed = compute_shooting_technique_frame(
        detections=detections,
        pose_keypoints=pose_keypoints,
        state=state,
    )

    output_image_path: Path | None = None
    if output_image_dir is not None:
        output_image_path = Path(output_image_dir) / output_image_name
        overlay = draw_technique_overlay(
            frame_bgr,
            detections,
            technique_score=float(computed["technique_score"]),
            phase=str(computed["phase"]),
            shot_power_kmh=float(computed["shot_power_kmh"]),
        )
        output_image_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_image_path), overlay)

    return computed, detections, output_image_path
