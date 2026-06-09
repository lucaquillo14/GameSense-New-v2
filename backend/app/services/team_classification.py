from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

TEAM_CALIBRATION_FRAMES = 10
REFEREE_DISTANCE_THRESHOLD = 0.55
TEAM_A_LABEL = "Team A"
TEAM_B_LABEL = "Team B"

TeamId = Literal["team_a", "team_b", "referee"]


@dataclass
class TeamTemplates:
    team_a_histogram: np.ndarray
    team_b_histogram: np.ndarray
    team_a_color_rgb: tuple[int, int, int]
    team_b_color_rgb: tuple[int, int, int]

    def to_dict(self) -> dict:
        return {
            "team_a": {
                "histogram": self.team_a_histogram.astype(float).tolist(),
                "display_color": {
                    "r": self.team_a_color_rgb[0],
                    "g": self.team_a_color_rgb[1],
                    "b": self.team_a_color_rgb[2],
                },
            },
            "team_b": {
                "histogram": self.team_b_histogram.astype(float).tolist(),
                "display_color": {
                    "r": self.team_b_color_rgb[0],
                    "g": self.team_b_color_rgb[1],
                    "b": self.team_b_color_rgb[2],
                },
            },
            "calibration_frames": TEAM_CALIBRATION_FRAMES,
            "referee_distance_threshold": REFEREE_DISTANCE_THRESHOLD,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> TeamTemplates:
        team_a = np.array(payload["team_a"]["histogram"], dtype=np.float32)
        team_b = np.array(payload["team_b"]["histogram"], dtype=np.float32)
        color_a = payload["team_a"]["display_color"]
        color_b = payload["team_b"]["display_color"]
        return cls(
            team_a_histogram=_normalize_histogram(team_a),
            team_b_histogram=_normalize_histogram(team_b),
            team_a_color_rgb=(int(color_a["r"]), int(color_a["g"]), int(color_a["b"])),
            team_b_color_rgb=(int(color_b["r"]), int(color_b["g"]), int(color_b["b"])),
        )


def _normalize_histogram(histogram: np.ndarray) -> np.ndarray:
    total = float(histogram.sum())
    if total <= 0:
        return histogram.astype(np.float32)
    return (histogram / total).astype(np.float32)


def extract_shirt_histogram(frame: np.ndarray, bbox: dict | tuple[float, float, float, float]) -> np.ndarray | None:
    if isinstance(bbox, dict):
        x = float(bbox["x"])
        y = float(bbox["y"])
        w = float(bbox["width"])
        h = float(bbox["height"])
    else:
        x, y, w, h = [float(value) for value in bbox]

    frame_h, frame_w = frame.shape[:2]
    left = min(max(int(round(x)), 0), max(frame_w - 1, 0))
    top = min(max(int(round(y)), 0), max(frame_h - 1, 0))
    right = min(max(int(round(x + w)), left + 1), frame_w)
    bottom = min(max(int(round(y + h)), top + 1), frame_h)
    crop_w = right - left
    crop_h = bottom - top
    if crop_w < 4 or crop_h < 8:
        return None

    shirt_bottom = top + max(int(crop_h * 0.50), 4)
    crop = frame[top:shirt_bottom, left:right]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
    return _normalize_histogram(histogram.flatten())


def histogram_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        cv2.compareHist(
            left.astype(np.float32),
            right.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA,
        )
    )


def classify_team(
    histogram: np.ndarray,
    templates: TeamTemplates,
) -> TeamId:
    distance_a = histogram_distance(histogram, templates.team_a_histogram)
    distance_b = histogram_distance(histogram, templates.team_b_histogram)
    if distance_a > REFEREE_DISTANCE_THRESHOLD and distance_b > REFEREE_DISTANCE_THRESHOLD:
        return "referee"
    return "team_a" if distance_a <= distance_b else "team_b"


def team_label(team_id: Literal["team_a", "team_b"]) -> str:
    return TEAM_A_LABEL if team_id == "team_a" else TEAM_B_LABEL


def team_color_hex(templates: TeamTemplates, team_id: Literal["team_a", "team_b"]) -> str:
    rgb = templates.team_a_color_rgb if team_id == "team_a" else templates.team_b_color_rgb
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _histogram_to_display_color(histogram: np.ndarray) -> tuple[int, int, int]:
    hist = histogram.reshape(16, 16)
    hue_weights = np.arange(16, dtype=np.float32) * (180.0 / 16.0)
    sat_weights = np.arange(16, dtype=np.float32) * (256.0 / 16.0)
    hue = float((hist.sum(axis=1) * hue_weights).sum() / max(hist.sum(), 1e-6))
    sat = float((hist.sum(axis=0) * sat_weights).sum() / max(hist.sum(), 1e-6))
    value = 200.0
    hsv_pixel = np.uint8([[[int(hue), int(min(sat, 255)), int(value)]]])
    bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def _default_templates() -> TeamTemplates:
    red = np.zeros(256, dtype=np.float32)
    red[0:128] = 1.0
    blue = np.zeros(256, dtype=np.float32)
    blue[128:256] = 1.0
    red = _normalize_histogram(red)
    blue = _normalize_histogram(blue)
    return TeamTemplates(
        team_a_histogram=red,
        team_b_histogram=blue,
        team_a_color_rgb=(220, 38, 38),
        team_b_color_rgb=(37, 99, 235),
    )


def build_team_templates(
    video_path: Path,
    detect_people,
    resize_frame,
    unscale_bbox,
    calibration_frames: int = TEAM_CALIBRATION_FRAMES,
) -> TeamTemplates:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return _default_templates()

    features: list[np.ndarray] = []
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames_to_scan = min(calibration_frames, max(frame_count, calibration_frames))

    for frame_id in range(frames_to_scan):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        if not ok:
            continue

        resized, scale = resize_frame(frame)
        for bbox in detect_people(resized):
            x, y, w, h, _confidence = bbox
            original_bbox = unscale_bbox((x, y, w, h), scale)
            histogram = extract_shirt_histogram(frame, original_bbox)
            if histogram is not None:
                features.append(histogram)

    capture.release()

    if len(features) < 2:
        return _default_templates()

    data = np.array(features, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    _compactness, _labels, centers = cv2.kmeans(
        data,
        2,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS,
    )

    team_a = _normalize_histogram(centers[0])
    team_b = _normalize_histogram(centers[1])
    return TeamTemplates(
        team_a_histogram=team_a,
        team_b_histogram=team_b,
        team_a_color_rgb=_histogram_to_display_color(team_a),
        team_b_color_rgb=_histogram_to_display_color(team_b),
    )
