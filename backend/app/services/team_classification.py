from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

TEAM_CALIBRATION_FRAMES_INITIAL = 30
TEAM_CALIBRATION_FRAMES_EXTENDED = 60
MIN_SAMPLES_PER_CLUSTER = 8
KMEANS_RESTARTS = 10
REFEREE_TEAM_COLLAPSE_DISTANCE = 15.0
MAX_BACKGROUND_BLEED_RATIO = 0.30
TEMPORAL_HISTORY_FRAMES = 8
TEMPORAL_CONFIRM_VOTES = 5
CONFLICT_MAX_SAME_TEAM = 7

TEAM_A_LABEL = "Team A"
TEAM_B_LABEL = "Team B"

TeamId = Literal["team_a", "team_b", "referee"]
ConfirmedTeam = Literal["team_a", "team_b", "unconfirmed", "referee"]


@dataclass
class TeamTemplates:
    team_a_lab: np.ndarray
    team_b_lab: np.ndarray
    team_a_color_rgb: tuple[int, int, int]
    team_b_color_rgb: tuple[int, int, int]
    referee_lab: np.ndarray | None = None
    referee_enabled: bool = False
    calibration_frames: int = TEAM_CALIBRATION_FRAMES_INITIAL
    warnings: list[str] = field(default_factory=list)

    @property
    def team_a_histogram(self) -> np.ndarray:
        return self.team_a_lab

    @property
    def team_b_histogram(self) -> np.ndarray:
        return self.team_b_lab

    def to_dict(self) -> dict:
        payload = {
            "team_a": {
                "lab": self.team_a_lab.astype(float).tolist(),
                "display_color": {
                    "r": self.team_a_color_rgb[0],
                    "g": self.team_a_color_rgb[1],
                    "b": self.team_a_color_rgb[2],
                },
            },
            "team_b": {
                "lab": self.team_b_lab.astype(float).tolist(),
                "display_color": {
                    "r": self.team_b_color_rgb[0],
                    "g": self.team_b_color_rgb[1],
                    "b": self.team_b_color_rgb[2],
                },
            },
            "calibration_frames": self.calibration_frames,
            "referee_enabled": self.referee_enabled,
            "warnings": self.warnings,
        }
        if self.referee_lab is not None:
            payload["referee"] = {"lab": self.referee_lab.astype(float).tolist()}
        return payload

    @classmethod
    def _lab_from_entry(cls, entry: dict) -> np.ndarray:
        if "lab" in entry:
            return np.array(entry["lab"], dtype=np.float32)
        return np.array(entry["histogram"], dtype=np.float32)

    @classmethod
    def from_dict(cls, payload: dict) -> TeamTemplates:
        team_a_lab = cls._lab_from_entry(payload["team_a"])
        team_b_lab = cls._lab_from_entry(payload["team_b"])
        color_a = payload["team_a"]["display_color"]
        color_b = payload["team_b"]["display_color"]
        referee_lab = None
        referee_enabled = bool(payload.get("referee_enabled", False))
        if payload.get("referee"):
            referee_lab = cls._lab_from_entry(payload["referee"])
            referee_enabled = True
        return cls(
            team_a_lab=team_a_lab,
            team_b_lab=team_b_lab,
            team_a_color_rgb=(int(color_a["r"]), int(color_a["g"]), int(color_a["b"])),
            team_b_color_rgb=(int(color_b["r"]), int(color_b["g"]), int(color_b["b"])),
            referee_lab=referee_lab,
            referee_enabled=referee_enabled,
            calibration_frames=int(payload.get("calibration_frames", TEAM_CALIBRATION_FRAMES_INITIAL)),
            warnings=list(payload.get("warnings") or []),
        )


def _bbox_tuple(bbox: dict | tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if isinstance(bbox, dict):
        return float(bbox["x"]), float(bbox["y"]), float(bbox["width"]), float(bbox["height"])
    return tuple(float(value) for value in bbox)


def _background_bleed_ratio(crop_bgr: np.ndarray) -> float:
    if crop_bgr.size == 0:
        return 1.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    white_mask = (saturation < 40) & (value > 200)
    black_mask = value < 45
    invalid = white_mask | black_mask
    return float(invalid.sum()) / float(invalid.size)


def extract_shirt_lab_sample(
    frame: np.ndarray,
    bbox: dict | tuple[float, float, float, float],
) -> np.ndarray | None:
    x, y, w, h = _bbox_tuple(bbox)
    frame_h, frame_w = frame.shape[:2]

    left = int(round(x + w * 0.20))
    right = int(round(x + w * 0.80))
    top = int(round(y + h * 0.15))
    bottom = int(round(y + h * 0.45))

    left = min(max(left, 0), max(frame_w - 1, 0))
    right = min(max(right, left + 1), frame_w)
    top = min(max(top, 0), max(frame_h - 1, 0))
    bottom = min(max(bottom, top + 1), frame_h)

    crop = frame[top:bottom, left:right]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None

    if _background_bleed_ratio(crop) > MAX_BACKGROUND_BLEED_RATIO:
        return None

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    mean_lab = lab.reshape(-1, 3).mean(axis=0).astype(np.float32)
    return mean_lab


def extract_shirt_histogram(frame: np.ndarray, bbox: dict | tuple[float, float, float, float]) -> np.ndarray | None:
    """Backward-compatible alias returning the Lab shirt sample."""
    return extract_shirt_lab_sample(frame, bbox)


def lab_distance(left: np.ndarray, right: np.ndarray) -> float:
    delta = left.astype(np.float32) - right.astype(np.float32)
    return float(np.linalg.norm(delta))


def histogram_distance(left: np.ndarray, right: np.ndarray) -> float:
    return lab_distance(left, right)


def classify_team(lab_sample: np.ndarray, templates: TeamTemplates) -> TeamId:
    distance_a = lab_distance(lab_sample, templates.team_a_lab)
    distance_b = lab_distance(lab_sample, templates.team_b_lab)
    if templates.referee_enabled and templates.referee_lab is not None:
        distance_ref = lab_distance(lab_sample, templates.referee_lab)
        nearest = min(
            (distance_a, "team_a"),
            (distance_b, "team_b"),
            (distance_ref, "referee"),
            key=lambda item: item[0],
        )
        return nearest[1]  # type: ignore[return-value]
    return "team_a" if distance_a <= distance_b else "team_b"


def team_label(team_id: Literal["team_a", "team_b"]) -> str:
    return TEAM_A_LABEL if team_id == "team_a" else TEAM_B_LABEL


def team_color_hex(templates: TeamTemplates, team_id: Literal["team_a", "team_b"]) -> str:
    rgb = templates.team_a_color_rgb if team_id == "team_a" else templates.team_b_color_rgb
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _lab_to_display_rgb(lab: np.ndarray) -> tuple[int, int, int]:
    patch = np.uint8([[[int(lab[0]), int(lab[1]), int(lab[2])]]])
    bgr = cv2.cvtColor(patch, cv2.COLOR_LAB2BGR)[0][0]
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def _kmeans_best_inertia(data: np.ndarray, cluster_count: int) -> tuple[np.ndarray, np.ndarray, float]:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.1)
    best_centers: np.ndarray | None = None
    best_labels: np.ndarray | None = None
    best_inertia = float("inf")

    for seed in range(KMEANS_RESTARTS):
        attempt = cv2.kmeans(
            data.astype(np.float32),
            cluster_count,
            None,
            criteria,
            1,
            cv2.KMEANS_RANDOM_CENTERS,
        )
        _compactness, labels, centers = attempt
        inertia = float(_compactness)
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers
            best_labels = labels.flatten()

    assert best_centers is not None and best_labels is not None
    return best_centers, best_labels, best_inertia


def _cluster_counts(labels: np.ndarray, cluster_count: int) -> dict[int, int]:
    counts: dict[int, int] = {index: 0 for index in range(cluster_count)}
    for label in labels.tolist():
        counts[int(label)] = counts.get(int(label), 0) + 1
    return counts


def _templates_from_samples(samples: list[np.ndarray], frames_used: int) -> TeamTemplates:
    warnings: list[str] = []
    if len(samples) < 2:
        return _default_templates(warnings=["Insufficient shirt samples for team calibration."])

    data = np.array(samples, dtype=np.float32)
    centers3, labels3, _ = _kmeans_best_inertia(data, 3)
    counts3 = _cluster_counts(labels3, 3)
    referee_cluster = min(counts3, key=counts3.get)
    team_clusters = [index for index in counts3 if index != referee_cluster]
    referee_centroid = centers3[referee_cluster]
    referee_enabled = True

    if len(team_clusters) < 2:
        return _default_templates(warnings=["Could not separate two team colour clusters."])

    team_a_cluster, team_b_cluster = sorted(team_clusters)[:2]
    if lab_distance(centers3[referee_cluster], centers3[team_a_cluster]) < REFEREE_TEAM_COLLAPSE_DISTANCE:
        referee_enabled = False
        warnings.append("Referee kit matched a team kit — using two-team mode for this video.")
    if lab_distance(centers3[referee_cluster], centers3[team_b_cluster]) < REFEREE_TEAM_COLLAPSE_DISTANCE:
        referee_enabled = False
        warnings.append("Referee kit matched a team kit — using two-team mode for this video.")

    if not referee_enabled:
        centers2, _labels2, _ = _kmeans_best_inertia(data, 2)
        team_a_lab = centers2[0].flatten().astype(np.float32)
        team_b_lab = centers2[1].flatten().astype(np.float32)
        referee_lab = None
    else:
        team_a_lab = centers3[team_a_cluster].flatten().astype(np.float32)
        team_b_lab = centers3[team_b_cluster].flatten().astype(np.float32)
        referee_lab = referee_centroid.flatten().astype(np.float32)

    return TeamTemplates(
        team_a_lab=team_a_lab,
        team_b_lab=team_b_lab,
        team_a_color_rgb=_lab_to_display_rgb(team_a_lab),
        team_b_color_rgb=_lab_to_display_rgb(team_b_lab),
        referee_lab=referee_lab,
        referee_enabled=referee_enabled,
        calibration_frames=frames_used,
        warnings=warnings,
    )


def _default_templates(warnings: list[str] | None = None) -> TeamTemplates:
    team_a_lab = np.array([60.0, 180.0, 150.0], dtype=np.float32)
    team_b_lab = np.array([40.0, 150.0, 40.0], dtype=np.float32)
    return TeamTemplates(
        team_a_lab=team_a_lab,
        team_b_lab=team_b_lab,
        team_a_color_rgb=(220, 38, 38),
        team_b_color_rgb=(37, 99, 235),
        referee_enabled=False,
        warnings=warnings or [],
    )


def _samples_per_cluster(samples: list[np.ndarray], templates: TeamTemplates) -> tuple[int, int]:
    team_a_count = 0
    team_b_count = 0
    for sample in samples:
        team_id = classify_team(sample, templates)
        if team_id == "team_a":
            team_a_count += 1
        elif team_id == "team_b":
            team_b_count += 1
    return team_a_count, team_b_count


def build_team_templates(
    video_path: Path,
    detect_people,
    resize_frame,
    unscale_bbox,
    calibration_frames: int = TEAM_CALIBRATION_FRAMES_INITIAL,
) -> TeamTemplates:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return _default_templates(["Could not open video for team calibration."])

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = TEAM_CALIBRATION_FRAMES_EXTENDED if calibration_frames >= TEAM_CALIBRATION_FRAMES_INITIAL else calibration_frames
    frames_to_scan = min(max_frames, frame_count if frame_count > 0 else max_frames)

    def collect_until(target_frames: int) -> list[np.ndarray]:
        samples: list[np.ndarray] = []
        for frame_id in range(target_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = capture.read()
            if not ok:
                continue
            resized, scale = resize_frame(frame)
            for bbox in detect_people(resized):
                x, y, w, h, _confidence = bbox
                original_bbox = unscale_bbox((x, y, w, h), scale)
                sample = extract_shirt_lab_sample(frame, original_bbox)
                if sample is not None:
                    samples.append(sample)
        return samples

    samples = collect_until(frames_to_scan)
    templates = _templates_from_samples(samples, frames_to_scan)
    team_a_count, team_b_count = _samples_per_cluster(samples, templates)

    if min(team_a_count, team_b_count) < MIN_SAMPLES_PER_CLUSTER and frames_to_scan < TEAM_CALIBRATION_FRAMES_EXTENDED:
        extended_samples = collect_until(TEAM_CALIBRATION_FRAMES_EXTENDED)
        if len(extended_samples) >= len(samples):
            samples = extended_samples
            templates = _templates_from_samples(samples, TEAM_CALIBRATION_FRAMES_EXTENDED)
            team_a_count, team_b_count = _samples_per_cluster(samples, templates)

    capture.release()

    if min(team_a_count, team_b_count) < MIN_SAMPLES_PER_CLUSTER:
        templates.warnings.append(
            "Team colour templates are low-confidence — fewer than 8 samples per team were collected."
        )
    return templates


class TeamTemporalSmoother:
    def __init__(self) -> None:
        self._history: dict[str, deque[TeamId | None]] = {}

    def reset(self) -> None:
        self._history.clear()

    def record(self, player_key: str, raw_team: TeamId | None) -> ConfirmedTeam:
        history = self._history.setdefault(player_key, deque(maxlen=TEMPORAL_HISTORY_FRAMES))
        history.append(raw_team)
        if raw_team == "referee":
            return "referee"

        valid = [team for team in history if team in {"team_a", "team_b"}]
        if not valid:
            return "unconfirmed"

        counts = Counter(valid)
        team, votes = counts.most_common(1)[0]
        if votes >= TEMPORAL_CONFIRM_VOTES:
            return team  # type: ignore[return-value]
        return "unconfirmed"


def detect_team_conflict(team_counts: dict[str, int]) -> bool:
    return max(team_counts.values(), default=0) > CONFLICT_MAX_SAME_TEAM


@dataclass
class TeamClassificationService:
    templates: TeamTemplates | None = None
    smoother: TeamTemporalSmoother = field(default_factory=TeamTemporalSmoother)
    warnings: list[str] = field(default_factory=list)

    def set_templates(self, templates: TeamTemplates, *, reset_history: bool = True) -> None:
        changed = self.templates is not templates
        self.templates = templates
        self.warnings = list(templates.warnings)
        if changed or reset_history:
            self.smoother.reset()

    def reset(self) -> None:
        self.smoother.reset()

    def classify_player(
        self,
        frame: np.ndarray,
        bbox: dict | tuple[float, float, float, float],
        player_key: str,
        *,
        apply_temporal: bool = True,
    ) -> tuple[ConfirmedTeam, np.ndarray | None]:
        if self.templates is None:
            return "unconfirmed", None

        sample = extract_shirt_lab_sample(frame, bbox)
        if sample is None:
            if apply_temporal:
                return self.smoother.record(player_key, None), None
            return "unconfirmed", None

        raw_team = classify_team(sample, self.templates)
        if raw_team == "referee":
            if apply_temporal:
                return self.smoother.record(player_key, "referee"), sample
            return "referee", sample

        if apply_temporal:
            return self.smoother.record(player_key, raw_team), sample
        return raw_team, sample

    def classify_frame_counts(self, confirmed_teams: list[ConfirmedTeam]) -> dict[str, int]:
        counts = {"team_a": 0, "team_b": 0, "unconfirmed": 0, "referee": 0}
        for team in confirmed_teams:
            counts[team] = counts.get(team, 0) + 1
        return counts
