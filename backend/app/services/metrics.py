import os
from dataclasses import dataclass
from math import hypot

import numpy as np

from app.models import ShotEvent


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name, "").strip()
        return float(raw) if raw else default
    except (TypeError, ValueError):
        return default


# Physical ceiling on human acceleration (m/s^2). Real sprint accelerations
# peak around 8-10 m/s^2; anything well above this between two samples is an
# ID switch or calibration jump, not real motion. Tune via env if needed.
MAX_ACCEL_MPS2 = _env_float("GAMESENSE_MAX_ACCEL_MPS2", 15.0)


@dataclass
class TrackPoint:
    frame_id: int
    time_s: float
    x_px: float
    y_px: float
    x_m: float | None = None
    y_m: float | None = None
    calibrated: bool = False
    confidence: float = 1.0
    track_state: str = "visible"
    left_foot_px: tuple[float, float] | None = None
    right_foot_px: tuple[float, float] | None = None
    cal_age_s: float = 0.0   # seconds since the last absolute calibration fix

    def coord_x(self, units: str) -> float:
        if units == "metric" and self.calibrated and self.x_m is not None:
            return self.x_m
        return self.x_px

    def coord_y(self, units: str) -> float:
        if units == "metric" and self.calibrated and self.y_m is not None:
            return self.y_m
        return self.y_px


@dataclass
class BallTrackPoint:
    frame_id: int
    time_s: float
    x_px: float
    y_px: float
    x_m: float = 0.0
    y_m: float = 0.0
    calibrated: bool = False
    confidence: float = 1.0
    interpolated: bool = False


def stabilize_track_points(
    points: list[TrackPoint],
    max_speed_kmh: float = 38.0,
    max_speed_px_per_s: float = 420.0,
    median_window: int = 5,
    units: str = "metric",
) -> tuple[list[TrackPoint], int]:
    raw_count = len(points)
    if len(points) < 3:
        print(f"[GameSense] stabilize_track_points: raw={raw_count} stabilized={raw_count} rejected=0")
        return sorted(points, key=lambda point: point.frame_id), 0

    ordered = sorted(points, key=lambda point: point.frame_id)
    smoothed: list[TrackPoint] = []
    half_window = max(median_window // 2, 1)
    for index, point in enumerate(ordered):
        start = max(index - half_window, 0)
        end = min(index + half_window + 1, len(ordered))
        window = ordered[start:end]
        smoothed.append(
            TrackPoint(
                frame_id=point.frame_id,
                time_s=point.time_s,
                x_px=float(np.median([item.x_px for item in window])),
                y_px=float(np.median([item.y_px for item in window])),
                x_m=float(np.median([item.x_m for item in window if item.x_m is not None])) if any(item.x_m is not None for item in window) else None,
                y_m=float(np.median([item.y_m for item in window if item.y_m is not None])) if any(item.y_m is not None for item in window) else None,
                calibrated=point.calibrated,
                confidence=float(np.mean([item.confidence for item in window])),
            )
        )

    cleaned = [smoothed[0]]
    rejected = 0
    max_speed_rate = max_speed_kmh / 3.6 if units == "metric" else max_speed_px_per_s
    prev_speed: float | None = None  # last accepted speed (m/s), metric units only
    for point in smoothed[1:]:
        previous = cleaned[-1]
        dt = max(point.time_s - previous.time_s, 1e-6)
        distance = hypot(
            point.coord_x(units) - previous.coord_x(units),
            point.coord_y(units) - previous.coord_y(units),
        )
        speed = distance / dt
        if speed > max_speed_rate:
            rejected += 1
            continue
        # Acceleration gate (metric only): a change in speed faster than the
        # human ceiling implies a tracking jump, not real motion. The speed
        # cap above misses these when the absolute speed stays in-band.
        if units == "metric" and prev_speed is not None:
            if abs(speed - prev_speed) / dt > MAX_ACCEL_MPS2:
                rejected += 1
                continue
        prev_speed = speed
        cleaned.append(point)

    print(
        f"[GameSense] stabilize_track_points: raw={raw_count} "
        f"stabilized={len(cleaned)} rejected={rejected}"
    )
    return cleaned, rejected


def build_speed_series(points: list[TrackPoint], units: str = "metric") -> list[dict]:
    series: list[dict] = []
    for previous, current in zip(points, points[1:]):
        if units == "metric" and (not current.calibrated or not previous.calibrated):
            continue
        dt = max(current.time_s - previous.time_s, 1e-6)
        distance = hypot(
            current.coord_x(units) - previous.coord_x(units),
            current.coord_y(units) - previous.coord_y(units),
        )
        if units == "metric":
            speed_value = (distance / dt) * 3.6
            series.append({
                "time_s": round(current.time_s, 2),
                "speed_kmh": round(speed_value, 2),
            })
        else:
            speed_value = distance / dt
            series.append({
                "time_s": round(current.time_s, 2),
                "speed_px_per_s": round(speed_value, 2),
            })
    return series


def compute_metrics(
    player_id: int,
    points: list[TrackPoint],
    rejected_jump_count: int = 0,
    raw_point_count: int | None = None,
    sprint_threshold_kmh: float = 25.0,
    standing_threshold_kmh: float = 1.0,
    sprint_threshold_px_per_s: float = 280.0,
    standing_threshold_px_per_s: float = 12.0,
    units: str = "metric",
) -> dict:
    sample = points[:5]
    print(
        "[GameSense] compute_metrics field coords (first 5): "
        + ", ".join(
            f"({point.x_m}, {point.y_m})"
            for point in sample
        )
        if sample
        else "(no points)"
    )

    empty = {
        "player_id": player_id,
        "units": units,
        "top_speed_kmh": 0.0,
        "avg_speed_kmh": 0.0,
        "top_speed_px_per_s": 0.0,
        "avg_speed_px_per_s": 0.0,
        "peak_acceleration_mps2": 0.0,
        "avg_acceleration_mps2": 0.0,
        "total_distance_m": 0.0,
        "active_distance_m": 0.0,
        "total_distance_px": 0.0,
        "active_distance_px": 0.0,
        "sprint_count": 0,
        "sprint_distance_m": 0.0,
        "sprint_distance_px": 0.0,
        "usable_track_points": len(points),
        "rejected_jump_count": rejected_jump_count,
        "calibrated_point_ratio": 0.0,
        "confidence_score": 0.0,
    }
    if len(points) < 2:
        return empty

    distances: list[float] = []
    speeds: list[float] = []
    accelerations: list[float] = []
    sprint_count = 0
    sprint_distance = 0.0
    in_sprint = False
    total_distance = 0.0
    active_distance = 0.0
    calibrated_pairs = 0
    total_pairs = 0

    previous_speed = 0.0
    sprint_threshold = sprint_threshold_kmh if units == "metric" else sprint_threshold_px_per_s
    standing_threshold = standing_threshold_kmh if units == "metric" else standing_threshold_px_per_s
    for prev, curr in zip(points, points[1:]):
        total_pairs += 1
        if units == "metric" and (not curr.calibrated or not prev.calibrated):
            continue
        calibrated_pairs += 1
        dt = max(curr.time_s - prev.time_s, 1e-6)
        distance = hypot(
            curr.coord_x(units) - prev.coord_x(units),
            curr.coord_y(units) - prev.coord_y(units),
        )
        speed = (distance / dt) * 3.6 if units == "metric" else distance / dt
        acceleration = (speed - previous_speed) / dt

        distances.append(distance)
        speeds.append(speed)
        accelerations.append(acceleration)
        total_distance += distance

        if speed >= standing_threshold:
            active_distance += distance

        if speed > sprint_threshold:
            sprint_distance += distance
            if not in_sprint:
                sprint_count += 1
                in_sprint = True
        else:
            in_sprint = False

        previous_speed = speed

    source_count = max(raw_point_count or len(points), 1)
    retention = min(len(points) / source_count, 1.0)
    calibrated_ratio = (
        sum(1 for point in points if point.calibrated) / max(len(points), 1)
        if units == "metric"
        else 1.0
    )
    pair_ratio = calibrated_pairs / max(total_pairs, 1) if units == "metric" else 1.0
    confidence = float(np.mean([point.confidence for point in points])) * retention * pair_ratio

    result = {
        "player_id": player_id,
        "units": units,
        "top_speed_kmh": 0.0,
        "avg_speed_kmh": 0.0,
        "top_speed_px_per_s": 0.0,
        "avg_speed_px_per_s": 0.0,
        "peak_acceleration_mps2": round(float(max(accelerations, default=0.0)), 2),
        "avg_acceleration_mps2": round(float(np.mean(np.abs(accelerations))), 2) if accelerations else 0.0,
        "sprint_count": sprint_count,
        "usable_track_points": len(points),
        "rejected_jump_count": rejected_jump_count,
        "calibrated_point_ratio": round(calibrated_ratio, 3),
        "confidence_score": round(max(min(confidence, 1.0), 0.0), 3),
    }
    if units == "metric":
        result.update({
            "top_speed_kmh": round(float(max(speeds, default=0.0)), 2),
            "avg_speed_kmh": round(float(np.mean(speeds)), 2) if speeds else 0.0,
            "total_distance_m": round(total_distance, 2),
            "active_distance_m": round(active_distance, 2),
            "sprint_distance_m": round(sprint_distance, 2),
            "total_distance_px": 0.0,
            "active_distance_px": 0.0,
            "sprint_distance_px": 0.0,
        })
    else:
        result.update({
            "top_speed_px_per_s": round(float(max(speeds, default=0.0)), 2),
            "avg_speed_px_per_s": round(float(np.mean(speeds)), 2) if speeds else 0.0,
            "total_distance_px": round(total_distance, 2),
            "active_distance_px": round(active_distance, 2),
            "sprint_distance_px": round(sprint_distance, 2),
            "total_distance_m": 0.0,
            "active_distance_m": 0.0,
            "sprint_distance_m": 0.0,
        })
    return result


def compute_shot_metrics(
    player_id: int,
    shots: list[ShotEvent],
    ball_points: list[BallTrackPoint],
    rejected_track_points: int = 0,
    raw_point_count: int | None = None,
    touch_count: int = 0,
    pass_count: int = 0,
) -> dict:
    speeds = [shot.ball_speed_kmh for shot in shots]
    best_shot = max(shots, key=lambda shot: shot.ball_speed_kmh) if shots else None

    source_count = max(raw_point_count or len(ball_points), 1)
    usable_count = len(ball_points)
    retention = min(usable_count / source_count, 1.0)
    mean_confidence = float(np.mean([point.confidence for point in ball_points])) if ball_points else 0.0
    confidence = mean_confidence * retention

    return {
        "player_id": player_id,
        "peak_shot_speed_kmh": round(float(max(speeds, default=0.0)), 2),
        "avg_shot_speed_kmh": round(float(np.mean(speeds)), 2) if speeds else 0.0,
        "shot_count": len(shots),
        "best_shot": best_shot.model_dump() if best_shot else None,
        "shots": [shot.model_dump() for shot in shots],
        "confidence_score": round(max(min(confidence, 1.0), 0.0), 3),
        "usable_track_points": usable_count,
        "rejected_track_points": rejected_track_points,
        "touch_count": touch_count,
        "pass_count": pass_count,
    }
