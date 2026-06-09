from dataclasses import dataclass
from math import hypot

import numpy as np

from app.models import ShotEvent


@dataclass
class TrackPoint:
    frame_id: int
    time_s: float
    x_m: float
    y_m: float
    confidence: float = 1.0
    left_foot_px: tuple[float, float] | None = None
    right_foot_px: tuple[float, float] | None = None


@dataclass
class BallTrackPoint:
    frame_id: int
    time_s: float
    x_px: float
    y_px: float
    x_m: float
    y_m: float
    confidence: float = 1.0
    interpolated: bool = False


def stabilize_track_points(
    points: list[TrackPoint],
    max_speed_kmh: float = 38.0,
    median_window: int = 5,
) -> tuple[list[TrackPoint], int]:
    if len(points) < 3:
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
                x_m=float(np.median([item.x_m for item in window])),
                y_m=float(np.median([item.y_m for item in window])),
                confidence=float(np.mean([item.confidence for item in window])),
            )
        )

    cleaned = [smoothed[0]]
    rejected = 0
    max_speed_mps = max_speed_kmh / 3.6
    for point in smoothed[1:]:
        previous = cleaned[-1]
        dt = max(point.time_s - previous.time_s, 1e-6)
        distance = hypot(point.x_m - previous.x_m, point.y_m - previous.y_m)
        if distance / dt > max_speed_mps:
            rejected += 1
            continue
        cleaned.append(point)

    return cleaned, rejected


def compute_metrics(
    player_id: int,
    points: list[TrackPoint],
    rejected_jump_count: int = 0,
    raw_point_count: int | None = None,
    sprint_threshold_kmh: float = 25.0,
    standing_threshold_kmh: float = 1.0,
) -> dict:
    if len(points) < 2:
        return {
            "player_id": player_id,
            "top_speed_kmh": 0.0,
            "avg_speed_kmh": 0.0,
            "peak_acceleration_mps2": 0.0,
            "avg_acceleration_mps2": 0.0,
            "total_distance_m": 0.0,
            "active_distance_m": 0.0,
            "sprint_count": 0,
            "sprint_distance_m": 0.0,
            "usable_track_points": len(points),
            "rejected_jump_count": rejected_jump_count,
            "confidence_score": 0.0,
        }

    distances: list[float] = []
    speeds_kmh: list[float] = []
    accelerations: list[float] = []
    sprint_count = 0
    sprint_distance = 0.0
    in_sprint = False
    total_distance = 0.0
    active_distance = 0.0

    previous_speed_mps = 0.0
    for prev, curr in zip(points, points[1:]):
        dt = max(curr.time_s - prev.time_s, 1e-6)
        distance = hypot(curr.x_m - prev.x_m, curr.y_m - prev.y_m)
        speed_mps = distance / dt
        speed_kmh = speed_mps * 3.6
        acceleration = (speed_mps - previous_speed_mps) / dt

        distances.append(distance)
        speeds_kmh.append(speed_kmh)
        accelerations.append(acceleration)
        total_distance += distance

        if speed_kmh >= standing_threshold_kmh:
            active_distance += distance

        if speed_kmh > sprint_threshold_kmh:
            sprint_distance += distance
            if not in_sprint:
                sprint_count += 1
                in_sprint = True
        else:
            in_sprint = False

        previous_speed_mps = speed_mps

    source_count = max(raw_point_count or len(points), 1)
    retention = min(len(points) / source_count, 1.0)
    confidence = float(np.mean([point.confidence for point in points])) * retention

    return {
        "player_id": player_id,
        "top_speed_kmh": round(float(max(speeds_kmh, default=0.0)), 2),
        "avg_speed_kmh": round(float(np.mean(speeds_kmh)), 2),
        "peak_acceleration_mps2": round(float(max(accelerations, default=0.0)), 2),
        "avg_acceleration_mps2": round(float(np.mean(np.abs(accelerations))), 2),
        "total_distance_m": round(total_distance, 2),
        "active_distance_m": round(active_distance, 2),
        "sprint_count": sprint_count,
        "sprint_distance_m": round(sprint_distance, 2),
        "usable_track_points": len(points),
        "rejected_jump_count": rejected_jump_count,
        "confidence_score": round(max(min(confidence, 1.0), 0.0), 3),
    }


def compute_shot_metrics(
    player_id: int,
    shots: list[ShotEvent],
    ball_points: list[BallTrackPoint],
    rejected_track_points: int = 0,
    raw_point_count: int | None = None,
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
    }
