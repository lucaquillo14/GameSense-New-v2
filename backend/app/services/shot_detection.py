from __future__ import annotations

from math import hypot

import numpy as np

from app.models import Point, ShotEvent
from app.services.metrics import BallTrackPoint, TrackPoint

SHOT_SPEED_THRESHOLD_KMH = 50.0
ROLLING_SPEED_THRESHOLD_KMH = 10.0
CONTACT_RADIUS_PX = 80.0
MIN_FLIGHT_FRAMES = 3


def _ball_speed_kmh(
    previous: BallTrackPoint,
    current: BallTrackPoint,
    homography_matrix: np.ndarray,
) -> float:
    dt = max(current.time_s - previous.time_s, 1e-6)
    pixel_distance = hypot(current.x_px - previous.x_px, current.y_px - previous.y_px)
    field_delta = np.array(
        [
            [current.x_m - previous.x_m],
            [current.y_m - previous.y_m],
        ],
        dtype=np.float64,
    )
    field_distance = float(np.linalg.norm(field_delta))
    if field_distance > 0 and pixel_distance > 0:
        metres_per_pixel = field_distance / pixel_distance
        speed_mps = (pixel_distance * metres_per_pixel) / dt
    else:
        speed_mps = field_distance / dt
    return speed_mps * 3.6


def _player_point_at_frame(player_points: list[TrackPoint], frame_id: int) -> TrackPoint | None:
    for point in player_points:
        if point.frame_id == frame_id:
            return point
    return None


def _foot_near_ball(
    foot: tuple[float, float] | None,
    ball: BallTrackPoint,
    radius_px: float,
) -> bool:
    if foot is None:
        return False
    return hypot(foot[0] - ball.x_px, foot[1] - ball.y_px) <= radius_px


def detect_shots(
    ball_points: list[BallTrackPoint],
    player_points: list[TrackPoint],
    fps: float,
    homography_matrix: np.ndarray,
) -> list[ShotEvent]:
    if len(ball_points) < MIN_FLIGHT_FRAMES + 2:
        return []

    ordered_ball = sorted(ball_points, key=lambda point: point.frame_id)
    speeds: list[tuple[int, float]] = []
    for index in range(1, len(ordered_ball)):
        previous = ordered_ball[index - 1]
        current = ordered_ball[index]
        if not previous.calibrated or not current.calibrated:
            continue
        speed_kmh = _ball_speed_kmh(
            previous,
            current,
            homography_matrix,
        )
        speeds.append((current.frame_id, speed_kmh))

    if not speeds:
        return []

    frame_to_ball = {point.frame_id: point for point in ordered_ball}
    confirmed: list[ShotEvent] = []
    used_peaks: set[int] = set()

    for peak_index, (peak_frame_id, peak_speed) in enumerate(speeds):
        if peak_speed < SHOT_SPEED_THRESHOLD_KMH or peak_frame_id in used_peaks:
            continue

        contact_frame_id = peak_frame_id
        for walk_index in range(peak_index, -1, -1):
            frame_id, speed_kmh = speeds[walk_index]
            if speed_kmh < ROLLING_SPEED_THRESHOLD_KMH:
                contact_frame_id = frame_id
                break
            contact_frame_id = frame_id

        contact_ball = frame_to_ball.get(contact_frame_id)
        if contact_ball is None:
            continue

        player_point = _player_point_at_frame(player_points, contact_frame_id)
        if player_point is None:
            continue

        has_contact = (
            _foot_near_ball(player_point.left_foot_px, contact_ball, CONTACT_RADIUS_PX)
            or _foot_near_ball(player_point.right_foot_px, contact_ball, CONTACT_RADIUS_PX)
        )
        if not has_contact:
            continue

        sustained = 0
        for check_index in range(peak_index, len(speeds)):
            if speeds[check_index][1] >= SHOT_SPEED_THRESHOLD_KMH:
                sustained += 1
                if sustained >= MIN_FLIGHT_FRAMES:
                    break
            else:
                sustained = 0
        if sustained < MIN_FLIGHT_FRAMES:
            continue

        confirmed.append(
            ShotEvent(
                frame_id=contact_frame_id,
                timestamp_s=contact_ball.time_s,
                ball_speed_kmh=round(peak_speed, 2),
                contact_point=Point(x=contact_ball.x_px, y=contact_ball.y_px),
            )
        )
        used_peaks.add(peak_frame_id)

    return confirmed
