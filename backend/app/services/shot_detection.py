from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import numpy as np

from app.models import Point, ShotEvent
from app.services.metrics import BallTrackPoint, TrackPoint

SHOT_SPEED_THRESHOLD_KMH = 50.0
ROLLING_SPEED_THRESHOLD_KMH = 10.0
CONTACT_RADIUS_PX = 80.0
CONTACT_RADIUS_M = 1.5
MIN_FLIGHT_FRAMES = 3
MAX_MERGE_FRAMES = 8
PASS_SPEED_KMH = 15.0


@dataclass
class FieldTouchEvent:
    frame_id: int
    time_s: float
    contact_x_m: float
    contact_y_m: float
    ball_speed_kmh: float
    is_pass: bool = False


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


def detect_touches_and_passes(
    player_points: list[TrackPoint],
    ball_points: list[BallTrackPoint],
    source_fps: float,
) -> tuple[list[FieldTouchEvent], int, int]:
    if len(ball_points) < 4 or len(player_points) < 2:
        return [], 0, 0

    ball_map = {point.frame_id: point for point in ball_points}
    player_map = {point.frame_id: point for point in player_points}
    ordered_ball = sorted(ball_points, key=lambda point: point.frame_id)

    ball_speeds: dict[int, float] = {}
    for previous, current in zip(ordered_ball, ordered_ball[1:]):
        if not previous.calibrated or not current.calibrated:
            continue
        dt = max(current.time_s - previous.time_s, 1e-6)
        distance_m = hypot(float(current.x_m) - float(previous.x_m), float(current.y_m) - float(previous.y_m))
        ball_speeds[current.frame_id] = distance_m / dt * 3.6

    raw_touch_frames: list[int] = []
    for ball_point in ordered_ball:
        if not ball_point.calibrated:
            continue
        player_point = _nearest_player_point(player_map, ball_point.frame_id)
        if player_point is None or not player_point.calibrated:
            continue
        foot_x = float(player_point.x_m if player_point.x_m is not None else player_point.x_px)
        foot_y = float(player_point.y_m if player_point.y_m is not None else player_point.y_px)
        if hypot(float(ball_point.x_m) - foot_x, float(ball_point.y_m) - foot_y) <= CONTACT_RADIUS_M:
            raw_touch_frames.append(ball_point.frame_id)

    touch_groups: list[list[int]] = []
    for frame_id in raw_touch_frames:
        if touch_groups and frame_id - touch_groups[-1][-1] <= MAX_MERGE_FRAMES:
            touch_groups[-1].append(frame_id)
        else:
            touch_groups.append([frame_id])

    touch_count = len(touch_groups)
    pass_count = 0
    events: list[FieldTouchEvent] = []

    for group in touch_groups:
        contact_frame = group[0]
        after_frames = [
            frame_id
            for frame_id in sorted(ball_speeds.keys())
            if contact_frame < frame_id <= contact_frame + 20
        ]
        if not after_frames:
            continue

        peak_speed = max(ball_speeds.get(frame_id, 0.0) for frame_id in after_frames)
        sustained = sum(1 for frame_id in after_frames if ball_speeds.get(frame_id, 0.0) > SHOT_SPEED_THRESHOLD_KMH)
        is_pass = peak_speed >= PASS_SPEED_KMH
        if is_pass:
            pass_count += 1

        ball_point = ball_map.get(contact_frame) or _nearest_ball_point(ball_map, contact_frame)
        if ball_point is None or not ball_point.calibrated:
            continue

        if peak_speed >= SHOT_SPEED_THRESHOLD_KMH and sustained >= MIN_FLIGHT_FRAMES:
            events.append(
                FieldTouchEvent(
                    frame_id=contact_frame,
                    time_s=contact_frame / max(source_fps, 1e-6),
                    contact_x_m=float(ball_point.x_m),
                    contact_y_m=float(ball_point.y_m),
                    ball_speed_kmh=round(peak_speed, 2),
                    is_pass=is_pass,
                )
            )

    return events, touch_count, pass_count


def _nearest_player_point(player_map: dict[int, TrackPoint], frame_id: int, max_gap: int = 4) -> TrackPoint | None:
    if frame_id in player_map:
        return player_map[frame_id]
    for offset in range(1, max_gap + 1):
        if frame_id - offset in player_map:
            return player_map[frame_id - offset]
        if frame_id + offset in player_map:
            return player_map[frame_id + offset]
    return None


def _nearest_ball_point(ball_map: dict[int, BallTrackPoint], frame_id: int, max_gap: int = 4) -> BallTrackPoint | None:
    if frame_id in ball_map:
        return ball_map[frame_id]
    for offset in range(1, max_gap + 1):
        if frame_id - offset in ball_map:
            return ball_map[frame_id - offset]
        if frame_id + offset in ball_map:
            return ball_map[frame_id + offset]
    return None
