"""Joint-angle and body-kinematics computations from 2D pose landmarks."""
from __future__ import annotations

from typing import Optional, List

import numpy as np

from .pose import PoseFrame


def angle_3pt(a, b, c) -> Optional[float]:
    """Interior angle ABC in degrees. None if any point missing."""
    if a is None or b is None or c is None:
        return None
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def knee_angle(pf: PoseFrame, side: str) -> Optional[float]:
    """Hip-knee-ankle angle. 180 = fully extended."""
    s = side[0]
    return angle_3pt(pf.get(f"{s}_hip"), pf.get(f"{s}_knee"), pf.get(f"{s}_ankle"))


def ankle_angle(pf: PoseFrame, side: str) -> Optional[float]:
    """Knee-ankle-toe angle: proxy for plantarflexion / ankle lock."""
    s = side[0]
    return angle_3pt(pf.get(f"{s}_knee"), pf.get(f"{s}_ankle"), pf.get(f"{s}_foot"))


def trunk_lean(pf: PoseFrame) -> Optional[float]:
    """Unsigned trunk angle vs vertical (shoulder-midpoint -> hip-midpoint)."""
    sh, hp = pf.mid("l_shoulder", "r_shoulder"), pf.mid("l_hip", "r_hip")
    if sh is None or hp is None:
        return None
    v = sh - hp                       # points from hips up to shoulders
    return float(np.degrees(np.arctan2(abs(v[0]), max(-v[1], 1e-6))))


def trunk_lean_signed(pf: PoseFrame, shot_dir_x: float) -> Optional[float]:
    """Trunk lean signed relative to shot direction: + leaning over the ball,
    - leaning back/away from the shot."""
    sh, hp = pf.mid("l_shoulder", "r_shoulder"), pf.mid("l_hip", "r_hip")
    if sh is None or hp is None:
        return None
    v = sh - hp
    mag = np.degrees(np.arctan2(abs(v[0]), max(-v[1], 1e-6)))
    sign = 1.0 if np.sign(v[0]) == np.sign(shot_dir_x) else -1.0
    return float(sign * mag)


def hip_line_angle(pf: PoseFrame) -> Optional[float]:
    """Orientation of the pelvis line (left hip -> right hip) in image plane, deg."""
    lh, rh = pf.get("l_hip"), pf.get("r_hip")
    if lh is None or rh is None:
        return None
    v = rh - lh
    return float(np.degrees(np.arctan2(v[1], v[0])))


def hip_width_px(pf: PoseFrame) -> Optional[float]:
    lh, rh = pf.get("l_hip"), pf.get("r_hip")
    if lh is None or rh is None:
        return None
    return float(np.linalg.norm(rh - lh))


def shoulder_width_px(pf: PoseFrame) -> Optional[float]:
    ls, rs = pf.get("l_shoulder"), pf.get("r_shoulder")
    if ls is None or rs is None:
        return None
    return float(np.linalg.norm(rs - ls))


def hip_rotation_proxy(frames: List[PoseFrame], i_from: int, i_to: int) -> Optional[float]:
    """Pelvis rotation proxy from apparent hip-width change + hip-line tilt.

    In a 2D side/diagonal view, pelvis rotation toward the camera changes the
    projected inter-hip distance (w_proj = w_true * cos(theta)). We scan the
    whole window (not just the endpoints, which are noise-sensitive) and
    combine the hip-line angle change with the normalised width change.
    Width is normalised by shoulder width where possible so that the player
    moving toward/away from the camera doesn't masquerade as rotation.
    """
    if not (0 <= i_from < len(frames) and 0 <= i_to < len(frames)) or i_to <= i_from:
        return None

    widths: List[float] = []
    angles: List[float] = []
    for i in range(i_from, i_to + 1):
        pf = frames[i]
        if not pf.ok:
            continue
        w, a = hip_width_px(pf), hip_line_angle(pf)
        if w is None or a is None:
            continue
        sw = shoulder_width_px(pf)
        if sw is not None and sw > 1e-6:
            w = w / sw                # scale-invariant projected hip width
        widths.append(w)
        angles.append(a)

    if len(widths) < 2:
        return None

    # Angle change across the window (wrap-safe), using smoothed extremes.
    ang_change = abs(angles[-1] - angles[0])
    ang_change = min(ang_change, 360.0 - ang_change)

    wmax = max(widths)
    if wmax < 1e-6:
        return None
    ratio = float(np.clip(min(widths) / wmax, 0.0, 1.0))
    width_rot = float(np.degrees(np.arccos(ratio)))
    return float(min(max(ang_change, width_rot), 110.0))


def leg_length_px(pf: PoseFrame, side: str) -> Optional[float]:
    s = side[0]
    hip, knee, ankle = pf.get(f"{s}_hip"), pf.get(f"{s}_knee"), pf.get(f"{s}_ankle")
    if hip is None or knee is None or ankle is None:
        return None
    return float(np.linalg.norm(hip - knee) + np.linalg.norm(knee - ankle))


def body_height_px(pf: PoseFrame) -> Optional[float]:
    """Approximate standing height in pixels from pose landmarks:
    ankle->hip->shoulder chain * 1.33 (head+neck correction).
    More robust than a bbox height, which inflates with raised arms."""
    best = None
    for s in ("l", "r"):
        hip, knee, ankle = pf.get(f"{s}_hip"), pf.get(f"{s}_knee"), pf.get(f"{s}_ankle")
        sh = pf.get(f"{s}_shoulder")
        if hip is None or knee is None or ankle is None or sh is None:
            continue
        chain = (np.linalg.norm(ankle - knee) + np.linalg.norm(knee - hip)
                 + np.linalg.norm(hip - sh))
        h = float(chain) * 1.33
        best = h if best is None else max(best, h)
    return best


def smooth(series: List[Optional[float]], window: int = 5) -> List[Optional[float]]:
    """Median smoothing tolerant of Nones."""
    out = []
    half = window // 2
    for i in range(len(series)):
        vals = [v for v in series[max(0, i - half):i + half + 1] if v is not None]
        out.append(float(np.median(vals)) if vals else None)
    return out


def speed_series(points: List[Optional[np.ndarray]]) -> List[Optional[float]]:
    """Per-frame speed in px/frame from a 2D point track."""
    out: List[Optional[float]] = [None]
    for i in range(1, len(points)):
        a, b = points[i - 1], points[i]
        out.append(float(np.linalg.norm(b - a)) if a is not None and b is not None else None)
    return out


def direction_change_deg(points: List[Optional[np.ndarray]], i: int) -> Optional[float]:
    """Angle (deg) between the incoming and outgoing velocity at frame i.
    Large values indicate a trajectory kink, e.g. ball contact."""
    if i - 1 < 0 or i + 1 >= len(points):
        return None
    a, b, c = points[i - 1], points[i], points[i + 1]
    if a is None or b is None or c is None:
        return None
    v_in, v_out = b - a, c - b
    n1, n2 = np.linalg.norm(v_in), np.linalg.norm(v_out)
    if n1 < 0.5 or n2 < 0.5:
        return None
    cosang = np.clip(np.dot(v_in, v_out) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))
