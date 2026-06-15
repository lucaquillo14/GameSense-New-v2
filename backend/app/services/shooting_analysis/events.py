"""Shot event detection: contact frame, plant frame, backswing peak, phases.

Improvements over the original:
  * Contact scoring now also rewards a trajectory direction change (the ball's
    path kinks at impact) — far fewer false positives on bouncing run-ups.
  * Kicking-foot identification combines proximity AND foot swing speed at
    contact, instead of position only.
  * Plant frame requires the ankle to be both slow and near its lowest point
    (foot actually on the ground), not merely slow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import EventConfig
from .kinematics import speed_series, knee_angle, smooth, direction_change_deg
from .pose import PoseFrame

PHASES = ["Approach", "Plant", "Backswing", "Contact", "Follow-through"]


@dataclass
class ShotEvents:
    contact: int
    plant: int
    backswing_peak: int
    kicking_foot: str          # "left" / "right"
    plant_foot: str
    shot_dir_x: float          # +1 ball travels right, -1 left

    def phase_at(self, i: int, n_frames: int) -> str:
        if i < self.plant:
            return "Approach"
        if i < self.backswing_peak:
            return "Plant"
        if i < self.contact:
            return "Backswing"
        if i <= self.contact + 2:
            return "Contact"
        return "Follow-through"


def find_contact_frame(ball_track: List[Optional[np.ndarray]],
                       poses: List[PoseFrame],
                       cfg: EventConfig,
                       frame_h: int = 720) -> Optional[int]:
    """Contact frame via a 3-tier cascade:
       1. strict:  ball speed spike (>=3x) with a foot within 10% of frame height
       2. relaxed: spike >=1.8x with a foot within 20%, pose optional
       3. kinematic fallback: frame of peak kicking-foot swing speed
       4. last resort: global max ball speed
    """
    speeds = speed_series(ball_track)
    for ratio, frac, need_pose in (
            (cfg.ball_speed_spike_ratio, cfg.foot_ball_radius_frac, True),
            (1.8, cfg.foot_ball_radius_frac_relaxed, False)):
        i = _spike_pass(ball_track, poses, speeds, ratio, frac * frame_h, need_pose, cfg)
        if i is not None:
            return i

    i = _foot_swing_peak(poses)
    if i is not None:
        return i

    valid = [(i, s) for i, s in enumerate(speeds) if s is not None]
    return max(valid, key=lambda t: t[1])[0] if valid else None


def _spike_pass(ball_track, poses, speeds, ratio, max_px, need_pose,
                cfg: EventConfig) -> Optional[int]:
    n = len(ball_track)
    best_i, best_score = None, -1.0
    for i in range(2, n - 1):
        s_now = speeds[i] if speeds[i] is not None else 0.0
        pre = [s for s in speeds[max(0, i - 4):i] if s is not None]
        s_pre = max(np.mean(pre), 0.5) if pre else 0.5
        if s_now < ratio * s_pre or s_now < 2.0:
            continue
        bpos = ball_track[i - 1] if ball_track[i - 1] is not None else ball_track[i]
        if bpos is None:
            continue
        pf = poses[i - 1] if poses[i - 1].ok else (poses[i] if poses[i].ok else None)
        if pf is None:
            if need_pose:
                continue
            d = max_px * 0.5            # no pose: neutral proximity assumption
        else:
            d = _min_foot_dist(pf, bpos)
            if d is None or d > max_px:
                continue
        score = s_now / s_pre - d / max_px
        # Trajectory kink bonus: the ball's direction changes sharply at impact.
        kink = direction_change_deg(ball_track, i - 1)
        if kink is not None and kink > 25.0:
            score += cfg.direction_change_bonus * min(kink / 90.0, 1.0)
        if score > best_score:
            best_score, best_i = score, i
    return best_i


def estimate_kick_frame(ball_track: List[Optional[np.ndarray]],
                        poses: List[PoseFrame]) -> Optional[int]:
    """Cheap, local estimate of where the kick happens — used to bound the
    expensive hosted-inference backfill window. Prefers the sparse ball-speed
    peak; falls back to the kinematic foot-swing peak (needs no ball at all)."""
    speeds = speed_series(ball_track)
    valid = [(i, s) for i, s in enumerate(speeds) if s is not None]
    if valid:
        i, s = max(valid, key=lambda t: t[1])
        if s >= 3.0:
            return i
    return _foot_swing_peak(poses)


def _foot_swing_peak(poses: List[PoseFrame]) -> Optional[int]:
    """Kinematic fallback: contact ≈ frame of maximum foot speed (the swing whips
    fastest at impact). Works even with zero ball detections."""
    best_i, best_v = None, 0.0
    for side in ("l", "r"):
        track = [pf.get(f"{side}_foot") if pf.ok else None for pf in poses]
        spd = speed_series(track)
        sm = smooth(spd, 3)
        for i in range(3, len(sm) - 2):      # skip clip edges
            if sm[i] is not None and sm[i] > best_v:
                best_v, best_i = sm[i], i
    return best_i if best_v > 4.0 else None


def _min_foot_dist(pf: PoseFrame, ball_pos: np.ndarray) -> Optional[float]:
    ds = []
    for name in ("l_foot", "r_foot", "l_ankle", "r_ankle"):
        p = pf.get(name)
        if p is not None:
            ds.append(float(np.linalg.norm(p - ball_pos)))
    return min(ds) if ds else None


def identify_kicking_foot(poses: List[PoseFrame], ball_track, contact: int,
                          forced: str = "auto") -> str:
    """Kicking foot = the foot that is both closest to the ball AND swinging
    fastest around contact. Combining the two signals is robust to pose jitter
    and to crossed-leg frames at impact."""
    if forced in ("left", "right"):
        return forced

    bpos = None
    for i in range(contact, max(contact - 4, -1), -1):
        if ball_track[i] is not None:
            bpos = ball_track[i]
            break

    scores = {"left": 0.0, "right": 0.0}
    sides = {"left": "l", "right": "r"}

    # Proximity vote
    if bpos is not None and poses[contact].ok:
        pf = poses[contact]
        d = {foot: _dist(pf.get(f"{s}_foot"), bpos) for foot, s in sides.items()}
        dl, dr = d["left"], d["right"]
        if dl is not None or dr is not None:
            closer = "left" if (dl or 1e9) < (dr or 1e9) else "right"
            scores[closer] += 1.0

    # Swing-speed vote: which foot moved fastest in the window before/at contact
    lo, hi = max(0, contact - 3), min(len(poses), contact + 2)
    for foot, s in sides.items():
        track = [poses[i].get(f"{s}_foot") if poses[i].ok else None for i in range(lo, hi)]
        spd = [v for v in speed_series(track) if v is not None]
        scores[foot] += 0.0 if not spd else max(spd) / 100.0
    fast = max(scores, key=lambda k: scores[k])
    scores[fast] += 0.5

    if scores["left"] == scores["right"]:
        return "right"
    return max(scores, key=lambda k: scores[k])


def _dist(p, q) -> Optional[float]:
    if p is None or q is None:
        return None
    return float(np.linalg.norm(p - q))


def find_backswing_peak(poses: List[PoseFrame], contact: int, foot: str) -> int:
    """Frame of maximum kicking-knee flexion (minimum knee angle) before contact."""
    lo = max(0, contact - 20)
    angles = smooth([knee_angle(poses[i], foot) if poses[i].ok else None
                     for i in range(lo, contact + 1)], 3)
    best_i, best_a = contact - 3, 1e9
    for k, a in enumerate(angles):
        if a is not None and a < best_a:
            best_a, best_i = a, lo + k
    return max(0, best_i)


def find_plant_frame(poses: List[PoseFrame], backswing_peak: int, contact: int,
                     plant_foot: str) -> int:
    """Frame where the plant ankle settles: slow AND near its lowest height
    (i.e. actually on the ground), scanning back from the backswing peak."""
    s = plant_foot[0]
    lo = max(0, backswing_peak - 15)
    track = [poses[i].get(f"{s}_ankle") if poses[i].ok else None
             for i in range(lo, backswing_peak + 1)]
    spd = speed_series(track)

    ys = [p[1] for p in track if p is not None]
    y_floor = max(ys) if ys else None          # image y grows downward

    best = None
    for k in range(len(spd) - 1, 0, -1):
        if spd[k] is None or spd[k] >= 2.5:
            continue
        p = track[k]
        if y_floor is not None and p is not None and (y_floor - p[1]) > 0.12 * y_floor:
            continue                            # ankle clearly off the ground
        best = lo + k
        break
    if best is not None:
        return best

    # fallback: slow only
    for k in range(len(spd) - 1, 0, -1):
        if spd[k] is not None and spd[k] < 2.5:
            return lo + k
    return max(0, backswing_peak - 4)


def shot_direction(ball_track, contact: int) -> float:
    after = [p for p in ball_track[contact:contact + 8] if p is not None]
    if len(after) >= 2:
        return 1.0 if (after[-1][0] - after[0][0]) >= 0 else -1.0
    return 1.0


def detect_events(ball_track, poses, cfg: EventConfig, forced_foot="auto",
                  frame_h: int = 720) -> Optional[ShotEvents]:
    contact = find_contact_frame(ball_track, poses, cfg, frame_h)
    if contact is None:
        return None
    contact = int(np.clip(contact, 1, len(poses) - 2))
    foot = identify_kicking_foot(poses, ball_track, contact, forced_foot)
    plant_foot = "left" if foot == "right" else "right"
    bsw = find_backswing_peak(poses, contact, foot)
    plant = find_plant_frame(poses, bsw, contact, plant_foot)
    return ShotEvents(
        contact=contact, plant=plant, backswing_peak=bsw,
        kicking_foot=foot, plant_foot=plant_foot,
        shot_dir_x=shot_direction(ball_track, contact),
    )
