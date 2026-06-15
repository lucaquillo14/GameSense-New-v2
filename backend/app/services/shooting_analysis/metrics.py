"""Metric extraction, shot-power estimation, scoring, and feedback generation.

Improvements over the original:
  * Scale fuses goal width AND height (median), with a pose-derived player
    height fallback that doesn't inflate when arms are raised.
  * Shot power uses a robust (median per-frame displacement) estimate instead
    of a least-squares fit that a single bad ball detection could wreck.
  * Feedback is guaranteed to contain 3–6 specific points.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import IDEALS, POWER_BANDS, PipelineConfig
from .events import ShotEvents
from .kinematics import (knee_angle, ankle_angle, trunk_lean_signed,
                         hip_rotation_proxy, leg_length_px, body_height_px,
                         smooth, speed_series)
from .pose import PoseFrame


@dataclass
class Analysis:
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    meters_per_px: float = 0.0
    scale_source: str = ""
    shot_speed_kmh: Optional[float] = None
    power_label: str = ""
    score: float = 0.0
    feedback: List[str] = field(default_factory=list)
    per_metric_ok: Dict[str, bool] = field(default_factory=dict)
    on_target: Optional[bool] = None


# --------------------------------------------------------------------- scale
def _goal_scale(goal_box, cfg: PipelineConfig) -> Optional[float]:
    if goal_box is None:
        return None
    x1, y1, x2, y2 = goal_box["xyxy"]
    w, h = abs(x2 - x1), abs(y2 - y1)
    estimates = []
    if w > 20:
        estimates.append(cfg.goal_width_m / w)
    if h > 12:
        estimates.append(cfg.goal_height_m / h)
    return float(np.median(estimates)) if estimates else None


def _pose_scale(poses, cfg: PipelineConfig) -> Optional[float]:
    if not poses:
        return None
    heights = [body_height_px(pf) for pf in poses if pf.ok]
    heights = [h for h in heights if h is not None and h > 40]
    if len(heights) < 3:
        return None
    return cfg.player_height_m / float(np.median(heights))


def _ball_scale(ball_diams_px, contact: Optional[int], cfg: PipelineConfig) -> Optional[float]:
    """metres-per-pixel from the ball's own apparent diameter (0.22 m FIFA
    size 5). Independent of player height AND measured exactly at the plane
    where the shot happens. Pre-contact detections only (the resting ball is
    sharp; the struck ball is blurred and its box inflates)."""
    from .config import BALL_DIAMETER_M
    if not ball_diams_px:
        return None
    diams = [entry[1] for entry in ball_diams_px
             if (contact is None or entry[0] <= contact) and 4.0 <= entry[1]]
    if len(diams) < 3:
        return None
    return BALL_DIAMETER_M / float(np.median(diams))


def _goal_plane_crossing(ball_track, ball_diams_px, contact: int, fps: float,
                         goal_box, frame_w: Optional[int],
                         cfg: PipelineConfig) -> Optional[dict]:
    """Depth-resolved goal-line crossing.

    Pinhole depth: Z = f_px * real_size / pixel_size. The ball's apparent
    diameter gives its depth every frame; the goal's apparent width gives the
    goal-plane depth. Fitting Z_ball(t) after contact and solving
    Z_ball(t) = Z_goal yields the crossing time; the ball's image position at
    that instant, measured against the goal box at goal-plane scale, gives
    where it crossed (height above the line, offset from centre) and a true
    3D speed (depth + lateral components).
    Post-contact diameters use the MIN box side — motion blur stretches the
    box along the flight direction but barely across it."""
    if goal_box is None or not ball_diams_px or not frame_w or fps <= 0:
        return None
    from .config import BALL_DIAMETER_M
    gx1, gy1, gx2, gy2 = goal_box["xyxy"]
    goal_w_px = float(gx2 - gx1)
    goal_h_px = float(max(gy2 - gy1, 1.0))
    if goal_w_px < 20:
        return None
    f_px = 0.8 * float(frame_w)
    z_goal = f_px * cfg.goal_width_m / goal_w_px

    samples: list[tuple[float, float]] = []
    for entry in ball_diams_px:
        fid = entry[0]
        diam = entry[2] if len(entry) > 2 and entry[2] else entry[1]
        if fid < contact or fid > contact + int(2.0 * fps) or diam < 3.0:
            continue
        samples.append(((fid - contact) / fps, f_px * BALL_DIAMETER_M / diam))
    if len(samples) < 3:
        return None

    t = np.array([s[0] for s in samples], dtype=float)
    z = np.array([s[1] for s in samples], dtype=float)
    coef = np.polyfit(t, z, 1)
    resid = z - np.polyval(coef, t)
    keep = np.abs(resid) <= max(1.5 * float(np.std(resid)), 0.5)
    if int(keep.sum()) >= 3:
        coef = np.polyfit(t[keep], z[keep], 1)
    vz, z0 = float(coef[0]), float(coef[1])
    if vz < 0.5:
        return None                       # ball not travelling toward the goal in depth
    t_cross = (z_goal - z0) / vz
    if not (0.05 <= t_cross <= 3.0):
        return None

    # ball image position at the crossing instant (interpolated)
    fc = contact + t_cross * fps
    pos = None
    lo, hi = int(np.floor(fc)), int(np.ceil(fc))
    if 0 <= lo < len(ball_track) and ball_track[lo] is not None:
        if hi < len(ball_track) and ball_track[hi] is not None and hi != lo:
            w = fc - lo
            pos = np.asarray(ball_track[lo], dtype=float) * (1 - w) + np.asarray(ball_track[hi], dtype=float) * w
        else:
            pos = np.asarray(ball_track[lo], dtype=float)
    elif 0 <= hi < len(ball_track) and ball_track[hi] is not None:
        pos = np.asarray(ball_track[hi], dtype=float)
    if pos is None:
        centre = int(round(fc))
        for d in range(1, max(int(fps), 2)):
            for j in (centre - d, centre + d):
                if 0 <= j < len(ball_track) and ball_track[j] is not None:
                    pos = np.asarray(ball_track[j], dtype=float)
                    break
            if pos is not None:
                break
    if pos is None:
        return None

    s_g_w = cfg.goal_width_m / goal_w_px
    s_g_h = cfg.goal_height_m / goal_h_px
    offset_m = float((pos[0] - (gx1 + gx2) / 2.0) * s_g_w)   # + = right of centre
    height_m = float((gy2 - pos[1]) * s_g_h)                 # above the goal line
    on_target = (
        (gx1 - 0.05 * goal_w_px) <= pos[0] <= (gx2 + 0.05 * goal_w_px)
        and pos[1] >= gy1 - 0.05 * goal_h_px
        and height_m >= -0.2
    )

    # 3D speed: depth gap + lateral image displacement at the mean plane scale
    ball_at_contact = None
    for j in range(contact, max(-1, contact - 6), -1):
        if ball_track[j] is not None:
            ball_at_contact = np.asarray(ball_track[j], dtype=float)
            break
    speed_kmh = None
    dist_m = None
    dz = max(z_goal - z0, 0.0)
    if ball_at_contact is not None:
        pre = [e[1] for e in ball_diams_px if e[0] <= contact and e[1] >= 3.0]
        s_ball = (BALL_DIAMETER_M / float(np.median(pre))) if len(pre) >= 2 else s_g_w
        lateral_m = float(np.linalg.norm(pos - ball_at_contact)) * (s_ball + s_g_w) / 2.0
        dist_m = float(np.hypot(dz, lateral_m))
        speed_kmh = dist_m / t_cross * 3.6

    return {
        "t_cross_s": float(t_cross),
        "speed_kmh": speed_kmh,
        "dist_m": dist_m,
        "height_m": height_m,
        "offset_m": offset_m,
        "on_target": bool(on_target),
    }


def _bbox_scale(player_boxes, cfg: PipelineConfig) -> Optional[float]:
    heights = [b["xyxy"][3] - b["xyxy"][1] for b in player_boxes if b is not None]
    heights = [h for h in heights if h > 40]
    if not heights:
        return None
    return cfg.player_height_m / float(np.median(heights))


def compute_scale(goal_box, player_boxes, cfg: PipelineConfig,
                  poses: Optional[List[PoseFrame]] = None):
    """metres-per-pixel with cross-validation.

    The goal-derived scale is preferred, BUT a mis-detected goal box (a partial
    post, a far-away goal, a fence) silently inflates every speed and distance.
    So the goal scale is sanity-checked against the player-derived scale: if
    they disagree by more than ~2.2x, the goal box is rejected.
    """
    goal = _goal_scale(goal_box, cfg)
    pose = _pose_scale(poses, cfg)
    bbox = _bbox_scale(player_boxes, cfg)
    reference = pose or bbox

    if goal is not None:
        if reference is not None:
            ratio = goal / reference
            if not (1 / 2.2 <= ratio <= 2.2):
                return reference, (
                    f"player height ({cfg.player_height_m:.2f} m) — goal box rejected "
                    f"(scale disagreed {ratio:.1f}x)"
                )
        return goal, f"goal dimensions ({cfg.goal_width_m:.2f}x{cfg.goal_height_m:.2f} m)"

    if pose is not None:
        return pose, f"player height ({cfg.player_height_m:.2f} m, pose)"
    if bbox is not None:
        return bbox, f"player height ({cfg.player_height_m:.2f} m)"
    return 0.0, "unavailable"


# --------------------------------------------------------------------- power
def shot_power_kmh(ball_track, contact: int, fps: float,
                   m_per_px: float, fit_frames: int,
                   player_boxes=None) -> Optional[float]:
    """Robust post-contact ball speed -> km/h.

    The launch speed is the LARGEST velocity supported by two consecutive,
    directionally consistent steps:
      * pre-launch crawl and a static mis-locked "ball" produce small steps
        -> they can never outvote the real flight;
      * a single mis-detected position produces two large steps in OPPOSITE
        directions -> rejected by the direction check;
      * the real flight produces consecutive large steps in the SAME
        direction -> accepted.
    This is far more failure-tolerant than a median (biases low when the
    detector loses the ball after impact) or a plain peak (inflated by any
    detection error)."""
    if m_per_px <= 0:
        return None

    def player_center(frame_id: int):
        if player_boxes is None or frame_id >= len(player_boxes):
            return None
        box = player_boxes[frame_id]
        if box is None:
            return None
        x1, y1, x2, y2 = box["xyxy"]
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

    # (frame_id_from, frame_id_to, velocity px/frame)
    # Duplicated video frames (30 fps content in a 60 fps container) repeat
    # the same ball position; skipping those WITHOUT advancing the anchor
    # keeps velocities correct against the metadata frame rate.
    vels: list[tuple[int, int, np.ndarray]] = []
    prev_i, prev_p = None, None
    for i in range(contact, min(len(ball_track), contact + fit_frames + 5)):
        p = ball_track[i]
        if p is None:
            continue
        if prev_p is not None and float(np.linalg.norm(
                np.asarray(p, dtype=float) - np.asarray(prev_p, dtype=float))) < 0.7:
            continue                                   # duplicated frame
        if prev_p is not None:
            gap = i - prev_i
            vels.append((prev_i, i,
                         (np.asarray(p, dtype=float) - np.asarray(prev_p, dtype=float)) / gap))
        prev_i, prev_p = i, p
    if len(vels) < 2:
        return None

    def player_speed_at(f1: int, f2: int) -> Optional[float]:
        pa, pb = player_center(f1), player_center(f2)
        if pa is None or pb is None or f2 <= f1:
            return None
        return float(np.linalg.norm(pb - pa)) / (f2 - f1)

    best = 0.0
    rejected_non_separating = False
    for (f1a, f1b, v1), (f2a, f2b, v2) in zip(vels, vels[1:]):
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        if max(n1, n2) / max(min(n1, n2), 1e-6) > 1.8:
            continue                                  # inconsistent magnitudes
        if float(np.dot(v1, v2)) / (n1 * n2) < 0.5:
            continue                                  # inconsistent direction
        cand = (n1 + n2) / 2.0
        # Anti-player-lock gate. Only applies when the candidate speed is in
        # the same range as the player's own movement — genuine ball flight
        # is far faster than anyone runs, and must not be filtered away just
        # because detections were too sparse to confirm separation.
        p_spd = player_speed_at(f1a, f2b)
        if p_spd is not None and cand < 3.0 * max(p_spd, 1.0):
            pc_a, pc_b = player_center(f1a), player_center(f2b)
            ball_a, ball_b = ball_track[f1a], ball_track[f2b]
            if pc_a is not None and pc_b is not None and ball_a is not None and ball_b is not None:
                d_start = float(np.linalg.norm(np.asarray(ball_a, dtype=float) - pc_a))
                d_end = float(np.linalg.norm(np.asarray(ball_b, dtype=float) - pc_b))
                travelled = cand * max(f2b - f1a, 1)
                if d_end - d_start < 0.15 * travelled:
                    rejected_non_separating = True
                    continue
        best = max(best, cand)

    if best <= 0.0:
        # Last resort: a single large step that clearly separates from the
        # player (the launch itself, when the detector lost the ball after).
        for f1, f2, v in vels:
            mag = float(np.linalg.norm(v))
            p_spd = player_speed_at(f1, f2) or 0.0
            if mag < max(4.0, 3.0 * p_spd):
                continue
            pc_a, pc_b = player_center(f1), player_center(f2)
            ball_a, ball_b = ball_track[f1], ball_track[f2]
            if pc_a is not None and pc_b is not None and ball_a is not None and ball_b is not None:
                d_start = float(np.linalg.norm(np.asarray(ball_a, dtype=float) - pc_a))
                d_end = float(np.linalg.norm(np.asarray(ball_b, dtype=float) - pc_b))
                if d_end - d_start < 0.3 * mag * (f2 - f1):
                    continue
            best = max(best, mag)
        if best <= 0.0:
            if rejected_non_separating:
                return None    # only player-co-moving motion found: not a ball flight
            mags = sorted(float(np.linalg.norm(v)) for _, _, v in vels)
            best = mags[len(mags) // 2]
    return best * fps * m_per_px * 3.6


def _time_of_flight_kmh(ball_track, contact: int, fps: float,
                        goal_box, shot_distance_m: Optional[float]) -> Optional[float]:
    """Average ball speed from contact until it reaches the goal mouth.

    Arrival = the frame where the ball is CLOSEST to the goal-mouth centre
    (not first box entry, which under-counts the distance travelled). Only
    accepted when the ball gets within 60% of the goal width and has clearly
    approached compared to where it started."""
    if goal_box is None or shot_distance_m is None or shot_distance_m < 3.0:
        return None
    gx1, gy1, gx2, gy2 = goal_box["xyxy"]
    mouth = np.array([(gx1 + gx2) / 2.0, gy2])
    goal_w_px = max(gx2 - gx1, 1.0)

    d_start = None
    best_i, best_d = None, None
    for i in range(contact, min(len(ball_track), contact + int(2.5 * fps))):
        p = ball_track[i]
        if p is None:
            continue
        d = float(np.linalg.norm(np.asarray(p, dtype=float) - mouth))
        if d_start is None:
            d_start = d
            continue
        if best_d is None or d < best_d:
            best_d, best_i = d, i

    if best_i is None or best_d is None or d_start is None:
        return None
    if best_d > 0.6 * goal_w_px:
        return None                      # never actually reached the goal
    if best_d > 0.6 * d_start:
        return None                      # barely approached: not a goal-bound shot
    dt = (best_i - contact) / max(fps, 1e-6)
    if dt < 0.08:
        return None                      # arrival in <0.08 s is a tracking glitch
    return float(shot_distance_m / dt * 3.6)


def power_label(kmh: Optional[float]) -> str:
    if kmh is None:
        return "n/a"
    for lo, hi, name in POWER_BANDS:
        if lo <= kmh < hi:
            return name
    return POWER_BANDS[-1][2]


# ------------------------------------------------------------------- metrics
def _shot_distance_m(bpos, goal_box, poses, player_boxes, cfg: PipelineConfig,
                     m_per_px: float, frame_w: Optional[int],
                     preferred_player_scale: Optional[float] = None) -> Optional[float]:
    """Depth-aware ball->goal distance.

    A flat image-plane measurement collapses to ~0 when shooting TOWARD the
    goal (ball and goal overlap on screen). Instead, the apparent-size scales
    at the two depths give the depth gap directly:
        Z = f_px * (real size / pixel size)  =>  dZ = f_px * (s_goal - s_player)
    with f_px approximated from a typical phone field of view (~65 deg).
    The lateral component comes from the image offset at the mean scale.
    Falls back to the flat estimate when either scale is unavailable.
    """
    if bpos is None or goal_box is None:
        return None
    gx1, gy1, gx2, gy2 = goal_box["xyxy"]
    goal_mouth = np.array([(gx1 + gx2) / 2.0, gy2])
    offset_px = float(np.linalg.norm(np.asarray(bpos, dtype=float) - goal_mouth))

    # The ball-diameter scale (when available) is the best player-plane
    # anchor: the ball sits exactly where the shot starts, whatever the
    # camera angle. Pose/bbox height are the fallbacks.
    s_p = preferred_player_scale or _pose_scale(poses, cfg) or _bbox_scale(player_boxes, cfg)
    s_g = _goal_scale(goal_box, cfg)
    if s_p and s_g and frame_w:
        f_px = 0.8 * float(frame_w)            # ~65 deg horizontal FOV assumption
        depth_m = f_px * abs(s_g - s_p)
        lateral_m = offset_px * (s_p + s_g) / 2.0
        dist = float(np.hypot(depth_m, lateral_m))
        if 1.0 <= dist <= 80.0:
            return dist
    if m_per_px > 0:
        flat = offset_px * m_per_px
        return flat if 1.0 <= flat <= 80.0 else None
    return None


def compute_metrics(poses: List[PoseFrame], ball_track, ev: ShotEvents,
                    cfg: PipelineConfig, fps: float,
                    goal_box, player_boxes,
                    frame_w: Optional[int] = None,
                    ball_diams_px=None) -> Analysis:
    a = Analysis()
    a.meters_per_px, a.scale_source = compute_scale(goal_box, player_boxes, cfg, poses)

    # The ball's own diameter is the best scale anchor for speed: it needs no
    # assumed player height and sits exactly at the shot's depth plane. Used
    # when it broadly agrees with the body-derived scale (guards against a
    # mis-sized blurry box).
    s_ball = _ball_scale(ball_diams_px, ev.contact, cfg)
    if s_ball is not None:
        ref = _pose_scale(poses, cfg) or _bbox_scale(player_boxes, cfg)
        if ref is None or 0.4 <= s_ball / ref <= 2.5:
            a.meters_per_px = s_ball
            a.scale_source = "ball diameter (0.22 m)"
    c, foot = ev.contact, ev.kicking_foot
    pf_c = poses[c] if poses[c].ok else _nearest_ok(poses, c)

    # Backswing knee flexion (report as flexion = 180 - min knee angle)
    ka_bsw = knee_angle(poses[ev.backswing_peak], foot) if poses[ev.backswing_peak].ok else None
    a.metrics["backswing_knee_flexion"] = (180.0 - ka_bsw) if ka_bsw is not None else None

    # Knee angle at contact (median over contact ±1 to suppress single-frame jitter)
    ka_win = [knee_angle(poses[i], foot)
              for i in range(max(0, c - 1), min(len(poses), c + 2)) if poses[i].ok]
    ka_win = [v for v in ka_win if v is not None]
    if ka_win:
        a.metrics["contact_knee_angle"] = float(np.median(ka_win))
    else:
        a.metrics["contact_knee_angle"] = knee_angle(pf_c, foot) if pf_c else None

    # Ankle lock: angular variation of the kicking ankle across contact ±2 frames
    win = [ankle_angle(poses[i], foot) for i in range(max(0, c - 2), min(len(poses), c + 3))
           if poses[i].ok]
    win = [v for v in win if v is not None]
    a.metrics["ankle_lock_variation"] = float(max(win) - min(win)) if len(win) >= 2 else None

    # Plant foot distance to ball at contact (metres)
    bpos = _last_ball_before(ball_track, c)
    plant_ankle = pf_c.get(f"{ev.plant_foot[0]}_ankle") if pf_c else None
    if bpos is not None and plant_ankle is not None and a.meters_per_px > 0:
        a.metrics["plant_foot_distance_m"] = float(np.linalg.norm(plant_ankle - bpos)) * a.meters_per_px
    else:
        a.metrics["plant_foot_distance_m"] = None

    # Approach angle: hip-midpoint travel direction vs ball->goal (or shot) line
    a.metrics["approach_angle"] = _approach_angle(poses, ball_track, ev, goal_box)

    # Shot distance: ball position at contact -> centre of the goal mouth.
    a.metrics["shot_distance_m"] = _shot_distance_m(
        bpos, goal_box, poses, player_boxes, cfg, a.meters_per_px, frame_w,
        preferred_player_scale=s_ball if a.scale_source.startswith("ball diameter") else None)

    # Hip rotation from plant to shortly after contact
    a.metrics["hip_rotation"] = hip_rotation_proxy(poses, ev.plant, min(c + 3, len(poses) - 1))

    # Trunk lean at contact (signed; + = over the ball)
    a.metrics["trunk_lean"] = trunk_lean_signed(pf_c, ev.shot_dir_x) if pf_c else None

    # Follow-through height: kicking foot rise above contact height / leg length
    a.metrics["follow_through_height"] = _follow_through(poses, ev, foot)

    # Shot power, with a plausibility guard: if the primary scale yields an
    # impossible speed, retry with the player-derived scale; if still
    # impossible, report n/a instead of a bogus number.
    from .config import MAX_PLAUSIBLE_SHOT_KMH, MIN_PLAUSIBLE_SHOT_KMH

    a.shot_speed_kmh = shot_power_kmh(ball_track, c, fps, a.meters_per_px,
                                      cfg.events.post_contact_fit_frames,
                                      player_boxes)
    if a.shot_speed_kmh is not None and not (
        MIN_PLAUSIBLE_SHOT_KMH <= a.shot_speed_kmh <= MAX_PLAUSIBLE_SHOT_KMH
    ):
        fallback = _pose_scale(poses, cfg) or _bbox_scale(player_boxes, cfg)
        if fallback and abs(fallback - a.meters_per_px) > 1e-9:
            retry = shot_power_kmh(ball_track, c, fps, fallback,
                                   cfg.events.post_contact_fit_frames,
                                   player_boxes)
            if retry is not None and MIN_PLAUSIBLE_SHOT_KMH <= retry <= MAX_PLAUSIBLE_SHOT_KMH:
                a.shot_speed_kmh = retry
                a.meters_per_px = fallback
                a.scale_source = f"player height ({cfg.player_height_m:.2f} m, power fallback)"
                # plant-foot distance was computed with the rejected scale; redo it
                if (a.metrics.get("plant_foot_distance_m") is not None
                        and bpos is not None and plant_ankle is not None):
                    a.metrics["plant_foot_distance_m"] = float(
                        np.linalg.norm(plant_ankle - bpos)) * fallback
            else:
                a.shot_speed_kmh = None
        else:
            a.shot_speed_kmh = None

    # Time-of-flight cross-estimate: depth-aware distance / time for the ball
    # to reach the goal mouth. Immune to the projection problem (a shot hit
    # AWAY from the camera looks slow in the image plane but its arrival time
    # doesn't lie). Both estimators only ever under-read, so take the larger.
    tof = _time_of_flight_kmh(ball_track, ev.contact, fps, goal_box,
                              a.metrics.get("shot_distance_m"))
    if tof is not None and MIN_PLAUSIBLE_SHOT_KMH <= tof <= MAX_PLAUSIBLE_SHOT_KMH:
        if a.shot_speed_kmh is None or tof > a.shot_speed_kmh:
            a.shot_speed_kmh = tof

    # Depth-resolved goal-plane crossing (ball size vs goal size): the most
    # physically grounded estimate of where and how fast the ball crossed.
    crossing = _goal_plane_crossing(ball_track, ball_diams_px, ev.contact, fps,
                                    goal_box, frame_w, cfg)
    if crossing is not None:
        a.metrics["goal_crossing_height_m"] = crossing["height_m"]
        a.metrics["goal_crossing_offset_m"] = crossing["offset_m"]
        a.on_target = crossing["on_target"]
        spd = crossing["speed_kmh"]
        if spd is not None and MIN_PLAUSIBLE_SHOT_KMH <= spd <= MAX_PLAUSIBLE_SHOT_KMH:
            if a.shot_speed_kmh is None or spd > a.shot_speed_kmh:
                a.shot_speed_kmh = spd
        if a.metrics.get("shot_distance_m") is None and crossing["dist_m"]:
            a.metrics["shot_distance_m"] = crossing["dist_m"]
    a.power_label = power_label(a.shot_speed_kmh)

    _score_and_feedback(a, ev)
    return a


def _nearest_ok(poses, i):
    for d in range(1, 6):
        for j in (i - d, i + d):
            if 0 <= j < len(poses) and poses[j].ok:
                return poses[j]
    return None


def _last_ball_before(ball_track, c):
    for i in range(c, max(-1, c - 6), -1):
        if ball_track[i] is not None:
            return ball_track[i]
    return None


def _approach_angle(poses, ball_track, ev: ShotEvents, goal_box) -> Optional[float]:
    lo = max(0, ev.plant - 12)
    mids = [poses[i].mid("l_hip", "r_hip") for i in range(lo, ev.plant + 1) if poses[i].ok]
    mids = [m for m in mids if m is not None]
    if len(mids) < 2:
        return None
    run = mids[-1] - mids[0]
    if np.linalg.norm(run) < 5:
        return None
    bpos = _last_ball_before(ball_track, ev.contact)
    if bpos is None:
        return None
    if goal_box is not None:
        gx1, gy1, gx2, gy2 = goal_box["xyxy"]
        target = np.array([(gx1 + gx2) / 2, (gy1 + gy2) / 2])
    else:
        after = [p for p in ball_track[ev.contact:ev.contact + 8] if p is not None]
        if len(after) < 2:
            return None
        target = bpos + (after[-1] - after[0])
    shot_line = target - bpos
    if np.linalg.norm(shot_line) < 1e-6:
        return None
    cosang = np.clip(np.dot(run, shot_line) /
                     (np.linalg.norm(run) * np.linalg.norm(shot_line)), -1, 1)
    return float(np.degrees(np.arccos(cosang)))


def _follow_through(poses, ev: ShotEvents, foot: str) -> Optional[float]:
    s = foot[0]
    pf_c = poses[ev.contact] if poses[ev.contact].ok else None
    if pf_c is None:
        return None
    base = pf_c.get(f"{s}_foot")
    leg = leg_length_px(pf_c, foot)
    if base is None or leg is None or leg < 1e-6:
        return None
    best = 0.0
    for i in range(ev.contact + 1, min(len(poses), ev.contact + 18)):
        if not poses[i].ok:
            continue
        p = poses[i].get(f"{s}_foot")
        if p is None:
            continue
        rise = (base[1] - p[1]) / leg          # image y is down; rise = positive up
        best = max(best, rise)
    return float(best)


# ------------------------------------------------------------ score+feedback
_FEEDBACK = {
    "backswing_knee_flexion": (
        "Your backswing knee flexion was {v:.0f}° (ideal {lo:.0f}–{hi:.0f}°). Cock your kicking "
        "leg further back — heel toward glute — to store more elastic energy before the strike.",
        "Backswing knee flexion {v:.0f}° is past the ideal {lo:.0f}–{hi:.0f}° window; an "
        "over-deep backswing delays the leg whip. Slightly shorten it and accelerate sooner."),
    "contact_knee_angle": (
        "Knee angle at contact was {v:.0f}° (ideal {lo:.0f}–{hi:.0f}°). You're striking with a bent "
        "knee — delay contact a fraction so the knee is snapping toward extension through the ball.",
        "Knee angle at contact was {v:.0f}° — close to locked. Contact should happen just BEFORE "
        "full extension ({lo:.0f}–{hi:.0f}°) so the snap finishes through the ball, not before it."),
    "ankle_lock_variation": (
        None,
        "Your ankle moved {v:.0f}° through contact (target < {hi:.0f}°). Lock it: toes pointed "
        "down and rigid like a hammer head — a loose ankle leaks power and accuracy."),
    "plant_foot_distance_m": (
        "Plant foot was only {v100:.0f} cm from the ball — too tight, cramping the hip swing. "
        "Plant a comfortable {lo100:.0f}–{hi100:.0f} cm beside the ball.",
        "Plant foot landed {v100:.0f} cm from the ball (ideal {lo100:.0f}–{hi100:.0f} cm). Planting "
        "too far makes you reach, opening the hips early and dragging the shot."),
    "approach_angle": (
        "Your approach was only {v:.0f}° off the shot line (ideal {lo:.0f}–{hi:.0f}°). A straighter "
        "run-up limits hip rotation — approach more diagonally, around 30–45°.",
        "Approach angle of {v:.0f}° is steeper than the ideal {lo:.0f}–{hi:.0f}°; it forces "
        "compensation in the plant step. Flatten the run-up slightly."),
    "hip_rotation": (
        "Hip rotation measured ~{v:.0f}° (ideal {lo:.0f}–{hi:.0f}°). Drive the kicking-side hip "
        "through the ball — power comes from the pelvis, the leg just delivers it.",
        None),
    "trunk_lean": (
        "Your trunk was leaning back ({v:.0f}°) at contact — the classic cause of ballooned shots. "
        "Get your head and chest over the ball ({lo:.0f}–{hi:.0f}° forward lean).",
        "Trunk lean of {v:.0f}° is excessive (ideal {lo:.0f}–{hi:.0f}°); over-leaning smothers the "
        "shot low and kills lift. Stay tall through the strike."),
    "follow_through_height": (
        "Follow-through peaked at {v:.2f}× leg length (ideal ≥ {lo:.2f}). Don't stab at the ball — "
        "swing through it and let the kicking foot finish high toward the target.",
        None),
}

_PRAISE = {
    "contact_knee_angle": "Knee extension timing at contact ({v:.0f}°) is in the ideal window — clean leg whip.",
    "ankle_lock_variation": "Excellent ankle lock — only {v:.0f}° of movement through contact.",
    "plant_foot_distance_m": "Plant foot placement ({v100:.0f} cm from the ball) is textbook.",
    "trunk_lean": "Good body position — {v:.0f}° forward over the ball keeps the shot down.",
    "follow_through_height": "Strong, high follow-through ({v:.2f}× leg length) — full hip drive.",
    "backswing_knee_flexion": "Deep backswing ({v:.0f}° flexion) — great elastic loading.",
    "approach_angle": "Approach angle of {v:.0f}° is right in the optimal diagonal zone.",
    "hip_rotation": "Strong hip rotation (~{v:.0f}°) driving the kick from the pelvis.",
}


def _fmt(template: str, v: float, ideal) -> str:
    return template.format(v=v, lo=ideal.lo, hi=ideal.hi,
                           v100=v * 100, lo100=ideal.lo * 100, hi100=ideal.hi * 100)


def _score_and_feedback(a: Analysis, ev: ShotEvents) -> None:
    total_w, lost = 0.0, 0.0
    issues, praises = [], []

    for key, ideal in IDEALS.items():
        v = a.metrics.get(key)
        if v is None:
            continue
        total_w += ideal.weight
        if ideal.lo <= v <= ideal.hi:
            a.per_metric_ok[key] = True
            if key in _PRAISE:
                praises.append((ideal.weight, _fmt(_PRAISE[key], v, ideal)))
            continue
        a.per_metric_ok[key] = False
        span = max(ideal.hi - ideal.lo, 1e-6)
        dev = (ideal.lo - v) / span if v < ideal.lo else (v - ideal.hi) / span
        sev = float(np.clip(dev, 0.15, 1.0))
        lost += ideal.weight * sev
        low_msg, high_msg = _FEEDBACK[key]
        msg = low_msg if v < ideal.lo else high_msg
        if msg is None:
            msg = (low_msg or high_msg)
        if msg:
            issues.append((ideal.weight * sev, _fmt(msg, v, ideal)))

    a.score = round(10.0 * (1.0 - lost / total_w), 1) if total_w > 0 else 0.0
    a.score = max(a.score, 1.0)

    # 3–6 specific points: worst issues first, then top praise, then guidance.
    issues.sort(key=lambda t: -t[0])
    praises.sort(key=lambda t: -t[0])
    fb = [m for _, m in issues[:5]]
    while len(fb) < 3 and praises:
        fb.append(praises.pop(0)[1])
    if len(fb) < 6 and praises and len(fb) < 4:
        fb.append(praises.pop(0)[1])
    if len(fb) < 3:
        fb.append(f"Detected a {ev.kicking_foot}-footed strike; record from a side-on angle "
                  "(~90° to the shot line) for the most reliable measurements.")
    a.feedback = fb[:6]
