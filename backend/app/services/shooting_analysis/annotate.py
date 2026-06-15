"""Render the annotated output video: skeleton, angles, phases, power, scorecard.

Improvements over the original:
  * Fading ball trail (recent path bright, older path dim) instead of an
    ever-growing opaque polyline that ends up covering the player.
  * Knee angle drawn with an arc between thigh and shank, not just text.
  * Labelled phase chips on the timeline; CONTACT flash with vignette.
  * Word-wrapped scorecard so feedback lines aren't truncated mid-sentence.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
import mediapipe as mp

from .config import PipelineConfig, IDEALS
from .events import ShotEvents
from .kinematics import knee_angle, ankle_angle, trunk_lean_signed
from .metrics import Analysis
from .pose import PoseFrame

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

FONT = cv2.FONT_HERSHEY_SIMPLEX
TRAIL_MAX = 28                 # number of recent ball positions kept in the trail


def _put(img, text, org, scale=0.55, color=(240, 240, 240), thick=1, shadow=True):
    if shadow:
        cv2.putText(img, text, (org[0] + 1, org[1] + 1), FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _panel(img, x, y, w, h, color, alpha=0.62):
    x, y = max(0, x), max(0, y)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)
    if w <= 0 or h <= 0:
        return
    roi = img[y:y + h, x:x + w]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def _wrap(text: str, width: int) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for word in words:
        cand = f"{cur} {word}".strip()
        if len(cand) <= width:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


class Annotator:
    def __init__(self, cfg: PipelineConfig, ev: ShotEvents, analysis: Analysis,
                 n_frames: int, fps: float, frame_size):
        self.cfg, self.ev, self.a = cfg, ev, analysis
        self.n, self.fps = n_frames, fps
        self.w, self.h = frame_size
        self.s = cfg.style
        self.trail: List[np.ndarray] = []
        self._scorecard_lines = self._build_scorecard_lines()

    # ------------------------------------------------------------------ main
    def draw(self, frame, i: int, pose: PoseFrame,
             ball: Optional[np.ndarray], goal_box) -> np.ndarray:
        img = frame
        if goal_box is not None:
            x1, y1, x2, y2 = map(int, goal_box["xyxy"])
            cv2.rectangle(img, (x1, y1), (x2, y2), self.s.goal_color, 2)
            _put(img, "GOAL", (x1, max(18, y1 - 8)), 0.55, self.s.goal_color, 2)

        if pose.ok and pose.raw is not None:
            mp_draw.draw_landmarks(
                img, pose.raw, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=self.s.joint_color, thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=self.s.skeleton_color, thickness=2),
            )
            self._angle_labels(img, i, pose)

        if ball is not None:
            self.trail.append(ball.copy())
            if len(self.trail) > TRAIL_MAX:
                self.trail = self.trail[-TRAIL_MAX:]
            cv2.circle(img, tuple(map(int, ball)), 9, self.s.ball_color, 2)
        self._draw_trail(img)

        self._phase_bar(img, i)
        if i >= self.ev.contact:
            self._power_overlay(img)
        if i == self.ev.contact:
            self._flash(img, "CONTACT")
        return img

    # ----------------------------------------------------------- components
    def _draw_trail(self, img):
        m = len(self.trail)
        for k in range(1, m):
            t = k / max(m - 1, 1)                     # 0 oldest -> 1 newest
            col = tuple(int(c * (0.35 + 0.65 * t)) for c in self.s.ball_trail_color)
            cv2.line(img, tuple(map(int, self.trail[k - 1])),
                     tuple(map(int, self.trail[k])), col, 1 + int(round(t)))

    def _angle_labels(self, img, i, pose: PoseFrame):
        foot = self.ev.kicking_foot
        s = foot[0]
        ka = knee_angle(pose, foot)
        hip, knee, ank = pose.get(f"{s}_hip"), pose.get(f"{s}_knee"), pose.get(f"{s}_ankle")
        if ka is not None and knee is not None:
            ok = IDEALS["contact_knee_angle"].lo <= ka <= IDEALS["contact_knee_angle"].hi
            col = self.s.angle_good if (ok or i < self.ev.backswing_peak) else self.s.angle_bad
            if hip is not None and ank is not None:
                self._angle_arc(img, hip, knee, ank, col)
            _put(img, f"knee {ka:.0f}", (int(knee[0]) + 10, int(knee[1])), 0.5, col, 1)

        aa = ankle_angle(pose, foot)
        ankle_pt = pose.get(f"{s}_ankle")
        if aa is not None and ankle_pt is not None and abs(i - self.ev.contact) <= 4:
            _put(img, f"ankle {aa:.0f}", (int(ankle_pt[0]) + 8, int(ankle_pt[1]) + 14), 0.45,
                 self.s.accent, 1)

        tl = trunk_lean_signed(pose, self.ev.shot_dir_x)
        sh = pose.mid("l_shoulder", "r_shoulder")
        if tl is not None and sh is not None and abs(i - self.ev.contact) <= 6:
            ok = IDEALS["trunk_lean"].lo <= tl <= IDEALS["trunk_lean"].hi
            _put(img, f"trunk {tl:+.0f}", (int(sh[0]) + 8, int(sh[1]) - 8), 0.5,
                 self.s.angle_good if ok else self.s.angle_bad, 1)

    @staticmethod
    def _angle_arc(img, a, b, c, color, radius=22):
        """Small arc at vertex b spanning rays b->a and b->c."""
        ang1 = np.degrees(np.arctan2(a[1] - b[1], a[0] - b[0]))
        ang2 = np.degrees(np.arctan2(c[1] - b[1], c[0] - b[0]))
        sweep = (ang2 - ang1) % 360.0
        if sweep > 180.0:
            ang1, ang2 = ang2, ang1
            sweep = 360.0 - sweep
        cv2.ellipse(img, tuple(map(int, b)), (radius, radius), 0.0,
                    float(ang1), float(ang1 + sweep), color, 2, cv2.LINE_AA)

    def _phase_bar(self, img, i):
        phase = self.ev.phase_at(i, self.n)
        _panel(img, 0, 0, self.w, 34, self.s.panel_bg)
        _put(img, f"PHASE: {phase.upper()}", (12, 23), 0.62, self.s.accent, 2)
        _put(img, f"{self.ev.kicking_foot.upper()} FOOT", (self.w - 150, 23), 0.5,
             self.s.text, 1)
        # mini timeline with labelled event markers
        bx, bw, by = int(self.w * 0.34), int(self.w * 0.38), 17
        cv2.line(img, (bx, by), (bx + bw, by), (90, 90, 90), 3)
        for f, col, tag in ((self.ev.plant, (200, 200, 80), "P"),
                            (self.ev.backswing_peak, (80, 160, 255), "B"),
                            (self.ev.contact, (60, 60, 230), "C")):
            px = bx + int(bw * f / max(1, self.n - 1))
            cv2.circle(img, (px, by), 4, col, -1)
            _put(img, tag, (px - 4, by + 13), 0.35, col, 1, shadow=False)
        cv2.circle(img, (bx + int(bw * i / max(1, self.n - 1)), by), 5, (255, 255, 255), 1)

    def _power_overlay(self, img):
        kmh = self.a.shot_speed_kmh
        x, y, w, h = self.w - 232, 44, 220, 86
        _panel(img, x, y, w, h, self.s.panel_bg)
        _put(img, "SHOT POWER", (x + 12, y + 22), 0.5, self.s.text, 1)
        if kmh is None:
            _put(img, "n/a", (x + 12, y + 52), 0.8, self.s.text, 2)
            return
        _put(img, f"{kmh:.0f} km/h", (x + 12, y + 52), 0.85, self.s.accent, 2)
        _put(img, self.a.power_label, (x + 12, y + 74), 0.5, self.s.text, 1)
        frac = float(np.clip(kmh / 110.0, 0, 1))
        cv2.rectangle(img, (x + 110, y + 62), (x + 208, y + 74), (70, 70, 70), -1)
        cv2.rectangle(img, (x + 110, y + 62), (x + 110 + int(98 * frac), y + 74),
                      self.s.accent, -1)

    def _flash(self, img, text):
        # darken edges slightly so the flash reads instantly
        _panel(img, 0, 0, self.w, self.h, (0, 0, 0), alpha=0.18)
        size = cv2.getTextSize(text, FONT, 1.4, 3)[0]
        _put(img, text, (int((self.w - size[0]) / 2), int(self.h * 0.5)), 1.4,
             (60, 60, 230), 3)

    def _build_scorecard_lines(self) -> List[str]:
        lines: List[str] = []
        for k, point in enumerate(self.a.feedback):
            wrapped = _wrap(f"{k + 1}. {point}", 80)
            lines.extend(wrapped[:2])          # max 2 lines per point
        return lines

    # ----------------------------------------------------------- end card
    _METRIC_LABELS = (
        ("backswing_knee_flexion", "Backswing knee flexion", "{v:.0f} deg"),
        ("contact_knee_angle", "Knee at contact", "{v:.0f} deg"),
        ("ankle_lock_variation", "Ankle lock variation", "{v:.0f} deg"),
        ("plant_foot_distance_m", "Plant foot distance", "{v100:.0f} cm"),
        ("approach_angle", "Approach angle", "{v:.0f} deg"),
        ("hip_rotation", "Hip rotation", "{v:.0f} deg"),
        ("trunk_lean", "Trunk lean", "{v:.0f} deg"),
        ("follow_through_height", "Follow-through height", "{v:.2f}x leg"),
    )

    def end_card(self) -> np.ndarray:
        """Full-frame, readable summary shown for a few seconds at the end of
        the annotated video: score, power, per-metric verdicts, feedback."""
        img = np.full((self.h, self.w, 3), self.s.panel_bg, dtype=np.uint8)
        margin = max(int(self.w * 0.05), 24)
        x = margin
        y = margin + 18

        scale_big = max(self.w / 1100.0, 0.7)
        scale_md = scale_big * 0.62
        scale_sm = scale_big * 0.5
        line_md = int(34 * scale_big)
        line_sm = int(26 * scale_big)

        _put(img, f"TECHNIQUE SCORE  {self.a.score:.1f}/10", (x, y),
             scale_big, self.s.accent, 3)
        y += line_md
        if self.a.shot_speed_kmh is not None:
            _put(img, f"SHOT POWER  {self.a.shot_speed_kmh:.0f} km/h  ({self.a.power_label})",
                 (x, y), scale_big * 0.8, self.s.text, 2)
        else:
            _put(img, "SHOT POWER  n/a (scale unreliable in this clip)",
                 (x, y), scale_big * 0.8, self.s.text, 2)
        y += line_sm
        _put(img, f"{self.ev.kicking_foot.upper()} FOOT  |  scale: {self.a.scale_source}",
             (x, y), scale_sm, (170, 170, 170), 1)
        y += int(line_md * 1.1)

        # Two-column layout: metrics left, feedback right (stacked if narrow)
        col2_x = x + int(self.w * 0.46) if self.w >= 900 else x
        metrics_bottom = y
        my = y
        _put(img, "MEASUREMENTS", (x, my), scale_md, self.s.accent, 2)
        my += line_sm
        from .config import IDEALS
        for key, label, fmt in self._METRIC_LABELS:
            v = self.a.metrics.get(key)
            ideal = IDEALS.get(key)
            if v is None:
                txt = f"{label}: --"
                col = (130, 130, 130)
            else:
                txt = f"{label}: " + fmt.format(v=v, v100=v * 100)
                ok = ideal is not None and ideal.lo <= v <= ideal.hi
                col = self.s.angle_good if ok else self.s.angle_bad
                txt += "  OK" if ok else "  FIX"
            _put(img, txt, (x, my), scale_sm, col, 1)
            my += line_sm
        metrics_bottom = my

        fy = y if self.w >= 900 else metrics_bottom + line_sm
        _put(img, "FEEDBACK", (col2_x, fy), scale_md, self.s.accent, 2)
        fy += line_sm
        wrap_w = 52 if self.w >= 900 else 80
        for k, point in enumerate(self.a.feedback):
            for j, line in enumerate(_wrap(f"{k + 1}. {point}", wrap_w)[:3]):
                indent = 0 if j == 0 else int(16 * scale_big)
                if fy > self.h - margin:
                    break
                _put(img, line, (col2_x + indent, fy), scale_sm, self.s.text, 1)
                fy += line_sm
            fy += int(line_sm * 0.25)
        return img
