"""MediaPipe Pose: full-body landmarks on every frame.

Improvements over the original:
  * Optional ROI processing — when a player bounding box is supplied, the pose
    model runs on a padded crop around the player. For wide framings where the
    player is small, this dramatically improves landmark accuracy.
  * Automatic full-frame fallback when the ROI yields no pose.
  * `interpolate_missing` accepts a configurable max gap.
"""
from __future__ import annotations

from typing import Optional, List, Tuple

import numpy as np

import mediapipe as mp

mp_pose = mp.solutions.pose
L = mp_pose.PoseLandmark  # shorthand

# Landmarks we use by name
JOINTS = {
    "l_shoulder": L.LEFT_SHOULDER, "r_shoulder": L.RIGHT_SHOULDER,
    "l_hip": L.LEFT_HIP,           "r_hip": L.RIGHT_HIP,
    "l_knee": L.LEFT_KNEE,         "r_knee": L.RIGHT_KNEE,
    "l_ankle": L.LEFT_ANKLE,       "r_ankle": L.RIGHT_ANKLE,
    "l_heel": L.LEFT_HEEL,         "r_heel": L.RIGHT_HEEL,
    "l_foot": L.LEFT_FOOT_INDEX,   "r_foot": L.RIGHT_FOOT_INDEX,
}

ROI_PAD_FRAC = 0.35          # padding around player bbox when cropping
MIN_ROI_SIZE = 192           # don't crop tighter than this (px)


class PoseFrame:
    """Pixel-space landmark snapshot for one frame."""

    __slots__ = ("ok", "pts", "vis", "raw")

    def __init__(self, ok: bool, pts: dict, vis: dict, raw=None):
        self.ok = ok
        self.pts = pts      # name -> np.array([x_px, y_px])
        self.vis = vis      # name -> visibility 0..1
        self.raw = raw      # raw mediapipe landmark list (for skeleton drawing)

    def get(self, name: str) -> Optional[np.ndarray]:
        p = self.pts.get(name)
        if p is None or self.vis.get(name, 0) < 0.4:
            return None
        return p

    def mid(self, a: str, b: str) -> Optional[np.ndarray]:
        pa, pb = self.get(a), self.get(b)
        if pa is None or pb is None:
            return None
        return (pa + pb) / 2.0


class PoseTracker:
    def __init__(self):
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,          # full model: best landmark accuracy
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # Separate instance for ROI crops: crop geometry changes frame to frame,
        # so temporal smoothing across crops would corrupt landmarks.
        self._pose_roi = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
        )

    def process(self, frame_bgr: np.ndarray,
                player_box: Optional[Tuple[float, float, float, float]] = None) -> PoseFrame:
        """Run pose on the frame. If `player_box` (x1,y1,x2,y2) is given and the
        player occupies a small part of the frame, run on a padded crop instead
        and map landmarks back to full-frame pixel coordinates."""
        h, w = frame_bgr.shape[:2]

        if player_box is not None:
            x1, y1, x2, y2 = player_box
            box_h = max(y2 - y1, 1.0)
            if box_h < 0.55 * h:                 # player small in frame -> ROI helps
                result = self._process_roi(frame_bgr, player_box)
                if result.ok:
                    return result

        return self._process_full(frame_bgr)

    # ------------------------------------------------------------------ internals
    def _process_full(self, frame_bgr: np.ndarray) -> PoseFrame:
        import cv2
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._pose.process(rgb)
        if not res.pose_landmarks:
            return PoseFrame(False, {}, {}, None)
        return self._to_pose_frame(res, 0.0, 0.0, w, h, w, h)

    def _process_roi(self, frame_bgr: np.ndarray,
                     player_box: Tuple[float, float, float, float]) -> PoseFrame:
        import cv2
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = player_box
        pad_x = max((x2 - x1) * ROI_PAD_FRAC, (MIN_ROI_SIZE - (x2 - x1)) / 2, 0)
        pad_y = max((y2 - y1) * ROI_PAD_FRAC, (MIN_ROI_SIZE - (y2 - y1)) / 2, 0)
        cx1 = int(max(0, x1 - pad_x))
        cy1 = int(max(0, y1 - pad_y))
        cx2 = int(min(w, x2 + pad_x))
        cy2 = int(min(h, y2 + pad_y))
        if cx2 - cx1 < 32 or cy2 - cy1 < 32:
            return PoseFrame(False, {}, {}, None)

        crop = frame_bgr[cy1:cy2, cx1:cx2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._pose_roi.process(rgb)
        if not res.pose_landmarks:
            return PoseFrame(False, {}, {}, None)
        return self._to_pose_frame(res, cx1, cy1, cx2 - cx1, cy2 - cy1, w, h)

    @staticmethod
    def _to_pose_frame(res, off_x: float, off_y: float,
                       roi_w: float, roi_h: float,
                       frame_w: int, frame_h: int) -> PoseFrame:
        lms = res.pose_landmarks.landmark
        pts, vis = {}, {}
        for name, idx in JOINTS.items():
            lm = lms[idx]
            pts[name] = np.array(
                [off_x + lm.x * roi_w, off_y + lm.y * roi_h], dtype=np.float64
            )
            vis[name] = lm.visibility
        # Remap normalised raw landmarks to full-frame coords so skeleton drawing
        # stays correct even for ROI runs.
        if off_x != 0.0 or off_y != 0.0 or roi_w != frame_w or roi_h != frame_h:
            for lm in res.pose_landmarks.landmark:
                lm.x = (off_x + lm.x * roi_w) / max(frame_w, 1)
                lm.y = (off_y + lm.y * roi_h) / max(frame_h, 1)
        return PoseFrame(True, pts, vis, res.pose_landmarks)

    def close(self):
        self._pose.close()
        self._pose_roi.close()


def interpolate_missing(track: List[Optional[np.ndarray]],
                        max_gap: int = 8) -> List[Optional[np.ndarray]]:
    """Linear-fill short gaps in a 2D point track (e.g. ball centres)."""
    n = len(track)
    out = list(track)
    i = 0
    while i < n:
        if out[i] is None:
            j = i
            while j < n and out[j] is None:
                j += 1
            if 0 < i and j < n and (j - i) <= max_gap:
                a, b = out[i - 1], out[j]
                for k in range(i, j):
                    t = (k - i + 1) / (j - i + 1)
                    out[k] = a + (b - a) * t
            i = j
        else:
            i += 1
    return out


def interpolate_boxes(boxes: List[Optional[dict]],
                      max_gap: int = 12) -> List[Optional[dict]]:
    """Linear-fill short gaps in a bbox track (player boxes sampled on a stride)."""
    n = len(boxes)
    out = list(boxes)
    i = 0
    while i < n:
        if out[i] is None:
            j = i
            while j < n and out[j] is None:
                j += 1
            if 0 < i and j < n and (j - i) <= max_gap:
                a = np.array(out[i - 1]["xyxy"], dtype=float)
                b = np.array(out[j]["xyxy"], dtype=float)
                for k in range(i, j):
                    t = (k - i + 1) / (j - i + 1)
                    xyxy = a + (b - a) * t
                    out[k] = {"label": out[i - 1].get("label", "person"),
                              "conf": 0.0,
                              "xyxy": tuple(float(v) for v in xyxy),
                              "interpolated": True}
            i = j
        else:
            i += 1
    return out
