"""Keypoint-based pitch registration and per-frame homography chaining.

Part 1 — absolute calibration from whatever pitch markings are visible:
the roboflow/sports pitch keypoint model outputs the pixel locations of the
~32 standard pitch landmarks (box corners, goal-area corners, penalty spots,
centre-circle extremes, line intersections). Any 4+ visible keypoints give a
homography into real-world pitch coordinates — no full pitch required, no
hand-written line/circle geometry.

Part 2 — continuous recalibration for camera motion: keypoints are detected
every N frames; between fixes the homography is propagated by chaining
frame-to-frame homographies estimated from background optical flow, and it
snaps back to absolute on the next keypoint fix. The result is H(t) per frame.

Part 3 — honesty about gaps: every conversion carries a freshness age
(frames since the last absolute fix). Long gaps are NOT bridged with guesses;
points older than MAX_AGE_S are reported uncalibrated and excluded from
speed metrics by the caller.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

# Real FIFA dimensions (metres). x runs along the pitch length, y along width.
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
PENALTY_BOX_W = 40.32
PENALTY_BOX_D = 16.5
GOAL_BOX_W = 18.32
GOAL_BOX_D = 5.5
PENALTY_SPOT = 11.0
CIRCLE_R = 9.15


def _build_vertices() -> list[tuple[float, float]]:
    """The 32-vertex layout used by the roboflow/sports field model, with
    real-world coordinates."""
    L, W = PITCH_LENGTH_M, PITCH_WIDTH_M
    return [
        (0.0, 0.0),
        (0.0, (W - PENALTY_BOX_W) / 2),
        (0.0, (W - GOAL_BOX_W) / 2),
        (0.0, (W + GOAL_BOX_W) / 2),
        (0.0, (W + PENALTY_BOX_W) / 2),
        (0.0, W),
        (GOAL_BOX_D, (W - GOAL_BOX_W) / 2),
        (GOAL_BOX_D, (W + GOAL_BOX_W) / 2),
        (PENALTY_SPOT, W / 2),
        (PENALTY_BOX_D, (W - PENALTY_BOX_W) / 2),
        (PENALTY_BOX_D, (W - GOAL_BOX_W) / 2),
        (PENALTY_BOX_D, (W + GOAL_BOX_W) / 2),
        (PENALTY_BOX_D, (W + PENALTY_BOX_W) / 2),
        (L / 2, 0.0),
        (L / 2, W / 2 - CIRCLE_R),
        (L / 2, W / 2 + CIRCLE_R),
        (L / 2, W),
        (L - PENALTY_BOX_D, (W - PENALTY_BOX_W) / 2),
        (L - PENALTY_BOX_D, (W - GOAL_BOX_W) / 2),
        (L - PENALTY_BOX_D, (W + GOAL_BOX_W) / 2),
        (L - PENALTY_BOX_D, (W + PENALTY_BOX_W) / 2),
        (L - PENALTY_SPOT, W / 2),
        (L - GOAL_BOX_D, (W - GOAL_BOX_W) / 2),
        (L - GOAL_BOX_D, (W + GOAL_BOX_W) / 2),
        (L, 0.0),
        (L, (W - PENALTY_BOX_W) / 2),
        (L, (W - GOAL_BOX_W) / 2),
        (L, (W + GOAL_BOX_W) / 2),
        (L, (W + PENALTY_BOX_W) / 2),
        (L, W),
        (L / 2 - CIRCLE_R, W / 2),
        (L / 2 + CIRCLE_R, W / 2),
    ]


PITCH_VERTICES = _build_vertices()

PITCH_MODEL_ID = os.environ.get("ROBOFLOW_PITCH_MODEL", "").strip() or "football-field-detection-f07vi/15"
KEYPOINT_CONF = 0.5
MIN_KEYPOINTS = 4
RANSAC_REPROJ_M = 1.0          # findHomography threshold, in metres (dst space)


def _extract_keypoints(result: dict) -> list[tuple[int, float, float, float]]:
    """Normalise hosted-inference keypoint output to (class_index, x, y, conf)."""
    raw: list[tuple[int, float, float, float]] = []
    predictions = result.get("predictions", []) if isinstance(result, dict) else []
    for pred in predictions:
        for kp in pred.get("keypoints", []) or []:
            cls = kp.get("class_id")
            if cls is None:
                name = str(kp.get("class", kp.get("class_name", ""))).strip()
                try:
                    cls = int(name)
                except ValueError:
                    continue
            raw.append((int(cls), float(kp.get("x", 0)), float(kp.get("y", 0)),
                        float(kp.get("confidence", 0))))
    if not raw:
        return []
    # Some exports index classes 1..32 instead of 0..31.
    max_id = max(item[0] for item in raw)
    if max_id >= len(PITCH_VERTICES):
        raw = [(cls - 1, x, y, c) for cls, x, y, c in raw]
    return raw


class PitchKeypointDetector:
    """Hosted pitch-keypoint model wrapper. Fails soft: if the model is
    unavailable the detector disables itself and calibration falls back to
    the static path."""

    def __init__(self) -> None:
        self._model = None
        self._failed = False

    def _get_model(self):
        if self._failed:
            return None
        if self._model is None:
            try:
                from app.services.roboflow_inference import get_model
                self._model = get_model(model_id=PITCH_MODEL_ID)
                print(f"[GameSense] pitch keypoint model ready ({PITCH_MODEL_ID})")
            except Exception as exc:
                print(f"[GameSense] pitch keypoint model unavailable: {exc}")
                self._failed = True
        return self._model

    @property
    def available(self) -> bool:
        return not self._failed

    def homography(self, frame_bgr: np.ndarray) -> tuple[np.ndarray | None, int]:
        """Absolute image->pitch homography from this frame's visible
        keypoints. Returns (H, inlier_count) or (None, n_found)."""
        model = self._get_model()
        if model is None:
            return None, 0
        try:
            result = model.infer(frame_bgr, confidence=0.3)[0]
        except Exception as exc:
            print(f"[GameSense] pitch keypoint inference failed: {exc}")
            self._failed = True
            return None, 0

        img_pts: list[list[float]] = []
        world_pts: list[tuple[float, float]] = []
        for cls, x, y, conf in _extract_keypoints(result):
            if conf < KEYPOINT_CONF or not (0 <= cls < len(PITCH_VERTICES)):
                continue
            img_pts.append([x, y])
            world_pts.append(PITCH_VERTICES[cls])
        if len(img_pts) < MIN_KEYPOINTS:
            return None, len(img_pts)

        H, inlier_mask = cv2.findHomography(
            np.array(img_pts, dtype=np.float32),
            np.array(world_pts, dtype=np.float32),
            cv2.RANSAC,
            RANSAC_REPROJ_M,
        )
        if H is None or not np.isfinite(H).all():
            return None, len(img_pts)
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else len(img_pts)
        if inliers < MIN_KEYPOINTS:
            return None, inliers
        return H, inliers


def interframe_homography(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    exclude_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> np.ndarray | None:
    """Homography mapping PREVIOUS-frame pixels to CURRENT-frame pixels,
    estimated from background feature flow (players masked out)."""
    mask = np.full(prev_gray.shape, 255, dtype=np.uint8)
    for bbox in exclude_bboxes or []:
        x, y, w, h = bbox
        cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y + h)), 0, -1)
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=150, qualityLevel=0.01,
                                  minDistance=12, mask=mask)
    if pts is None or len(pts) < 12:
        return None
    nxt, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts, None)
    if nxt is None or status is None:
        return None
    ok = status.flatten() == 1
    good_prev = pts[ok].reshape(-1, 2)
    good_next = nxt[ok].reshape(-1, 2)
    if len(good_prev) < 12:
        return None
    H, _mask = cv2.findHomography(good_prev, good_next, cv2.RANSAC, 2.0)
    if H is None or not np.isfinite(H).all():
        return None
    return H


class DynamicPitchCalibration:
    """Per-frame homography H(t): absolute keypoint fixes chained with
    inter-frame flow homographies, with an explicit freshness age.

    Anchor mode (`set_anchor`): seeds the chain with a trusted static
    homography — e.g. the user's manual pitch polygon — and tracks the
    cumulative camera pose. Whenever the camera returns near the anchor pose
    the chain snaps back to the exact anchor and the age resets, so a static
    camera keeps the manual calibration exactly valid indefinitely while a
    panning camera gets motion-compensated coordinates."""

    MAX_AGE_S = 3.0            # beyond this, points are uncalibrated (no guessing)
    FLOW_FAIL_PENALTY_S = 0.25 # a missed flow estimate ages the fix faster
    SNAPBACK_CORNER_FRAC = 0.005  # of frame diagonal: max corner drift for re-anchor

    def __init__(self, fps: float) -> None:
        self.fps = max(fps, 1e-6)
        self._H: np.ndarray | None = None
        self._age_frames: float | None = None
        self.fix_count = 0
        self._anchor: np.ndarray | None = None
        self._cum: np.ndarray | None = None
        self._frame_size: tuple[int, int] | None = None

    @property
    def has_fix(self) -> bool:
        return self._H is not None

    def set_absolute(self, H: np.ndarray) -> None:
        self._H = H
        self._age_frames = 0.0
        self.fix_count += 1

    def set_anchor(self, H: np.ndarray, frame_size: tuple[int, int]) -> None:
        """Seed with a trusted static image->pitch homography and enable
        snap-back re-anchoring (see class docstring)."""
        self._anchor = np.asarray(H, dtype=np.float64).copy()
        self._frame_size = (int(frame_size[0]), int(frame_size[1]))
        self._cum = np.eye(3, dtype=np.float64)
        self.set_absolute(self._anchor.copy())

    def propagate(self, h_prev_to_cur: np.ndarray | None) -> None:
        """Advance one frame: chain the inter-frame homography onto the fix."""
        if self._H is None or self._age_frames is None:
            return
        if h_prev_to_cur is None:
            self._age_frames += 1.0 + self.FLOW_FAIL_PENALTY_S * self.fps
            return
        try:
            self._H = self._H @ np.linalg.inv(h_prev_to_cur)
        except np.linalg.LinAlgError:
            self._age_frames += 1.0 + self.FLOW_FAIL_PENALTY_S * self.fps
            return
        self._age_frames += 1.0
        if self._anchor is not None and self._cum is not None:
            # _cum tracks the camera pose relative to the ANCHOR frame and is
            # never reset on snap-back: a slow continuous pan must accumulate
            # rather than being silently re-anchored a few pixels at a time.
            self._cum = h_prev_to_cur @ self._cum
            if self._near_anchor_pose():
                self._H = self._anchor.copy()
                self._age_frames = 0.0

    def _near_anchor_pose(self) -> bool:
        if self._frame_size is None or self._cum is None:
            return False
        w, h = self._frame_size
        corners = np.array(
            [[[0.0, 0.0]], [[float(w), 0.0]], [[0.0, float(h)]], [[float(w), float(h)]]],
            dtype=np.float32,
        )
        try:
            mapped = cv2.perspectiveTransform(corners, self._cum)
        except cv2.error:
            return False
        if mapped is None or not np.isfinite(mapped).all():
            return False
        max_disp = float(np.max(np.linalg.norm(mapped - corners, axis=2)))
        return max_disp <= self.SNAPBACK_CORNER_FRAC * float(np.hypot(w, h))

    def current(self) -> tuple[np.ndarray | None, float | None]:
        """(H, age_seconds) — H is None when there has never been a fix or
        the fix has gone stale beyond MAX_AGE_S."""
        if self._H is None or self._age_frames is None:
            return None, None
        age_s = self._age_frames / self.fps
        if age_s > self.MAX_AGE_S:
            return None, age_s
        return self._H, age_s


def pitch_point(H: np.ndarray, point_xy: tuple[float, float]) -> tuple[float, float]:
    src = np.array([[[point_xy[0], point_xy[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H)[0][0]
    return float(dst[0]), float(dst[1])
