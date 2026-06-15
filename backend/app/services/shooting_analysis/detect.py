"""RF-DETR detection of player, ball, and goal via Roboflow.

Two interchangeable backends:
  * "hosted": Roboflow serverless inference (inference-sdk + your API key).
              Uses the `rfdetr-base` COCO checkpoint for person / sports ball,
              plus an optional Roboflow Universe model for the goal.
  * "local":  the `rfdetr` pip package running RF-DETR locally.

All results are normalised to: {"label": str, "conf": float, "xyxy": (x1,y1,x2,y2)}

Improvements over the original:
  * Hosted calls retry with backoff on transient network errors.
  * Goal detection aggregates several frames and returns the median box —
    a single-frame goal box is noisy and skews the px->m scale.
  * Ball picking rejects implausible jumps and oversized "balls".
  * `detect_with_model` is the public way to query an arbitrary model
    (the old private `_detect_hosted` is kept as an alias).
"""
from __future__ import annotations

import os
import time
from typing import List, Dict, Optional

import numpy as np

from .config import DetectorConfig

PERSON, BALL, GOAL = "person", "sports ball", "goal"


class Detector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.api_key = cfg.api_key or os.environ.get("ROBOFLOW_API_KEY", "")
        self._client = None
        self._local_model = None
        self._coco_classes = None

        if cfg.backend == "hosted":
            if not self.api_key:
                raise RuntimeError(
                    "Roboflow API key missing. Pass --api-key or set ROBOFLOW_API_KEY."
                )
            from inference_sdk import InferenceHTTPClient
            self._client = InferenceHTTPClient(api_url=cfg.api_url, api_key=self.api_key)
        elif cfg.backend == "local":
            from rfdetr import RFDETRBase
            from rfdetr.util.coco_classes import COCO_CLASSES
            self._local_model = RFDETRBase()
            self._coco_classes = COCO_CLASSES
        else:
            raise ValueError(f"Unknown backend: {cfg.backend}")

    # ------------------------------------------------------------------ core
    def detect(self, frame_bgr: np.ndarray) -> List[Dict]:
        """Detect person + sports ball on one frame."""
        if self.cfg.backend == "hosted":
            return self.detect_with_model(frame_bgr, self.cfg.core_model_id)
        return self._detect_local(frame_bgr)

    def detect_goal(self, frame_bgr: np.ndarray) -> Optional[Dict]:
        """Detect the goal in one frame using the configured Universe model."""
        if not self.cfg.goal_model_id or self.cfg.backend != "hosted":
            return None
        dets = self.detect_with_model(frame_bgr, self.cfg.goal_model_id)
        goals = [d for d in dets
                 if d["label"].lower() in self.cfg.goal_class_names
                 and d["conf"] > self.cfg.conf_goal]
        if not goals:
            return None
        best = max(goals, key=lambda d: d["conf"])
        best["label"] = GOAL
        return best

    def detect_goal_robust(self, frames: List[np.ndarray]) -> Optional[Dict]:
        """Median goal box over several sampled frames — much more stable than a
        single-frame detection, which directly improves the px->m scale."""
        boxes = []
        for frame in frames:
            box = self.detect_goal(frame)
            if box is not None:
                boxes.append(box)
        if not boxes:
            return None
        xyxy = np.median(np.array([b["xyxy"] for b in boxes], dtype=float), axis=0)
        conf = float(np.median([b["conf"] for b in boxes]))
        return {"label": GOAL, "conf": conf, "xyxy": tuple(float(v) for v in xyxy),
                "samples": len(boxes)}

    # --------------------------------------------------------------- hosted
    def detect_with_model(self, frame_bgr: np.ndarray, model_id: str) -> List[Dict]:
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                result = self._client.infer(rgb, model_id=model_id)
                break
            except Exception as exc:                       # transient network/API error
                last_exc = exc
                if attempt >= self.cfg.max_retries:
                    return []
                time.sleep(self.cfg.retry_backoff_s * (attempt + 1))
        else:                                              # pragma: no cover
            return []
        if isinstance(result, list):
            result = result[0]
        out = []
        for p in result.get("predictions", []):
            x, y, w, h = p["x"], p["y"], p["width"], p["height"]
            out.append({
                "label": p.get("class", "").lower(),
                "conf": float(p.get("confidence", 0.0)),
                "xyxy": (x - w / 2, y - h / 2, x + w / 2, y + h / 2),
            })
        return out

    # Backwards-compatible alias (older callers used the private name).
    _detect_hosted = detect_with_model

    # ---------------------------------------------------------------- local
    def _detect_local(self, frame_bgr: np.ndarray) -> List[Dict]:
        import cv2
        from PIL import Image
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        det = self._local_model.predict(Image.fromarray(rgb), threshold=0.25)
        out = []
        for xyxy, cls_id, conf in zip(det.xyxy, det.class_id, det.confidence):
            out.append({
                "label": str(self._coco_classes[int(cls_id)]).lower(),
                "conf": float(conf),
                "xyxy": tuple(float(v) for v in xyxy),
            })
        return out


# ------------------------------------------------------------------ helpers
def pick_player(dets: List[Dict], cfg: DetectorConfig,
                prev_center: Optional[tuple] = None) -> Optional[Dict]:
    """Largest confident person box, biased toward the previously tracked one."""
    people = [d for d in dets if d["label"] == PERSON and d["conf"] >= cfg.conf_player]
    if not people:
        return None
    if prev_center is not None:
        def cost(d):
            x1, y1, x2, y2 = d["xyxy"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            dist = np.hypot(cx - prev_center[0], cy - prev_center[1])
            area = (x2 - x1) * (y2 - y1)
            return dist - 0.0005 * area
        return min(people, key=cost)
    return max(people, key=lambda d: (d["xyxy"][2] - d["xyxy"][0]) * (d["xyxy"][3] - d["xyxy"][1]))


def pick_ball(dets: List[Dict], cfg: DetectorConfig,
              prev_center: Optional[tuple] = None,
              frame_diag: Optional[float] = None) -> Optional[Dict]:
    """Best ball candidate. Rejects boxes that are implausibly large for a ball
    and (when a previous position is known) candidates that teleport across the
    frame in a single step."""
    balls = [d for d in dets if d["label"] in (BALL, "ball") and d["conf"] >= cfg.conf_ball]
    if frame_diag:
        max_side = 0.18 * frame_diag           # a ball is never ~1/5 of the frame diagonal
        balls = [d for d in balls
                 if max(d["xyxy"][2] - d["xyxy"][0], d["xyxy"][3] - d["xyxy"][1]) <= max_side]
    if not balls:
        return None
    if prev_center is not None:
        def dist(d):
            x1, y1, x2, y2 = d["xyxy"]
            return np.hypot((x1 + x2) / 2 - prev_center[0], (y1 + y2) / 2 - prev_center[1])
        if frame_diag:
            max_jump = cfg.max_ball_jump_frac * frame_diag
            plausible = [d for d in balls if dist(d) <= max_jump]
            if plausible:
                return min(plausible, key=dist)
            # All candidates jumped — trust the most confident one only if it's strong.
            best = max(balls, key=lambda d: d["conf"])
            return best if best["conf"] >= 0.6 else None
        return min(balls, key=dist)
    return max(balls, key=lambda d: d["conf"])


def center(xyxy) -> tuple:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
