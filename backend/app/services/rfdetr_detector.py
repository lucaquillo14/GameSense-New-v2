from __future__ import annotations

from threading import Lock

import cv2
import numpy as np

from app.services.roboflow_inference import get_model, require_roboflow_api_key

_rfdetr_model = None
_model_lock = Lock()

RFDETR_MODEL_ID = "rfdetr-base"
INFERENCE_MAX_DIM = 1280

# COCO class IDs used by rfdetr-base.
COCO_PERSON = 0
COCO_SPORTS_BALL = 32

GOAL_CLASS_NAMES = frozenset({"goal", "goalpost", "goal_post", "soccer goal"})


def get_rfdetr():
    global _rfdetr_model
    if _rfdetr_model is None:
        with _model_lock:
            if _rfdetr_model is None:
                _rfdetr_model = get_model(model_id=RFDETR_MODEL_ID, api_key=require_roboflow_api_key())
                print("[GameSense] RF-DETR model loaded")
    return _rfdetr_model


def _resize_for_inference(frame: np.ndarray, max_dim: int = INFERENCE_MAX_DIM) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return frame, 1.0
    scale = max_dim / float(longest)
    resized = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def _scale_bbox_xywh(
    bbox: tuple[float, float, float, float],
    scale: float,
) -> tuple[float, float, float, float]:
    if scale == 1.0:
        return bbox
    inv = 1.0 / scale
    x, y, w, h = bbox
    return x * inv, y * inv, w * inv, h * inv


def _scale_bbox_xyxy(
    bbox: tuple[float, float, float, float],
    scale: float,
) -> tuple[float, float, float, float]:
    if scale == 1.0:
        return bbox
    inv = 1.0 / scale
    x1, y1, x2, y2 = bbox
    return x1 * inv, y1 * inv, x2 * inv, y2 * inv


def _parse_inference_predictions(result: dict, scale: float) -> dict:
    predictions = result.get("predictions") or []
    players: list[tuple[float, float, float, float, float]] = []
    ball: tuple[float, float, float, float, float] | None = None
    goal: tuple[float, float, float, float, float] | None = None

    for prediction in predictions:
        confidence = float(prediction.get("confidence") or 0.0)
        class_name = str(prediction.get("class") or prediction.get("class_name") or "").lower()
        class_id = prediction.get("class_id")
        if class_id is not None:
            class_id = int(class_id)

        if "x" in prediction and "y" in prediction:
            x = float(prediction["x"])
            y = float(prediction["y"])
            w = float(prediction.get("width") or prediction.get("w") or 0.0)
            h = float(prediction.get("height") or prediction.get("h") or 0.0)
            bbox = _scale_bbox_xywh((x - w / 2.0, y - h / 2.0, w, h), scale)
        else:
            x1 = float(prediction.get("x_min") or prediction.get("x1") or 0.0)
            y1 = float(prediction.get("y_min") or prediction.get("y1") or 0.0)
            x2 = float(prediction.get("x_max") or prediction.get("x2") or 0.0)
            y2 = float(prediction.get("y_max") or prediction.get("y2") or 0.0)
            scaled = _scale_bbox_xyxy((x1, y1, x2, y2), scale)
            bbox = (scaled[0], scaled[1], scaled[2] - scaled[0], scaled[3] - scaled[1])

        if class_name in GOAL_CLASS_NAMES or "goal" in class_name:
            goal = (*bbox, confidence)
            continue
        if class_id == COCO_PERSON or class_name in {"person", "player", "goalkeeper"}:
            players.append((*bbox, confidence))
            continue
        if class_id == COCO_SPORTS_BALL or class_name in {"ball", "sports ball", "soccer ball", "football"}:
            if ball is None or confidence > ball[4]:
                ball = (*bbox, confidence)

    return {"players": players, "ball": ball, "goal": goal}


def _detect_goal_heuristic(frame: np.ndarray) -> tuple[float, float, float, float, float] | None:
    height, width = frame.shape[:2]
    roi = frame[0 : int(height * 0.55), :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    merged = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, float, float, float, float] | None = None
    best_score = 0.0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.25 or h < height * 0.04:
            continue
        aspect = w / max(h, 1.0)
        if aspect < 2.5:
            continue
        area = w * h
        score = area * aspect
        if score > best_score:
            best_score = score
            best = (float(x), float(y), float(x + w), float(y + h), 0.35)
    return best


def detect_objects(frame: np.ndarray) -> dict:
    """Detect players, ball, and goal in a single frame."""
    resized, scale = _resize_for_inference(frame)
    result = get_rfdetr().infer(resized, confidence=0.25)[0]
    parsed = _parse_inference_predictions(result, scale)
    if parsed["goal"] is None:
        parsed["goal"] = _detect_goal_heuristic(frame)
    if parsed["ball"] is None:
        parsed["ball"] = _detect_ball_heuristic(frame)
    return parsed


def _detect_ball_heuristic(frame: np.ndarray) -> tuple[float, float, float, float, float] | None:
    height, width = frame.shape[:2]
    lower = frame[int(height * 0.35) :, :]
    hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, float, float, float, float] | None = None
    best_score = 0.0
    y_offset = int(height * 0.35)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        diameter = max(w, h)
        if diameter < 6 or diameter > height / 12:
            continue
        aspect = w / max(h, 1.0)
        if aspect < 0.6 or aspect > 1.6:
            continue
        score = diameter
        if score > best_score:
            best_score = score
            best = (float(x), float(y + y_offset), float(w), float(h), 0.2)
    return best
