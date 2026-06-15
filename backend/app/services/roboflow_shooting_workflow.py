"""Roboflow Workflow client: Soccer Shooting Technique Analyzer."""

from __future__ import annotations

import base64
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.services.roboflow_inference import (
    RoboflowConfigError,
    inference_available,
    require_roboflow_api_key,
)

try:
    from inference_sdk import InferenceHTTPClient
    from inference_sdk.http.errors import HTTPCallErrorError
except ImportError:  # pragma: no cover
    InferenceHTTPClient = None
    HTTPCallErrorError = Exception

ROBOFLOW_SERVERLESS_URL = "https://serverless.roboflow.com"
WORKSPACE_NAME = "lucass-workspace-fn5cc"
WORKFLOW_ID = "soccer-shooting-technique-analyzer-1781109926364"
WORKFLOW_INPUT_CANDIDATES = ("input_frame", "image")

# Declared in workflows_get (published + draft share the same output names).
WORKFLOW_OUTPUT_KEYS = frozenset({
    "output_image",
    "technique_score",
    "feedback",
    "metrics",
    "shot_power_kmh",
    "phase",
    "predictions",
    "pose_predictions",
})

DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF_S = 1.5

_serverless_client: InferenceHTTPClient | None = None


class ShootingTechniqueWorkflowError(RoboflowConfigError):
    """Base error for shooting-technique workflow integration."""


class ShootingTechniqueWorkflowExecutionError(ShootingTechniqueWorkflowError):
    """Remote workflow execution failed after retries."""


@dataclass(frozen=True)
class DetectionSummary:
    class_names: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    boxes_xyxy: list[tuple[float, float, float, float]] = field(default_factory=list)


@dataclass(frozen=True)
class ShootingTechniqueFrameResult:
    """Parsed, lightweight view of one workflow response entry."""

    technique_score: float
    shot_power_kmh: float
    phase: str
    feedback: list[str]
    metrics: dict[str, Any]
    output_image_path: Path | None = None
    detections: DetectionSummary | None = None
    raw_output_keys: tuple[str, ...] = field(default_factory=tuple)


def _get_serverless_client(api_key: str) -> InferenceHTTPClient:
    global _serverless_client
    if InferenceHTTPClient is None:
        raise ShootingTechniqueWorkflowError(
            "inference-sdk is not installed. Run: py -3.12 -m pip install inference-sdk"
        )
    if _serverless_client is None:
        _serverless_client = InferenceHTTPClient(
            api_url=ROBOFLOW_SERVERLESS_URL,
            api_key=api_key,
        )
    return _serverless_client


def _workflow_input_name() -> tuple[str, ...]:
    override = os.environ.get("ROBOFLOW_SHOOTING_WORKFLOW_INPUT", "").strip()
    if override:
        return (override, *tuple(name for name in WORKFLOW_INPUT_CANDIDATES if name != override))
    return WORKFLOW_INPUT_CANDIDATES


def _should_retry_input_name(error: Exception, attempted: str) -> bool:
    message = str(error).lower()
    if attempted == "image":
        return False
    if "workflowimage" in message and "image" in message and "not provided" in message:
        return True
    if "runtime parameter `image`" in message:
        return True
    return False


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _coerce_feedback(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _slim_metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keep = (
        "knee_angle_deg",
        "ankle_lock_deg",
        "plant_foot_distance_player_heights",
        "approach_angle_deg",
        "hip_rotation_deg",
        "trunk_lean_deg",
        "follow_through_height_ratio",
        "goal_scale_note",
    )
    return {key: value[key] for key in keep if key in value}


def _slim_detection_summary(value: Any) -> DetectionSummary | None:
    if value is None:
        return None
    if isinstance(value, dict) and "predictions" in value and "class_name" not in value:
        nested = _slim_detection_summary(value.get("predictions"))
        if nested is not None:
            return nested
    if isinstance(value, dict):
        class_names = value.get("class_name") or value.get("class") or value.get("classes")
        confidences = value.get("confidence") or value.get("confidences")
        xyxy = value.get("xyxy")
        if isinstance(class_names, np.ndarray):
            class_names = class_names.tolist()
        if isinstance(confidences, np.ndarray):
            confidences = confidences.tolist()
        if isinstance(xyxy, np.ndarray):
            xyxy = xyxy.tolist()
        names = [str(name) for name in (class_names or [])]
        confs = [_coerce_float(item) for item in (confidences or [])]
        boxes: list[tuple[float, float, float, float]] = []
        if isinstance(xyxy, list):
            for row in xyxy:
                if isinstance(row, (list, tuple)) and len(row) >= 4:
                    boxes.append(
                        (
                            float(row[0]),
                            float(row[1]),
                            float(row[2]),
                            float(row[3]),
                        )
                    )
        if not names and not boxes:
            predictions = value.get("predictions")
            if isinstance(predictions, list):
                for pred in predictions:
                    if not isinstance(pred, dict):
                        continue
                    label = pred.get("class") or pred.get("class_name") or pred.get("label")
                    if label is not None:
                        names.append(str(label))
                    if "confidence" in pred:
                        confs.append(_coerce_float(pred.get("confidence")))
                    for key in ("x", "y", "width", "height"):
                        if key not in pred:
                            break
                    else:
                        x = _coerce_float(pred.get("x"))
                        y = _coerce_float(pred.get("y"))
                        w = _coerce_float(pred.get("width"))
                        h = _coerce_float(pred.get("height"))
                        boxes.append((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
        if not names and not boxes:
            return None
        return DetectionSummary(class_names=names, confidences=confs, boxes_xyxy=boxes)
    if isinstance(value, list):
        names: list[str] = []
        confs: list[float] = []
        boxes: list[tuple[float, float, float, float]] = []
        for pred in value:
            if not isinstance(pred, dict):
                continue
            label = pred.get("class") or pred.get("class_name") or pred.get("label")
            if label is not None:
                names.append(str(label))
            if "confidence" in pred:
                confs.append(_coerce_float(pred.get("confidence")))
        if not names and not boxes:
            return None
        return DetectionSummary(class_names=names, confidences=confs, boxes_xyxy=boxes)
    return None


def write_output_image(
    output_image: Any,
    destination: Path,
) -> Path:
    """Decode a workflow image output and write it to disk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(output_image, dict):
        if output_image.get("type") == "base64":
            payload = output_image.get("value")
            if not isinstance(payload, str):
                raise ShootingTechniqueWorkflowError("Workflow image output missing base64 value.")
            destination.write_bytes(base64.b64decode(payload))
            return destination
        nested = output_image.get("value")
        if isinstance(nested, str) and len(nested) > 100:
            destination.write_bytes(base64.b64decode(nested))
            return destination
    if isinstance(output_image, str):
        if output_image.startswith("data:image"):
            _, _, payload = output_image.partition(",")
            destination.write_bytes(base64.b64decode(payload))
            return destination
        if len(output_image) > 100:
            destination.write_bytes(base64.b64decode(output_image))
            return destination
    raise ShootingTechniqueWorkflowError("Unsupported workflow image output format.")


def parse_shooting_technique_output(
    raw: dict[str, Any],
    *,
    output_image_dir: Path | None = None,
    output_image_name: str = "workflow_output.jpg",
) -> ShootingTechniqueFrameResult:
    """Parse one workflow result entry using whatever keys the workflow returned."""
    if not isinstance(raw, dict):
        raise ShootingTechniqueWorkflowError("Workflow response entry must be a dict.")

    output_image_path: Path | None = None
    if "output_image" in raw and output_image_dir is not None:
        output_image_path = write_output_image(
            raw["output_image"],
            output_image_dir / output_image_name,
        )

    metrics = _slim_metrics(raw.get("metrics"))
    detections = _slim_detection_summary(raw.get("predictions"))

    return ShootingTechniqueFrameResult(
        technique_score=_coerce_float(raw.get("technique_score")),
        shot_power_kmh=_coerce_float(raw.get("shot_power_kmh")),
        phase=_coerce_str(raw.get("phase"), default="approach"),
        feedback=_coerce_feedback(raw.get("feedback")),
        metrics=metrics,
        output_image_path=output_image_path,
        detections=detections,
        raw_output_keys=tuple(raw.keys()),
    )


def run_soccer_shooting_technique_analyzer(
    image: str | Path | np.ndarray,
    *,
    api_key: str | None = None,
    parameters: dict[str, Any] | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    output_image_dir: Path | None = None,
    output_image_name: str = "workflow_output.jpg",
    ball_state: Any | None = None,
    frame_bgr: np.ndarray | None = None,
) -> ShootingTechniqueFrameResult:
    """Run the saved Roboflow workflow on one image with retries and typed errors."""
    if not inference_available():
        raise ShootingTechniqueWorkflowError(
            "inference-sdk is not installed. Run: py -3.12 -m pip install inference-sdk"
        )

    key = api_key or require_roboflow_api_key()
    client = _get_serverless_client(key)
    image_input = _prepare_image_input(image)
    if frame_bgr is None and isinstance(image, np.ndarray):
        frame_bgr = image
    if frame_bgr is None and isinstance(image, (str, Path)):
        path = Path(image)
        if path.exists():
            loaded = cv2.imread(str(path))
            if loaded is not None:
                frame_bgr = loaded

    last_error: Exception | None = None
    raw_entry: dict[str, Any] | None = None
    for attempt in range(max_retries + 1):
        for input_name in _workflow_input_name():
            try:
                raw_results = client.run_workflow(
                    workspace_name=WORKSPACE_NAME,
                    workflow_id=WORKFLOW_ID,
                    images={input_name: image_input},
                    parameters=parameters or {},
                )
                if not isinstance(raw_results, list) or not raw_results:
                    raise ShootingTechniqueWorkflowExecutionError(
                        "Workflow returned an empty result list."
                    )
                entry = raw_results[0]
                if not isinstance(entry, dict):
                    raise ShootingTechniqueWorkflowExecutionError(
                        "Workflow result entry is not a dict."
                    )
                raw_entry = entry
                break
            except HTTPCallErrorError as exc:
                last_error = exc
                if _should_retry_input_name(exc, input_name):
                    continue
                break
            except Exception as exc:
                last_error = exc
                break
        if raw_entry is not None:
            break
        if attempt < max_retries:
            time.sleep(DEFAULT_RETRY_BACKOFF_S * (attempt + 1))

    if raw_entry is None:
        message = str(last_error) if last_error else "Unknown workflow failure"
        if frame_bgr is None:
            raise ShootingTechniqueWorkflowExecutionError(
                f"Soccer Shooting Technique Analyzer workflow failed after "
                f"{max_retries + 1} attempt(s): {message}"
            ) from last_error
        raw_entry = {}

    if frame_bgr is not None:
        from app.services.shooting_technique_metrics import (
            BallTrackerState,
            analyze_frame_with_local_detections,
        )

        state = ball_state if ball_state is not None else BallTrackerState()
        computed, detections, output_image_path = analyze_frame_with_local_detections(
            frame_bgr,
            raw_entry,
            state,
            output_image_dir=output_image_dir,
            output_image_name=output_image_name,
        )
        return ShootingTechniqueFrameResult(
            technique_score=_coerce_float(computed.get("technique_score")),
            shot_power_kmh=_coerce_float(computed.get("shot_power_kmh")),
            phase=_coerce_str(computed.get("phase"), default="approach"),
            feedback=_coerce_feedback(computed.get("feedback")),
            metrics=_slim_metrics(computed.get("metrics")),
            output_image_path=output_image_path,
            detections=detections,
            raw_output_keys=tuple(raw_entry.keys()),
        )

    return parse_shooting_technique_output(
        raw_entry,
        output_image_dir=output_image_dir,
        output_image_name=output_image_name,
    )


def _prepare_image_input(image: str | Path | np.ndarray) -> str | np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    path = Path(image)
    if path.exists():
        return str(path)
    return str(image)


def _cleanup_temp_image(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def run_soccer_shooting_technique_analyzer_from_bgr_frame(
    frame_bgr: np.ndarray,
    **kwargs: Any,
) -> ShootingTechniqueFrameResult:
    """Encode an OpenCV BGR frame to a temp JPEG and run the workflow."""
    temp_dir = Path(tempfile.mkdtemp(prefix="gamesense-workflow-"))
    temp_path = temp_dir / "frame.jpg"
    try:
        if not cv2.imwrite(str(temp_path), frame_bgr):
            raise ShootingTechniqueWorkflowError("Could not encode frame for workflow input.")
        return run_soccer_shooting_technique_analyzer(
            temp_path,
            frame_bgr=frame_bgr,
            output_image_dir=kwargs.pop("output_image_dir", None),
            **kwargs,
        )
    finally:
        _cleanup_temp_image(temp_path)
        try:
            temp_dir.rmdir()
        except OSError:
            pass


def _is_contact_phase(phase: str) -> bool:
    return "contact" in phase.lower()


def _detection_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _ball_center_from_detections(detections: DetectionSummary | None) -> tuple[float, float] | None:
    if not detections:
        return None
    for name, box in zip(detections.class_names, detections.boxes_xyxy):
        if "ball" in name.lower():
            return _detection_center(box)
    return None


def _player_height_px_from_detections(detections: DetectionSummary | None) -> float | None:
    if not detections:
        return None
    best: float | None = None
    for name, box in zip(detections.class_names, detections.boxes_xyxy):
        if name.lower() == "person":
            height = max(1.0, float(box[3] - box[1]))
            best = height if best is None else max(best, height)
    return best


def _estimate_shot_power_kmh(
    frame_results: list[tuple[int, float, ShootingTechniqueFrameResult]],
) -> float:
    """Estimate shot speed from ball displacement across sampled frames."""
    import math

    ball_track: list[tuple[float, float, float]] = []
    player_heights: list[float] = []
    for _frame_id, time_s, result in frame_results:
        center = _ball_center_from_detections(result.detections)
        if center is not None:
            ball_track.append((time_s, center[0], center[1]))
        height = _player_height_px_from_detections(result.detections)
        if height is not None:
            player_heights.append(height)

    if len(ball_track) < 2 or not player_heights:
        return 0.0

    max_speed_px = 0.0
    for index in range(1, len(ball_track)):
        t1, x1, y1 = ball_track[index - 1]
        t2, x2, y2 = ball_track[index]
        dt = max(t2 - t1, 1.0 / 30.0)
        speed = math.hypot(x2 - x1, y2 - y1) / dt
        max_speed_px = max(max_speed_px, speed)

    if max_speed_px <= 0:
        return 0.0

    player_height_px = max(player_heights)
    meters_per_px = 1.75 / player_height_px
    return round(max_speed_px * meters_per_px * 3.6, 1)


def build_shooting_feedback_from_workflow_frames(
    frame_results: list[tuple[int, float, ShootingTechniqueFrameResult]],
    *,
    annotated_image_url: str | None = None,
) -> "ShootingFeedback":
    """Aggregate per-frame workflow results into the API ShootingFeedback model."""
    from app.models import BodyAngle, ShootingFeedback, TechniqueFrame

    if not frame_results:
        return ShootingFeedback(
            feedback_points=[
                "No frames were analyzed. Upload a clip with the kicker and ball clearly visible."
            ],
            confidence=0.0,
        )

    contact_entry = max(
        frame_results,
        key=lambda item: (
            1 if _is_contact_phase(item[2].phase) else 0,
            len(item[2].metrics),
            item[2].technique_score,
        ),
    )
    contact_frame_id, _, contact_result = contact_entry

    workflow_shot_power = max(item[2].shot_power_kmh for item in frame_results)
    tracked_shot_power = _estimate_shot_power_kmh(frame_results)
    shot_power_kmh = max(workflow_shot_power, tracked_shot_power)
    metrics = contact_result.metrics
    plant_heights = metrics.get("plant_foot_distance_player_heights")
    plant_foot_distance_cm = 0.0
    if isinstance(plant_heights, (int, float)):
        plant_foot_distance_cm = round(float(plant_heights) * 175.0, 1)

    follow_ratio = metrics.get("follow_through_height_ratio")
    if isinstance(follow_ratio, (int, float)):
        if follow_ratio < 0.05:
            follow_through = "low"
        elif follow_ratio > 0.15:
            follow_through = "high"
        else:
            follow_through = "medium"
    else:
        follow_through = "medium"

    frame_analysis: list[TechniqueFrame] = []
    for frame_id, time_s, result in frame_results:
        angles: list[BodyAngle] = []
        mapping = (
            ("knee_angle", "knee_angle_deg"),
            ("trunk_lean", "trunk_lean_deg"),
            ("hip_rotation", "hip_rotation_deg"),
            ("approach_angle", "approach_angle_deg"),
            ("ankle_lock", "ankle_lock_deg"),
        )
        for angle_name, metric_key in mapping:
            value = result.metrics.get(metric_key)
            if isinstance(value, (int, float)):
                angles.append(
                    BodyAngle(
                        name=angle_name,
                        value_deg=float(value),
                        frame_id=frame_id,
                        time_s=time_s,
                    )
                )
        frame_analysis.append(
            TechniqueFrame(
                frame_id=frame_id,
                time_s=time_s,
                angles=angles,
                ball_visible=bool(result.detections and "ball" in " ".join(result.detections.class_names).lower()),
                phase=result.phase,
            )
        )

    confidences = [
        max(result.detections.confidences)
        for _, _, result in frame_results
        if result.detections and result.detections.confidences
    ]
    confidence = float(sum(confidences) / len(confidences)) if confidences else 0.5

    return ShootingFeedback(
        shot_power_kmh=round(float(shot_power_kmh), 1),
        technique_score=round(float(contact_result.technique_score), 1),
        approach_angle_deg=_coerce_float(metrics.get("approach_angle_deg")),
        plant_foot_distance_cm=plant_foot_distance_cm,
        knee_bend_at_contact_deg=_coerce_float(metrics.get("knee_angle_deg")),
        hip_rotation_deg=_coerce_float(metrics.get("hip_rotation_deg")),
        follow_through_height=follow_through,
        feedback_points=list(contact_result.feedback),
        annotated_video_url=annotated_image_url,
        frame_analysis=frame_analysis,
        confidence=round(confidence, 3),
        contact_frame_id=contact_frame_id,
    )

