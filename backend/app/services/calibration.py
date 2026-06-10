from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

FIELD_LENGTH_M = 105.0
FIELD_WIDTH_M = 68.0
PENALTY_BOX_WIDTH_M = 40.32
PENALTY_BOX_DEPTH_M = 16.5
CENTRE_CIRCLE_RADIUS_M = 9.15
PLAYER_HEIGHT_MIN_M = 1.4
PLAYER_HEIGHT_MAX_M = 2.2
PLAYER_HEIGHT_EXPECTED_MIN_M = 1.65
PLAYER_HEIGHT_EXPECTED_MAX_M = 1.95
AXIS_ANGLE_TOLERANCE_DEG = 15.0
PENALTY_BOX_RATIO = PENALTY_BOX_WIDTH_M / PENALTY_BOX_DEPTH_M
PENALTY_BOX_RATIO_TOLERANCE = 0.35


@dataclass
class CalibrationResult:
    matrix: np.ndarray | None
    scale_known: bool
    frame_width: int = 0
    frame_height: int = 0
    units: Literal["metric", "pixels"] = "pixels"
    region_polygon_px: list[tuple[float, float]] | None = None
    visible_field_bounds_m: tuple[float, float, float, float] | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    median_player_height_m: float | None = None
    auto_detected: bool = False
    detection_method: str | None = None

    def to_dict(self) -> dict:
        return {
            "scale_known": self.scale_known,
            "units": self.units,
            "region_polygon_px": self.region_polygon_px,
            "visible_field_bounds_m": self.visible_field_bounds_m,
            "warnings": self.warnings,
            "errors": self.errors,
            "median_player_height_m": self.median_player_height_m,
            "auto_detected": self.auto_detected,
            "detection_method": self.detection_method,
        }


def _order_quad(points: np.ndarray) -> np.ndarray:
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]
    top_right = points[np.argmin(diffs)]
    bottom_left = points[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _field_destination() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [FIELD_LENGTH_M, 0.0],
            [FIELD_LENGTH_M, FIELD_WIDTH_M],
            [0.0, FIELD_WIDTH_M],
        ],
        dtype=np.float32,
    )


def _penalty_box_destination() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [PENALTY_BOX_WIDTH_M, 0.0],
            [PENALTY_BOX_WIDTH_M, PENALTY_BOX_DEPTH_M],
            [0.0, PENALTY_BOX_DEPTH_M],
        ],
        dtype=np.float32,
    )


def pixel_to_field(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    src = np.array([[[point[0], point[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, matrix)[0][0]
    return float(dst[0]), float(dst[1])


def visible_field_bounds_from_matrix(
    matrix: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float, float]:
    corners = [
        (0.0, 0.0),
        (float(frame_width), 0.0),
        (float(frame_width), float(frame_height)),
        (0.0, float(frame_height)),
    ]
    field_points = [pixel_to_field(matrix, corner) for corner in corners]
    xs = [point[0] for point in field_points]
    ys = [point[1] for point in field_points]
    return min(xs), max(xs), min(ys), max(ys)


def is_pixel_in_calibrated_region(
    x_px: float,
    y_px: float,
    region_polygon_px: list[tuple[float, float]] | None,
    frame_width: int,
    frame_height: int,
) -> bool:
    if not region_polygon_px or len(region_polygon_px) < 3:
        return 0.0 <= x_px <= frame_width and 0.0 <= y_px <= frame_height
    contour = np.array(region_polygon_px, dtype=np.float32)
    return cv2.pointPolygonTest(contour, (float(x_px), float(y_px)), False) >= 0


def player_height_metres(matrix: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
    x, y, w, h = bbox
    top = pixel_to_field(matrix, (x + w / 2.0, y))
    bottom = pixel_to_field(matrix, (x + w / 2.0, y + h))
    return float(np.hypot(bottom[0] - top[0], bottom[1] - top[1]))


def validate_calibration_scale(
    matrix: np.ndarray,
    player_bboxes: list[tuple[float, float, float, float]],
) -> tuple[bool, float | None, str | None]:
    if not player_bboxes:
        return True, None, None

    heights = [player_height_metres(matrix, bbox) for bbox in player_bboxes]
    median_height = float(np.median(heights))
    if PLAYER_HEIGHT_MIN_M <= median_height <= PLAYER_HEIGHT_MAX_M:
        return True, median_height, None

    message = (
        f"Pitch calibration scale looks wrong: median player height is {median_height:.2f}m "
        f"(expected {PLAYER_HEIGHT_EXPECTED_MIN_M:.2f}–{PLAYER_HEIGHT_EXPECTED_MAX_M:.2f}m). "
        "Please re-mark the pitch polygon."
    )
    return False, median_height, message


def _line_angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    return abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))


def _cluster_line_positions(
    segments: list[tuple[float, float, float, float]],
    axis: Literal["horizontal", "vertical"],
    tolerance: float = 18.0,
) -> list[float]:
    if not segments:
        return []
    positions = []
    for x1, y1, x2, y2 in segments:
        positions.append((y1 + y2) / 2.0 if axis == "horizontal" else (x1 + x2) / 2.0)
    positions.sort()
    clusters: list[float] = []
    current = [positions[0]]
    for position in positions[1:]:
        if position - current[-1] <= tolerance:
            current.append(position)
        else:
            clusters.append(float(np.mean(current)))
            current = [position]
    clusters.append(float(np.mean(current)))
    return clusters


def _segments_from_hough(frame_gray: np.ndarray) -> tuple[list[tuple[float, float, float, float]], list[tuple[float, float, float, float]]]:
    edges = cv2.Canny(frame_gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=80, minLineLength=80, maxLineGap=20)
    horizontals: list[tuple[float, float, float, float]] = []
    verticals: list[tuple[float, float, float, float]] = []
    if lines is None:
        return horizontals, verticals

    for raw in lines:
        x1, y1, x2, y2 = [float(value) for value in raw[0]]
        angle = _line_angle_deg(x1, y1, x2, y2)
        if angle <= AXIS_ANGLE_TOLERANCE_DEG or angle >= 180.0 - AXIS_ANGLE_TOLERANCE_DEG:
            horizontals.append((x1, y1, x2, y2))
        elif 90.0 - AXIS_ANGLE_TOLERANCE_DEG <= angle <= 90.0 + AXIS_ANGLE_TOLERANCE_DEG:
            verticals.append((x1, y1, x2, y2))
    return horizontals, verticals


def _quad_from_penalty_box_clusters(
    top: float,
    bottom: float,
    left: float,
    right: float,
) -> np.ndarray:
    return np.array(
        [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ],
        dtype=np.float32,
    )


def detect_pitch_markings(frame_bgr: np.ndarray) -> tuple[np.ndarray | None, str | None, float]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    horizontals, verticals = _segments_from_hough(gray)
    h_positions = _cluster_line_positions(horizontals, "horizontal")
    v_positions = _cluster_line_positions(verticals, "vertical")

    best_quad: np.ndarray | None = None
    best_score = 0.0
    best_method = "penalty_box"

    for index_a, top in enumerate(h_positions):
        for bottom in h_positions[index_a + 1 :]:
            height_px = bottom - top
            if height_px < 30:
                continue
            for index_c, left in enumerate(v_positions):
                for right in v_positions[index_c + 1 :]:
                    width_px = right - left
                    if width_px < 50:
                        continue
                    ratio = width_px / height_px
                    if abs(ratio - PENALTY_BOX_RATIO) > PENALTY_BOX_RATIO_TOLERANCE:
                        continue
                    score = width_px * height_px
                    if score > best_score:
                        best_score = score
                        best_quad = _quad_from_penalty_box_clusters(top, bottom, left, right)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=100,
        param2=30,
        minRadius=25,
        maxRadius=220,
    )
    if circles is not None and len(circles[0]) > 0:
        circle = max(circles[0], key=lambda item: item[2])
        cx, cy, radius = float(circle[0]), float(circle[1]), float(circle[2])
        circle_score = radius * radius
        if circle_score > best_score * 0.6:
            quad = np.array(
                [
                    [cx - radius, cy - radius],
                    [cx + radius, cy - radius],
                    [cx + radius, cy + radius],
                    [cx - radius, cy + radius],
                ],
                dtype=np.float32,
            )
            return _order_quad(quad), "centre_circle", circle_score

    if best_quad is not None:
        return _order_quad(best_quad), best_method, best_score
    return None, None, 0.0


def _homography_from_polygon(
    polygon: list[dict],
    frame_width: int,
    frame_height: int,
) -> tuple[np.ndarray | None, list[tuple[float, float]], list[str]]:
    warnings: list[str] = []
    points = np.array([[p["x"], p["y"]] for p in polygon], dtype=np.float32)
    if len(points) < 4:
        return None, [], ["Pitch calibration insufficient - at least 4 points are required."]

    hull = cv2.convexHull(points).reshape(-1, 2)
    area = cv2.contourArea(hull.astype(np.float32))
    if area < 10_000:
        warnings.append("Pitch calibration insufficient - selected polygon area is very small.")

    quad = _order_quad(hull if len(hull) >= 4 else points)
    region_polygon_px = [(float(x), float(y)) for x, y in points.tolist()]
    matrix = cv2.getPerspectiveTransform(quad, _field_destination())
    if not np.isfinite(matrix).all():
        return None, region_polygon_px, ["Pitch calibration insufficient - homography could not be computed."]

    if len(polygon) < 6:
        warnings.append("Pitch calibration has only 4-5 points; 6-8 points are recommended for accuracy.")
    return matrix, region_polygon_px, warnings


def compute_calibration(
    polygon: list[dict] | None,
    frame_width: int,
    frame_height: int,
    calibration_frame: np.ndarray | None = None,
    player_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> CalibrationResult:
    warnings: list[str] = []
    errors: list[str] = []

    if polygon:
        matrix, region_polygon_px, polygon_warnings = _homography_from_polygon(polygon, frame_width, frame_height)
        warnings.extend(polygon_warnings)
        if matrix is None:
            return CalibrationResult(
                matrix=None,
                scale_known=False,
                units="pixels",
                warnings=warnings,
                errors=polygon_warnings,
            )

        valid, median_height, height_error = validate_calibration_scale(matrix, player_bboxes or [])
        if not valid and height_error:
            return CalibrationResult(
                matrix=None,
                scale_known=False,
                units="pixels",
                region_polygon_px=region_polygon_px,
                warnings=warnings,
                errors=[height_error],
                median_player_height_m=median_height,
            )

        visible_bounds = visible_field_bounds_from_matrix(matrix, frame_width, frame_height)
        return CalibrationResult(
            matrix=matrix,
            scale_known=True,
            frame_width=frame_width,
            frame_height=frame_height,
            units="metric",
            region_polygon_px=region_polygon_px,
            visible_field_bounds_m=visible_bounds,
            warnings=warnings,
            median_player_height_m=median_height,
            auto_detected=False,
            detection_method="manual_polygon",
        )

    if calibration_frame is not None:
        quad, method, score = detect_pitch_markings(calibration_frame)
        if quad is not None and method is not None and score > 0:
            destination = _penalty_box_destination() if method == "penalty_box" else np.array(
                [
                    [0.0, 0.0],
                    [2 * CENTRE_CIRCLE_RADIUS_M, 0.0],
                    [2 * CENTRE_CIRCLE_RADIUS_M, 2 * CENTRE_CIRCLE_RADIUS_M],
                    [0.0, 2 * CENTRE_CIRCLE_RADIUS_M],
                ],
                dtype=np.float32,
            )
            matrix = cv2.getPerspectiveTransform(quad, destination)
            if np.isfinite(matrix).all():
                region_polygon_px = [(float(x), float(y)) for x, y in quad.tolist()]
                valid, median_height, height_error = validate_calibration_scale(matrix, player_bboxes or [])
                if not valid and height_error:
                    return CalibrationResult(
                        matrix=None,
                        scale_known=False,
                        units="pixels",
                        region_polygon_px=region_polygon_px,
                        warnings=warnings,
                        errors=[height_error],
                        median_player_height_m=median_height,
                        auto_detected=True,
                        detection_method=method,
                    )
                visible_bounds = visible_field_bounds_from_matrix(matrix, frame_width, frame_height)
                warnings.append(
                    f"Pitch markings auto-detected ({method.replace('_', ' ')}). "
                    "Metrics use the visible calibrated region only."
                )
                return CalibrationResult(
                    matrix=matrix,
                    scale_known=True,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    units="metric",
                    region_polygon_px=region_polygon_px,
                    visible_field_bounds_m=visible_bounds,
                    warnings=warnings,
                    median_player_height_m=median_height,
                    auto_detected=True,
                    detection_method=method,
                )

    warnings.append(
        "No pitch calibration is available. Positions and speeds are reported in pixels until you mark the pitch polygon."
    )
    return CalibrationResult(
        matrix=None,
        scale_known=False,
        frame_width=frame_width,
        frame_height=frame_height,
        units="pixels",
        warnings=warnings,
        errors=errors,
    )


def build_track_point(
    calibration: CalibrationResult,
    frame_id: int,
    time_s: float,
    foot_px: tuple[float, float],
    confidence: float,
) -> "TrackPoint":
    from app.services.metrics import TrackPoint

    calibrated = False
    x_m: float | None = None
    y_m: float | None = None
    if calibration.scale_known and calibration.matrix is not None:
        in_region = is_pixel_in_calibrated_region(
            foot_px[0],
            foot_px[1],
            calibration.region_polygon_px,
            calibration.frame_width,
            calibration.frame_height,
        )
        if in_region:
            x_m, y_m = pixel_to_field(calibration.matrix, foot_px)
            calibrated = True

    return TrackPoint(
        frame_id=frame_id,
        time_s=time_s,
        x_px=foot_px[0],
        y_px=foot_px[1],
        x_m=x_m,
        y_m=y_m,
        calibrated=calibrated,
        confidence=max(min(float(confidence), 1.0), 0.0),
    )


def compute_homography(
    polygon: list[dict] | None,
    frame_width: int | None = None,
    frame_height: int | None = None,
    calibration_frame: np.ndarray | None = None,
    player_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> tuple[np.ndarray | None, list[str]]:
    """Backward-compatible wrapper returning only the homography matrix and warnings."""
    if not frame_width or not frame_height:
        return None, ["Pitch calibration unavailable - video dimensions are missing."]
    result = compute_calibration(
        polygon,
        int(frame_width),
        int(frame_height),
        calibration_frame=calibration_frame,
        player_bboxes=player_bboxes,
    )
    warnings = list(result.warnings)
    warnings.extend(result.errors)
    if not result.scale_known:
        return None, warnings
    return result.matrix, warnings
