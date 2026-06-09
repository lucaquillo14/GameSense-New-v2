import cv2
import numpy as np

FIELD_LENGTH_M = 105.0
FIELD_WIDTH_M = 68.0


def _order_quad(points: np.ndarray) -> np.ndarray:
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]
    top_right = points[np.argmin(diffs)]
    bottom_left = points[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def compute_homography(
    polygon: list[dict] | None,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> tuple[np.ndarray | None, list[str]]:
    warnings: list[str] = []
    if not polygon:
        if not frame_width or not frame_height:
            return None, ["Auto pitch calibration failed - video dimensions are unavailable."]
        source = np.array(
            [
                [0.0, 0.0],
                [float(frame_width), 0.0],
                [float(frame_width), float(frame_height)],
                [0.0, float(frame_height)],
            ],
            dtype=np.float32,
        )
        destination = _field_destination()
        warnings.append(
            "Auto pitch calibration used full-frame scale. Metrics are stabilized but less accurate than manual pitch marking."
        )
        return cv2.getPerspectiveTransform(source, destination), warnings

    points = np.array([[p["x"], p["y"]] for p in polygon], dtype=np.float32)
    if len(points) < 4:
        return None, ["Pitch calibration insufficient - at least 4 points are required."]

    hull = cv2.convexHull(points).reshape(-1, 2)
    area = cv2.contourArea(hull.astype(np.float32))
    if area < 10_000:
        warnings.append("Pitch calibration insufficient - selected polygon area is very small.")

    quad = _order_quad(hull if len(hull) >= 4 else points)
    destination = _field_destination()

    matrix = cv2.getPerspectiveTransform(quad, destination)
    if not np.isfinite(matrix).all():
        return None, ["Pitch calibration insufficient - homography could not be computed."]

    if len(polygon) < 6:
        warnings.append("Pitch calibration has only 4-5 points; 6-8 points are recommended for accuracy.")

    return matrix, warnings


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


def pixel_to_field(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    src = np.array([[[point[0], point[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, matrix)[0][0]
    return float(dst[0]), float(dst[1])
