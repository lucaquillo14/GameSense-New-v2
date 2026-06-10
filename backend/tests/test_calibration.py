import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.calibration import (
    build_track_point,
    compute_calibration,
    detect_pitch_markings,
    is_pixel_in_calibrated_region,
    player_height_metres,
    validate_calibration_scale,
)
from app.services.metrics import compute_metrics, stabilize_track_points


def _synthetic_pitch_frame() -> np.ndarray:
    frame = np.zeros((480, 720, 3), dtype=np.uint8)
    frame[:, :] = (20, 90, 20)
    cv2.rectangle(frame, (180, 140), (520, 300), (255, 255, 255), 3)
    cv2.line(frame, (180, 140), (180, 300), (255, 255, 255), 3)
    cv2.line(frame, (520, 140), (520, 300), (255, 255, 255), 3)
    cv2.line(frame, (180, 140), (520, 140), (255, 255, 255), 3)
    cv2.line(frame, (180, 300), (520, 300), (255, 255, 255), 3)
    return frame


class CalibrationTests(unittest.TestCase):
    def test_unknown_scale_when_no_polygon_or_markings(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = compute_calibration(None, 320, 240, calibration_frame=frame)
        self.assertFalse(result.scale_known)
        self.assertEqual(result.units, "pixels")
        self.assertIsNone(result.matrix)
        self.assertTrue(any("pitch calibration" in warning.lower() for warning in result.warnings))

    def test_manual_polygon_produces_metric_calibration(self):
        polygon = [
            {"x": 40.0, "y": 40.0},
            {"x": 600.0, "y": 50.0},
            {"x": 620.0, "y": 420.0},
            {"x": 30.0, "y": 430.0},
        ]
        result = compute_calibration(polygon, 720, 480, player_bboxes=[])
        self.assertTrue(result.scale_known)
        self.assertIsNotNone(result.matrix)
        self.assertEqual(result.units, "metric")
        self.assertIsNotNone(result.visible_field_bounds_m)

    def test_validate_calibration_scale_rejects_implausible_player_height(self):
        polygon = [
            {"x": 0.0, "y": 0.0},
            {"x": 10.0, "y": 0.0},
            {"x": 10.0, "y": 10.0},
            {"x": 0.0, "y": 10.0},
        ]
        bboxes = [(2.0, 2.0, 4.0, 6.0)]
        result = compute_calibration(polygon, 720, 480, player_bboxes=bboxes)
        self.assertFalse(result.scale_known)
        self.assertTrue(result.errors)

    def test_build_track_point_marks_uncalibrated_points_outside_region(self):
        from app.services.calibration import CalibrationResult

        calibration = CalibrationResult(
            matrix=np.eye(3, dtype=np.float32),
            scale_known=True,
            frame_width=320,
            frame_height=240,
            units="metric",
            region_polygon_px=[(50.0, 50.0), (250.0, 50.0), (250.0, 200.0), (50.0, 200.0)],
        )
        inside = build_track_point(calibration, 1, 0.04, (120.0, 180.0), 0.9)
        outside = build_track_point(calibration, 2, 0.08, (10.0, 10.0), 0.9)
        self.assertTrue(inside.calibrated)
        self.assertFalse(outside.calibrated)

    def test_pixel_mode_metrics_use_pixel_units(self):
        from app.services.metrics import TrackPoint

        points = [
            TrackPoint(frame_id=0, time_s=0.0, x_px=0.0, y_px=0.0, calibrated=False),
            TrackPoint(frame_id=1, time_s=0.1, x_px=30.0, y_px=0.0, calibrated=False),
            TrackPoint(frame_id=2, time_s=0.2, x_px=60.0, y_px=0.0, calibrated=False),
        ]
        metrics = compute_metrics(1, points, units="pixels")
        self.assertEqual(metrics["units"], "pixels")
        self.assertGreater(metrics["top_speed_px_per_s"], 0.0)
        self.assertEqual(metrics["top_speed_kmh"], 0.0)

    def test_detect_pitch_markings_finds_penalty_box_pattern(self):
        frame = _synthetic_pitch_frame()
        quad, method, score = detect_pitch_markings(frame)
        self.assertIsNotNone(quad)
        self.assertIn(method, {"penalty_box", "centre_circle"})
        self.assertGreater(score, 0.0)

    def test_is_pixel_in_calibrated_region(self):
        polygon = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        self.assertTrue(is_pixel_in_calibrated_region(50.0, 50.0, polygon, 200, 200))
        self.assertFalse(is_pixel_in_calibrated_region(150.0, 150.0, polygon, 200, 200))


if __name__ == "__main__":
    unittest.main()
