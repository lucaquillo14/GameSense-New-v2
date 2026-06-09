import sys
import unittest
from pathlib import Path

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.cv_pipeline import (
    BALL_KALMAN_MEASUREMENT_NOISE,
    BALL_KALMAN_PROCESS_NOISE,
    BALL_MAX_CONSECUTIVE_MISSES,
    BallKalmanTracker,
)


class BallKalmanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.homography = np.eye(3, dtype=np.float64)
        self.tracker = BallKalmanTracker()

    def test_constants(self):
        self.assertEqual(BALL_KALMAN_PROCESS_NOISE, 25.0)
        self.assertEqual(BALL_KALMAN_MEASUREMENT_NOISE, 10.0)
        self.assertEqual(BALL_MAX_CONSECUTIVE_MISSES, 15)

    def test_marks_interpolated_points_when_detection_missing(self):
        first = self.tracker.step(0, 0.0, self.homography, (100.0, 200.0, 0.9))
        second = self.tracker.step(1, 1 / 30.0, self.homography, None)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertFalse(first.interpolated)
        self.assertTrue(second.interpolated)

    def test_resets_after_fifteen_consecutive_misses(self):
        self.tracker.step(0, 0.0, self.homography, (100.0, 200.0, 0.9))
        last_point = None
        for frame_id in range(1, BALL_MAX_CONSECUTIVE_MISSES + 2):
            last_point = self.tracker.step(frame_id, frame_id / 30.0, self.homography, None)
        self.assertIsNone(last_point)
        self.assertFalse(self.tracker.initialized)

    def test_predicted_position_available_after_init(self):
        self.tracker.step(0, 0.0, self.homography, (100.0, 200.0, 0.9))
        predicted = self.tracker.predicted_position()
        self.assertIsNotNone(predicted)
        self.assertEqual(len(predicted), 2)


if __name__ == "__main__":
    unittest.main()
