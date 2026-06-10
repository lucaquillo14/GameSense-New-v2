import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.calibration import CalibrationResult
from app.services.robust_player_tracker import (
    PREDICTED_MAX_CONSECUTIVE,
    REID_COSINE_THRESHOLD,
    track_player_robust,
)
from app.services.video_streaming import ANALYSIS_SAMPLE_FPS, compute_frame_interval


class RobustPlayerTrackerTests(unittest.TestCase):
    def test_sampling_interval_targets_eight_fps(self):
        self.assertEqual(compute_frame_interval(50.0), 6)
        self.assertEqual(compute_frame_interval(30.0), 4)
        self.assertEqual(compute_frame_interval(8.0), 1)

    def test_predicted_limit_and_reid_threshold_constants(self):
        self.assertEqual(PREDICTED_MAX_CONSECUTIVE, 45)
        self.assertEqual(REID_COSINE_THRESHOLD, 0.65)

    @patch("app.services.robust_player_tracker.iter_all_frames")
    def test_track_player_robust_returns_metric_fields(self, mock_iter):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        mock_iter.return_value = [(0, 30.0, frame, 120)]

        pipeline = MagicMock()
        pipeline._detect_people.return_value = [(40.0, 40.0, 60.0, 120.0, 0.92)]
        pipeline._appearance_feature.return_value = np.ones(32, dtype=np.float32)
        pipeline._blend_feature.side_effect = lambda left, right, alpha=0.7: left

        calibration = CalibrationResult(
            units="metric",
            scale_known=True,
            matrix=np.eye(3, dtype=np.float64),
            frame_width=320,
            frame_height=240,
            warnings=[],
            errors=[],
        )

        result = track_player_robust(
            pipeline,
            Path("fake.mp4"),
            {"x": 70.0, "y": 160.0},
            calibration,
            initial_bbox={"x": 40.0, "y": 40.0, "width": 60.0, "height": 120.0},
        )

        self.assertEqual(result.sampling_fps, ANALYSIS_SAMPLE_FPS)
        self.assertGreaterEqual(result.stats.visible_frames, 1)
        self.assertIn("max_speed_kmh", result.__dict__)
        self.assertGreaterEqual(result.confidence_score, 0.0)


if __name__ == "__main__":
    unittest.main()
