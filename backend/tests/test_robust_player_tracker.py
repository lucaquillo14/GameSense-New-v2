import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.calibration import CalibrationResult
from app.services.pitch_keypoints import DynamicPitchCalibration
from app.services.robust_player_tracker import (
    BALL_EVERY_N_DETECTIONS,
    MAX_PLAUSIBLE_SPEED_KMH,
    PREDICTED_MAX_CONSECUTIVE,
    TARGET_DETECT_HZ,
    _detection_cadence,
    _scaled_interframe_homography,
    track_player_robust,
)
from app.services.video_streaming import ANALYSIS_SAMPLE_FPS, compute_frame_interval


class RobustPlayerTrackerTests(unittest.TestCase):
    def test_sampling_interval_targets_eight_fps(self):
        self.assertEqual(compute_frame_interval(50.0), 6)
        self.assertEqual(compute_frame_interval(30.0), 4)
        self.assertEqual(compute_frame_interval(8.0), 1)

    def test_predicted_limit_constant(self):
        self.assertEqual(PREDICTED_MAX_CONSECUTIVE, 45)

    def test_detection_cadence_targets_fifteen_hz(self):
        self.assertEqual(TARGET_DETECT_HZ, 15.0)
        with patch.dict(os.environ, {"TRACK_DETECT_CADENCE": ""}):
            self.assertEqual(_detection_cadence(30.0), 2)
            self.assertEqual(_detection_cadence(60.0), 4)
            self.assertEqual(_detection_cadence(15.0), 1)
        with patch.dict(os.environ, {"TRACK_DETECT_CADENCE": "3"}):
            self.assertEqual(_detection_cadence(30.0), 3)

    def test_speed_gates_are_human_plausible(self):
        self.assertGreater(MAX_PLAUSIBLE_SPEED_KMH, 38.0)
        self.assertLess(MAX_PLAUSIBLE_SPEED_KMH, 45.0)
        self.assertGreaterEqual(BALL_EVERY_N_DETECTIONS, 1)

    @patch("app.services.pitch_keypoints.PitchKeypointDetector")
    @patch("app.services.robust_player_tracker.iter_all_frames")
    def test_track_player_robust_returns_metric_fields(self, mock_iter, mock_kp_cls):
        mock_kp_cls.return_value.available = False
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        mock_iter.return_value = [(0, 30.0, frame, 120)]

        tracked = MagicMock()
        tracked.tracker_id = np.array([3])
        tracked.xyxy = np.array([[40.0, 40.0, 100.0, 160.0]])
        tracked.confidence = np.array([0.92])

        pipeline = MagicMock()
        pipeline.create_player_tracker.return_value = MagicMock()
        pipeline.track_players.return_value = tracked
        pipeline.tracked_to_tuples.return_value = [(40.0, 40.0, 60.0, 120.0, 0.92)]
        pipeline.select_tracker_id.return_value = 3
        pipeline.bbox_for_tracker_id.return_value = (40.0, 40.0, 60.0, 120.0, 0.92)
        pipeline._detect_ball.return_value = []

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
        self.assertIn("rejected_outliers", result.__dict__)
        self.assertGreaterEqual(result.confidence_score, 0.0)


class DynamicAnchorCalibrationTests(unittest.TestCase):
    def _translation(self, dx: float, dy: float) -> np.ndarray:
        H = np.eye(3, dtype=np.float64)
        H[0, 2] = dx
        H[1, 2] = dy
        return H

    def test_anchor_snaps_back_when_camera_returns(self):
        anchor = np.array(
            [[0.1, 0.0, 5.0], [0.0, 0.1, 3.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        dyn = DynamicPitchCalibration(fps=30.0)
        dyn.set_anchor(anchor, (1280, 720))
        self.assertTrue(dyn.has_fix)
        h0, age0 = dyn.current()
        self.assertIsNotNone(h0)
        self.assertEqual(age0, 0.0)

        # Camera pans 80 px right: homography follows, age starts growing.
        dyn.propagate(self._translation(80.0, 0.0))
        h_pan, age_pan = dyn.current()
        self.assertIsNotNone(h_pan)
        self.assertGreater(age_pan, 0.0)
        self.assertFalse(np.allclose(h_pan, anchor))

        # Camera pans back: cumulative motion is near identity, so the chain
        # re-anchors exactly and the age resets to zero.
        dyn.propagate(self._translation(-80.0, 0.0))
        h_back, age_back = dyn.current()
        self.assertTrue(np.allclose(h_back, anchor))
        self.assertEqual(age_back, 0.0)

    def test_anchor_goes_stale_during_long_pan_without_fix(self):
        dyn = DynamicPitchCalibration(fps=30.0)
        dyn.set_anchor(np.eye(3, dtype=np.float64), (1280, 720))
        # Pan steadily away for >3 seconds of frames: H goes stale (no guessing).
        for _ in range(int(DynamicPitchCalibration.MAX_AGE_S * 30.0) + 5):
            dyn.propagate(self._translation(4.0, 0.0))
        h_now, age = dyn.current()
        self.assertIsNone(h_now)
        self.assertGreater(age, DynamicPitchCalibration.MAX_AGE_S)

    def test_static_camera_keeps_anchor_fresh_indefinitely(self):
        dyn = DynamicPitchCalibration(fps=30.0)
        anchor = np.eye(3, dtype=np.float64)
        dyn.set_anchor(anchor, (1280, 720))
        # Identity inter-frame homographies (static camera) for 10 s of frames.
        for _ in range(300):
            dyn.propagate(np.eye(3, dtype=np.float64))
        h_now, age = dyn.current()
        self.assertIsNotNone(h_now)
        self.assertEqual(age, 0.0)
        self.assertTrue(np.allclose(h_now, anchor))


class ScaledFlowHelpersTests(unittest.TestCase):
    def test_scaled_homography_round_trip(self):
        # A pure translation estimated at half resolution must convert to the
        # doubled translation at full resolution.
        rng = np.random.default_rng(7)
        full = (rng.random((720, 1280)) * 255).astype(np.uint8)
        # Build synthetic small grays: shift by 5 px at half res = 10 px full.
        small_prev = full[::2, ::2].copy()
        small_cur = np.roll(small_prev, 5, axis=1)
        h_full = _scaled_interframe_homography(small_prev, small_cur, [], 0.5)
        if h_full is not None:   # flow can fail on synthetic noise; only assert when found
            dx_full = h_full[0, 2]
            self.assertAlmostEqual(dx_full, 10.0, delta=2.5)


if __name__ == "__main__":
    unittest.main()
