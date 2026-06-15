import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.detections_export import collect_overlay_detections
from app.services.team_classification import TeamTemplates


class DetectionsExportTests(unittest.TestCase):
    def test_collect_overlay_detections_returns_normalized_compact_entries(self):
        templates = TeamTemplates(
            team_a_lab=np.array([60.0, 180.0, 150.0], dtype=np.float32),
            team_b_lab=np.array([40.0, 150.0, 40.0], dtype=np.float32),
            team_a_color_rgb=(220, 38, 38),
            team_b_color_rgb=(37, 99, 235),
        )

        tracked = MagicMock()
        tracked.tracker_id = np.array([7])
        tracked.xyxy = np.array([[40.0, 40.0, 100.0, 160.0]])
        tracked.confidence = np.array([0.92])

        pipeline = MagicMock()
        pipeline.create_player_tracker.return_value = MagicMock()
        pipeline.track_players.return_value = tracked
        pipeline.tracked_to_tuples.return_value = [(40.0, 40.0, 60.0, 120.0, 0.92)]
        pipeline._classify_player_detection.return_value = ("team_a", (220, 38, 38))
        pipeline.team_service = MagicMock()

        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        def fake_iter_all_frames(_video_path, start_frame_id=0):
            yield 0, 50.0, frame, 100

        with patch("app.services.detections_export.iter_all_frames", fake_iter_all_frames):
            payload = collect_overlay_detections(
                pipeline,
                Path("fake.mp4"),
                templates,
                {"fps": 50.0, "frame_count": 100, "width": 320, "height": 240},
                target_player={"player_id": "A1"},
            )

        self.assertEqual(payload["interval"], 6)
        self.assertEqual(payload["target_id"], "A1")
        self.assertIn("0", payload["frames"])
        entry = payload["frames"]["0"][0]
        self.assertEqual(entry["team"], "Team A")
        self.assertEqual(len(entry["b"]), 4)
        self.assertLessEqual(entry["b"][0], 1.0)
        self.assertEqual(entry["c"], 0.92)
        self.assertFalse(entry["interpolated"])


if __name__ == "__main__":
    unittest.main()
