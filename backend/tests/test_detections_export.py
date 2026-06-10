import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

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

        pipeline = MagicMock()
        pipeline.sample_fps = 8.0
        pipeline.max_width = 1280
        pipeline._resize.return_value = (np.zeros((240, 320, 3), dtype=np.uint8), 1.0)
        pipeline._detect_people.return_value = [(40.0, 40.0, 60.0, 120.0, 0.92)]
        pipeline._unscale_bbox.return_value = {"x": 40.0, "y": 40.0, "width": 60.0, "height": 120.0}
        pipeline._classify_player_detection.return_value = ("team_a", (220, 38, 38))
        pipeline.team_service = MagicMock()

        class FakeCapture:
            def __init__(self):
                self._open = True

            def isOpened(self):
                return self._open

            def get(self, prop):
                import cv2

                if prop == cv2.CAP_PROP_FPS:
                    return 50.0
                if prop == cv2.CAP_PROP_FRAME_COUNT:
                    return 100
                return 0

            def set(self, _prop, _value):
                return True

            def read(self):
                return True, np.zeros((240, 320, 3), dtype=np.uint8)

            def release(self):
                self._open = False

        import cv2
        from app.services import detections_export as export_module

        original_capture = cv2.VideoCapture
        cv2.VideoCapture = lambda _path: FakeCapture()
        try:
            payload = collect_overlay_detections(
                pipeline,
                Path("fake.mp4"),
                templates,
                {"fps": 50.0, "frame_count": 100, "width": 320, "height": 240},
                target_player={"player_id": "A1"},
            )
        finally:
            cv2.VideoCapture = original_capture

        self.assertEqual(payload["interval"], 6)
        self.assertEqual(payload["target_id"], "A1")
        self.assertIn("0", payload["frames"])
        entry = payload["frames"]["0"][0]
        self.assertEqual(entry["team"], "Team A")
        self.assertEqual(len(entry["b"]), 4)
        self.assertLessEqual(entry["b"][0], 1.0)
        self.assertEqual(entry["c"], 0.92)


if __name__ == "__main__":
    unittest.main()
