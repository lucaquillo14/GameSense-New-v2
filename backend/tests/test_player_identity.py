import sys
import unittest
from pathlib import Path

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.player_identity import (
    COLOR_REID_THRESHOLD,
    NEW_ID_CONFIRMATION_FRAMES,
    PlayerIdentityManager,
)
from app.services.team_classification import TeamTemplates, extract_shirt_histogram


def _solid_player_frame(shirt_bgr: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :] = (30, 30, 30)
    frame[40:100, 40:100] = shirt_bgr
    frame[100:160, 40:100] = (20, 20, 20)
    return frame


class PlayerIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        red_hist = extract_shirt_histogram(_solid_player_frame((0, 0, 220)), {"x": 40, "y": 40, "width": 60, "height": 120})
        blue_hist = extract_shirt_histogram(_solid_player_frame((220, 0, 0)), {"x": 40, "y": 40, "width": 60, "height": 120})
        self.templates = TeamTemplates(
            team_a_histogram=red_hist,
            team_b_histogram=blue_hist,
            team_a_color_rgb=(220, 0, 0),
            team_b_color_rgb=(0, 0, 220),
        )
        self.manager = PlayerIdentityManager(self.templates)

    def test_registers_team_prefixed_ids(self):
        frame = _solid_player_frame((0, 0, 220))
        bbox = (40.0, 40.0, 60.0, 120.0)
        stable_id = self.manager.register_immediate(frame, bbox, 0, 0.0, byte_track_id=7, team="team_a")
        self.assertEqual(stable_id, "A1")

        frame_b = _solid_player_frame((220, 0, 0))
        stable_id_b = self.manager.register_immediate(frame_b, bbox, 0, 0.0, byte_track_id=8, team="team_b")
        self.assertEqual(stable_id_b, "B1")

    def test_restores_original_id_after_byte_track_switch(self):
        frame = _solid_player_frame((0, 0, 220))
        bbox = (40.0, 40.0, 60.0, 120.0)
        stable_id = self.manager.register_immediate(frame, bbox, 10, 0.33, byte_track_id=1, team="team_a")
        self.assertEqual(stable_id, "A1")

        restored = self.manager.assign_identity(
            frame,
            bbox,
            11,
            0.37,
            byte_track_id=99,
            team="team_a",
        )
        self.assertEqual(restored, "A1")

    def test_new_id_requires_five_frames(self):
        frame = _solid_player_frame((0, 0, 220))
        bbox = (40.0, 40.0, 60.0, 120.0)
        for frame_id in range(NEW_ID_CONFIRMATION_FRAMES - 1):
            stable_id = self.manager.assign_identity(frame, bbox, frame_id, frame_id / 30.0, byte_track_id=5, team="team_a")
            self.assertIsNone(stable_id)

        stable_id = self.manager.assign_identity(
            frame,
            bbox,
            NEW_ID_CONFIRMATION_FRAMES - 1,
            (NEW_ID_CONFIRMATION_FRAMES - 1) / 30.0,
            byte_track_id=5,
            team="team_a",
        )
        self.assertEqual(stable_id, "A1")

    def test_constants(self):
        self.assertEqual(COLOR_REID_THRESHOLD, 0.38)
        self.assertEqual(NEW_ID_CONFIRMATION_FRAMES, 5)


if __name__ == "__main__":
    unittest.main()
