import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.team_classification import (
    REFEREE_DISTANCE_THRESHOLD,
    TEAM_A_LABEL,
    TEAM_B_LABEL,
    TeamTemplates,
    build_team_templates,
    classify_team,
    extract_shirt_histogram,
    team_label,
)


def _solid_player_frame(
    shirt_bgr: tuple[int, int, int],
    x: int = 40,
    y: int = 40,
    width: int = 60,
    height: int = 120,
) -> np.ndarray:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :] = (30, 30, 30)
    top = y
    bottom = y + height
    shirt_bottom = top + int(height * 0.50)
    frame[top:shirt_bottom, x : x + width] = shirt_bgr
    frame[shirt_bottom:bottom, x : x + width] = (20, 20, 20)
    return frame


def _bbox(x: int, y: int, width: int, height: int) -> dict:
    return {"x": float(x), "y": float(y), "width": float(width), "height": float(height)}


class TeamClassificationTests(unittest.TestCase):
    def test_extract_shirt_histogram_uses_upper_body_only(self):
        frame = _solid_player_frame((0, 0, 220))
        histogram = extract_shirt_histogram(frame, _bbox(40, 40, 60, 120))
        self.assertIsNotNone(histogram)
        self.assertEqual(histogram.shape[0], 256)
        self.assertAlmostEqual(float(histogram.sum()), 1.0, places=3)

    def test_classify_team_assigns_distinct_colours(self):
        red_hist = extract_shirt_histogram(frame := _solid_player_frame((0, 0, 220)), _bbox(40, 40, 60, 120))
        blue_hist = extract_shirt_histogram(_solid_player_frame((220, 0, 0)), _bbox(40, 40, 60, 120))
        self.assertIsNotNone(red_hist)
        self.assertIsNotNone(blue_hist)

        templates = TeamTemplates(
            team_a_histogram=red_hist,
            team_b_histogram=blue_hist,
            team_a_color_rgb=(220, 0, 0),
            team_b_color_rgb=(0, 0, 220),
        )

        self.assertEqual(classify_team(red_hist, templates), "team_a")
        self.assertEqual(classify_team(blue_hist, templates), "team_b")
        self.assertEqual(team_label("team_a"), TEAM_A_LABEL)
        self.assertEqual(team_label("team_b"), TEAM_B_LABEL)

    def test_referee_excluded_when_far_from_both_teams(self):
        red_hist = extract_shirt_histogram(_solid_player_frame((0, 0, 220)), _bbox(40, 40, 60, 120))
        blue_hist = extract_shirt_histogram(_solid_player_frame((220, 0, 0)), _bbox(40, 40, 60, 120))
        yellow_hist = extract_shirt_histogram(_solid_player_frame((0, 220, 220)), _bbox(40, 40, 60, 120))
        self.assertIsNotNone(yellow_hist)

        templates = TeamTemplates(
            team_a_histogram=red_hist,
            team_b_histogram=blue_hist,
            team_a_color_rgb=(220, 0, 0),
            team_b_color_rgb=(0, 0, 220),
        )

        self.assertEqual(classify_team(yellow_hist, templates), "referee")

    def test_team_templates_round_trip(self):
        red_hist = extract_shirt_histogram(_solid_player_frame((0, 0, 220)), _bbox(40, 40, 60, 120))
        blue_hist = extract_shirt_histogram(_solid_player_frame((220, 0, 0)), _bbox(40, 40, 60, 120))
        templates = TeamTemplates(
            team_a_histogram=red_hist,
            team_b_histogram=blue_hist,
            team_a_color_rgb=(220, 0, 0),
            team_b_color_rgb=(0, 0, 220),
        )
        restored = TeamTemplates.from_dict(templates.to_dict())
        self.assertEqual(classify_team(red_hist, restored), "team_a")
        self.assertEqual(classify_team(blue_hist, restored), "team_b")

    def test_kmeans_calibration_separates_two_team_colours(self):
        frames: list[np.ndarray] = []
        for _ in range(10):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            frame[:, :] = (30, 30, 30)
            frame[40:100, 30:90] = (0, 0, 220)
            frame[40:100, 120:180] = (220, 0, 0)
            frame[100:160, 30:90] = (20, 20, 20)
            frame[100:160, 120:180] = (20, 20, 20)
            frames.append(frame)

        frame_index = {"value": 0}

        def read_frame(_frame_id: int) -> np.ndarray:
            frame = frames[min(frame_index["value"], len(frames) - 1)]
            frame_index["value"] += 1
            return frame

        def detect_people(frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
            players = []
            for x in (30, 120):
                players.append((float(x), 40.0, 60.0, 120.0, 0.95))
            return players

        capture_frames = iter(frames)

        class FakeCapture:
            def __init__(self):
                self.opened = True

            def isOpened(self):
                return self.opened

            def get(self, prop):
                if prop == cv2.CAP_PROP_FRAME_COUNT:
                    return len(frames)
                return 0

            def set(self, _prop, _value):
                return True

            def read(self):
                try:
                    return True, next(capture_frames)
                except StopIteration:
                    return False, None

            def release(self):
                self.opened = False

        original_video_capture = cv2.VideoCapture
        cv2.VideoCapture = lambda _path: FakeCapture()
        try:
            templates = build_team_templates(
                Path("synthetic.mp4"),
                detect_people,
                lambda frame: (frame, 1.0),
                lambda bbox, _scale: {
                    "x": bbox[0],
                    "y": bbox[1],
                    "width": bbox[2],
                    "height": bbox[3],
                },
            )
        finally:
            cv2.VideoCapture = original_video_capture

        red_hist = extract_shirt_histogram(frames[0], _bbox(30, 40, 60, 120))
        blue_hist = extract_shirt_histogram(frames[0], _bbox(120, 40, 60, 120))
        self.assertIsNotNone(red_hist)
        self.assertIsNotNone(blue_hist)

        red_team = classify_team(red_hist, templates)
        blue_team = classify_team(blue_hist, templates)
        self.assertIn(red_team, {"team_a", "team_b"})
        self.assertIn(blue_team, {"team_a", "team_b"})
        self.assertNotEqual(red_team, blue_team)

    def test_referee_threshold_constant(self):
        self.assertEqual(REFEREE_DISTANCE_THRESHOLD, 0.55)


if __name__ == "__main__":
    unittest.main()
