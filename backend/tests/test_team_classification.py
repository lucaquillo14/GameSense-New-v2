import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.team_classification import (
    CONFLICT_MAX_SAME_TEAM,
    MIN_SAMPLES_PER_CLUSTER,
    TEAM_A_LABEL,
    TEAM_B_LABEL,
    TEMPORAL_CONFIRM_VOTES,
    TEMPORAL_HISTORY_FRAMES,
    TeamClassificationService,
    TeamTemplates,
    TeamTemporalSmoother,
    build_team_templates,
    classify_team,
    detect_team_conflict,
    extract_shirt_histogram,
    extract_shirt_lab_sample,
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
    def test_extract_shirt_lab_sample_returns_mean_lab_vector(self):
        frame = _solid_player_frame((0, 0, 220))
        sample = extract_shirt_lab_sample(frame, _bbox(40, 40, 60, 120))
        self.assertIsNotNone(sample)
        self.assertEqual(sample.shape, (3,))

    def test_extract_shirt_histogram_alias_matches_lab_sample(self):
        frame = _solid_player_frame((0, 0, 220))
        bbox = _bbox(40, 40, 60, 120)
        self.assertTrue(np.allclose(extract_shirt_histogram(frame, bbox), extract_shirt_lab_sample(frame, bbox)))

    def test_rejects_crops_with_excessive_background_bleed(self):
        frame = np.full((240, 320, 3), 255, dtype=np.uint8)
        sample = extract_shirt_lab_sample(frame, _bbox(40, 40, 60, 120))
        self.assertIsNone(sample)

    def test_classify_team_assigns_distinct_colours(self):
        red_sample = extract_shirt_lab_sample(_solid_player_frame((0, 0, 220)), _bbox(40, 40, 60, 120))
        blue_sample = extract_shirt_lab_sample(_solid_player_frame((220, 0, 0)), _bbox(40, 40, 60, 120))
        self.assertIsNotNone(red_sample)
        self.assertIsNotNone(blue_sample)

        templates = TeamTemplates(
            team_a_lab=red_sample,
            team_b_lab=blue_sample,
            team_a_color_rgb=(220, 0, 0),
            team_b_color_rgb=(0, 0, 220),
        )

        self.assertEqual(classify_team(red_sample, templates), "team_a")
        self.assertEqual(classify_team(blue_sample, templates), "team_b")
        self.assertEqual(team_label("team_a"), TEAM_A_LABEL)
        self.assertEqual(team_label("team_b"), TEAM_B_LABEL)

    def test_referee_excluded_when_third_cluster_enabled(self):
        red_sample = extract_shirt_lab_sample(_solid_player_frame((0, 0, 220)), _bbox(40, 40, 60, 120))
        blue_sample = extract_shirt_lab_sample(_solid_player_frame((220, 0, 0)), _bbox(40, 40, 60, 120))
        yellow_sample = extract_shirt_lab_sample(_solid_player_frame((0, 220, 220)), _bbox(40, 40, 60, 120))
        self.assertIsNotNone(yellow_sample)

        templates = TeamTemplates(
            team_a_lab=red_sample,
            team_b_lab=blue_sample,
            team_a_color_rgb=(220, 0, 0),
            team_b_color_rgb=(0, 0, 220),
            referee_lab=yellow_sample,
            referee_enabled=True,
        )

        self.assertEqual(classify_team(yellow_sample, templates), "referee")

    def test_team_templates_round_trip(self):
        red_sample = extract_shirt_lab_sample(_solid_player_frame((0, 0, 220)), _bbox(40, 40, 60, 120))
        blue_sample = extract_shirt_lab_sample(_solid_player_frame((220, 0, 0)), _bbox(40, 40, 60, 120))
        templates = TeamTemplates(
            team_a_lab=red_sample,
            team_b_lab=blue_sample,
            team_a_color_rgb=(220, 0, 0),
            team_b_color_rgb=(0, 0, 220),
        )
        restored = TeamTemplates.from_dict(templates.to_dict())
        self.assertEqual(classify_team(red_sample, restored), "team_a")
        self.assertEqual(classify_team(blue_sample, restored), "team_b")

    def test_temporal_smoother_requires_majority_votes(self):
        smoother = TeamTemporalSmoother()
        player_key = "p1"
        for _ in range(TEMPORAL_CONFIRM_VOTES - 1):
            self.assertEqual(smoother.record(player_key, "team_a"), "unconfirmed")
        self.assertEqual(smoother.record(player_key, "team_a"), "team_a")

    def test_temporal_smoother_marks_referee_immediately(self):
        smoother = TeamTemporalSmoother()
        self.assertEqual(smoother.record("ref", "referee"), "referee")

    def test_team_classification_service_applies_temporal_smoothing(self):
        red_sample = extract_shirt_lab_sample(_solid_player_frame((0, 0, 220)), _bbox(40, 40, 60, 120))
        blue_sample = extract_shirt_lab_sample(_solid_player_frame((220, 0, 0)), _bbox(40, 40, 60, 120))
        templates = TeamTemplates(
            team_a_lab=red_sample,
            team_b_lab=blue_sample,
            team_a_color_rgb=(220, 0, 0),
            team_b_color_rgb=(0, 0, 220),
        )
        service = TeamClassificationService()
        service.set_templates(templates)
        frame = _solid_player_frame((0, 0, 220))
        bbox = _bbox(40, 40, 60, 120)

        for _ in range(TEMPORAL_CONFIRM_VOTES - 1):
            confirmed, _ = service.classify_player(frame, bbox, "player-1")
            self.assertEqual(confirmed, "unconfirmed")
        confirmed, _ = service.classify_player(frame, bbox, "player-1")
        self.assertEqual(confirmed, "team_a")

    def test_detect_team_conflict_flags_overloaded_team(self):
        self.assertFalse(detect_team_conflict({"team_a": 7, "team_b": 3}))
        self.assertTrue(detect_team_conflict({"team_a": CONFLICT_MAX_SAME_TEAM + 1, "team_b": 2}))

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

        def detect_people(_frame: np.ndarray) -> list[tuple[float, float, float, float, float]]:
            return [
                (30.0, 40.0, 60.0, 120.0, 0.95),
                (120.0, 40.0, 60.0, 120.0, 0.95),
            ]

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

        red_sample = extract_shirt_lab_sample(frames[0], _bbox(30, 40, 60, 120))
        blue_sample = extract_shirt_lab_sample(frames[0], _bbox(120, 40, 60, 120))
        self.assertIsNotNone(red_sample)
        self.assertIsNotNone(blue_sample)

        red_team = classify_team(red_sample, templates)
        blue_team = classify_team(blue_sample, templates)
        self.assertIn(red_team, {"team_a", "team_b"})
        self.assertIn(blue_team, {"team_a", "team_b"})
        self.assertNotEqual(red_team, blue_team)

    def test_calibration_constants(self):
        self.assertGreaterEqual(MIN_SAMPLES_PER_CLUSTER, 8)
        self.assertEqual(TEMPORAL_HISTORY_FRAMES, 8)
        self.assertEqual(TEMPORAL_CONFIRM_VOTES, 5)


if __name__ == "__main__":
    unittest.main()
