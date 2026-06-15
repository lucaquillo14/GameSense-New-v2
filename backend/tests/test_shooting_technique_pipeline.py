"""Tests for shooting technique pipeline adapter."""
from __future__ import annotations

import unittest
from unittest.mock import Mock

from app.models import ShootingFeedback
from app.services.shooting_technique_pipeline import (
    _follow_through_label,
    analysis_to_shooting_feedback,
)


class ShootingTechniquePipelineTests(unittest.TestCase):
    def test_follow_through_label(self) -> None:
        self.assertEqual(_follow_through_label(None), "medium")
        self.assertEqual(_follow_through_label(0.4), "low")
        self.assertEqual(_follow_through_label(0.6), "medium")
        self.assertEqual(_follow_through_label(0.9), "high")

    def test_analysis_to_shooting_feedback(self) -> None:
        analysis = Mock()
        analysis.metrics = {
            "backswing_knee_flexion": 95.0,
            "contact_knee_angle": 155.0,
            "ankle_lock_variation": 8.0,
            "plant_foot_distance_m": 0.18,
            "approach_angle": 38.0,
            "hip_rotation": 42.0,
            "trunk_lean": 12.0,
            "follow_through_height": 0.72,
        }
        analysis.scale_source = "player height (1.75 m)"
        analysis.shot_speed_kmh = 68.5
        analysis.on_target = True
        analysis.power_label = "Strong"
        analysis.score = 7.8
        analysis.feedback = [
            "Knee extension timing at contact (155°) is in the ideal window — clean leg whip."
        ]
        events = Mock()
        events.contact = 40
        events.plant = 28
        events.backswing_peak = 34
        events.kicking_foot = "right"
        events.plant_foot = "left"
        events.shot_dir_x = 1.0
        events.phase_at = lambda i, n: "contact"

        contact_pose = Mock(ok=True)
        contact_pose.get = Mock(return_value=None)
        contact_pose.mid = Mock(return_value=None)
        poses = [Mock(ok=False) for _ in range(50)]
        poses[40] = contact_pose

        feedback = analysis_to_shooting_feedback(
            analysis=analysis,
            ev=events,
            poses=poses,
            fps=30.0,
            annotated_video_url="/media/demo/workflow/annotated.mp4",
        )

        self.assertIsInstance(feedback, ShootingFeedback)
        self.assertEqual(feedback.technique_score, 7.8)
        self.assertEqual(feedback.shot_power_kmh, 68.5)
        self.assertEqual(feedback.plant_foot_distance_cm, 18.0)
        self.assertEqual(feedback.knee_bend_at_contact_deg, 155.0)
        self.assertEqual(feedback.backswing_knee_flexion_deg, 95.0)
        self.assertEqual(feedback.ankle_lock_variation_deg, 8.0)
        self.assertEqual(feedback.follow_through_height_ratio, 0.72)
        self.assertEqual(feedback.power_rating, "Strong")
        self.assertEqual(feedback.kicking_foot, "right")
        self.assertEqual(feedback.contact_frame_id, 40)
        self.assertTrue(feedback.feedback_points)


if __name__ == "__main__":
    unittest.main()
