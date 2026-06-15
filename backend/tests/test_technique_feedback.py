import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.models import BodyAngle, TechniqueFrame
from app.services.technique_feedback import generate_feedback


class TechniqueFeedbackTests(unittest.TestCase):
    def _frame(self, frame_id: int, phase: str, angles: list[BodyAngle]) -> TechniqueFrame:
        return TechniqueFrame(
            frame_id=frame_id,
            time_s=frame_id / 30.0,
            angles=angles,
            ball_visible=True,
            foot_to_ball_px=40.0,
            phase=phase,
        )

    def test_generates_ankle_feedback_when_below_ideal(self):
        frames = [
            self._frame(
                10,
                "contact",
                [
                    BodyAngle(name="ankle_angle", value_deg=95.0, frame_id=10, time_s=10 / 30.0),
                    BodyAngle(name="knee_angle", value_deg=145.0, frame_id=10, time_s=10 / 30.0),
                ],
            )
        ]
        feedback = generate_feedback(frames, 55.0, 10, plant_foot_distance_cm=22.0)
        self.assertTrue(any("ankle" in point.lower() for point in feedback.feedback_points))
        self.assertLess(feedback.technique_score, 10.0)

    def test_positive_feedback_when_all_ideal(self):
        frames = [
            self._frame(
                5,
                "contact",
                [
                    BodyAngle(name="ankle_angle", value_deg=120.0, frame_id=5, time_s=5 / 30.0),
                    BodyAngle(name="knee_angle", value_deg=150.0, frame_id=5, time_s=5 / 30.0),
                    BodyAngle(name="approach_angle", value_deg=35.0, frame_id=5, time_s=5 / 30.0),
                    BodyAngle(name="shoulder_hip_rotation", value_deg=10.0, frame_id=5, time_s=5 / 30.0),
                    BodyAngle(name="trunk_lean", value_deg=15.0, frame_id=5, time_s=5 / 30.0),
                ],
            )
        ]
        feedback = generate_feedback(frames, 72.0, 5, plant_foot_distance_cm=20.0)
        self.assertEqual(feedback.technique_score, 10.0)
        self.assertTrue(any("well-structured" in point.lower() for point in feedback.feedback_points))


if __name__ == "__main__":
    unittest.main()
