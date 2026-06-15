from __future__ import annotations

from app.models import ShootingFeedback, TechniqueFrame

IDEAL = {
    "ankle_angle": (110.0, 130.0),
    "knee_angle": (135.0, 160.0),
    "plant_foot_distance_cm": (15.0, 30.0),
    "approach_angle": (30.0, 45.0),
    "shoulder_hip_rotation": (0.0, 45.0),
    "trunk_lean": (10.0, 20.0),
}


def _angle_value(frames: list[TechniqueFrame], name: str, phase: str | None = None) -> float | None:
    for frame in frames:
        if phase is not None and frame.phase != phase:
            continue
        for angle in frame.angles:
            if angle.name == name:
                return angle.value_deg
    return None


def _follow_through_height(frames: list[TechniqueFrame], contact_frame_id: int) -> str:
    post_contact = [frame for frame in frames if frame.frame_id > contact_frame_id and frame.phase == "follow_through"]
    if not post_contact:
        return "medium"
    ankle_values = [
        angle.value_deg
        for frame in post_contact[:8]
        for angle in frame.angles
        if angle.name == "ankle_angle"
    ]
    if not ankle_values:
        return "medium"
    avg = sum(ankle_values) / len(ankle_values)
    if avg < 120:
        return "low"
    if avg > 150:
        return "high"
    return "medium"


def _plant_foot_distance_cm(frames: list[TechniqueFrame], contact_frame_id: int, pixels_per_meter: float) -> float:
    for frame in frames:
        if frame.frame_id != contact_frame_id:
            continue
        if frame.foot_to_ball_px is None or pixels_per_meter <= 0:
            return 0.0
        return (frame.foot_to_ball_px / pixels_per_meter) * 100.0
    return 0.0


def generate_feedback(
    pose_frames: list[TechniqueFrame],
    shot_power_kmh: float,
    contact_frame_id: int,
    *,
    plant_foot_distance_cm: float = 0.0,
    pixels_per_meter: float = 0.0,
) -> ShootingFeedback:
    if plant_foot_distance_cm <= 0 and pixels_per_meter > 0:
        plant_foot_distance_cm = _plant_foot_distance_cm(pose_frames, contact_frame_id, pixels_per_meter)

    contact_frames = [frame for frame in pose_frames if frame.frame_id == contact_frame_id]
    contact_frame = contact_frames[0] if contact_frames else None

    knee_at_contact = _angle_value(pose_frames, "knee_angle", "contact")
    ankle_at_contact = _angle_value(pose_frames, "ankle_angle", "contact")
    approach_angle = _angle_value(pose_frames, "approach_angle")
    shoulder_hip_rotation = _angle_value(pose_frames, "shoulder_hip_rotation", "contact")
    trunk_lean = _angle_value(pose_frames, "trunk_lean", "contact")
    follow_through = _follow_through_height(pose_frames, contact_frame_id)

    feedback_points: list[tuple[int, str, str]] = []

    if ankle_at_contact is not None and ankle_at_contact < IDEAL["ankle_angle"][0]:
        feedback_points.append(
            (
                1,
                "Ankle lock at contact",
                f"Your ankle was not fully locked at contact ({ankle_at_contact:.0f}° — aim for 110–130°). "
                "Push your toes down and hold them firm through the kick. A floppy ankle loses 15–20% of power transfer.",
            )
        )

    if knee_at_contact is not None and knee_at_contact < 130:
        feedback_points.append(
            (
                2,
                "Knee bend at contact",
                f"Your kicking knee was too bent at contact ({knee_at_contact:.0f}° — aim for 135–160°). "
                "Let the leg extend more naturally through the ball. Over-bending reduces the lever arm length.",
            )
        )

    if knee_at_contact is not None and knee_at_contact > 170:
        feedback_points.append(
            (
                2,
                "Knee extension at contact",
                f"Your kicking leg was almost straight at contact ({knee_at_contact:.0f}°). "
                "A slightly bent knee (135–160°) creates a whip effect that generates more power.",
            )
        )

    if plant_foot_distance_cm > 40:
        feedback_points.append(
            (
                3,
                "Plant foot distance",
                f"Your plant foot landed {plant_foot_distance_cm:.0f}cm from the ball (aim for 15–30cm to the side). "
                "Move your standing foot closer to the ball to improve balance and contact quality.",
            )
        )

    if 0 < plant_foot_distance_cm < 10:
        feedback_points.append(
            (
                3,
                "Plant foot distance",
                f"Your plant foot was too close to the ball ({plant_foot_distance_cm:.0f}cm — aim for 15–30cm). "
                "This restricts hip rotation and reduces power.",
            )
        )

    if approach_angle is not None and approach_angle < 20:
        feedback_points.append(
            (
                4,
                "Approach angle",
                f"You approached the ball almost straight on ({approach_angle:.0f}° — aim for 30–45°). "
                "A slight angled run-up allows better hip rotation and more powerful contact.",
            )
        )

    if approach_angle is not None and approach_angle > 60:
        feedback_points.append(
            (
                4,
                "Approach angle",
                f"Your approach angle was very wide ({approach_angle:.0f}°). "
                "This makes it harder to drive through the ball. Aim for 30–45 degrees off straight.",
            )
        )

    if shoulder_hip_rotation is not None and shoulder_hip_rotation < 0:
        feedback_points.append(
            (
                5,
                "Shoulder–hip rotation",
                "Your hips were rotating ahead of your shoulders at contact. "
                "Lead with your shoulders slightly to maintain power through the swing.",
            )
        )

    if follow_through == "low":
        feedback_points.append(
            (
                6,
                "Follow-through height",
                "Your follow-through was low, which typically produces a ground shot. "
                "For more power and elevation, let your kicking leg continue upward to hip height after contact.",
            )
        )

    if trunk_lean is not None and trunk_lean > 25:
        feedback_points.append(
            (
                5,
                "Trunk lean",
                f"You were leaning away from the ball at contact ({trunk_lean:.0f}°). "
                "Stay over the ball to keep the shot low and maximise power. Leaning back sends it high and weak.",
            )
        )

    feedback_points.sort(key=lambda item: item[0])
    messages = [text for _priority, _title, text in feedback_points[:6]]

    issue_count = len(feedback_points)
    if issue_count == 0:
        messages = [
            "Your technique is well-structured. Focus on generating more speed in your approach run to increase power further."
        ]

    technique_score = max(1.0, 10.0 - float(issue_count))

    visible_frames = sum(1 for frame in pose_frames if frame.ball_visible)
    confidence = min(1.0, visible_frames / max(len(pose_frames), 1))

    return ShootingFeedback(
        shot_power_kmh=round(shot_power_kmh, 1),
        technique_score=round(technique_score, 1),
        approach_angle_deg=round(approach_angle or 0.0, 1),
        plant_foot_distance_cm=round(plant_foot_distance_cm, 1),
        knee_bend_at_contact_deg=round(knee_at_contact or 0.0, 1),
        hip_rotation_deg=round(shoulder_hip_rotation or 0.0, 1),
        follow_through_height=follow_through,
        feedback_points=messages,
        frame_analysis=pose_frames,
        confidence=round(confidence, 2),
        contact_frame_id=contact_frame_id,
    )
