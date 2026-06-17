from typing import Literal

from pydantic import BaseModel, Field

AnalysisMode = Literal["max_speed", "max_shot_power", "shooting_technique"]


class Point(BaseModel):
    x: float
    y: float


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class TeamColor(BaseModel):
    r: int
    g: int
    b: int


class TeamClassificationInfo(BaseModel):
    team_a: dict
    team_b: dict
    calibration_frames: int = 30
    referee_enabled: bool = False
    warnings: list[str] = Field(default_factory=list)


class Detection(BaseModel):
    id: str
    label: str
    confidence: float
    bbox: BoundingBox
    team: Literal["team_a", "team_b"] | None = None
    team_label: Literal["Team A", "Team B"] | None = None
    team_color: TeamColor | None = None
    player_id: str | None = None


class UploadResponse(BaseModel):
    video_id: str
    filename: str
    setup_frame_url: str
    source_url: str
    metadata: dict


class PlayerSelectionRequest(BaseModel):
    video_id: str
    click: Point
    frame_id: int = 0
    player_id_target: int | None = None
    detection_id: str | None = None
    bbox: BoundingBox | None = None


class GoalPosts(BaseModel):
    """Four corners of one goal frame (both post bases + both post tops),
    in any order — the backend normalises left/right and top/base."""

    points: list[Point] = Field(min_length=4, max_length=4)


class PitchSetupRequest(BaseModel):
    video_id: str
    pitch_polygon: list[Point] = Field(default_factory=list)
    frame_id: int = 0
    goal_left: Point | None = None
    goal_right: Point | None = None
    goal_posts: GoalPosts | None = None


class FrameResponse(BaseModel):
    video_id: str
    frame_id: int
    frame_url: str
    detections: list[Detection] = []


class FrameDetectionsResponse(BaseModel):
    video_id: str
    frame_id: int
    detections: list[Detection]


class BodyAngle(BaseModel):
    name: str
    value_deg: float
    frame_id: int
    time_s: float


class TechniqueFrame(BaseModel):
    frame_id: int
    time_s: float
    angles: list[BodyAngle] = Field(default_factory=list)
    ball_visible: bool = False
    foot_to_ball_px: float | None = None
    phase: str = "approach"


class ShootingFeedback(BaseModel):
    shot_power_kmh: float = 0.0
    technique_score: float = 0.0
    approach_angle_deg: float = 0.0
    plant_foot_distance_cm: float = 0.0
    knee_bend_at_contact_deg: float = 0.0
    hip_rotation_deg: float = 0.0
    follow_through_height: str = "medium"
    feedback_points: list[str] = Field(default_factory=list)
    annotated_video_url: str | None = None
    contact_frame_url: str | None = None
    frame_analysis: list[TechniqueFrame] = Field(default_factory=list)
    confidence: float = 0.0
    contact_frame_id: int | None = None
    backswing_knee_flexion_deg: float = 0.0
    ankle_lock_variation_deg: float = 0.0
    follow_through_height_ratio: float = 0.0
    power_rating: str = ""
    kicking_foot: str = ""
    scale_source: str = ""
    shot_distance_m: float = 0.0
    on_target: bool | None = None
    goal_crossing_height_m: float = 0.0
    goal_crossing_offset_m: float = 0.0


class ProcessRequest(BaseModel):
    video_id: str
    mode: AnalysisMode = "max_speed"
    player_height_cm: float | None = None


class SpeedSeriesPoint(BaseModel):
    time_s: float
    speed_kmh: float


class Metrics(BaseModel):
    player_id: int
    player_label: str | None = None
    team_label: str | None = None
    units: Literal["metric", "pixels"] = "metric"
    speed_series: list[SpeedSeriesPoint] = Field(default_factory=list)
    max_speed_kmh: float = 0.0
    top_speed_kmh: float = 0.0
    avg_speed_kmh: float = 0.0
    distance_m: float = 0.0
    tracked_frames: int = 0
    predicted_frames: int = 0
    lost_frames: int = 0
    top_speed_px_per_s: float = 0.0
    avg_speed_px_per_s: float = 0.0
    calibrated_point_ratio: float = 0.0
    peak_acceleration_mps2: float = 0.0
    avg_acceleration_mps2: float = 0.0
    total_distance_m: float = 0.0
    active_distance_m: float = 0.0
    sprint_count: int = 0
    sprint_distance_m: float = 0.0
    usable_track_points: int = 0
    rejected_jump_count: int = 0
    confidence_score: float = 0.0
    touch_count: int = 0
    pass_count: int = 0


class ShotEvent(BaseModel):
    frame_id: int
    timestamp_s: float
    ball_speed_kmh: float
    contact_point: Point


class ShotMetrics(BaseModel):
    player_id: int
    player_label: str | None = None
    team_label: str | None = None
    peak_shot_speed_kmh: float
    avg_shot_speed_kmh: float
    shot_count: int
    best_shot: ShotEvent | None = None
    shots: list[ShotEvent] = Field(default_factory=list)
    confidence_score: float = 0.0
    usable_track_points: int = 0
    rejected_track_points: int = 0
    touch_count: int = 0
    pass_count: int = 0


class ResultAssets(BaseModel):
    sprint_highlights: list[str] = []
    detections_json: str | None = None
    position_heatmap: str | None = None
    speed_heatmap: str | None = None
    movement_heatmap: str | None = None
    touch_heatmap: str | None = None


class VideoResult(BaseModel):
    video_id: str
    filename: str | None = None
    status: Literal["uploaded", "player_selected", "setup_complete", "processing", "complete", "failed"]
    mode: AnalysisMode = "max_speed"
    team_classification: TeamClassificationInfo | None = None
    setup_frame: str | None = None
    setup_frame_id: int | None = None
    source_url: str | None = None
    video_metadata: dict | None = None
    target_player: dict | None = None
    pitch_setup: dict | None = None
    results: Metrics | ShotMetrics | None = None
    shooting_result: ShootingFeedback | None = None
    assets: ResultAssets | None = None
    warnings: list[str] = []
    progress: dict | None = None
    # Membership-gated features the owner's tier does not include (e.g.
    # "heatmaps"). The frontend uses this to render upgrade prompts.
    locked_features: list[str] = []
