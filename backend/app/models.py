from typing import Literal

from pydantic import BaseModel, Field

AnalysisMode = Literal["max_speed", "max_shot_power"]


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
    calibration_frames: int = 10
    referee_distance_threshold: float = 0.55


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


class PitchSetupRequest(BaseModel):
    video_id: str
    pitch_polygon: list[Point] = Field(default_factory=list)
    frame_id: int = 0
    goal_left: Point | None = None
    goal_right: Point | None = None


class FrameResponse(BaseModel):
    video_id: str
    frame_id: int
    frame_url: str
    detections: list[Detection] = []


class FrameDetectionsResponse(BaseModel):
    video_id: str
    frame_id: int
    detections: list[Detection]


class ProcessRequest(BaseModel):
    video_id: str
    mode: AnalysisMode = "max_speed"


class Metrics(BaseModel):
    player_id: int
    top_speed_kmh: float
    avg_speed_kmh: float
    peak_acceleration_mps2: float
    avg_acceleration_mps2: float
    total_distance_m: float
    active_distance_m: float
    sprint_count: int
    sprint_distance_m: float
    usable_track_points: int = 0
    rejected_jump_count: int = 0
    confidence_score: float = 0.0


class ShotEvent(BaseModel):
    frame_id: int
    timestamp_s: float
    ball_speed_kmh: float
    contact_point: Point


class ShotMetrics(BaseModel):
    player_id: int
    peak_shot_speed_kmh: float
    avg_shot_speed_kmh: float
    shot_count: int
    best_shot: ShotEvent | None = None
    shots: list[ShotEvent] = Field(default_factory=list)
    confidence_score: float = 0.0
    usable_track_points: int = 0
    rejected_track_points: int = 0


class ResultAssets(BaseModel):
    sprint_highlights: list[str] = []


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
    assets: ResultAssets | None = None
    warnings: list[str] = []
    progress: dict | None = None
