"""Configuration: biomechanical ideals, detection settings, scoring weights.

Ideal ranges are for an instep-drive power shot, drawn from soccer kicking
biomechanics literature (Lees & Nolan 1998; Kellis & Katis 2007; Barfield 1998;
Lees et al. 2010). All angles in degrees, distances in metres unless noted.
"""
from dataclasses import dataclass, field

# -----------------------------------------------------------------------------
# Real-world reference dimensions (used to convert pixels -> metres)
# -----------------------------------------------------------------------------
FULL_SIZE_GOAL_WIDTH_M = 7.32     # FIFA full-size goal
FULL_SIZE_GOAL_HEIGHT_M = 2.44
DEFAULT_PLAYER_HEIGHT_M = 1.75    # fallback scale if no goal detected
BALL_DIAMETER_M = 0.22            # size-5 ball; sanity check for scale

# -----------------------------------------------------------------------------
# Roboflow / RF-DETR
# -----------------------------------------------------------------------------
@dataclass
class DetectorConfig:
    backend: str = "hosted"               # "hosted" (inference-sdk) or "local" (rfdetr pip pkg)
    api_key: str = ""                     # or set env ROBOFLOW_API_KEY
    api_url: str = "https://serverless.roboflow.com"
    core_model_id: str = "rfdetr-base"    # RF-DETR COCO checkpoint on Roboflow hosted inference
    # Roboflow Universe: "Ball and Goalpost Detection 2" (MIT, 1.3k images, 97% mAP@50,
    # classes: ball + goalpost). Also used to backfill missed ball detections.
    goal_model_id: str = "ball-and-goalpost-detection-2/10"
    goal_class_names: tuple = ("goal", "goalpost", "goal-post", "net")
    conf_player: float = 0.5
    conf_ball: float = 0.30               # balls are small & blurry; keep low
    conf_goal: float = 0.32               # median over many samples filters flukes
    detect_stride: int = 1                # run detector every N frames (raise to 2-3 for hosted speed)
    max_retries: int = 2                  # hosted-inference retry attempts per frame
    retry_backoff_s: float = 0.6
    goal_sample_count: int = 9            # frames sampled across the clip for goal detection
    max_ball_jump_frac: float = 0.35      # reject ball candidates jumping > this frac of frame diag/frame
    parallel_requests: int = 6            # concurrent hosted-inference calls (I/O bound)
    backfill_window_s: float = 2.5        # backfill ball detections only ± this around the kick
    backfill_budget: int = 200            # hard cap on backfill inference calls per clip

# -----------------------------------------------------------------------------
# Biomechanical ideals for an instep drive
# -----------------------------------------------------------------------------
@dataclass
class Ideal:
    lo: float
    hi: float
    weight: float          # contribution to the /10 score
    unit: str = "deg"

IDEALS = {
    # Kicking-knee flexion at peak backswing: deep flexion stores elastic energy.
    "backswing_knee_flexion": Ideal(75, 115, 1.5),
    # Kicking-knee angle AT ball contact: nearly extended but not locked.
    "contact_knee_angle":     Ideal(140, 170, 1.5),
    # Ankle lock: plantarflexed, rigid foot through contact (angular change ±2 frames).
    "ankle_lock_variation":   Ideal(0, 12, 1.5),
    # Plant-foot distance: lateral distance from plant ankle to ball at contact.
    "plant_foot_distance_m":  Ideal(0.05, 0.30, 1.5, unit="m"),
    # Approach angle relative to shot line (45 deg classically optimal).
    "approach_angle":         Ideal(25, 50, 1.0),
    # Hip rotation: pelvis rotation from plant to follow-through (proxy, image plane).
    "hip_rotation":           Ideal(25, 70, 1.0),
    # Trunk lean at contact: slight forward lean over the ball. Negative = leaning back.
    "trunk_lean":             Ideal(5, 25, 1.0),
    # Follow-through height: kicking foot peak height after contact, as fraction of leg
    # length above the ground reference. High follow-through = full hip drive.
    "follow_through_height":  Ideal(0.55, 1.40, 1.0, unit="x leg length"),
}

# -----------------------------------------------------------------------------
# Shot power bands (km/h) — 2D ground-plane estimate
# -----------------------------------------------------------------------------
POWER_BANDS = [
    (0,   40,  "Developing"),
    (40,  65,  "Solid"),
    (65,  85,  "Strong"),
    (85,  200, "Elite"),
]

# Hardest recorded shots are ~130 km/h; anything beyond this means the
# px->m scale is wrong, so the estimate is discarded rather than reported.
MAX_PLAUSIBLE_SHOT_KMH = 160.0
MIN_PLAUSIBLE_SHOT_KMH = 3.0

# -----------------------------------------------------------------------------
# Event detection
# -----------------------------------------------------------------------------
@dataclass
class EventConfig:
    ball_speed_spike_ratio: float = 3.0    # strict pass; relaxed pass uses 1.8
    foot_ball_radius_frac: float = 0.10    # of frame height (strict pass)
    foot_ball_radius_frac_relaxed: float = 0.20
    post_contact_fit_frames: int = 6       # frames of ball trajectory used for power fit
    smooth_window: int = 5
    min_ball_coverage: float = 0.25        # below this, backfill with goal model's ball class
    max_interp_gap: int = 8                # longest ball-track gap (frames) to linear-fill
    direction_change_bonus: float = 0.5    # contact-candidate score bonus for trajectory kink

# -----------------------------------------------------------------------------
# Annotation styling (BGR)
# -----------------------------------------------------------------------------
@dataclass
class Style:
    skeleton_color: tuple = (80, 220, 80)
    joint_color: tuple = (255, 255, 255)
    ball_color: tuple = (0, 200, 255)
    ball_trail_color: tuple = (0, 140, 255)
    goal_color: tuple = (255, 120, 0)
    angle_good: tuple = (90, 220, 90)
    angle_bad: tuple = (60, 60, 230)
    panel_bg: tuple = (28, 24, 20)
    accent: tuple = (60, 180, 255)
    text: tuple = (240, 240, 240)

@dataclass
class PipelineConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    events: EventConfig = field(default_factory=EventConfig)
    style: Style = field(default_factory=Style)
    goal_width_m: float = FULL_SIZE_GOAL_WIDTH_M
    goal_height_m: float = FULL_SIZE_GOAL_HEIGHT_M
    player_height_m: float = DEFAULT_PLAYER_HEIGHT_M
    kicking_foot: str = "auto"            # "auto" | "left" | "right"
