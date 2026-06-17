import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


_load_env_file()
if os.environ.get("ROBOFLOW_API_KEY", "").strip():
    print("[GameSense] Roboflow API key loaded from backend/.env")
else:
    print("[GameSense] WARNING: ROBOFLOW_API_KEY missing in backend/.env")
from threading import Lock
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_team_calibration_lock = Lock()
_cv_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cv-worker")
_frame_detection_cache: dict[tuple[str, int], list[dict]] = {}

from app.models import (
    FrameDetectionsResponse,
    PitchSetupRequest,
    FrameResponse,
    PlayerSelectionRequest,
    ProcessRequest,
    VideoResult,
)
from app.services.roboflow_inference import inference_available, require_roboflow_api_key

try:
    from app.services.cv_pipeline import get_ball_model, get_cv_pipeline, get_player_model
except ImportError as exc:
    print(f"[GameSense] CV pipeline import skipped: {exc}")
    get_ball_model = None
    get_cv_pipeline = None
    get_player_model = None

from app.services.processing import process_video
from app.services.team_classification import TeamTemplates
from app.services.storage import (
    MEDIA_ROOT,
    get_video_record,
    save_frame,
    save_setup_frame,
    update_video_record,
    video_dir,
    video_metadata,
)
from app import auth, db
from app.services import scoring, subscriptions
from app.social_routes import router as social_router
from app.billing_routes import router as billing_router

app = FastAPI(title="GameSense AI Phase 1 API", version="0.1.0")


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 250 * 1024 * 1024:
            return JSONResponse({"detail": "File too large. Maximum is 250MB."}, status_code=413)
        return await call_next(request)


# Allowed frontend origins come from GAMESENSE_ALLOWED_ORIGINS (comma-separated)
# so production can point at the real domain without code changes.
_allowed_origins = [
    origin.strip()
    for origin in os.environ.get(
        "GAMESENSE_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(MaxBodySizeMiddleware)

MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=MEDIA_ROOT), name="media")

# Accounts, leaderboard, and leagues.
app.include_router(social_router)
# Membership tiers, Stripe checkout, and billing.
app.include_router(billing_router)


@app.on_event("startup")
def init_database() -> None:
    db.init_db()
    print("[GameSense] Leaderboard/leagues database ready.")


@app.on_event("startup")
def warmup_cv_pipeline() -> None:
    if not inference_available():
        print(f"[GameSense] CV pipeline warmup skipped — inference-sdk missing for {sys.executable}")
        return
    if get_player_model is None:
        print("[GameSense] CV pipeline warmup skipped — cv_pipeline could not be imported.")
        return
    try:
        require_roboflow_api_key()
        get_player_model()
        get_ball_model()
        get_cv_pipeline()
        print("[GameSense] CV pipeline warmed up.")
    except Exception as exc:
        print(f"[GameSense] CV pipeline warmup skipped: {exc}")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "GameSense AI API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "python": sys.executable,
        "inference_sdk": inference_available(),
        "roboflow_api_key_set": bool(os.environ.get("ROBOFLOW_API_KEY", "").strip()),
        "shooting_technique_engine": "rfdetr-mediapipe-v2-parallel",
        "shooting_workflow_id": False,
    }


def _calibrate_team_classification(video_id: str) -> None:
    with _team_calibration_lock:
        record = get_video_record(video_id)
        if not record or record.get("team_classification"):
            return
        video_path = Path(record["video_path"])

        def on_progress(done: int, total: int) -> None:
            if done % 5 != 0 and done != total:
                return
            current = get_video_record(video_id)
            if not current:
                return
            current["progress"] = {
                "stage": "team_calibration",
                "percent": 2,
                "message": f"Detecting team colours — frame {done} of {total}",
                "setup_percent": int(done / max(total, 1) * 100),
            }
            update_video_record(video_id, current)

        templates = get_cv_pipeline().calibrate_team_templates(video_path, progress_callback=on_progress)
        record = get_video_record(video_id)
        if not record:
            return
        record["team_classification"] = templates.to_dict()
        record["progress"] = {
            "stage": "team_ready",
            "percent": 3,
            "message": "Team colours calibrated",
            "setup_percent": 100,
        }
        update_video_record(video_id, record)


def _ensure_team_classification(video_id: str, record: dict, video_path: Path) -> TeamTemplates:
    cached = record.get("team_classification")
    if cached:
        return TeamTemplates.from_dict(cached)

    _calibrate_team_classification(video_id)
    record = get_video_record(video_id) or record
    cached = record.get("team_classification")
    if cached:
        return TeamTemplates.from_dict(cached)

    templates = get_cv_pipeline().calibrate_team_templates(video_path)
    record["team_classification"] = templates.to_dict()
    update_video_record(video_id, record)
    return templates


def _detect_frame(
    video_path: Path,
    frame_id: int,
    team_templates: TeamTemplates | None = None,
    assign_player_ids: bool = False,
) -> list[dict]:
    return get_cv_pipeline().detect_frame_objects(
        video_path,
        frame_id,
        team_templates,
        assign_player_ids=assign_player_ids,
    )


def _select_detection(detections: list[dict], click: dict, detection_id: str | None = None) -> dict | None:
    if detection_id:
        for detection in detections:
            if detection["id"] == detection_id:
                return detection

    player_detections = [
        detection for detection in detections
        if detection.get("team") in {"team_a", "team_b"} or detection.get("label") in {"team_a", "team_b", "player"}
    ]
    containing = []
    for detection in player_detections:
        bbox = detection["bbox"]
        if bbox["x"] <= click["x"] <= bbox["x"] + bbox["width"] and bbox["y"] <= click["y"] <= bbox["y"] + bbox["height"]:
            containing.append(detection)

    candidates = containing or player_detections
    if not candidates:
        return None

    def score(detection: dict) -> float:
        bbox = detection["bbox"]
        center_x = bbox["x"] + bbox["width"] / 2
        center_y = bbox["y"] + bbox["height"] / 2
        distance = (center_x - click["x"]) ** 2 + (center_y - click["y"]) ** 2
        return distance - detection["confidence"] * 500.0

    return min(candidates, key=score)


def _extract_setup_frame(video_id: str, video_path: Path) -> None:
    try:
        setup_frame = save_setup_frame(video_id, video_path)
        record = get_video_record(video_id)
        if record:
            record["setup_frame"] = setup_frame
            update_video_record(video_id, record)
        _cv_executor.submit(_calibrate_team_classification, video_id)
    except Exception:
        pass


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...), authorization: str | None = Header(default=None)):
    owner_id = auth.user_id_from_header(authorization)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov"}:
        raise HTTPException(status_code=400, detail="Only .mp4 and .mov files are supported.")

    video_id = str(uuid4())
    vdir = MEDIA_ROOT / video_id
    vdir.mkdir(parents=True, exist_ok=True)
    video_path = vdir / f"source{suffix}"

    with video_path.open("wb") as f:
        while chunk := await file.read(4 * 1024 * 1024):
            f.write(chunk)

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration_s = frame_count / fps if fps > 0 else 0.0
    metadata = {"fps": fps, "frame_count": frame_count, "duration_s": duration_s, "width": width, "height": height}

    record = {
        "video_id": video_id,
        "owner_id": owner_id,
        "filename": file.filename,
        "status": "uploaded",
        "video_path": str(video_path),
        "source_url": f"/media/{video_id}/source{suffix}",
        "setup_frame": None,
        "setup_frame_id": 0,
        "video_metadata": metadata,
        "target_player": None,
        "pitch_setup": None,
        "mode": "max_speed",
        "results": None,
        "assets": None,
        "warnings": [],
        "team_colors": None,
        "progress": {"stage": "uploaded", "percent": 0, "message": "Video uploaded"},
    }

    record_file = vdir / "record.json"
    record_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

    # Save the setup frame immediately and start team-colour calibration in
    # the background so the setup page is ready (or visibly progressing) by
    # the time the user lands on it.
    _extract_setup_frame(video_id, video_path)

    return {
        "video_id": video_id,
        "filename": file.filename,
        "setup_frame_url": f"/media/{video_id}/setup-frame.jpg",
        "source_url": f"/media/{video_id}/source{suffix}",
        "metadata": metadata,
    }


@app.get("/frames/{video_id}/{frame_id}", response_model=FrameResponse)
def get_frame(video_id: str, frame_id: int) -> FrameResponse:
    record = get_video_record(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found.")

    metadata = record.get("video_metadata") or video_metadata(Path(record["video_path"]))
    record["video_metadata"] = metadata
    frame_count = int(metadata.get("frame_count") or 0)
    if frame_id < 0 or (frame_count and frame_id >= frame_count):
        raise HTTPException(status_code=400, detail="Frame is outside the video range.")

    video_path = Path(record["video_path"])
    frame_url = save_frame(video_id, video_path, frame_id)
    team_templates = _ensure_team_classification(video_id, record, video_path)
    detections = _detect_frame(video_path, frame_id, team_templates)
    record["setup_frame"] = frame_url
    record["setup_frame_id"] = frame_id
    record["last_detections"] = {"frame_id": frame_id, "detections": detections}
    update_video_record(video_id, record)
    return FrameResponse(video_id=video_id, frame_id=frame_id, frame_url=frame_url, detections=detections)


@app.get("/frames/{video_id}/{frame_id}/detections", response_model=FrameDetectionsResponse)
def get_frame_detections(video_id: str, frame_id: int) -> FrameDetectionsResponse:
    record = get_video_record(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found.")

    metadata = record.get("video_metadata") or video_metadata(Path(record["video_path"]))
    frame_count = int(metadata.get("frame_count") or 0)
    if frame_id < 0 or (frame_count and frame_id >= frame_count):
        raise HTTPException(status_code=400, detail="Frame is outside the video range.")

    video_path = Path(record["video_path"])
    team_templates = _ensure_team_classification(video_id, record, video_path)
    cache_key = (video_id, frame_id)
    if cache_key in _frame_detection_cache:
        detections = _frame_detection_cache[cache_key]
    else:
        detections = _detect_frame(video_path, frame_id, team_templates)
        _frame_detection_cache[cache_key] = detections

    return FrameDetectionsResponse(video_id=video_id, frame_id=frame_id, detections=detections)


@app.post("/select-player")
def select_player(payload: PlayerSelectionRequest) -> dict[str, object]:
    record = get_video_record(payload.video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found.")

    click_payload = payload.click.model_dump()
    selected_detection = None
    bbox_payload = payload.bbox.model_dump() if payload.bbox else None
    video_path = Path(record["video_path"])
    team_templates = _ensure_team_classification(payload.video_id, record, video_path)
    if not bbox_payload:
        detections = _detect_frame(video_path, payload.frame_id, team_templates)
        selected_detection = _select_detection(detections, click_payload, payload.detection_id)
        bbox_payload = selected_detection["bbox"] if selected_detection else None

    player_id_target = payload.player_id_target or 1
    record["target_player"] = {
        "player_id_target": player_id_target,
        "click": click_payload,
        "frame_id": payload.frame_id,
        "detection_id": payload.detection_id or (selected_detection or {}).get("id"),
        "bbox": bbox_payload,
        "confidence": (selected_detection or {}).get("confidence", 0.5 if bbox_payload else 0.25),
        "team": (selected_detection or {}).get("team"),
        "team_label": (selected_detection or {}).get("team_label"),
        "player_id": (selected_detection or {}).get("player_id"),
    }
    record["status"] = "player_selected"
    record["progress"] = {"stage": "player_selected", "percent": 5, "message": "Target player selected"}
    update_video_record(payload.video_id, record)

    return {"video_id": payload.video_id, "player_id_target": player_id_target}


@app.post("/set-pitch-polygon")
def set_pitch_polygon(payload: PitchSetupRequest) -> dict[str, object]:
    record = get_video_record(payload.video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found.")

    if 0 < len(payload.pitch_polygon) < 4:
        raise HTTPException(
            status_code=400,
            detail="Pitch polygon requires at least 4 points, or leave it empty to attempt pitch-line auto-detection.",
        )

    record["pitch_setup"] = payload.model_dump()
    record["status"] = "setup_complete"
    record["progress"] = {"stage": "setup_complete", "percent": 8, "message": "Calibration saved"}
    update_video_record(payload.video_id, record)

    return {"video_id": payload.video_id, "pitch_polygon_points": len(payload.pitch_polygon)}


@app.post("/process-video")
def process(payload: ProcessRequest, authorization: str | None = Header(default=None)) -> dict[str, str]:
    record = get_video_record(payload.video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found.")
    if payload.mode != "shooting_technique" and not record.get("target_player"):
        raise HTTPException(status_code=400, detail="Target player must be selected before processing.")

    # Server-side membership enforcement: analyses require a signed-in user and
    # are metered against the user's tier limits.
    user_id = auth.user_id_from_header(authorization) or record.get("owner_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Sign in to run an analysis.")

    quota = subscriptions.check_quota(user_id, payload.mode)
    if not quota["allowed"]:
        # 402 Payment Required — the frontend detects this to show an upgrade prompt.
        raise HTTPException(status_code=402, detail=quota["reason"])

    # Ensure the clip is owned by the processing user (older anonymous uploads).
    if not record.get("owner_id"):
        record["owner_id"] = user_id

    subscriptions.record_analysis(user_id, payload.mode, payload.video_id)

    record["status"] = "processing"
    record["mode"] = payload.mode
    record["player_height_cm"] = payload.player_height_cm
    record["warnings"] = []
    record["results"] = None
    record["shooting_result"] = None
    record["assets"] = None
    if payload.mode == "shooting_technique":
        record["progress"] = {"stage": "detection", "percent": 10, "message": "Starting shooting technique analysis"}
    else:
        record["progress"] = {"stage": "calibration", "percent": 2, "message": "Calibrating pitch"}
    update_video_record(payload.video_id, record)
    _cv_executor.submit(process_video, payload.video_id)

    return {"video_id": payload.video_id, "status": "processing"}


@app.get("/shooting-result/{video_id}")
def shooting_result(video_id: str) -> dict:
    record = get_video_record(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found.")
    return record.get("shooting_result") or {}


@app.get("/results/{video_id}", response_model=VideoResult)
def results(video_id: str) -> VideoResult:
    record = get_video_record(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found.")
    if not record.get("video_metadata") and record.get("video_path"):
        record["video_metadata"] = video_metadata(Path(record["video_path"]))
        update_video_record(video_id, record)
    if not record.get("source_url") and record.get("video_path"):
        video_path = Path(record["video_path"])
        record["source_url"] = f"/media/{video_id}/{video_path.name}"
        update_video_record(video_id, record)
    detections_file = MEDIA_ROOT / video_id / "detections.json"
    if detections_file.exists():
        assets = dict(record.get("assets") or {})
        assets["detections_json"] = f"/media/{video_id}/detections.json"
        if record.get("assets") != assets:
            record["assets"] = assets
            update_video_record(video_id, record)

    # Award leaderboard points once a result is complete and the clip belongs
    # to a signed-in user. Idempotent — scored at most once per video.
    if record.get("status") == "complete" and record.get("owner_id"):
        try:
            scoring.record_upload_score(record["owner_id"], record)
        except Exception as exc:  # never break results delivery over scoring
            print(f"[GameSense] Scoring skipped for {video_id}: {exc}")

    # Membership gating: heatmaps are a Pro+ feature. Strip the URLs (so they
    # can't be fetched) and flag the locked feature for an upgrade prompt when
    # the clip owner is on a tier without access.
    locked: list[str] = []
    owner_id = record.get("owner_id")
    if owner_id and not subscriptions.has_feature(owner_id, "heatmaps"):
        assets = record.get("assets")
        if isinstance(assets, dict) and (assets.get("movement_heatmap") or assets.get("touch_heatmap")):
            assets = dict(assets)
            assets["movement_heatmap"] = None
            assets["touch_heatmap"] = None
            assets["position_heatmap"] = None
            assets["speed_heatmap"] = None
            record = {**record, "assets": assets}
            locked.append("heatmaps")
    record = {**record, "locked_features": locked}

    return VideoResult.model_validate(record)


@app.get("/preview/{video_id}")
def preview_frame(video_id: str) -> FileResponse:
    preview_path = MEDIA_ROOT / video_id / "preview-frame.jpg"
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview frame not ready yet.")
    return FileResponse(
        preview_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )
