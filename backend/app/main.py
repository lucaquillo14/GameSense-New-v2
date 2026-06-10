from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from uuid import uuid4

_team_calibration_lock = Lock()
_cv_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cv-worker")

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.models import (
    FrameDetectionsResponse,
    PitchSetupRequest,
    FrameResponse,
    PlayerSelectionRequest,
    ProcessRequest,
    UploadResponse,
    VideoResult,
)
from app.services.cv_pipeline import get_cv_pipeline
from app.services.processing import process_video
from app.services.upload_limits import MAX_UPLOAD_BYTES, MAX_VIDEO_DURATION_S
from app.services.team_classification import TeamTemplates
from app.services.storage import (
    MEDIA_ROOT,
    create_video_record,
    get_video_record,
    save_frame,
    save_setup_frame,
    save_upload,
    update_video_record,
    video_dir,
    video_metadata,
)

app = FastAPI(title="GameSense AI Phase 1 API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=MEDIA_ROOT), name="media")


@app.on_event("startup")
def warmup_cv_pipeline() -> None:
    get_cv_pipeline()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _calibrate_team_classification(video_id: str) -> None:
    with _team_calibration_lock:
        record = get_video_record(video_id)
        if not record or record.get("team_classification"):
            return
        video_path = Path(record["video_path"])
        templates = get_cv_pipeline().calibrate_team_templates(video_path)
        record = get_video_record(video_id)
        if not record:
            return
        record["team_classification"] = templates.to_dict()
        record["progress"] = {
            "stage": "team_ready",
            "percent": 3,
            "message": "Team colours calibrated",
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


@app.post("/upload-video", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()) -> UploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov"}:
        raise HTTPException(status_code=400, detail="Only mp4 and mov uploads are supported.")

    video_id = str(uuid4())
    video_path = await save_upload(video_id, file, suffix)

    if video_path.stat().st_size > MAX_UPLOAD_BYTES:
        video_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )

    metadata = video_metadata(video_path)
    duration_s = float(metadata.get("duration_s") or 0.0)
    if duration_s > MAX_VIDEO_DURATION_S:
        video_path.unlink(missing_ok=True)
        try:
            import shutil
            shutil.rmtree(video_dir(video_id), ignore_errors=True)
        except OSError:
            pass
        raise HTTPException(
            status_code=400,
            detail=(
                f"This clip is {duration_s:.1f} seconds long. "
                f"Maximum allowed duration is {int(MAX_VIDEO_DURATION_S)} seconds."
            ),
        )

    source_url = f"/media/{video_id}/{video_path.name}"

    create_video_record(
        video_id,
        {
            "video_id": video_id,
            "filename": file.filename,
            "status": "uploaded",
            "video_path": str(video_path),
            "source_url": source_url,
            "setup_frame": None,
            "setup_frame_id": 0,
            "video_metadata": metadata,
            "target_player": None,
            "pitch_setup": None,
            "results": None,
            "assets": None,
            "warnings": [],
            "progress": {"stage": "uploaded", "percent": 1, "message": "Video uploaded"},
        },
    )
    background_tasks.add_task(_extract_setup_frame, video_id, video_path)

    return UploadResponse(
        video_id=video_id,
        filename=file.filename or video_path.name,
        setup_frame_url=f"/media/{video_id}/setup-frame.jpg",
        source_url=source_url,
        metadata=metadata,
    )


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
    cached = record.get("last_detections") or {}
    if cached.get("frame_id") == frame_id:
        detections = cached.get("detections") or []
    else:
        detections = _detect_frame(video_path, frame_id, team_templates)
        record["last_detections"] = {"frame_id": frame_id, "detections": detections}
        update_video_record(video_id, record)

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
def process(payload: ProcessRequest) -> dict[str, str]:
    record = get_video_record(payload.video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found.")
    if not record.get("target_player"):
        raise HTTPException(status_code=400, detail="Target player must be selected before processing.")

    record["status"] = "processing"
    record["mode"] = payload.mode
    record["warnings"] = []
    record["results"] = None
    record["assets"] = None
    record["progress"] = {"stage": "calibration", "percent": 2, "message": "Calibrating pitch"}
    update_video_record(payload.video_id, record)
    _cv_executor.submit(process_video, payload.video_id)

    return {"video_id": payload.video_id, "status": "processing"}


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
