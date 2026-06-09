import json
import os
import tempfile
from pathlib import Path

import cv2
from fastapi import UploadFile

ROOT = Path(__file__).resolve().parents[3]
MEDIA_ROOT = ROOT / "storage"


def video_dir(video_id: str) -> Path:
    path = MEDIA_ROOT / video_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def record_path(video_id: str) -> Path:
    return video_dir(video_id) / "record.json"


async def save_upload(video_id: str, file: UploadFile, suffix: str) -> Path:
    path = video_dir(video_id) / f"source{suffix}"
    with path.open("wb") as output:
        while chunk := await file.read(1024 * 1024):
            output.write(chunk)
    return path


def save_setup_frame(video_id: str, video_path: Path) -> str:
    return save_frame(video_id, video_path, 0, "setup-frame.jpg")


def save_frame(video_id: str, video_path: Path, frame_id: int, filename: str | None = None) -> str:
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count > 0:
        frame_id = min(max(frame_id, 0), frame_count - 1)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_id} from video.")

    frame_path = video_dir(video_id) / (filename or f"frame-{frame_id}.jpg")
    cv2.imwrite(str(frame_path), frame)
    return f"/media/{video_id}/{frame_path.name}"


def video_metadata(video_path: Path) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    duration_s = frame_count / fps if fps > 0 else 0.0
    return {
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "duration_s": round(duration_s, 3),
        "width": width,
        "height": height,
    }


def create_video_record(video_id: str, record: dict) -> None:
    update_video_record(video_id, record)


def get_video_record(video_id: str) -> dict | None:
    path = record_path(video_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def update_video_record(video_id: str, record: dict) -> None:
    path = record_path(video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="record-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
            temp_file.write(json.dumps(record, indent=2))
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_json(video_id: str, filename: str, payload: dict) -> str:
    path = video_dir(video_id) / filename
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return f"/media/{video_id}/{filename}"
