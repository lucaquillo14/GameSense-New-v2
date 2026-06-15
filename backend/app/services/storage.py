import json
import os
import tempfile
import threading
import time
from pathlib import Path

import cv2
from fastapi import UploadFile

ROOT = Path(__file__).resolve().parents[3]
# OneDrive/Dropbox sync clients lock files mid-sync and cause intermittent
# [Errno 13] errors when the project lives in a synced folder. Set
# GAMESENSE_MEDIA_ROOT in backend/.env to move storage somewhere unsynced,
# e.g. GAMESENSE_MEDIA_ROOT=C:\GameSenseStorage
_env_root = os.environ.get("GAMESENSE_MEDIA_ROOT", "").strip()
MEDIA_ROOT = Path(_env_root) if _env_root else (ROOT / "storage")
_record_write_lock = threading.Lock()

_LOCK_RETRIES = 40            # ~10 s total — OneDrive sync locks can last seconds
_LOCK_BACKOFF_S = 0.25


def _retry_locked(operation, *, retries: int = _LOCK_RETRIES):
    """Run a filesystem operation, riding out transient sync-client locks."""
    last_exc: PermissionError | None = None
    for attempt in range(retries):
        try:
            return operation()
        except PermissionError as exc:
            last_exc = exc
            time.sleep(min(_LOCK_BACKOFF_S * (attempt + 1), 1.0))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry_locked: no operation result")


def video_dir(video_id: str) -> Path:
    path = MEDIA_ROOT / video_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def record_path(video_id: str) -> Path:
    return video_dir(video_id) / "record.json"


async def save_upload(video_id: str, file: UploadFile, suffix: str) -> Path:
    path = video_dir(video_id) / f"source{suffix}"
    with path.open("wb") as output:
        while chunk := await file.read(4 * 1024 * 1024):
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
    return _retry_locked(lambda: json.loads(path.read_text(encoding="utf-8")))


def update_video_record(video_id: str, record: dict) -> None:
    path = record_path(video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, indent=2)

    with _record_write_lock:
        fd, temp_name = tempfile.mkstemp(prefix="record-", suffix=".json", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
                temp_file.write(payload)
                temp_file.flush()
                os.fsync(temp_file.fileno())

            try:
                _retry_locked(lambda: os.replace(temp_name, path))
            except PermissionError:
                # Final fallback: direct write (also retried).
                _retry_locked(lambda: path.write_text(payload, encoding="utf-8"))
        finally:
            if os.path.exists(temp_name):
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass


def write_json(video_id: str, filename: str, payload: dict, *, compact: bool = False) -> str:
    path = video_dir(video_id) / filename
    if compact:
        text = json.dumps(payload, separators=(",", ":"))
    else:
        text = json.dumps(payload, indent=2)
    _retry_locked(lambda: path.write_text(text, encoding="utf-8"))
    return f"/media/{video_id}/{filename}"
