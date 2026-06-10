from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.services.storage import video_dir

PREVIEW_MAX_WIDTH = 640
PREVIEW_JPEG_QUALITY = 72


def save_preview_frame(
    video_id: str,
    frame_bgr: np.ndarray,
    player_boxes: list[dict],
    ball_box: tuple[float, float, float, float] | None = None,
) -> str:
    preview = frame_bgr.copy()
    for entry in player_boxes:
        x, y, w, h = [int(round(v)) for v in entry["bbox"]]
        color = entry.get("color_bgr", (59, 130, 246))
        cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
        label = entry.get("label", "Player")
        cv2.putText(
            preview,
            label,
            (x, max(y - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    if ball_box is not None:
        x, y, w, h = [int(round(v)) for v in ball_box]
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(preview, "Ball", (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    height, width = preview.shape[:2]
    if width > PREVIEW_MAX_WIDTH:
        scale = PREVIEW_MAX_WIDTH / float(width)
        preview = cv2.resize(
            preview,
            (PREVIEW_MAX_WIDTH, max(int(round(height * scale)), 1)),
            interpolation=cv2.INTER_AREA,
        )

    output_path = video_dir(video_id) / "preview-frame.jpg"
    cv2.imwrite(str(output_path), preview, [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_JPEG_QUALITY])
    return str(output_path)
