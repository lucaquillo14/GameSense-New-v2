#!/usr/bin/env python3
"""Upload extracted frames to your Roboflow project for labeling + training.

Bulk-uploads a folder of images to your Roboflow dataset via the official
Roboflow Python SDK. Run it on YOUR machine — it needs the frames on disk and
internet access. Your SAM3 Rapid workflow then auto-labels the new images into
the review queue.

One-time setup:
    py -3.12 -m pip install roboflow

Usage (from the backend folder):
    py -3.12 scripts/upload_to_roboflow.py --folder "C:\\training_frames"

The Roboflow API key is read from backend/.env automatically.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Your workspace + dataset (the dataset the Rapid/SAM3 workflow feeds).
WORKSPACE = "lucass-workspace-fn5cc"
PROJECT = "find-player-goal-and-more-qomuc"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _load_backend_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _upload_one(project, image_path: str, batch_name: str) -> None:
    """Upload a single image, tolerating SDK versions without batch_name."""
    try:
        project.upload(image_path, batch_name=batch_name)
    except TypeError:
        project.upload(image_path)


def main() -> None:
    _load_backend_env()
    parser = argparse.ArgumentParser(description="Upload frames to a Roboflow project.")
    parser.add_argument("--folder", default=r"C:\training_frames", help="Folder of images to upload")
    parser.add_argument("--workspace", default=WORKSPACE)
    parser.add_argument("--project", default=PROJECT, help="Roboflow project/dataset id")
    parser.add_argument("--batch-name", default="extracted-frames")
    parser.add_argument("--limit", type=int, default=0, help="Upload at most N images (0 = all)")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print("ROBOFLOW_API_KEY is not set (add it to backend/.env).")
        sys.exit(1)

    folder = Path(args.folder).expanduser()
    if not folder.exists():
        print(f"Folder not found: {folder}")
        sys.exit(1)
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        print(f"No .jpg/.png images found in {folder}")
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Roboflow SDK not installed. Run:  py -3.12 -m pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    try:
        project = rf.workspace(args.workspace).project(args.project)
    except Exception as exc:
        print(f"Could not open project {args.workspace}/{args.project}: {exc}")
        print("Double-check the project id on app.roboflow.com (Projects list).")
        sys.exit(1)

    print(f"Uploading {len(images)} image(s) to {args.workspace}/{args.project} ...")
    uploaded = 0
    failed = 0
    for index, image in enumerate(images, 1):
        try:
            _upload_one(project, str(image), args.batch_name)
            uploaded += 1
        except Exception as exc:
            failed += 1
            print(f"  ! {image.name}: {exc}")
        if index % 25 == 0:
            print(f"  {index}/{len(images)} ...")

    print(f"\nDone: {uploaded} uploaded, {failed} failed.")
    print("Next in Roboflow: review/approve the SAM3 auto-labels (focus on the "
          "'player' class), then Generate a Version and Train a YOLOv8n/s model.")


if __name__ == "__main__":
    main()
