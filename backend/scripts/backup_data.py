#!/usr/bin/env python3
"""Back up the GameSense database (and optionally the videos).

The accounts/leagues/scores live in a single SQLite file (gamesense.db) — that's
the critical, irreplaceable data, and it's tiny, so it's backed up every run
using SQLite's online-backup API (safe even while the backend is running / WAL
is active). Videos are large, so they're only mirrored when you pass --media.

Backups are timestamped; the newest --keep are retained, older ones pruned.

Usage (from the backend folder):
    py -3.12 scripts/backup_data.py                      # DB only -> C:\\GameSenseBackups
    py -3.12 scripts/backup_data.py --media              # DB + videos
    py -3.12 scripts/backup_data.py --dest "D:\\Backups" --keep 30

Run it on a schedule (Windows Task Scheduler) for real protection.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


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


def _media_root() -> Path:
    _load_backend_env()
    root = os.environ.get("GAMESENSE_MEDIA_ROOT", "").strip()
    if root:
        return Path(root)
    return Path(__file__).resolve().parents[2] / "storage"


def backup_database(media_root: Path, dest_dir: Path) -> Path | None:
    db_path = media_root / "gamesense.db"
    if not db_path.exists():
        print(f"  (no database at {db_path} — skipping DB backup)")
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = dest_dir / f"gamesense_{stamp}.db"
    # Online backup API: consistent snapshot even with the backend running.
    src = sqlite3.connect(str(db_path))
    try:
        dst = sqlite3.connect(str(out_path))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    size_kb = out_path.stat().st_size / 1024
    print(f"  DB backed up -> {out_path.name} ({size_kb:.0f} KB)")
    return out_path


def prune_db_backups(dest_dir: Path, keep: int) -> None:
    backups = sorted(dest_dir.glob("gamesense_*.db"))
    extra = backups[:-keep] if keep > 0 else []
    for old in extra:
        try:
            old.unlink()
            print(f"  pruned old backup {old.name}")
        except OSError as exc:
            print(f"  ! could not prune {old.name}: {exc}")


def mirror_media(media_root: Path, dest_dir: Path) -> None:
    """Copy video folders that are new or changed (skips the DB + WAL files)."""
    media_dest = dest_dir / "media"
    media_dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for item in media_root.iterdir():
        if not item.is_dir():
            continue  # skip gamesense.db / -wal / -shm at the root
        target = media_dest / item.name
        for src_file in item.rglob("*"):
            if not src_file.is_file():
                continue
            rel = src_file.relative_to(media_root)
            dst_file = media_dest / rel
            if dst_file.exists() and dst_file.stat().st_mtime >= src_file.stat().st_mtime:
                skipped += 1
                continue
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            copied += 1
        del target
    print(f"  media mirrored -> {media_dest}  ({copied} copied, {skipped} up-to-date)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Back up the GameSense database (and optionally videos).")
    parser.add_argument("--dest", default=r"C:\GameSenseBackups", help="Backup destination folder")
    parser.add_argument("--media", action="store_true", help="Also mirror the video files (large)")
    parser.add_argument("--keep", type=int, default=14, help="How many DB snapshots to retain")
    args = parser.parse_args()

    media_root = _media_root()
    dest_dir = Path(args.dest).expanduser()
    if not media_root.exists():
        print(f"Media root not found: {media_root}")
        sys.exit(1)

    print(f"Source : {media_root}")
    print(f"Backup : {dest_dir}\n")

    db_dir = dest_dir / "db"
    backup_database(media_root, db_dir)
    prune_db_backups(db_dir, args.keep)
    if args.media:
        mirror_media(media_root, dest_dir)
    else:
        print("  (videos not backed up — pass --media to include them)")

    print("\nBackup complete.")


if __name__ == "__main__":
    main()
