"""SQLite data layer for users, uploads/scores, and leagues.

Uses the Python standard library `sqlite3` only — no extra dependencies.
The database file lives next to the media storage so it travels with the
existing `storage/` directory.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.services.storage import MEDIA_ROOT

DB_PATH = Path(MEDIA_ROOT) / "gamesense.db"

_local = threading.local()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def get_conn() -> sqlite3.Connection:
    """Return a thread-local connection (sqlite connections aren't shareable)."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _connect()
        _local.conn = conn
    return conn


@contextmanager
def cursor() -> Iterator[sqlite3.Cursor]:
    conn = get_conn()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    email         TEXT UNIQUE NOT NULL,
    display_name  TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    avatar_url    TEXT,
    created_at    TEXT NOT NULL
);

-- One row per completed analysis tied to a user. Primary key is the
-- video_id so scoring is naturally idempotent (a video is scored once).
CREATE TABLE IF NOT EXISTS uploads (
    video_id        TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    mode            TEXT,
    filename        TEXT,
    max_speed_kmh   REAL DEFAULT 0,
    shot_power_kmh  REAL DEFAULT 0,
    technique_score REAL DEFAULT 0,
    points          INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_uploads_user ON uploads (user_id);

CREATE TABLE IF NOT EXISTS leagues (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    owner_id    TEXT NOT NULL,
    invite_code TEXT UNIQUE NOT NULL,
    created_at  TEXT NOT NULL,
    FOREIGN KEY (owner_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS league_members (
    league_id TEXT NOT NULL,
    user_id   TEXT NOT NULL,
    joined_at TEXT NOT NULL,
    PRIMARY KEY (league_id, user_id),
    FOREIGN KEY (league_id) REFERENCES leagues (id) ON DELETE CASCADE,
    FOREIGN KEY (user_id)   REFERENCES users (id) ON DELETE CASCADE
);

-- One row per follow relationship (one-directional).
CREATE TABLE IF NOT EXISTS follows (
    follower_id  TEXT NOT NULL,
    following_id TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (follower_id, following_id),
    FOREIGN KEY (follower_id)  REFERENCES users (id) ON DELETE CASCADE,
    FOREIGN KEY (following_id) REFERENCES users (id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_follows_following ON follows (following_id);
"""


def init_db() -> None:
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()
    _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply lightweight, idempotent migrations for databases created before
    newer columns/tables existed."""
    cur = conn.cursor()
    try:
        cur.execute("PRAGMA table_info(users)")
        user_columns = {row[1] for row in cur.fetchall()}
        if "avatar_url" not in user_columns:
            cur.execute("ALTER TABLE users ADD COLUMN avatar_url TEXT")
        conn.commit()
    finally:
        cur.close()
