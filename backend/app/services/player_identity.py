from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.services.team_classification import (
    TeamTemplates,
    classify_team,
    extract_shirt_histogram,
    histogram_distance,
)

COLOR_REID_THRESHOLD = 0.38
NEW_ID_CONFIRMATION_FRAMES = 5
SPATIAL_GATE_HEIGHT_MULTIPLIER = 2.0


@dataclass
class PlayerGalleryEntry:
    stable_id: str
    team: str
    appearance_histogram: np.ndarray
    last_bbox: tuple[float, float, float, float]
    last_center: tuple[float, float]
    last_velocity: tuple[float, float] = (0.0, 0.0)
    last_frame_id: int = 0
    last_time_s: float = 0.0


@dataclass
class PendingPlayer:
    team: str
    appearance_histogram: np.ndarray
    bbox: tuple[float, float, float, float]
    byte_track_id: int | None
    frames_seen: int = 1
    first_frame_id: int = 0


@dataclass
class PlayerIdentityManager:
    team_templates: TeamTemplates
    gallery: dict[str, PlayerGalleryEntry] = field(default_factory=dict)
    byte_track_map: dict[int, str] = field(default_factory=dict)
    pending_players: dict[int, PendingPlayer] = field(default_factory=dict)
    team_counters: dict[str, int] = field(default_factory=lambda: {"team_a": 0, "team_b": 0})

    def _predict_center(self, entry: PlayerGalleryEntry, frame_id: int, time_s: float) -> tuple[float, float]:
        dt = max(time_s - entry.last_time_s, 1e-6)
        frame_dt = max(frame_id - entry.last_frame_id, 1)
        velocity = entry.last_velocity
        if frame_dt > 0 and entry.last_time_s > 0:
            scale = min(dt * 30.0, frame_dt)
        else:
            scale = 1.0
        return (
            entry.last_center[0] + velocity[0] * scale,
            entry.last_center[1] + velocity[1] * scale,
        )

    def _bbox_center(self, bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        x, y, w, h = bbox
        return (x + w / 2.0, y + h / 2.0)

    def _spatial_gate(
        self,
        entry: PlayerGalleryEntry,
        bbox: tuple[float, float, float, float],
        frame_id: int,
        time_s: float,
    ) -> bool:
        predicted = self._predict_center(entry, frame_id, time_s)
        center = self._bbox_center(bbox)
        distance = ((center[0] - predicted[0]) ** 2 + (center[1] - predicted[1]) ** 2) ** 0.5
        max_jump = max(entry.last_bbox[3] * SPATIAL_GATE_HEIGHT_MULTIPLIER, 40.0)
        return distance <= max_jump

    def _color_gate(self, entry: PlayerGalleryEntry, histogram: np.ndarray) -> bool:
        return histogram_distance(histogram, entry.appearance_histogram) < COLOR_REID_THRESHOLD

    def _allocate_stable_id(self, team: str) -> str:
        prefix = "A" if team == "team_a" else "B"
        self.team_counters[team] += 1
        return f"{prefix}{self.team_counters[team]}"

    def _update_entry(
        self,
        entry: PlayerGalleryEntry,
        histogram: np.ndarray,
        bbox: tuple[float, float, float, float],
        frame_id: int,
        time_s: float,
    ) -> None:
        center = self._bbox_center(bbox)
        dt = max(time_s - entry.last_time_s, 1e-6)
        if entry.last_time_s > 0:
            entry.last_velocity = (
                (center[0] - entry.last_center[0]) / dt,
                (center[1] - entry.last_center[1]) / dt,
            )
        entry.last_center = center
        entry.last_bbox = bbox
        entry.last_frame_id = frame_id
        entry.last_time_s = time_s
        entry.appearance_histogram = entry.appearance_histogram * 0.72 + histogram * 0.28

    def _register_player(
        self,
        team: str,
        histogram: np.ndarray,
        bbox: tuple[float, float, float, float],
        frame_id: int,
        time_s: float,
        byte_track_id: int | None = None,
    ) -> str:
        stable_id = self._allocate_stable_id(team)
        center = self._bbox_center(bbox)
        self.gallery[stable_id] = PlayerGalleryEntry(
            stable_id=stable_id,
            team=team,
            appearance_histogram=histogram.copy(),
            last_bbox=bbox,
            last_center=center,
            last_frame_id=frame_id,
            last_time_s=time_s,
        )
        if byte_track_id is not None:
            self.byte_track_map[byte_track_id] = stable_id
        return stable_id

    def _match_gallery(
        self,
        team: str,
        histogram: np.ndarray,
        bbox: tuple[float, float, float, float],
        frame_id: int,
        time_s: float,
    ) -> str | None:
        best_id: str | None = None
        best_distance = float("inf")
        for stable_id, entry in self.gallery.items():
            if entry.team != team:
                continue
            if not self._color_gate(entry, histogram):
                continue
            if not self._spatial_gate(entry, bbox, frame_id, time_s):
                continue
            distance = histogram_distance(histogram, entry.appearance_histogram)
            if distance < best_distance:
                best_distance = distance
                best_id = stable_id
        return best_id

    def register_immediate(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        frame_id: int,
        time_s: float,
        byte_track_id: int | None = None,
        team: str | None = None,
    ) -> str | None:
        bbox_dict = {"x": bbox[0], "y": bbox[1], "width": bbox[2], "height": bbox[3]}
        histogram = extract_shirt_histogram(frame, bbox_dict)
        if histogram is None:
            return None

        resolved_team = team
        if resolved_team is None:
            classified = classify_team(histogram, self.team_templates)
            if classified == "referee":
                return None
            resolved_team = classified

        matched_id = self._match_gallery(resolved_team, histogram, bbox, frame_id, time_s)
        if matched_id is not None:
            entry = self.gallery[matched_id]
            self._update_entry(entry, histogram, bbox, frame_id, time_s)
            if byte_track_id is not None:
                self.byte_track_map[byte_track_id] = matched_id
            return matched_id

        return self._register_player(
            resolved_team,
            histogram,
            bbox,
            frame_id,
            time_s,
            byte_track_id,
        )

    def assign_identity(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        frame_id: int,
        time_s: float,
        byte_track_id: int | None = None,
        team: str | None = None,
    ) -> str | None:
        bbox_dict = {"x": bbox[0], "y": bbox[1], "width": bbox[2], "height": bbox[3]}
        histogram = extract_shirt_histogram(frame, bbox_dict)
        if histogram is None:
            return None

        resolved_team = team
        if resolved_team is None:
            classified = classify_team(histogram, self.team_templates)
            if classified == "referee":
                return None
            resolved_team = classified

        if byte_track_id is not None and byte_track_id in self.byte_track_map:
            stable_id = self.byte_track_map[byte_track_id]
            entry = self.gallery.get(stable_id)
            if entry and self._color_gate(entry, histogram) and self._spatial_gate(entry, bbox, frame_id, time_s):
                self._update_entry(entry, histogram, bbox, frame_id, time_s)
                return stable_id
            self.byte_track_map.pop(byte_track_id, None)

        matched_id = self._match_gallery(resolved_team, histogram, bbox, frame_id, time_s)
        if matched_id is not None:
            entry = self.gallery[matched_id]
            self._update_entry(entry, histogram, bbox, frame_id, time_s)
            if byte_track_id is not None:
                self.byte_track_map[byte_track_id] = matched_id
            if byte_track_id is not None and byte_track_id in self.pending_players:
                self.pending_players.pop(byte_track_id, None)
            return matched_id

        if byte_track_id is None:
            return self._register_player(resolved_team, histogram, bbox, frame_id, time_s)

        pending = self.pending_players.get(byte_track_id)
        if pending is None:
            self.pending_players[byte_track_id] = PendingPlayer(
                team=resolved_team,
                appearance_histogram=histogram.copy(),
                bbox=bbox,
                byte_track_id=byte_track_id,
                frames_seen=1,
                first_frame_id=frame_id,
            )
            return None

        if pending.team != resolved_team:
            self.pending_players[byte_track_id] = PendingPlayer(
                team=resolved_team,
                appearance_histogram=histogram.copy(),
                bbox=bbox,
                byte_track_id=byte_track_id,
                frames_seen=1,
                first_frame_id=frame_id,
            )
            return None

        pending.frames_seen += 1
        pending.bbox = bbox
        pending.appearance_histogram = pending.appearance_histogram * 0.6 + histogram * 0.4
        if pending.frames_seen < NEW_ID_CONFIRMATION_FRAMES:
            return None

        stable_id = self._register_player(
            resolved_team,
            pending.appearance_histogram,
            bbox,
            frame_id,
            time_s,
            byte_track_id,
        )
        self.pending_players.pop(byte_track_id, None)
        return stable_id
