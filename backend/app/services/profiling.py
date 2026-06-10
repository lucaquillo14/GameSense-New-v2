from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class StageTimer:
    timings: dict[str, float] = field(default_factory=dict)
    _active: str | None = field(default=None, repr=False)
    _started_at: float = field(default=0.0, repr=False)

    def start(self, stage: str) -> None:
        self._active = stage
        self._started_at = time.perf_counter()

    def stop(self, stage: str | None = None) -> float:
        name = stage or self._active
        if not name:
            return 0.0
        elapsed = time.perf_counter() - self._started_at
        self.timings[name] = self.timings.get(name, 0.0) + elapsed
        print(f"[profile] {name}: {elapsed:.2f}s")
        self._active = None
        return elapsed

    def log_summary(self, video_id: str) -> None:
        total = sum(self.timings.values())
        print(f"[profile] video={video_id} total={total:.2f}s")
        for stage, elapsed in self.timings.items():
            print(f"[profile]   {stage}: {elapsed:.2f}s ({(elapsed / max(total, 1e-6)) * 100:.1f}%)")
