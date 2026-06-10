from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.services.storage import video_dir

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def generate_heatmaps(
    video_id: str,
    position_samples: list[tuple[float, float]],
    speed_samples: list[tuple[float, float, float]],
    frame_width: int,
    frame_height: int,
) -> dict[str, str]:
    if plt is None:
        return {}

    urls: dict[str, str] = {}
    output_dir = video_dir(video_id)

    if position_samples:
        xs = [item[0] for item in position_samples]
        ys = [item[1] for item in position_samples]
        fig, axis = plt.subplots(figsize=(8, 4.5), dpi=100)
        axis.set_facecolor("#0a0a0f")
        fig.patch.set_facecolor("#0a0a0f")
        heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=40)
        axis.imshow(
            heatmap.T,
            origin="lower",
            cmap="hot",
            extent=[0, max(frame_width, 1), 0, max(frame_height, 1)],
            aspect="auto",
            alpha=0.9,
        )
        axis.set_title("Player position heatmap", color="#f1f5f9")
        axis.tick_params(colors="#64748b")
        position_path = output_dir / "position-heatmap.png"
        fig.savefig(position_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        urls["position_heatmap"] = f"/media/{video_id}/position-heatmap.png"

    if speed_samples:
        xs = [item[0] for item in speed_samples]
        ys = [item[1] for item in speed_samples]
        speeds = [item[2] for item in speed_samples]
        fig, axis = plt.subplots(figsize=(8, 4.5), dpi=100)
        axis.set_facecolor("#0a0a0f")
        fig.patch.set_facecolor("#0a0a0f")
        scatter = axis.scatter(xs, ys, c=speeds, cmap="plasma", s=18, alpha=0.85)
        axis.set_xlim(0, max(frame_width, 1))
        axis.set_ylim(0, max(frame_height, 1))
        axis.set_title("Speed heatmap", color="#f1f5f9")
        axis.tick_params(colors="#64748b")
        colorbar = fig.colorbar(scatter, ax=axis)
        colorbar.set_label("km/h", color="#f1f5f9")
        colorbar.ax.yaxis.set_tick_params(color="#64748b")
        speed_path = output_dir / "speed-heatmap.png"
        fig.savefig(speed_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        urls["speed_heatmap"] = f"/media/{video_id}/speed-heatmap.png"

    return urls
