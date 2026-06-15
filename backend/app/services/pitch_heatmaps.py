from __future__ import annotations

from pathlib import Path

import numpy as np

FIELD_LENGTH_M = 105.0
FIELD_WIDTH_M = 68.0

_PITCH_MARKINGS = {
    "penalty_left": (0, 13.84, 16.5, 54.16),
    "penalty_right": (88.5, 13.84, 105, 54.16),
    "goal_area_left": (0, 24.84, 5.5, 43.16),
    "goal_area_right": (99.5, 24.84, 105, 43.16),
}
_CENTRE = (52.5, 34.0)
_CENTRE_RADIUS = 9.15
_PENALTY_SPOTS = [(11.0, 34.0), (94.0, 34.0)]


def generate_movement_heatmap(track_points: list[dict], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    xs = np.array([point["x_m"] for point in track_points], dtype=float)
    ys = np.array([point["y_m"] for point in track_points], dtype=float)

    fig, axis = plt.subplots(figsize=(12, 8), facecolor="#07090f")
    axis.set_facecolor("#07090f")
    axis.set_xlim(0, FIELD_LENGTH_M)
    axis.set_ylim(0, FIELD_WIDTH_M)
    axis.set_aspect("equal")
    axis.axis("off")

    if len(xs) >= 4:
        try:
            kde = gaussian_kde(np.vstack([xs, ys]), bw_method=2.5 / np.std(xs + ys + 1))
            xi = np.linspace(0, FIELD_LENGTH_M, 210)
            yi = np.linspace(0, FIELD_WIDTH_M, 136)
            xi_grid, yi_grid = np.meshgrid(xi, yi)
            density = kde(np.vstack([xi_grid.ravel(), yi_grid.ravel()])).reshape(xi_grid.shape)
            density = (density - density.min()) / (density.max() - density.min() + 1e-9)
            axis.contourf(xi_grid, yi_grid, density, levels=20, cmap="RdYlGn_r", alpha=0.85)
        except Exception:
            pass

    _draw_pitch_lines(axis)
    plt.tight_layout(pad=0)
    fig.savefig(str(output_path), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_touch_heatmap(
    touch_events: list[dict],
    pass_events: list[dict],
    touch_count: int,
    pass_count: int,
    output_path: Path,
) -> None:
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axis = plt.subplots(figsize=(12, 8), facecolor="#07090f")
    axis.set_facecolor("#07090f")
    axis.set_xlim(0, FIELD_LENGTH_M)
    axis.set_ylim(0, FIELD_WIDTH_M)
    axis.set_aspect("equal")
    axis.axis("off")

    _draw_pitch_lines(axis)

    for event in touch_events:
        nearby = sum(
            1
            for other in touch_events
            if abs(other["x_m"] - event["x_m"]) < 3 and abs(other["y_m"] - event["y_m"]) < 3
        )
        radius = 1.5 + nearby * 0.3
        circle = plt.Circle((event["x_m"], event["y_m"]), radius, color="white", alpha=0.65, zorder=3)
        axis.add_patch(circle)

    for event in pass_events:
        angle_rad = math.radians(event.get("angle_deg", 0))
        arrow_len = 3.0
        dx = arrow_len * math.cos(angle_rad)
        dy = arrow_len * math.sin(angle_rad)
        circle = plt.Circle((event["x_m"], event["y_m"]), 1.5, color="#3b82f6", alpha=0.80, zorder=4)
        axis.add_patch(circle)
        axis.annotate(
            "",
            xy=(event["x_m"] + dx, event["y_m"] + dy),
            xytext=(event["x_m"], event["y_m"]),
            arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=1.8),
            zorder=5,
        )

    axis.text(
        FIELD_LENGTH_M - 1,
        FIELD_WIDTH_M - 2,
        f"Touches: {touch_count}",
        color="white",
        fontsize=13,
        ha="right",
        va="top",
        fontfamily="sans-serif",
        fontweight="bold",
    )
    axis.text(
        FIELD_LENGTH_M - 1,
        FIELD_WIDTH_M - 6,
        f"Passes: {pass_count}",
        color="#3b82f6",
        fontsize=13,
        ha="right",
        va="top",
        fontfamily="sans-serif",
        fontweight="bold",
    )

    plt.tight_layout(pad=0)
    fig.savefig(str(output_path), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _draw_pitch_lines(axis) -> None:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    line_kw = dict(color="white", alpha=0.35, linewidth=1.2, zorder=2)
    axis.add_patch(
        mpatches.Rectangle(
            (0, 0),
            FIELD_LENGTH_M,
            FIELD_WIDTH_M,
            fill=False,
            edgecolor="white",
            linewidth=1.5,
            alpha=0.4,
            zorder=2,
        )
    )
    axis.plot([52.5, 52.5], [0, FIELD_WIDTH_M], **line_kw)
    axis.add_patch(plt.Circle(_CENTRE, _CENTRE_RADIUS, fill=False, **line_kw))
    axis.plot(*_CENTRE, "o", color="white", markersize=3, alpha=0.4, zorder=2)

    for x1, y1, x2, y2 in _PITCH_MARKINGS.values():
        axis.add_patch(
            mpatches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="white",
                linewidth=1.2,
                alpha=0.35,
                zorder=2,
            )
        )

    for spot in _PENALTY_SPOTS:
        axis.plot(*spot, "o", color="white", markersize=3, alpha=0.4, zorder=2)
