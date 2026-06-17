"""Generate a downloadable PDF report for a shooting-technique analysis.

Built with matplotlib (already a dependency) so it needs no extra packages.
The report is a single A4 page: header, headline metrics, a measurements
table with ideal ranges, coaching feedback, and the point-of-contact frame
when available.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless / server-side rendering
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from app.services.storage import MEDIA_ROOT

# (label, key on the shooting_result, formatter, ideal-range text)
_METRICS = [
    ("Backswing Knee Flexion", "backswing_knee_flexion_deg", "{:.0f}°", "75–115°"),
    ("Knee at Contact", "knee_bend_at_contact_deg", "{:.0f}°", "140–170°"),
    ("Ankle Lock", "ankle_lock_variation_deg", "{:.0f}° var", "< 12° variation"),
    ("Plant Foot Distance", "plant_foot_distance_cm", "{:.0f} cm", "5–30 cm"),
    ("Approach Angle", "approach_angle_deg", "{:.0f}°", "25–50°"),
    ("Hip Rotation", "hip_rotation_deg", "{:.0f}°", "25–70°"),
    ("Follow-through Height", "follow_through_height_ratio", "{:.2f}x leg", ">= 0.55x leg"),
    ("Shot Distance", "shot_distance_m", "{:.1f} m", "—"),
]

_INK = "#0f1117"
_MUTED = "#5b6678"
_ACCENT = "#0891b2"


def _fmt(value, template: str) -> str:
    try:
        if value is None or float(value) <= 0:
            return "—"
        return template.format(float(value))
    except (TypeError, ValueError):
        return "—"


def build_pdf(record: dict) -> bytes:
    """Render the technique report for a finished video record into PDF bytes."""
    shooting = record.get("shooting_result") or {}
    video_id = record.get("video_id", "")
    filename = record.get("filename") or "clip"
    created = datetime.now(timezone.utc).strftime("%d %b %Y")

    score = float(shooting.get("technique_score") or 0.0)
    power = float(shooting.get("shot_power_kmh") or 0.0)
    power_rating = shooting.get("power_rating") or ""
    kicking_foot = shooting.get("kicking_foot") or ""
    feedback_points = shooting.get("feedback_points") or []

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.patch.set_facecolor("white")

        # --- Header -------------------------------------------------------
        fig.text(0.06, 0.955, "GameSense", fontsize=22, fontweight="bold", color=_ACCENT)
        fig.text(0.06, 0.935, "Shooting Technique Report", fontsize=13, color=_INK)
        fig.text(0.94, 0.952, created, fontsize=9, color=_MUTED, ha="right")
        fig.text(0.94, 0.936, f"Clip: {filename}", fontsize=8, color=_MUTED, ha="right")
        fig.add_artist(plt.Line2D([0.06, 0.94], [0.92, 0.92], color="#e2e8f0", lw=1))

        # --- Headline metrics --------------------------------------------
        fig.text(0.06, 0.885, "TECHNIQUE SCORE", fontsize=8.5, color=_MUTED, fontweight="bold")
        score_color = "#10b981" if score >= 7 else "#f59e0b" if score >= 4 else "#ef4444"
        fig.text(0.06, 0.845, f"{score:.1f}", fontsize=34, color=score_color, fontweight="bold")
        fig.text(0.165, 0.852, "/ 10", fontsize=12, color=_MUTED)

        fig.text(0.55, 0.885, "SHOT POWER", fontsize=8.5, color=_MUTED, fontweight="bold")
        power_str = f"{power:.0f} km/h" if power > 0 else "—"
        fig.text(0.55, 0.845, power_str, fontsize=26, color=_INK, fontweight="bold")
        if power_rating:
            fig.text(0.55, 0.825, power_rating, fontsize=9, color=_MUTED)

        tags = []
        if kicking_foot:
            tags.append(f"{'Left' if kicking_foot == 'left' else 'Right'}-footed")
        if shooting.get("on_target") is not None:
            tags.append("On target" if shooting.get("on_target") else "Off target")
        conf = float(shooting.get("confidence") or 0.0)
        if conf > 0:
            tags.append(f"Confidence {conf * 100:.0f}%")
        if tags:
            fig.text(0.06, 0.80, "   •   ".join(tags), fontsize=9, color=_MUTED)

        # --- Measurements table ------------------------------------------
        table_top = 0.76
        fig.text(0.06, table_top, "Measurements", fontsize=12, color=_INK, fontweight="bold")
        rows = []
        for label, key, template, ideal in _METRICS:
            rows.append([label, _fmt(shooting.get(key), template), ideal])

        ax = fig.add_axes([0.06, 0.44, 0.88, table_top - 0.45])
        ax.axis("off")
        table = ax.table(
            cellText=rows,
            colLabels=["Metric", "Measured", "Ideal range"],
            colWidths=[0.45, 0.27, 0.28],
            cellLoc="left",
            loc="upper center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1, 1.5)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#e2e8f0")
            if r == 0:
                cell.set_facecolor("#f1f5f9")
                cell.set_text_props(color=_INK, fontweight="bold")
            else:
                cell.set_facecolor("white")
                cell.set_text_props(color=_INK)

        # --- Feedback -----------------------------------------------------
        fb_top = 0.40
        fig.text(0.06, fb_top, "Coaching feedback", fontsize=12, color=_INK, fontweight="bold")
        y = fb_top - 0.035
        if feedback_points:
            for point in feedback_points[:8]:
                wrapped = _wrap(point, 95)
                fig.text(0.07, y, "•", fontsize=9, color=_ACCENT)
                fig.text(0.09, y, wrapped, fontsize=8.7, color="#334155", va="top")
                y -= 0.022 + 0.016 * wrapped.count("\n")
        else:
            fig.text(0.07, y, "No specific feedback was generated for this clip.", fontsize=9, color=_MUTED)

        # --- Contact frame image -----------------------------------------
        img_path = _contact_frame_path(video_id, shooting)
        if img_path and img_path.exists():
            try:
                img = mpimg.imread(str(img_path))
                ax_img = fig.add_axes([0.55, 0.04, 0.39, 0.18])
                ax_img.imshow(img)
                ax_img.axis("off")
                ax_img.set_title("Point of contact", fontsize=8, color=_MUTED)
            except Exception:
                pass

        fig.text(
            0.06, 0.03,
            "Generated by GameSense AI — gamesense.app",
            fontsize=7.5, color=_MUTED,
        )

        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

    return buf.getvalue()


def _contact_frame_path(video_id: str, shooting: dict) -> Optional[Path]:
    url = shooting.get("contact_frame_url")
    if url:
        # url like /media/<video_id>/contact-frame.jpg -> strip the /media prefix
        rel = url.split("/media/", 1)[-1] if "/media/" in url else url.lstrip("/")
        candidate = MEDIA_ROOT / rel
        if candidate.exists():
            return candidate
    # Fallback to the conventional filename.
    fallback = MEDIA_ROOT / video_id / "contact-frame.jpg"
    return fallback if fallback.exists() else None


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width=width)) or text
