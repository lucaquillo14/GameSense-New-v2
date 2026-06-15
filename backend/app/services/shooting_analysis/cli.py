"""Shooting Technique Mode — main pipeline.

Usage:
    python -m shooting_analysis.cli clip.mp4 --api-key YOUR_ROBOFLOW_KEY \
        [--goal-model workspace/goal-model/2] [--backend hosted|local] \
        [--kicking-foot auto|left|right] [--player-height 1.80] \
        [--goal-width 7.32] [--stride 1] [-o annotated.mp4]
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

import cv2
import numpy as np

from .config import PipelineConfig
from .detect import Detector, pick_player, pick_ball, center
from .pose import PoseTracker, interpolate_missing
from .events import detect_events
from .metrics import compute_metrics
from .annotate import Annotator


def analyze(video_path: str, cfg: PipelineConfig, out_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    n = len(frames)
    if n < 10:
        raise ValueError("Clip too short — need at least ~0.5 s of video.")
    print(f"[1/5] Loaded {n} frames @ {fps:.1f} fps ({W}x{H})")

    # ---------------------------------------------------------- detection
    detector = Detector(cfg.detector)
    ball_track: List[Optional[np.ndarray]] = [None] * n
    player_boxes: List[Optional[dict]] = [None] * n
    prev_p, prev_b = None, None
    stride = max(1, cfg.detector.detect_stride)
    for i in range(0, n, stride):
        dets = detector.detect(frames[i])
        p = pick_player(dets, cfg.detector, prev_p)
        b = pick_ball(dets, cfg.detector, prev_b)
        if p is not None:
            player_boxes[i] = p
            prev_p = center(p["xyxy"])
        if b is not None:
            ball_track[i] = np.array(center(b["xyxy"]))
            prev_b = tuple(ball_track[i])
        if i % (stride * 15) == 0:
            print(f"      RF-DETR frame {i}/{n}", end="\r")
    ball_track = interpolate_missing(ball_track)
    coverage = sum(b is not None for b in ball_track) / n
    print(f"\n[2/5] Detection done — ball seen on "
          f"{sum(b is not None for b in ball_track)}/{n} frames ({coverage:.0%})")

    # Backfill missed ball frames using the goal model (it also has a 'ball' class)
    if coverage < cfg.events.min_ball_coverage and cfg.detector.goal_model_id \
            and cfg.detector.backend == "hosted":
        print("      Low ball coverage — backfilling with the goalpost model's ball class…")
        prev_b = None
        for i in range(n):
            if ball_track[i] is not None:
                prev_b = tuple(ball_track[i])
                continue
            dets = detector._detect_hosted(frames[i], cfg.detector.goal_model_id)
            b = pick_ball(dets, cfg.detector, prev_b)
            if b is not None:
                ball_track[i] = np.array(center(b["xyxy"]))
                prev_b = tuple(ball_track[i])
        ball_track = interpolate_missing(ball_track)
        coverage = sum(b is not None for b in ball_track) / n
        print(f"      Ball coverage after backfill: {coverage:.0%}")

    # Goal: static, so sample a few frames
    goal_box = None
    if cfg.detector.goal_model_id:
        for i in (0, n // 2, n - 1):
            g = detector.detect_goal(frames[i])
            if g is not None:
                goal_box = g
                break
        print(f"      Goal {'detected' if goal_box else 'NOT detected — falling back to player-height scale'}")

    # --------------------------------------------------------------- pose
    tracker = PoseTracker()
    poses = []
    for i, f in enumerate(frames):
        poses.append(tracker.process(f))
        if i % 20 == 0:
            print(f"      MediaPipe frame {i}/{n}", end="\r")
    tracker.close()
    print(f"\n[3/5] Pose tracked on {sum(p.ok for p in poses)}/{n} frames")

    # ------------------------------------------------------------- events
    ev = detect_events(ball_track, poses, cfg.events, cfg.kicking_foot, frame_h=H)
    if ev is None:
        pose_cov = sum(p.ok for p in poses) / n
        raise RuntimeError(
            "Could not locate the contact frame.\n"
            f"  Diagnostics: ball coverage {coverage:.0%}, pose coverage {pose_cov:.0%}.\n"
            "  Try: --ball-conf 0.2 (blurry/small ball) · --stride 1 · trim the clip so the\n"
            "  kick is the main action · film closer/side-on so the ball is bigger in frame."
        )
    print(f"[4/5] Events: plant f{ev.plant} → backswing f{ev.backswing_peak} → "
          f"CONTACT f{ev.contact} ({ev.kicking_foot} foot)")

    # ------------------------------------------------------------ metrics
    analysis = compute_metrics(poses, ball_track, ev, cfg, fps, goal_box, player_boxes)

    # ------------------------------------------------------------- render
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    ann = Annotator(cfg, ev, analysis, n, fps, (W, H))
    for i, f in enumerate(frames):
        writer.write(ann.draw(f, i, poses[i], ball_track[i], goal_box))
        # slow-motion replay of the strike window (3x) right after contact passes
    # append a 3x slow-mo replay of plant -> follow-through
    for i in range(max(0, ev.plant - 3), min(n, ev.contact + 12)):
        img = ann.draw(frames[i].copy(), i, poses[i], ball_track[i], goal_box)
        cv2.putText(img, "REPLAY 1/3x", (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (60, 180, 255), 2, cv2.LINE_AA)
        for _ in range(3):
            writer.write(img)
    writer.release()
    print(f"[5/5] Annotated video -> {out_path}")

    report = {
        "score_out_of_10": analysis.score,
        "shot_speed_kmh": None if analysis.shot_speed_kmh is None else round(analysis.shot_speed_kmh, 1),
        "power_rating": analysis.power_label,
        "kicking_foot": ev.kicking_foot,
        "contact_frame": ev.contact,
        "scale_source": analysis.scale_source,
        "metrics": {k: (None if v is None else round(v, 2)) for k, v in analysis.metrics.items()},
        "feedback": analysis.feedback,
    }
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description="Shooting Technique Mode")
    p.add_argument("video")
    p.add_argument("-o", "--output", default="annotated.mp4")
    p.add_argument("--api-key", default="", help="Roboflow API key (or env ROBOFLOW_API_KEY)")
    p.add_argument("--backend", choices=["hosted", "local"], default="hosted")
    p.add_argument("--goal-model", default="ball-and-goalpost-detection-2/10",
                   help="Roboflow Universe model id for goal detection ('' to disable)")
    p.add_argument("--ball-conf", type=float, default=0.30,
                   help="ball confidence threshold (lower to 0.2 for blurry/small balls)")
    p.add_argument("--kicking-foot", choices=["auto", "left", "right"], default="auto")
    p.add_argument("--player-height", type=float, default=1.75, help="metres, for scale fallback")
    p.add_argument("--goal-width", type=float, default=7.32, help="metres (5.0 for 5-a-side)")
    p.add_argument("--stride", type=int, default=1, help="run RF-DETR every N frames")
    p.add_argument("--json", default="", help="also write the report to this JSON path")
    args = p.parse_args(argv)

    cfg = PipelineConfig()
    cfg.detector.api_key = args.api_key
    cfg.detector.backend = args.backend
    cfg.detector.goal_model_id = args.goal_model
    cfg.detector.conf_ball = args.ball_conf
    cfg.detector.detect_stride = args.stride
    cfg.kicking_foot = args.kicking_foot
    cfg.player_height_m = args.player_height
    cfg.goal_width_m = args.goal_width

    report = analyze(args.video, cfg, args.output)

    print("\n================ SHOOTING TECHNIQUE REPORT ================")
    print(f"  Score: {report['score_out_of_10']}/10   "
          f"Power: {report['shot_speed_kmh']} km/h ({report['power_rating']})")
    print(f"  Foot: {report['kicking_foot']}   Scale: {report['scale_source']}")
    print("  Metrics:")
    for k, v in report["metrics"].items():
        print(f"    - {k}: {v}")
    print("  Feedback:")
    for i, fb in enumerate(report["feedback"], 1):
        print(f"    {i}. {fb}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  JSON report -> {args.json}")


if __name__ == "__main__":
    sys.exit(main())
