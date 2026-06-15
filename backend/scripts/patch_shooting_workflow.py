"""Patch Roboflow shooting workflow for 15-keypoint YOLO pose models."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

BACKEND = Path(__file__).resolve().parents[1]
ENV_PATH = BACKEND / ".env"
WORKSPACE = "lucass-workspace-fn5cc"
WORKFLOW_SLUG = "soccer-shooting-technique-analyzer-1781109926364"

# yolo26s-pose-640 exposes 15 keypoints (indices 0-14). COCO ankle indices 15/16 break visualization.
SAFE_POSE_EDGES = [
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],
    [5, 6],
    [5, 11],
    [6, 12],
    [11, 12],
    [11, 13],
    [12, 14],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
]

DETECTION_SELECTOR = "$steps.detect_player_ball.predictions"

# Insert after pts extraction in the custom block.
POSE_COCO_NORMALIZE = '''
    def build_coco_pts(raw_kps):
        try:
            items = as_list(raw_kps)
            if not items:
                return None
            if isinstance(items[0], dict):
                coco = _np.full((17, 2), _np.nan, dtype=float)
                for item in items:
                    cid = int(item.get("class_id", -1))
                    if 0 <= cid < 17:
                        coco[cid, 0] = float(item.get("x", _np.nan))
                        coco[cid, 1] = float(item.get("y", _np.nan))
                return coco if _np.isfinite(coco).any() else None
            arr = _np.asarray(items, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2]
        except Exception:
            return None
        return None

    if pts is None and kps is not None:
        pts = build_coco_pts(kps)
    if pts is not None:
        try:
            arr = _np.asarray(pts, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 17:
                pts = arr[:, :2]
        except Exception:
            pass
'''

POSE_GUARD = '''
    def pt_count(pts):
        try:
            if pts is None:
                return 0
            arr = _np.asarray(pts)
            if arr.ndim == 1:
                return 1 if arr.size >= 2 else 0
            return int(arr.shape[0])
        except Exception:
            return 0

    def pt_at(pts, idx):
        try:
            if pts is None or idx < 0:
                return None
            arr = _np.asarray(pts, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 17 and idx < 17:
                row = arr[idx]
            elif idx >= pt_count(pts):
                return None
            else:
                row = arr[idx]
            if row is None or len(row) < 2:
                return None
            if not (_np.isfinite(row[0]) and _np.isfinite(row[1])):
                return None
            return row
        except Exception:
            return None
'''

NEW_POSE_BLOCK = """    pose_ready = (
        valid(pt_at(pts, 11))
        and valid(pt_at(pts, 12))
        and (valid(pt_at(pts, 13)) or valid(pt_at(pts, 14)))
    )
    if pose_ready:
        LHIP = pt_at(pts, 11)
        LKNEE = pt_at(pts, 13)
        LANK = pt_at(pts, 15)
        RHIP = pt_at(pts, 12)
        RKNEE = pt_at(pts, 14)
        RANK = pt_at(pts, 16)
        LSHO = pt_at(pts, 5)
        RSHO = pt_at(pts, 6)"""


def _load_env() -> None:
    if not ENV_PATH.exists():
        return
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _patch_run_code(code: str) -> str:
    anchor = "    except Exception:\n        pts = None"
    if anchor in code and "def build_coco_pts(raw_kps):" not in code:
        code = code.replace(anchor, anchor + "\n" + POSE_COCO_NORMALIZE, 1)

    if "def pt_count(pts):" not in code:
        insert_after = "    def valid(p):"
        if insert_after not in code:
            raise RuntimeError("Could not locate valid() in workflow code")
        code = code.replace(insert_after, POSE_GUARD + "\n\n    def valid(p):", 1)
    else:
        # Refresh guard helpers when re-patching.
        start = code.find("    def pt_count(pts):")
        end = code.find("\n\n    def valid(p):", start)
        if start != -1 and end != -1:
            code = code[:start] + POSE_GUARD.strip() + "\n\n" + code[end + 2 :]

    old_blocks = [
        """    if pts is not None and len(pts) >= 17:
        LHIP, LKNEE, LANK = pts[11], pts[13], pts[15]
        RHIP, RKNEE, RANK = pts[12], pts[14], pts[16]
        LSHO, RSHO = pts[5], pts[6]""",
        """    pose_count = pt_count(pts)
    pose_ready = pose_count >= 17
    if pose_ready:
        LHIP = pt_at(pts, 11)
        LKNEE = pt_at(pts, 13)
        LANK = pt_at(pts, 15)
        RHIP = pt_at(pts, 12)
        RKNEE = pt_at(pts, 14)
        RANK = pt_at(pts, 16)
        LSHO = pt_at(pts, 5)
        RSHO = pt_at(pts, 6)""",
    ]
    for old in old_blocks:
        if old in code:
            code = code.replace(old, NEW_POSE_BLOCK)
            break

    return code


def _rewire_detection_selector(spec: dict, selector: str) -> None:
    tracked = "$steps.track_player_ball.tracked_detections"
    for step in spec.get("steps", []):
        for key, value in list(step.items()):
            if value == tracked:
                step[key] = selector
    for output in spec.get("outputs", []):
        if output.get("selector") == tracked:
            output["selector"] = selector


def _patch_spec(spec: dict) -> dict:
    steps = spec.get("steps", [])
    spec["steps"] = [step for step in steps if step.get("name") != "pose_skeleton"]
    for step in spec.get("steps", []):
        if step.get("name") == "detect_player_ball":
            step["model_id"] = "yolov11s-640"
            step["custom_confidence"] = 0.25
            step.pop("class_filter", None)
        if step.get("name") == "technique_overlay":
            step["image"] = "$steps.draw_labels.image"

    # ByteTrack returns empty boxes on single-frame serverless calls; use raw detections.
    _rewire_detection_selector(spec, DETECTION_SELECTOR)

    for block in spec.get("dynamic_blocks_definitions", []):
        code_obj = block.get("code", {})
        run_code = code_obj.get("run_function_code")
        if isinstance(run_code, str) and "def run(self, object_predictions, pose_predictions)" in run_code:
            code_obj["run_function_code"] = _patch_run_code(run_code)
    return spec


def main() -> int:
    _load_env()
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print("ROBOFLOW_API_KEY missing", file=sys.stderr)
        return 1

    get_resp = requests.get(
        f"https://api.roboflow.com/{WORKSPACE}/workflows/{WORKFLOW_SLUG}",
        params={"api_key": api_key},
        timeout=60,
    )
    get_resp.raise_for_status()
    workflow = get_resp.json()
    if "config" not in workflow and "workflow" in workflow:
        workflow = workflow["workflow"]
    config_raw = workflow.get("config") or workflow.get("lastVersionConfig")
    config = json.loads(config_raw) if isinstance(config_raw, str) else config_raw
    spec = config["specification"]
    config["specification"] = _patch_spec(spec)

    update_resp = requests.post(
        f"https://api.roboflow.com/{WORKSPACE}/updateWorkflow",
        params={"api_key": api_key},
        json={
            "id": workflow["id"],
            "url": workflow.get("url", WORKFLOW_SLUG),
            "name": workflow.get("name", "Soccer Shooting Technique Analyzer"),
            "config": json.dumps(config),
        },
        timeout=120,
    )
    if not update_resp.ok:
        print(update_resp.text[:2000], file=sys.stderr)
        update_resp.raise_for_status()
    print("Workflow draft patched (detector rewire + pose normalization).")

    publish_resp = requests.post(
        f"https://api.roboflow.com/{WORKSPACE}/workflows/{WORKFLOW_SLUG}/publish",
        params={"api_key": api_key},
        timeout=120,
    )
    if publish_resp.ok:
        print("Workflow published to serverless.")
    else:
        # Fall back to agent publish endpoint shape used by Roboflow MCP.
        alt = requests.post(
            "https://api.roboflow.com/agent/workflow/publish",
            params={"api_key": api_key},
            json={"workflow_url": WORKFLOW_SLUG},
            timeout=120,
        )
        if alt.ok:
            print("Workflow published via agent endpoint.")
        else:
            print(
                "Draft updated but publish failed — run agent_workflow_publish in Roboflow:",
                publish_resp.text[:500],
                alt.text[:500],
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
