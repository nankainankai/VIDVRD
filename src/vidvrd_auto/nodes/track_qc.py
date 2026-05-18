from __future__ import annotations

"""轨迹质检节点。

输入：轨迹 JSONL、窗口 JSON 和 QC 配置。
输出：`track_qc.json`，记录短轨迹、类别漂移、框跳变和可选 VL 复核建议。
该节点只标记风险，不直接修改轨迹。
"""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List
import json

from vidvrd_auto.models.vl_client import VLClient
from vidvrd_auto.prompts.templates import track_qc_prompt
from vidvrd_auto.utils.io import iter_jsonl, read_json, write_json


def _bbox_center(bbox: List[float]) -> tuple[float, float]:
    return (float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0


def run_track_qc(*, tracks_jsonl: Path, windows_json: Path, out_json: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    min_frames = int(config.get("min_track_frames", 2) or 2)
    max_class_changes = int(config.get("max_class_changes", 1) or 1)
    max_center_jump_ratio = float(config.get("max_center_jump_ratio", 0.45) or 0.45)

    track_frames: Dict[int, int] = Counter()
    track_classes: Dict[int, Counter[str]] = defaultdict(Counter)
    last_center: Dict[int, tuple[float, float]] = {}
    large_jumps: List[Dict[str, Any]] = []
    frame_count = 0

    for row in iter_jsonl(tracks_jsonl):
        frame_count += 1
        frame_idx = int(row.get("frame", frame_count - 1) or 0)
        tracks = row.get("tracks", []) or []
        for track in tracks:
            if not isinstance(track, dict):
                continue
            try:
                tid = int(track.get("track_id"))
            except Exception:
                continue
            bbox = track.get("bbox_observed", track.get("bbox"))
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            track_frames[tid] += 1
            track_classes[tid][str(track.get("class_name", "unknown") or "unknown")] += 1
            center = _bbox_center([float(x) for x in bbox])
            prev = last_center.get(tid)
            if prev is not None:
                width = max(1.0, abs(float(bbox[2]) - float(bbox[0])))
                height = max(1.0, abs(float(bbox[3]) - float(bbox[1])))
                diag = (width * width + height * height) ** 0.5
                jump_ratio = (((center[0] - prev[0]) ** 2 + (center[1] - prev[1]) ** 2) ** 0.5) / diag
                if jump_ratio > max_center_jump_ratio:
                    large_jumps.append({"track_id": tid, "frame": frame_idx, "jump_ratio": round(jump_ratio, 4)})
            last_center[tid] = center

    windows_obj = read_json(windows_json) if windows_json.exists() else {}
    windows = windows_obj.get("windows", []) if isinstance(windows_obj, dict) else []
    short_tracks = [tid for tid, count in sorted(track_frames.items()) if count < min_frames]
    class_drift = [
        {"track_id": tid, "classes": dict(counter)}
        for tid, counter in sorted(track_classes.items())
        if len(counter) > max_class_changes
    ]
    risk_items: List[Dict[str, Any]] = []
    risk_items.extend({"track_id": tid, "type": "short_track", "frame_count": track_frames[tid]} for tid in short_tracks[:50])
    risk_items.extend({"track_id": item["track_id"], "type": "class_drift", "classes": item["classes"]} for item in class_drift[:50])
    risk_items.extend({"track_id": item["track_id"], "type": "large_jump", "frame": item["frame"], "jump_ratio": item["jump_ratio"]} for item in large_jumps[:50])

    vl_enabled = bool(config.get("vl_enabled", False))
    vl_dry_run = bool(config.get("vl_dry_run", False))
    vl_review: Dict[str, Any] = {
        "enabled": vl_enabled,
        "state": "disabled",
        "items": [],
    }
    if vl_enabled and risk_items:
        client_cfg = {
            "model": str(config.get("vl_model", "qwen-vl-max") or "qwen-vl-max"),
            "api_key_env": str(config.get("api_key_env", "DASHSCOPE_API_KEY") or "DASHSCOPE_API_KEY"),
            "retries": int(config.get("vl_retries", 2) or 2),
            "backoff_sec": float(config.get("vl_backoff_sec", 1.5) or 1.5),
            "sleep_sec": float(config.get("vl_sleep_sec", 0.0) or 0.0),
            "dry_run": vl_dry_run,
        }
        vl_result = VLClient(client_cfg).call(prompt=track_qc_prompt(risk_items), dry_run=vl_dry_run)
        vl_review.update(vl_result.to_dict())
        vl_review["state"] = "succeeded" if vl_result.ok else "failed"
        if vl_result.ok:
            try:
                parsed = json.loads(vl_result.text)
            except Exception:
                parsed = {}
            if isinstance(parsed.get("items"), list):
                vl_review["items"] = parsed["items"]
        else:
            vl_review["fallback"] = "use_rule_risks"

    result = {
        "track_count": len(track_frames),
        "frame_count": frame_count,
        "window_count": len(windows),
        "short_track_count": len(short_tracks),
        "short_tracks": short_tracks[:100],
        "class_drift_count": len(class_drift),
        "class_drift": class_drift[:100],
        "large_jump_count": len(large_jumps),
        "large_jumps": large_jumps[:100],
        "risk_items": risk_items[:150],
        "vl_review": vl_review,
        "passed": len(class_drift) == 0,
    }
    write_json(out_json, result)
    return result
