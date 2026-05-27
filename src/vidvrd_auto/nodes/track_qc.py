from __future__ import annotations

"""轨迹质检节点。

输入：轨迹 JSONL、窗口 JSON 和 QC 配置。
输出：`track_qc.json`，记录短轨迹、类别漂移、框跳变和可选 VL 复核建议。
VL 开启时会对风险帧抽帧拼图后调用多模态模型。
"""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set
import json

from vidvrd_auto.prompts.templates import track_qc_prompt
from vidvrd_auto.relations.storyboard import load_tracks_for_frames
from vidvrd_auto.utils.io import iter_jsonl, read_json, write_json
from vidvrd_auto.utils.vl_frames import build_storyboard_from_video, call_vl_with_storyboard, video_path_from_windows


def _bbox_center(bbox: List[float]) -> tuple[float, float]:
    return (float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0


def run_track_qc(
    *,
    tracks_jsonl: Path,
    windows_json: Path,
    out_json: Path,
    config: Dict[str, Any],
    api_key: str = "",
) -> Dict[str, Any]:
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
    risk_items.extend(
        {"track_id": item["track_id"], "type": "large_jump", "frame": item["frame"], "jump_ratio": item["jump_ratio"]}
        for item in large_jumps[:50]
    )

    vl_enabled = bool(config.get("vl_enabled", False))
    vl_review: Dict[str, Any] = {
        "enabled": vl_enabled,
        "state": "disabled",
        "items": [],
        "used_images": False,
    }
    if vl_enabled and risk_items:
        video_path = video_path_from_windows(windows_json)
        storyboard = None
        vl_max_frames = int(config.get("vl_max_frames", 6) or 6)
        frame_indices: List[int] = []
        risk_track_ids: Set[int] = set()
        for item in risk_items:
            if item.get("type") == "large_jump" and "frame" in item:
                frame_indices.append(int(item["frame"]))
            try:
                risk_track_ids.add(int(item.get("track_id")))
            except Exception:
                pass
        frame_indices = sorted(set(frame_indices))[:vl_max_frames]
        if video_path is not None and frame_indices:
            tracks_map = load_tracks_for_frames(tracks_jsonl, set(frame_indices))
            storyboard = build_storyboard_from_video(
                video_path=video_path,
                frame_indices=frame_indices,
                tracks_by_frame=tracks_map,
                allowed_track_ids=risk_track_ids if risk_track_ids else None,
                tile_h=int(config.get("vl_tile_h", 320) or 320),
            )
            if storyboard is not None:
                vl_review["used_images"] = True
                vl_review["video_path"] = str(video_path)
                vl_review["sampled_frames"] = frame_indices
        else:
            vl_review["image_fallback"] = "no_risk_frames_or_video"

        vl_result = call_vl_with_storyboard(
            config=config,
            api_key=api_key,
            prompt=track_qc_prompt(risk_items, has_images=bool(storyboard is not None)),
            storyboard_bgr=storyboard,
        )
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
