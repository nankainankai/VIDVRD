from __future__ import annotations

"""全局关系节点。

输入：窗口级关系结果。
输出：`relations_global.json`，用于跨窗口聚合动态关系并减少碎片。
VL 开启时可结合全片抽帧拼图做视频级复核。
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

from vidvrd_auto.prompts.templates import global_relation_prompt
from vidvrd_auto.utils.io import read_json, write_json
from vidvrd_auto.utils.vl_frames import (
    build_storyboard_from_video,
    call_vl_with_storyboard,
    uniform_frame_indices,
    video_path_from_windows,
)


def run_global_relation(
    *,
    video_id: str,
    relations_json: Path,
    out_json: Path,
    config: Dict[str, Any],
    windows_json: Optional[Path] = None,
    api_key: str = "",
) -> Dict[str, Any]:
    obj = read_json(relations_json) if relations_json.exists() else {video_id: []}
    items = obj.get(video_id, []) if isinstance(obj, dict) else []
    if not isinstance(items, list):
        items = []

    vl_enabled = bool(config.get("vl_enabled", False))
    min_segments = int(config.get("min_segments", 2) or 2)
    dynamic_preds = set(config.get("dynamic_predicates", ["toward", "away", "follow", "chase"]) or [])
    grouped: Dict[Tuple[int, str, int], List[Dict[str, Any]]] = defaultdict(list)
    passthrough: List[Dict[str, Any]] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            key = (
                int(item.get("subject_track_id", item.get("subject_id"))),
                str(item.get("predicate", "") or "").strip().lower(),
                int(item.get("object_track_id", item.get("object_id"))),
            )
        except Exception:
            passthrough.append(dict(item))
            continue
        grouped[key].append(dict(item))

    for (sid, pred, oid), rels in sorted(grouped.items()):
        if pred in dynamic_preds and len(rels) >= min_segments:
            passthrough.append(
                {
                    "subject_track_id": sid,
                    "predicate": pred,
                    "object_track_id": oid,
                    "start_frame": min(int(r.get("start_frame", 0) or 0) for r in rels),
                    "end_frame": max(int(r.get("end_frame", 0) or 0) for r in rels),
                    "confidence": min(1.0, max(float(r.get("confidence", 0.0) or 0.0) for r in rels) + 0.05),
                    "source": "global_relation",
                    "sources": sorted({s for r in rels for s in (r.get("sources", []) or [r.get("source", "unknown")])}),
                    "evidence": f"aggregated {len(rels)} segments",
                    "segment_count": len(rels),
                }
            )
            continue
        passthrough.extend(rels)

    model_review: Dict[str, Any] = {
        "enabled": vl_enabled,
        "state": "disabled",
        "relations": [],
        "used_images": False,
    }
    if vl_enabled and passthrough:
        storyboard = None
        if windows_json is not None and windows_json.exists():
            video_path = video_path_from_windows(windows_json)
            windows_obj = read_json(windows_json)
            video_meta = windows_obj.get("video", {}) if isinstance(windows_obj, dict) else {}
            total_frames = int(video_meta.get("total_frames", 0) or 0)
            vl_sample = int(config.get("vl_sample_frames", 8) or 8)
            if video_path is not None and total_frames > 0:
                indices = uniform_frame_indices(total_frames=total_frames, count=vl_sample)
                storyboard = build_storyboard_from_video(
                    video_path=video_path,
                    frame_indices=indices,
                    tile_h=int(config.get("vl_tile_h", 320) or 320),
                )
                if storyboard is not None:
                    model_review["used_images"] = True
                    model_review["video_path"] = str(video_path)
                    model_review["sampled_frames"] = indices
            else:
                model_review["image_fallback"] = "video_meta_unavailable"
        else:
            model_review["image_fallback"] = "windows_json_missing"

        vl_result = call_vl_with_storyboard(
            config=config,
            api_key=api_key,
            prompt=global_relation_prompt(passthrough, has_images=bool(storyboard is not None)),
            storyboard_bgr=storyboard,
        )
        model_review.update(vl_result.to_dict())
        model_review["state"] = "succeeded" if vl_result.ok else "failed"
        if vl_result.ok:
            try:
                parsed = json.loads(vl_result.text)
            except Exception:
                parsed = {}
            if isinstance(parsed.get("relations"), list):
                model_items = [dict(x, source="global_relation_model") for x in parsed["relations"] if isinstance(x, dict)]
                passthrough.extend(model_items)
                model_review["relations"] = model_items
        else:
            model_review["fallback"] = "use_rule_aggregation"

    out = {video_id: passthrough, "_global_review": model_review}
    write_json(out_json, out)
    return out
