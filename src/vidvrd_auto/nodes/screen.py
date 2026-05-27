from __future__ import annotations

"""关键帧粗筛节点。

输入：检测结果 JSONL 和筛选配置。
输出：`screen_result.json`，包含 keep/drop/crop 决策、裁剪建议和可选 VL 判断结果。
VL 开启时会读取 `video_meta.json` 抽帧拼图后调用多模态模型。
"""

from pathlib import Path
from typing import Any, Dict, List
import json

from vidvrd_auto.prompts.templates import keyframe_screen_prompt
from vidvrd_auto.utils.io import iter_jsonl, write_json
from vidvrd_auto.utils.vl_frames import (
    build_storyboard_from_video,
    call_vl_with_storyboard,
    video_path_from_meta,
)


def screen_keyframes(
    *,
    detections_jsonl: Path,
    out_json: Path,
    config: Dict[str, Any],
    api_key: str = "",
) -> Dict[str, Any]:
    sample_frames = int(config.get("sample_frames", 8) or 8)
    max_frame_index = int(config.get("max_frame_index", 0) or 0)
    min_objects = int(config.get("min_objects", 2) or 2)
    min_confidence = float(config.get("min_confidence", 0.0) or 0.0)
    required_positive_frames = int(config.get("required_positive_frames", 1) or 1)
    enabled = bool(config.get("enabled", True))
    vl_enabled = bool(config.get("vl_enabled", False))
    vl_dry_run = bool(config.get("vl_dry_run", False))

    rows: List[Dict[str, Any]] = []
    detections_by_frame: Dict[int, List[Dict[str, Any]]] = {}

    if enabled and sample_frames > 0:
        if max_frame_index > 0:
            buckets: List[Any] = [None] * sample_frames
            filled = 0
            denom = float(max_frame_index + 1)
            bucket_size = denom / float(sample_frames)
            for row in iter_jsonl(detections_jsonl):
                try:
                    fr = int(row.get("frame", 0) or 0)
                except Exception:
                    fr = 0
                objs = row.get("objects", []) if isinstance(row.get("objects"), list) else []
                detections_by_frame[fr] = [o for o in objs if isinstance(o, dict)]
                if fr > max_frame_index:
                    break
                idx = int(fr / bucket_size) if bucket_size > 0 else 0
                idx = max(0, min(sample_frames - 1, idx))
                if buckets[idx] is None:
                    buckets[idx] = row
                    filled += 1
                    if filled >= sample_frames:
                        break
            rows = [r for r in buckets if r is not None]
        else:
            for row in iter_jsonl(detections_jsonl):
                fr = int(row.get("frame", 0) or 0)
                objs = row.get("objects", []) if isinstance(row.get("objects"), list) else []
                detections_by_frame[fr] = [o for o in objs if isinstance(o, dict)]
                rows.append(row)
                if len(rows) >= sample_frames:
                    break

    frame_results = []
    positive = 0
    for row in rows:
        valid = 0
        crop_boxes = []
        for obj in row.get("objects", []) or []:
            if not isinstance(obj, dict):
                continue
            bbox = obj.get("bbox")
            conf = float(obj.get("confidence", 0.0) or 0.0)
            if conf >= min_confidence and isinstance(bbox, list) and len(bbox) == 4:
                valid += 1
                crop_boxes.append([float(x) for x in bbox])
        passed = valid >= min_objects
        positive += int(passed)
        crop_suggestion = None
        if crop_boxes:
            crop_suggestion = [
                min(b[0] for b in crop_boxes),
                min(b[1] for b in crop_boxes),
                max(b[2] for b in crop_boxes),
                max(b[3] for b in crop_boxes),
            ]
        frame_results.append(
            {
                "frame": int(row.get("frame", 0) or 0),
                "valid_objects": valid,
                "passed": passed,
                "crop_suggestion": crop_suggestion,
            }
        )

    rule_passed = (not enabled) or positive >= required_positive_frames
    decision = "keep" if rule_passed else "drop"
    reason = "ok" if rule_passed else "not_enough_valid_objects"
    crop_suggestion = next((x.get("crop_suggestion") for x in frame_results if x.get("crop_suggestion")), None)
    vl_screen: Dict[str, Any] = {
        "enabled": vl_enabled,
        "state": "disabled",
        "model": str(config.get("vl_model", "") or ""),
        "used_images": False,
    }

    if vl_enabled and rule_passed:
        meta_path = detections_jsonl.parent / "video_meta.json"
        video_path = video_path_from_meta(meta_path)
        storyboard = None
        frame_indices = [int(x.get("frame", 0) or 0) for x in frame_results]
        if video_path is not None and frame_indices:
            storyboard = build_storyboard_from_video(
                video_path=video_path,
                frame_indices=frame_indices,
                detections_by_frame=detections_by_frame,
                tile_h=int(config.get("vl_tile_h", 320) or 320),
            )
            if storyboard is not None:
                vl_screen["used_images"] = True
                vl_screen["video_path"] = str(video_path)
        else:
            vl_screen["image_fallback"] = "video_or_frames_unavailable"

        vl_result = call_vl_with_storyboard(
            config=config,
            api_key=api_key,
            prompt=keyframe_screen_prompt(frame_results, has_images=bool(storyboard is not None)),
            storyboard_bgr=storyboard,
        )
        vl_screen.update(vl_result.to_dict())
        vl_screen["state"] = "succeeded" if vl_result.ok else "failed"
        if vl_result.ok:
            try:
                parsed = json.loads(vl_result.text)
            except Exception:
                parsed = {}
            model_decision = str(parsed.get("decision", "") or "").strip().lower()
            if model_decision in {"keep", "drop", "crop"}:
                decision = model_decision
                reason = str(parsed.get("reason", reason) or reason)
                if isinstance(parsed.get("crop_suggestion"), list) and len(parsed["crop_suggestion"]) == 4:
                    crop_suggestion = [float(x) for x in parsed["crop_suggestion"]]
        else:
            vl_screen["fallback"] = "use_rule_decision"

    result = {
        "enabled": enabled,
        "passed": decision != "drop",
        "decision": decision,
        "reason": reason,
        "positive_frames": positive,
        "required_positive_frames": required_positive_frames,
        "min_objects": min_objects,
        "min_confidence": min_confidence,
        "max_frame_index": max_frame_index,
        "sampled_frames": frame_results,
        "vl_screen": vl_screen,
        "crop_suggestion": crop_suggestion,
    }
    write_json(out_json, result)
    return result
