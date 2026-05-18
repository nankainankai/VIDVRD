from __future__ import annotations

"""关键帧粗筛节点。

输入：检测结果 JSONL 和筛选配置。
输出：`screen_result.json`，包含 keep/drop/crop 决策、裁剪建议和可选 VL 判断结果。
该节点支持规则筛选和大模型筛选，失败原因会进入 manifest。
"""

from pathlib import Path
from typing import Any, Dict
import json

from vidvrd_auto.models.vl_client import VLClient
from vidvrd_auto.prompts.templates import keyframe_screen_prompt
from vidvrd_auto.utils.io import iter_jsonl, write_json


def screen_keyframes(*, detections_jsonl: Path, out_json: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    sample_frames = int(config.get("sample_frames", 8) or 8)
    max_frame_index = int(config.get("max_frame_index", 0) or 0)
    min_objects = int(config.get("min_objects", 2) or 2)
    min_confidence = float(config.get("min_confidence", 0.0) or 0.0)
    required_positive_frames = int(config.get("required_positive_frames", 1) or 1)
    enabled = bool(config.get("enabled", True))
    vl_enabled = bool(config.get("vl_enabled", False))
    vl_dry_run = bool(config.get("vl_dry_run", False))

    rows = []
    if enabled and sample_frames > 0:
        if max_frame_index > 0:
            # Stratified sampling across timeline (0..max_frame_index) to avoid front-bias.
            buckets = [None] * sample_frames
            filled = 0
            denom = float(max_frame_index + 1)
            bucket_size = denom / float(sample_frames)
            for row in iter_jsonl(detections_jsonl):
                try:
                    fr = int(row.get("frame", 0) or 0)
                except Exception:
                    fr = 0
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
            # Backward compatible: take first N rows.
            for row in iter_jsonl(detections_jsonl):
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
    }

    if vl_enabled and rule_passed:
        client_cfg = {
            "model": str(config.get("vl_model", "qwen-vl-max") or "qwen-vl-max"),
            "api_key_env": str(config.get("api_key_env", "DASHSCOPE_API_KEY") or "DASHSCOPE_API_KEY"),
            "retries": int(config.get("vl_retries", 2) or 2),
            "backoff_sec": float(config.get("vl_backoff_sec", 1.5) or 1.5),
            "sleep_sec": float(config.get("vl_sleep_sec", 0.0) or 0.0),
            "dry_run": vl_dry_run,
        }
        vl_result = VLClient(client_cfg).call(prompt=keyframe_screen_prompt(frame_results), dry_run=vl_dry_run)
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
