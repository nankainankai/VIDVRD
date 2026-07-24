from __future__ import annotations

"""Build a per-video open object vocabulary before detection."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from vidvrd_auto.core.ontology import normalize_vocabulary, object_names
from vidvrd_auto.providers import DashScopeProvider
from vidvrd_auto.utils.io import write_json


def _storyboard(video_path: Path, output_path: Path, count: int) -> List[int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video for vocabulary discovery: {video_path}")
    total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    indices = sorted({int(round(value)) for value in np.linspace(0, total - 1, min(count, total))})
    tiles: List[Any] = []
    try:
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            tile = cv2.resize(frame, (360, 203))
            cv2.putText(tile, f"frame {frame_index}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            tiles.append(tile)
    finally:
        capture.release()
    if not tiles:
        raise RuntimeError("no frames available for vocabulary discovery")
    columns = min(4, len(tiles))
    blank = np.zeros_like(tiles[0])
    rows: List[Any] = []
    for offset in range(0, len(tiles), columns):
        current = tiles[offset : offset + columns]
        current.extend(blank.copy() for _ in range(columns - len(current)))
        rows.append(cv2.hconcat(current))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", cv2.vconcat(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("vocabulary storyboard encoding failed")
    output_path.write_bytes(encoded.tobytes())
    return indices


def _parse_objects(text: str) -> List[str]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text or "").strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    items = value.get("objects", []) if isinstance(value, dict) else []
    return [str(item).strip() for item in items if str(item).strip()] if isinstance(items, list) else []


def build_vocabulary(
    *, video_path: Path, out_json: Path, evidence_path: Path, config: Dict[str, Any], api_key: str = ""
) -> Dict[str, Any]:
    base = object_names()
    extras = [str(value) for value in config.get("extra_categories", [])]
    discovery_enabled = bool(config.get("discovery_enabled", False))
    discovered: List[str] = []
    review: Dict[str, Any] = {"enabled": discovery_enabled, "state": "disabled", "evidence_path": ""}
    sampled_frames: List[int] = []

    if discovery_enabled:
        sampled_frames = _storyboard(video_path, evidence_path, max(1, int(config.get("sample_frames", 8) or 8)))
        review["evidence_path"] = str(evidence_path)
        provider = DashScopeProvider(
            {
                "model": str(config.get("discovery_model", "qwen-vl-max") or "qwen-vl-max"),
                "api_key_env": str(config.get("api_key_env", "DASHSCOPE_API_KEY") or "DASHSCOPE_API_KEY"),
                "retries": int(config.get("retries", 2) or 2),
                "backoff_sec": float(config.get("backoff_sec", 1.5) or 1.5),
                "dry_run": bool(config.get("dry_run", False)),
            },
            api_key=api_key,
        )
        prompt = (
            "List every visually salient object category that should be tracked across this video. "
            "Use concise singular English nouns, include animals, vehicles, people, tools and scene objects, "
            "and do not describe actions or attributes. Return JSON only: {\"objects\":[\"person\",\"zebra\"]}."
        )
        response = provider.call(prompt=prompt, image_paths=[evidence_path], dry_run=bool(config.get("dry_run", False)))
        review.update(response.to_dict())
        review["state"] = "succeeded" if response.ok else "failed"
        if response.ok:
            discovered = _parse_objects(response.text)[: max(0, int(config.get("max_discovered_categories", 24) or 24))]
        else:
            review["fallback"] = "vidvrd_base_vocabulary"

    entries = normalize_vocabulary([*base, *extras, *discovered])
    categories = [entry["raw_label"] for entry in entries]
    label_map = {entry["raw_label"].lower(): entry["canonical_label"] for entry in entries}
    result = {
        "mode": "open" if discovery_enabled else "vidvrd_base",
        "categories": categories,
        "label_map": label_map,
        "entries": entries,
        "base_count": len(base),
        "discovered_count": sum(entry["ontology_source"] == "discovered" for entry in entries),
        "sampled_frames": sampled_frames,
        "discovery": review,
    }
    write_json(out_json, result)
    return result
