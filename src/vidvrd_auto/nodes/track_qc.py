from __future__ import annotations

"""Deterministic trajectory risk assessment for downstream relation review."""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from vidvrd_auto.utils.io import iter_jsonl, read_json, write_json


def _center(bbox: List[float]) -> tuple[float, float]:
    return (float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0


def run_track_qc(
    *, tracks_jsonl: Path, windows_json: Path, out_json: Path, config: Dict[str, Any]
) -> Dict[str, Any]:
    min_frames = max(1, int(config.get("min_track_frames", 2) or 2))
    max_class_changes = max(0, int(config.get("max_class_changes", 1) or 0))
    max_jump_ratio = float(config.get("max_center_jump_ratio", 0.45) or 0.45)
    frame_counts: Counter[int] = Counter()
    class_votes: Dict[int, Counter[str]] = defaultdict(Counter)
    class_sequence: Dict[int, List[str]] = defaultdict(list)
    last_observed: Dict[int, tuple[float, float]] = {}
    jumps: List[Dict[str, Any]] = []
    row_count = 0

    for row in iter_jsonl(tracks_jsonl):
        row_count += 1
        frame = int(row.get("frame", row_count - 1) or 0)
        for track in row.get("tracks", []) or []:
            if not isinstance(track, dict):
                continue
            try:
                track_id = int(track["track_id"])
            except (KeyError, TypeError, ValueError):
                continue
            frame_counts[track_id] += 1
            name = str(track.get("class_name", "unknown") or "unknown")
            class_votes[track_id][name] += 1
            if not class_sequence[track_id] or class_sequence[track_id][-1] != name:
                class_sequence[track_id].append(name)
            bbox = track.get("bbox_observed")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            current = _center([float(value) for value in bbox])
            previous = last_observed.get(track_id)
            if previous is not None:
                width = max(1.0, abs(float(bbox[2]) - float(bbox[0])))
                height = max(1.0, abs(float(bbox[3]) - float(bbox[1])))
                diagonal = (width * width + height * height) ** 0.5
                ratio = (((current[0] - previous[0]) ** 2 + (current[1] - previous[1]) ** 2) ** 0.5) / diagonal
                if ratio > max_jump_ratio:
                    jumps.append({"track_id": track_id, "frame": frame, "jump_ratio": round(ratio, 4)})
            last_observed[track_id] = current

    short_tracks = sorted(track_id for track_id, count in frame_counts.items() if count < min_frames)
    drift = [
        {"track_id": track_id, "changes": len(sequence) - 1, "sequence": sequence, "classes": dict(class_votes[track_id])}
        for track_id, sequence in sorted(class_sequence.items())
        if len(sequence) - 1 > max_class_changes
    ]
    risk_items: List[Dict[str, Any]] = []
    risk_items.extend({"track_id": track_id, "type": "short_track", "frame_count": frame_counts[track_id]} for track_id in short_tracks)
    risk_items.extend({"track_id": item["track_id"], "type": "class_drift", "changes": item["changes"]} for item in drift)
    risk_items.extend({"track_id": item["track_id"], "type": "large_jump", "frame": item["frame"], "jump_ratio": item["jump_ratio"]} for item in jumps)
    risk_ids = sorted({int(item["track_id"]) for item in risk_items})
    windows = read_json(windows_json).get("windows", []) if windows_json.exists() else []
    result = {
        "track_count": len(frame_counts),
        "frame_count": row_count,
        "window_count": len(windows) if isinstance(windows, list) else 0,
        "short_track_count": len(short_tracks),
        "short_tracks": short_tracks,
        "class_drift_count": len(drift),
        "class_drift": drift,
        "large_jump_count": len(jumps),
        "large_jumps": jumps,
        "risk_items": risk_items,
        "risk_track_ids": risk_ids,
        "needs_strong_review": bool(risk_ids),
        "passed": not risk_ids,
    }
    write_json(out_json, result)
    return result
