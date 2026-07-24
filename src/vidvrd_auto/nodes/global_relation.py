from __future__ import annotations

"""Deterministically aggregate overlapping window-level relations."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from vidvrd_auto.utils.io import read_json, write_json
from vidvrd_auto.core.ontology import predicate_components


def _key(item: Dict[str, Any]) -> Tuple[int, str, int] | None:
    try:
        subject_id = int(item.get("subject_track_id", item.get("subject_id")))
        object_id = int(item.get("object_track_id", item.get("object_id")))
    except (TypeError, ValueError):
        return None
    predicate = str(item.get("predicate", "") or "").strip().lower()
    return (subject_id, predicate, object_id) if predicate else None


def _span(item: Dict[str, Any]) -> Tuple[int, int]:
    start = int(item.get("start_frame", 0) or 0)
    end = int(item.get("end_frame", start) or start)
    return (start, end) if start <= end else (end, start)


def _confidence(item: Dict[str, Any]) -> float:
    try:
        return max(0.0, min(1.0, float(item.get("confidence", 0.0) or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _sources(items: List[Dict[str, Any]]) -> List[str]:
    values: List[str] = []
    for item in items:
        raw = item.get("sources") or [item.get("source", "unknown")]
        for source in raw:
            value = str(source or "unknown")
            if value not in values:
                values.append(value)
    return values


def _aggregate(key: Tuple[int, str, int], segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    subject_id, predicate, object_id = key
    starts, ends = zip(*(_span(item) for item in segments))
    weights = [max(1, end - start + 1) for start, end in zip(starts, ends)]
    confidence = sum(_confidence(item) * weight for item, weight in zip(segments, weights)) / sum(weights)
    return {
        "subject_track_id": subject_id,
        "predicate": predicate,
        "predicate_components": predicate_components(predicate),
        "object_track_id": object_id,
        "start_frame": min(starts),
        "end_frame": max(ends),
        "confidence": round(confidence, 4),
        "source": "cross_window_aggregate",
        "sources": _sources(segments),
        "segment_count": len(segments),
        "segment_ids": [item.get("segment_id") for item in segments if item.get("segment_id") is not None],
        "evidence": f"aggregated {len(segments)} overlapping window segment(s)",
    }


def run_global_relation(
    *, video_id: str, relations_json: Path, out_json: Path, config: Dict[str, Any]
) -> Dict[str, Any]:
    obj = read_json(relations_json) if relations_json.exists() else {video_id: []}
    items = obj.get(video_id, []) if isinstance(obj, dict) else []
    max_gap = max(0, int(config.get("max_window_gap", 1) or 0))
    grouped: Dict[Tuple[int, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for raw in items if isinstance(items, list) else []:
        if not isinstance(raw, dict):
            continue
        key = _key(raw)
        if key is not None:
            grouped[key].append(dict(raw))

    aggregated: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        ordered = sorted(grouped[key], key=lambda item: (_span(item)[0], _span(item)[1]))
        cluster: List[Dict[str, Any]] = []
        cluster_end = -1
        for item in ordered:
            start, end = _span(item)
            if cluster and start > cluster_end + max_gap:
                aggregated.append(_aggregate(key, cluster))
                cluster = []
                cluster_end = -1
            cluster.append(item)
            cluster_end = max(cluster_end, end)
        if cluster:
            aggregated.append(_aggregate(key, cluster))

    aggregated.sort(key=lambda item: (item["start_frame"], item["end_frame"], item["subject_track_id"], item["predicate"], item["object_track_id"]))
    for index, item in enumerate(aggregated, start=1):
        item["relation_id"] = f"r{index:06d}"
    result = {video_id: aggregated}
    write_json(out_json, result)
    return result
