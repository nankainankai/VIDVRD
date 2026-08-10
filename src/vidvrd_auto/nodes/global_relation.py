from __future__ import annotations

"""Deterministically aggregate overlapping window-level relations."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from vidvrd_auto.utils.io import read_json, write_json
from vidvrd_auto.core.ontology import predicate_components
from vidvrd_auto.core.schema import serialize_relation_artifact


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


def _ranking_score(item: Dict[str, Any]) -> float:
    for field in ("ranking_score", "agent_score", "rule_support", "confidence"):
        if item.get(field) is not None:
            return max(0.0, min(1.0, float(item[field])))
    return 0.0


def _evidence_frames(item: Dict[str, Any]) -> List[int]:
    frames = item.get("evidence_frames", [])
    if isinstance(frames, list) and frames:
        return sorted({int(frame) for frame in frames})
    start, end = _span(item)
    return [start, end] if start != end else [start]


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
    evidence_frames = sorted({frame for item in segments for frame in _evidence_frames(item)})
    rule_supports = [float(item["rule_support"]) for item in segments if item.get("rule_support") is not None]
    agent_scores = [float(item["agent_score"]) for item in segments if item.get("agent_score") is not None]
    ranking_score = max((_ranking_score(item) for item in segments), default=0.0)
    result = {
        "subject_track_id": subject_id,
        "predicate": predicate,
        "predicate_components": predicate_components(predicate),
        "object_track_id": object_id,
        "start_frame": min(starts),
        "end_frame": max(ends),
        "evidence_frames": evidence_frames,
        "ranking_score": round(ranking_score, 4),
        "score_kind": (
            "mixed_ranking" if rule_supports and agent_scores
            else "rule_support" if rule_supports
            else "agent_ranking" if agent_scores
            else "legacy_confidence"
        ),
        "source": "cross_window_aggregate",
        "sources": _sources(segments),
        "segment_count": len(segments),
        "segment_ids": [item.get("segment_id") for item in segments if item.get("segment_id") is not None],
        "evidence": f"aggregated {len(segments)} overlapping window segment(s)",
    }
    if rule_supports:
        result["rule_support"] = round(max(rule_supports), 4)
    if agent_scores:
        result["agent_score"] = round(max(agent_scores), 4)
    return result


def run_global_relation(
    *, video_id: str, relations_json: Path, out_json: Path, config: Dict[str, Any]
) -> Dict[str, Any]:
    obj = read_json(relations_json) if relations_json.exists() else {video_id: []}
    items = obj.get(video_id, []) if isinstance(obj, dict) else []
    max_gap = max(0, int(config.get("max_relation_gap_frames", config.get("max_window_gap", 1)) or 0))
    max_evidence_gap = max(0, int(config.get("max_evidence_gap", 4) or 0))
    grouped: Dict[Tuple[int, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for raw in items if isinstance(items, list) else []:
        if not isinstance(raw, dict):
            continue
        key = _key(raw)
        if key is not None:
            grouped[key].append(serialize_relation_artifact(raw))

    aggregated: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        ordered = sorted(grouped[key], key=lambda item: (_span(item)[0], _span(item)[1]))
        cluster: List[Dict[str, Any]] = []
        cluster_end = -1
        cluster_evidence_end = -1
        for item in ordered:
            start, end = _span(item)
            evidence_start = _evidence_frames(item)[0]
            if cluster and (start > cluster_end + max_gap or evidence_start > cluster_evidence_end + max_evidence_gap):
                aggregated.append(_aggregate(key, cluster))
                cluster = []
                cluster_end = -1
                cluster_evidence_end = -1
            cluster.append(item)
            cluster_end = max(cluster_end, end)
            cluster_evidence_end = max(cluster_evidence_end, _evidence_frames(item)[-1])
        if cluster:
            aggregated.append(_aggregate(key, cluster))

    aggregated.sort(key=lambda item: (item["start_frame"], item["end_frame"], item["subject_track_id"], item["predicate"], item["object_track_id"]))
    for index, item in enumerate(aggregated, start=1):
        item["relation_id"] = f"r{index:06d}"
    result = {video_id: [serialize_relation_artifact(item) for item in aggregated]}
    write_json(out_json, result)
    return result
