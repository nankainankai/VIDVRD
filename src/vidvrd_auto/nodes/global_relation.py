from __future__ import annotations

"""Deterministically aggregate overlapping window-level relations."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from vidvrd_auto.utils.io import read_json, write_json
from vidvrd_auto.core.ontology import predicate_components
from vidvrd_auto.core.schema import serialize_relation_artifact
from vidvrd_auto.relations.scoring import calibrated_score_fields


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


def _weighted_average(segments: List[Dict[str, Any]], field: str) -> float | None:
    values = [item for item in segments if item.get(field) is not None]
    if not values:
        return None
    weights = [max(1, len(_evidence_frames(item))) for item in values]
    return sum(float(item[field]) * weight for item, weight in zip(values, weights)) / sum(weights)


def _aggregate(
    key: Tuple[int, str, int], segments: List[Dict[str, Any]], config: Dict[str, Any]
) -> Dict[str, Any]:
    subject_id, predicate, object_id = key
    starts, ends = zip(*(_span(item) for item in segments))
    evidence_frames = sorted({frame for item in segments for frame in _evidence_frames(item)})
    rule_support = _weighted_average(segments, "rule_support")
    agent_score = _weighted_average(segments, "agent_score")
    result = {
        "subject_track_id": subject_id,
        "predicate": predicate,
        "predicate_components": predicate_components(predicate),
        "object_track_id": object_id,
        "start_frame": min(starts),
        "end_frame": max(ends),
        "evidence_frames": evidence_frames,
        "source": "cross_window_aggregate",
        "sources": _sources(segments),
        "segment_count": len(segments),
        "segment_ids": [item.get("segment_id") for item in segments if item.get("segment_id") is not None],
        "evidence": f"aggregated {len(segments)} overlapping window segment(s)",
    }
    if any(bool(item.get("split_on_track_jump")) for item in segments):
        result["split_on_track_jump"] = True
    if rule_support is not None:
        result["rule_support"] = round(rule_support, 6)
    if agent_score is not None:
        result["agent_score"] = round(agent_score, 6)
    result.update(calibrated_score_fields(
        result,
        track_quality=config.get("track_quality", {}),
        source_coefficients=config.get("source_coefficients", {}),
    ))
    return result


def _split_on_track_jumps(item: Dict[str, Any], jump_frames: Dict[str, Any]) -> List[Dict[str, Any]]:
    start, end = _span(item)
    subject_id, _, object_id = _key(item) or (-1, "", -1)
    boundaries = sorted({
        int(frame)
        for track_id in (subject_id, object_id)
        for frame in jump_frames.get(str(track_id), [])
        if start < int(frame) <= end
    })
    if not boundaries:
        return [item]
    evidence = _evidence_frames(item)
    output: List[Dict[str, Any]] = []
    left = start
    for right in boundaries + [end + 1]:
        frames = [frame for frame in evidence if left <= frame < right]
        if frames:
            piece = dict(item)
            piece["start_frame"] = max(left, frames[0])
            piece["end_frame"] = min(right - 1, frames[-1])
            piece["evidence_frames"] = frames
            piece["split_on_track_jump"] = True
            output.append(piece)
        left = right
    return output


def _aggregate_items(items: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    max_gap = max(0, int(config.get("max_relation_gap_frames", config.get("max_window_gap", 1)) or 0))
    max_evidence_gap = max(0, int(config.get("max_evidence_gap", 4) or 0))
    max_merged_span = max(0, int(config.get("max_merged_span_frames", 0) or 0))
    grouped: Dict[Tuple[int, str, int], List[Dict[str, Any]]] = defaultdict(list)
    jump_frames = config.get("track_jump_frames", {})
    for raw in items if isinstance(items, list) else []:
        if not isinstance(raw, dict):
            continue
        for item in _split_on_track_jumps(serialize_relation_artifact(raw), jump_frames):
            key = _key(item)
            if key is not None:
                grouped[key].append(item)

    aggregated: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        ordered = sorted(grouped[key], key=lambda item: (_span(item)[0], _span(item)[1]))
        cluster: List[Dict[str, Any]] = []
        cluster_end = -1
        cluster_evidence_end = -1
        for item in ordered:
            start, end = _span(item)
            evidence_start = _evidence_frames(item)[0]
            track_boundaries = {
                int(frame)
                for track_id in (key[0], key[2])
                for frame in jump_frames.get(str(track_id), [])
            }
            crosses_track_jump = any(cluster_end < frame <= start for frame in track_boundaries)
            if cluster and (
                start > cluster_end + max_gap
                or evidence_start > cluster_evidence_end + max_evidence_gap
                or crosses_track_jump
                or (max_merged_span and max(cluster_end, end) - _span(cluster[0])[0] + 1 > max_merged_span)
            ):
                aggregated.append(_aggregate(key, cluster, config))
                cluster = []
                cluster_end = -1
                cluster_evidence_end = -1
            cluster.append(item)
            cluster_end = max(cluster_end, end)
            cluster_evidence_end = max(cluster_evidence_end, _evidence_frames(item)[-1])
        if cluster:
            aggregated.append(_aggregate(key, cluster, config))

    return aggregated


def _multiscale_proposals(items: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Keep local and coherent event spans, discounting redundant hypotheses.

    Uses only predicted segments and track QC, never annotation boundaries.
    The highest-scoring hypothesis represents each event first; alternatives
    remain available for short events instead of imposing a universal duration.
    """
    short = _aggregate_items(items, config)
    long = _aggregate_items(items, {**config, "max_merged_span_frames": 0})
    decay = float(config.get("secondary_proposal_decay", 0.5))
    if not 0.0 < decay <= 1.0:
        raise ValueError("secondary_proposal_decay must be in (0, 1]")
    proposals: List[Dict[str, Any]] = []
    for event_index, event in enumerate(long):
        start, end = _span(event)
        alternatives = [event] + [
            item for item in short
            if _key(item) == _key(event) and start <= _span(item)[0] and _span(item)[1] <= end
        ]
        unique = {}
        for item in alternatives:
            unique.setdefault(_span(item), item)
        ordered = sorted(unique.values(), key=lambda item: (
            -_ranking_score(item), -(_span(item)[1] - _span(item)[0]), _span(item)[0],
        ))
        for rank, raw in enumerate(ordered):
            item = dict(raw)
            item.update(
                temporal_proposal_mode="multiscale_v1",
                proposal_event_id=f"e{event_index + 1:06d}",
                proposal_rank=rank,
                pre_diversity_ranking_score=_ranking_score(raw),
                redundancy_coefficient=1.0 if rank == 0 else decay,
                ranking_score=round(_ranking_score(raw) * (1.0 if rank == 0 else decay), 6),
                score_kind="calibrated_relation_multiscale_v1",
            )
            proposals.append(item)
    return proposals


def run_global_relation(
    *, video_id: str, relations_json: Path, out_json: Path, config: Dict[str, Any]
) -> Dict[str, Any]:
    obj = read_json(relations_json) if relations_json.exists() else {video_id: []}
    items = obj.get(video_id, []) if isinstance(obj, dict) else []
    mode = str(config.get("temporal_proposal_mode", "single"))
    if mode not in {"single", "multiscale"}:
        raise ValueError(f"unsupported temporal_proposal_mode: {mode}")
    aggregated = _multiscale_proposals(items, config) if mode == "multiscale" else _aggregate_items(items, config)
    aggregated.sort(key=lambda item: (item["start_frame"], item["end_frame"], item["subject_track_id"], item["predicate"], item["object_track_id"]))
    for index, item in enumerate(aggregated, start=1):
        item["relation_id"] = f"r{index:06d}"
    result = {video_id: [serialize_relation_artifact(item) for item in aggregated]}
    write_json(out_json, result)
    return result
