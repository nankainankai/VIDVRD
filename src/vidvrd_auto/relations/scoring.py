from __future__ import annotations

"""Cross-source relation score calibration with explicit components."""

import math
from typing import Any, Mapping

from vidvrd_auto.core.schema import relation_span


def temporal_coverage_quality(item: Mapping[str, Any]) -> float:
    span = relation_span(dict(item))
    duration = max(1, span.end_frame - span.start_frame)
    frames = sorted({int(value) for value in item.get("evidence_frames", [])})
    if not frames:
        return 0.25
    evidence_span = max(1, min(span.end_frame, frames[-1] + 1) - max(span.start_frame, frames[0]))
    span_coverage = min(1.0, evidence_span / duration)
    evidence_volume = 1.0 - math.exp(-len(frames) / 3.0)
    return max(0.05, min(1.0, math.sqrt(span_coverage * evidence_volume)))


def calibrated_score_fields(
    item: Mapping[str, Any],
    *,
    track_quality: Mapping[str, Any] | None = None,
    source_coefficients: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    qualities = track_quality or {}
    coefficients = {
        "semantic": 1.0,
        "mixed": 0.9,
        "rule": 0.55,
        "coupling": 0.5,
        **{str(key): float(value) for key, value in (source_coefficients or {}).items()},
    }
    agent = item.get("agent_score")
    rule = item.get("rule_support")
    if agent is not None and rule is not None:
        source_kind = "mixed"
        evidence_score = 0.75 * float(agent) + 0.25 * float(rule)
    elif agent is not None:
        source_kind = "semantic"
        evidence_score = float(agent)
    else:
        source_kind = "coupling" if "coupling" in item.get("sources", []) else "rule"
        evidence_score = float(rule if rule is not None else item.get("confidence", 0.0) or 0.0)
    try:
        subject_id = str(int(item.get("subject_track_id", item.get("subject_id"))))
        object_id = str(int(item.get("object_track_id", item.get("object_id"))))
    except (TypeError, ValueError):
        subject_id = object_id = ""

    def quality(track_id: str) -> float:
        value = qualities.get(track_id, 1.0)
        if isinstance(value, Mapping):
            value = value.get("quality", 1.0)
        return max(0.0, min(1.0, float(value)))

    pair_quality = min(quality(subject_id), quality(object_id))
    def is_risky(track_id: str) -> bool:
        value = qualities.get(track_id, 1.0)
        if isinstance(value, Mapping):
            return bool(value.get("risk_types"))
        return float(value) < 0.5

    temporal = temporal_coverage_quality(item)
    coefficient = max(0.0, min(1.0, coefficients.get(source_kind, 1.0)))
    ranking = max(0.0, min(1.0, evidence_score * temporal * pair_quality * coefficient))
    return {
        "evidence_score": round(evidence_score, 6),
        "temporal_quality": round(temporal, 6),
        "track_pair_quality": round(pair_quality, 6),
        "source_calibration": round(coefficient, 6),
        "source_kind": source_kind,
        "ranking_score": round(ranking, 6),
        "score_kind": "calibrated_relation_v1",
        "risk_track_involved": is_risky(subject_id) or is_risky(object_id),
    }
