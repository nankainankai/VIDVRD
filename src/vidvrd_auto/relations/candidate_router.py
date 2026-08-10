from __future__ import annotations

"""Evidence-ranked, open-object predicate routing."""

import math
from typing import Any, Collection

from vidvrd_auto.relations.predicate_hierarchy import (
    confusion_siblings,
    family_descriptions,
    neighboring_families,
    predicate_records,
)


def family_scores(evidence: dict[str, Any]) -> dict[str, float]:
    proximity = max(0.0, 1.0 - float(evidence["edge_gap_min"]))
    contact = max(float(evidence["iou_max"]), proximity)
    motion = min(1.0, abs(float(evidence["approach_rate"])) * 12.0 + float(evidence["relative_motion"]) * 4.0)
    alignment = max(0.0, float(evidence["motion_alignment"]))
    size_ratio = float(evidence["size_ratio"])
    comparative = min(1.0, abs(math.log(max(1e-6, size_ratio), 2)))
    return {
        "geometry": 0.55,
        "relative_motion": max(0.15, motion, 0.8 * alignment),
        "locomotion_state": max(0.20, min(1.0, float(evidence["subject_speed"]) * 5.0)),
        "contact": max(0.10, contact),
        "manipulation": max(0.10, 0.65 * contact + 0.35 * alignment),
        "attention_social": max(0.20, 0.40 * proximity + 0.35 * alignment),
        "comparative": max(0.10, comparative),
    }


def _spatial_support(spatial: str, evidence: dict[str, Any]) -> float:
    dx, dy = float(evidence["median_dx"]), float(evidence["median_dy"])
    if spatial == "left":
        return max(0.0, min(1.0, -dx))
    if spatial == "right":
        return max(0.0, min(1.0, dx))
    if spatial == "above":
        return max(0.0, min(1.0, -dy))
    if spatial == "beneath":
        return max(0.0, min(1.0, dy))
    if spatial == "toward":
        return max(0.0, min(1.0, float(evidence["approach_rate"]) * 12.0))
    if spatial == "away":
        return max(0.0, min(1.0, -float(evidence["approach_rate"]) * 12.0))
    if spatial == "past":
        return 1.0 if evidence["crosses_horizontal_axis"] else 0.0
    if spatial == "next_to":
        return max(0.0, 1.0 - float(evidence["distance_min"]))
    if spatial == "with":
        return max(0.0, float(evidence["motion_alignment"]))
    return 0.25


def _action_support(action: str, evidence: dict[str, Any]) -> float:
    speed = float(evidence["subject_speed"])
    if action in {"lie", "sit", "stand", "stop"}:
        return max(0.0, 1.0 - speed * 8.0)
    if action in {"creep", "fly", "jump", "move", "run", "swim", "walk"}:
        return min(1.0, speed * 8.0)
    if action in {"chase", "follow"}:
        return min(
            1.0,
            speed * 5.0
            + max(0.0, float(evidence["approach_rate"])) * 8.0
            + max(0.0, float(evidence["motion_alignment"])) * 0.3,
        )
    return 0.5 if action else 0.25


def route_predicates(
    evidence: dict[str, Any], *, split: str = "all", limit: int = 14, exclude: Collection[str] = ()
) -> dict[str, Any]:
    scores = family_scores(evidence)
    ranked: list[tuple[float, str, list[str]]] = []
    records = [record for record in predicate_records(split) if record["name"] not in exclude]
    for record in records:
        components = record["components"]
        # A composite predicate needs support for all participating families;
        # a left/right cue alone must not promote "run_left" for a static pair.
        family_score = min(scores[family] for family in record["families"])
        spatial_score = _spatial_support(str(components.get("spatial", "")), evidence)
        action_score = _action_support(str(components.get("action", "")), evidence)
        score = 0.60 * family_score + 0.25 * spatial_score + 0.15 * action_score
        ranked.append((score, str(record["name"]), list(record["families"])))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    selected: list[str] = []
    selected_families = [name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:3]]
    selected_family_set = set(selected_families)
    eligible = [
        (score, name, families)
        for score, name, families in ranked
        if set(families).issubset(selected_family_set)
    ]
    for family in selected_families:
        family_candidates = [name for _, name, families in eligible if family in families]
        selected.extend(family_candidates[:2])
    selected = list(dict.fromkeys(selected))
    for predicate in list(selected):
        for sibling in confusion_siblings(predicate):
            if sibling not in selected and any(record["name"] == sibling for record in records):
                selected.append(sibling)
    selected.extend(name for _, name, _ in eligible)
    selected = list(dict.fromkeys(selected))[:limit]
    selected = selected[:limit]
    expandable = neighboring_families(selected_families)
    return {
        "candidate_policy": "hierarchical_predicate_v1",
        "candidate_predicates": selected,
        "candidate_families": selected_families,
        "family_scores": {name: round(value, 4) for name, value in sorted(scores.items())},
        "family_descriptions": {name: family_descriptions()[name] for name in selected_families},
        "expandable_families": expandable,
        "ranked_predicates": [name for _, name, _ in ranked],
    }


def expand_route(
    route: dict[str, Any], families: list[str], *, limit: int = 24, split: str = "all"
) -> dict[str, Any]:
    allowed = set(route["expandable_families"])
    requested = [family for family in families if family in allowed]
    family_names = {
        str(record["name"])
        for record in predicate_records(split)
        if any(family in record["families"] for family in requested)
    }
    additions = [name for name in route["ranked_predicates"] if name in family_names]
    output = dict(route)
    output["candidate_predicates"] = list(dict.fromkeys(list(route["candidate_predicates"]) + additions))[:limit]
    output["candidate_families"] = list(dict.fromkeys(list(route["candidate_families"]) + requested))
    descriptions = family_descriptions()
    output["family_descriptions"] = {
        family: descriptions[family] for family in output["candidate_families"]
    }
    output["expandable_families"] = [
        family for family in neighboring_families(output["candidate_families"])
        if family not in output["candidate_families"]
    ]
    output["expanded_families"] = requested
    return output
