from __future__ import annotations

"""Evidence-ranked, open-object predicate routing."""

import math
from typing import Any, Collection, Mapping

from vidvrd_auto.relations.candidate_priors import predicate_prior_scores

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
        # Co-occurrence can be meaningful even when both objects are stationary.
        proximity = max(0.0, 1.0 - float(evidence["edge_gap_min"]))
        return max(0.25, float(evidence["motion_alignment"]) * proximity)
    if spatial in {"front", "behind", "inside"}:
        # 2-D boxes cannot establish depth; keep both hypotheses for visual review.
        return 0.30
    return 0.10


def _action_support(action: str, evidence: dict[str, Any]) -> float:
    speed = float(evidence["subject_speed"])
    if action in {"lie", "sit", "stand", "stop"}:
        return max(0.0, 1.0 - speed * 8.0)
    if action == "jump":
        vertical = float(evidence.get("subject_vertical_speed", 0.0))
        reversals = int(evidence.get("subject_vertical_direction_changes", 0))
        return min(1.0, speed * 5.0 + vertical * 10.0 + min(0.35, reversals * 0.2))
    if action in {"creep", "fly", "move", "run", "swim", "walk"}:
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
    evidence: dict[str, Any], *, split: str = "all", limit: int = 14, exclude: Collection[str] = (),
    subject_category: str = "unknown", object_category: str = "unknown",
    predicate_priors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scores = family_scores(evidence)
    priors = predicate_prior_scores(predicate_priors, subject_category, object_category)
    ranked: list[tuple[float, str, list[str]]] = []
    records = [record for record in predicate_records(split) if record["name"] not in exclude]
    for record in records:
        components = record["components"]
        # A composite predicate needs support for all participating families;
        # a left/right cue alone must not promote "run_left" for a static pair.
        family_score = min(scores[family] for family in record["families"])
        spatial_score = _spatial_support(str(components.get("spatial", "")), evidence)
        action_score = _action_support(str(components.get("action", "")), evidence)
        jump_bonus = (
            min(0.12, float(evidence.get("subject_vertical_speed", 0.0)) * 0.8)
            if str(components.get("action", "")) == "jump"
            else 0.0
        )
        geometric_score = 0.60 * family_score + 0.25 * spatial_score + 0.15 * action_score + jump_bonus
        score = 0.72 * geometric_score + 0.28 * priors.get(str(record["name"]), 0.0)
        ranked.append((score, str(record["name"]), list(record["families"])))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    selected_families = [name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:3]]
    by_name = {str(record["name"]): record for record in records}

    def composite(record: dict[str, Any]) -> bool:
        components = record["components"]
        return sum(bool(components.get(key)) for key in ("action", "spatial", "comparative")) >= 2

    # Family routing is descriptive, not a hard gate: a visually supported
    # semantic action must not disappear because its family ranked fourth.
    atomic_ranked = [name for _, name, _ in ranked if not composite(by_name[name])]
    composite_ranked_all = []
    for _, name, _ in ranked:
        record = by_name[name]
        if not composite(record):
            continue
        components = record["components"]
        action = str(components.get("action", ""))
        spatial = str(components.get("spatial", ""))
        if action and _action_support(action, evidence) < 0.12:
            continue
        if spatial and _spatial_support(spatial, evidence) < 0.12:
            continue
        composite_ranked_all.append(name)

    # Avoid exhausting the composite quota on alphabetical variants of one
    # action/spatial cue. This is candidate diversity, not a prediction.
    composite_ranked: list[str] = []
    remaining = list(composite_ranked_all)
    ranked_scores = {name: score for score, name, _ in ranked}
    while remaining:
        action_counts = {name: sum(
            by_name[item]["components"].get("action") == name for item in composite_ranked
        ) for name in {by_name[item]["components"].get("action") for item in remaining}}
        spatial_counts = {name: sum(
            by_name[item]["components"].get("spatial") == name for item in composite_ranked
        ) for name in {by_name[item]["components"].get("spatial") for item in remaining}}
        choice = max(remaining, key=lambda name: (
            ranked_scores[name]
            - 0.08 * action_counts.get(by_name[name]["components"].get("action"), 0)
            - (0.10 if by_name[name]["components"].get("spatial") == "with" else 0.01)
            * spatial_counts.get(by_name[name]["components"].get("spatial"), 0),
            ranked_scores[name], name,
        ))
        composite_ranked.append(choice)
        remaining.remove(choice)

    atomic_quota = min(6, limit)
    composite_quota = min(4, max(0, limit - atomic_quota))
    confusion_quota = min(2, max(0, limit - atomic_quota - composite_quota))
    selected = list(dict.fromkeys(atomic_ranked[:atomic_quota] + composite_ranked[:composite_quota]))
    confusion: list[str] = []
    sibling_lists = [confusion_siblings(predicate) for predicate in composite_ranked[:composite_quota] + atomic_ranked[:atomic_quota]]
    for sibling_index in range(max((len(values) for values in sibling_lists), default=0)):
        for siblings in sibling_lists:
            if sibling_index >= len(siblings):
                continue
            sibling = siblings[sibling_index]
            if sibling not in selected and sibling not in confusion and sibling in by_name:
                confusion.append(sibling)
    selected.extend(confusion[:confusion_quota])
    supplemental = [name for _, name, _ in ranked if name not in selected]
    selected.extend(supplemental[: max(0, limit - len(selected))])
    selected = list(dict.fromkeys(selected))[:limit]
    expandable = neighboring_families(selected_families)
    return {
        "candidate_policy": "quota_composite_predicate_v3_train_prior",
        "candidate_predicates": selected,
        "candidate_families": selected_families,
        "family_scores": {name: round(value, 4) for name, value in sorted(scores.items())},
        "family_descriptions": {name: family_descriptions()[name] for name in selected_families},
        "expandable_families": expandable,
        "ranked_predicates": [name for _, name, _ in ranked],
        "candidate_buckets": {
            "atomic": [name for name in selected if name in atomic_ranked[:atomic_quota]],
            "composite": [name for name in selected if name in composite_ranked[:composite_quota]],
            "confusion": [name for name in selected if name in confusion[:confusion_quota]],
            "supplemental": [name for name in selected if name in supplemental],
        },
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
