from __future__ import annotations

"""Training-only predicate priors for bounded candidate routing."""

from collections import Counter, defaultdict
from typing import Any, Mapping

from vidvrd_auto.core.ontology import normalize_object


def learn_predicate_priors(
    relations: Mapping[str, Any], trajectories: Mapping[str, Any], video_ids: set[str]
) -> dict[str, Any]:
    global_counts: Counter[str] = Counter()
    pair_counts: dict[str, Counter[str]] = defaultdict(Counter)
    subject_counts: dict[str, Counter[str]] = defaultdict(Counter)
    seen: set[tuple[str, int, str, int]] = set()
    for video_id in sorted(video_ids):
        categories = {
            int(item["track_id"]): normalize_object(str(item.get("category", "unknown")))
            for item in trajectories.get(video_id, [])
            if isinstance(item, dict) and "track_id" in item
        }
        for relation in relations.get(video_id, []):
            if not isinstance(relation, dict):
                continue
            try:
                subject = int(relation["subject_track_id"])
                obj = int(relation["object_track_id"])
                predicate = str(relation["predicate"])
            except (KeyError, TypeError, ValueError):
                continue
            identity = (video_id, subject, predicate, obj)
            if identity in seen:
                continue
            seen.add(identity)
            subject_category = categories.get(subject, "unknown")
            object_category = categories.get(obj, "unknown")
            global_counts[predicate] += 1
            subject_counts[subject_category][predicate] += 1
            pair_counts[f"{subject_category}|{object_category}"][predicate] += 1
    return {
        "source": "training_annotations_only",
        "training_video_count": len(video_ids),
        "global": dict(sorted(global_counts.items())),
        "subject": {key: dict(sorted(value.items())) for key, value in sorted(subject_counts.items())},
        "pair": {key: dict(sorted(value.items())) for key, value in sorted(pair_counts.items())},
    }


def predicate_prior_scores(
    priors: Mapping[str, Any] | None, subject_category: str, object_category: str
) -> dict[str, float]:
    if not priors:
        return {}
    subject = normalize_object(subject_category)
    obj = normalize_object(object_category)
    global_counts = priors.get("global", {})
    subject_counts = priors.get("subject", {}).get(subject, {})
    pair_counts = priors.get("pair", {}).get(f"{subject}|{obj}", {})
    if not isinstance(global_counts, dict) or not isinstance(subject_counts, dict) or not isinstance(pair_counts, dict):
        return {}
    global_total = sum(global_counts.values()) or 1
    subject_total = sum(subject_counts.values())
    pair_total = sum(pair_counts.values())
    subject_weight = subject_total / (subject_total + 10.0)
    pair_weight = pair_total / (pair_total + 5.0)
    values: dict[str, float] = {}
    for name in set(global_counts) | set(subject_counts) | set(pair_counts):
        global_rate = float(global_counts.get(name, 0)) / global_total
        subject_rate = float(subject_counts.get(name, 0)) / subject_total if subject_total else 0.0
        pair_rate = float(pair_counts.get(name, 0)) / pair_total if pair_total else 0.0
        values[name] = (
            (1.0 - pair_weight) * ((1.0 - subject_weight) * global_rate + subject_weight * subject_rate)
            + pair_weight * pair_rate
        )
    maximum = max(values.values(), default=0.0)
    return {name: value / maximum for name, value in values.items()} if maximum else {}
