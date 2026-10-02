from __future__ import annotations

"""Optimistic, label-agnostic ceiling imposed by predicted object trajectories."""

from collections import Counter
from typing import Any, Mapping, Sequence

from vidvrd_auto.core.ontology import normalize_object


def _area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0]) + 1.0) * max(0.0, float(box[3]) - float(box[1]) + 1.0)


def _intersection(left: Sequence[float], right: Sequence[float]) -> float:
    width = max(0.0, min(float(left[2]), float(right[2])) - max(float(left[0]), float(right[0])) + 1.0)
    height = max(0.0, min(float(left[3]), float(right[3])) - max(float(left[1]), float(right[1])) + 1.0)
    return width * height


def _best_role_coverage(
    category: str, duration: Sequence[int], gold_boxes: Sequence[Sequence[float]],
    predicted_tracks: Sequence[Mapping[str, Any]],
) -> tuple[float, int | None, bool]:
    candidates = [
        track for track in predicted_tracks
        if normalize_object(str(track.get("category", "unknown"))) == normalize_object(category)
    ]
    if not candidates:
        return 0.0, None, False
    gold_volume = sum(_area(box) for box in gold_boxes)
    if gold_volume <= 0:
        return 0.0, None, True
    best = (0.0, None)
    for track in candidates:
        trajectory = track.get("trajectory", {})
        overlap = sum(
            _intersection(gold_box, trajectory[str(frame)])
            for frame, gold_box in zip(range(int(duration[0]), int(duration[1])), gold_boxes)
            if str(frame) in trajectory
        )
        coverage = overlap / gold_volume
        if coverage > best[0]:
            best = (coverage, int(track["track_id"]))
    return best[0], best[1], True


def diagnose_track_ceiling(
    official_gold: Mapping[str, Sequence[Mapping[str, Any]]],
    predicted_trajectories: Mapping[str, Sequence[Mapping[str, Any]]],
    *, threshold: float = 0.5,
) -> dict[str, Any]:
    """Bound achievable relation recall even with oracle predicates/timestamps.

    For each role, intersection volume with the best same-category track is
    divided only by Gold volume, omitting every prediction-only penalty. The
    resulting value is an *upper* bound on official tube vIoU. Subject and
    object tracks are selected independently, so this is intentionally loose.
    """

    rows = []
    reasons: Counter[str] = Counter()
    for video_id, relations in official_gold.items():
        tracks = predicted_trajectories.get(video_id, [])
        for index, target in enumerate(relations):
            subject, predicate, obj = target["triplet"]
            duration = target["duration"]
            subject_bound, subject_id, subject_exists = _best_role_coverage(
                subject, duration, target["sub_traj"], tracks
            )
            object_bound, object_id, object_exists = _best_role_coverage(
                obj, duration, target["obj_traj"], tracks
            )
            bound = min(subject_bound, object_bound)
            reason = (
                "category_absent" if not subject_exists or not object_exists else
                "insufficient_track_coverage" if bound < threshold else
                "track_ceiling_allows_match"
            )
            reasons[reason] += 1
            rows.append({
                "video_id": video_id,
                "groundtruth_index": index,
                "triplet": [subject, predicate, obj],
                "duration": list(duration),
                "subject_coverage_upper_bound": round(subject_bound, 6),
                "object_coverage_upper_bound": round(object_bound, 6),
                "pair_viou_upper_bound": round(bound, 6),
                "best_subject_track_id": subject_id,
                "best_object_track_id": object_id,
                "reason": reason,
            })
    total = len(rows)
    return {
        "scope": "optimistic official-vIoU upper bound from predicted tracks; not mAP or achieved recall",
        "viou_threshold": threshold,
        "groundtruth_relation_count": total,
        "maximum_matchable_relations": reasons["track_ceiling_allows_match"],
        "maximum_recall_fraction": reasons["track_ceiling_allows_match"] / total if total else 0.0,
        "reason_counts": dict(sorted(reasons.items())),
        "rows": rows,
    }


def summarize_track_ceiling_outcomes(
    ceiling: Mapping[str, Any], official_errors: Mapping[str, Any],
    routing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Separate trajectory-limited misses from misses after adequate coverage."""

    outcomes = {
        (str(row["video_id"]), int(row["groundtruth_index"])): row
        for row in official_errors.get("groundtruth", [])
    }
    reachable_failures: Counter[str] = Counter()
    missing_by_route: Counter[str] = Counter()
    route_by_key = {
        (str(row["video_id"]), int(row["groundtruth_index"])): str(row["status"])
        for row in (routing or {}).get("rows", [])
    }
    true_positives = 0
    inconsistent = 0
    for row in ceiling.get("rows", []):
        outcome = outcomes.get((str(row["video_id"]), int(row["groundtruth_index"])))
        if outcome is None:
            continue
        matched = outcome.get("true_positive_rank") is not None
        reachable = row["reason"] == "track_ceiling_allows_match"
        if matched:
            true_positives += 1
            if not reachable:
                inconsistent += 1
        elif reachable:
            failure = str(outcome.get("failure_type", "unknown"))
            reachable_failures[failure] += 1
            if failure == "candidate_missing" and routing is not None:
                key = (str(row["video_id"]), int(row["groundtruth_index"]))
                missing_by_route[route_by_key.get(key, "routing_row_missing")] += 1
    possible = int(ceiling["maximum_matchable_relations"])
    return {
        "actual_true_positive_count_any_rank": true_positives,
        "conditional_realized_fraction": true_positives / possible if possible else 0.0,
        "optimistic_bound_inconsistency_count": inconsistent,
        "potentially_matchable_failure_counts": dict(sorted(reachable_failures.items())),
        "potentially_matchable_candidate_missing_by_route": dict(sorted(missing_by_route.items())),
    }
