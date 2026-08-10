from __future__ import annotations

"""ImageNet-VidVRD evaluator compatible with the official helper protocol."""

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from vidvrd_auto.core.schema import relation_span


def _score(item: Dict[str, Any]) -> float:
    for field in ("ranking_score", "agent_score", "rule_support", "confidence", "score"):
        if item.get(field) is not None:
            return float(item[field])
    return 0.0


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    recall_curve = np.concatenate(([0.0], recall, [1.0]))
    precision_curve = np.concatenate(([0.0], precision, [0.0]))
    for index in range(precision_curve.size - 1, 0, -1):
        precision_curve[index - 1] = max(precision_curve[index - 1], precision_curve[index])
    changes = np.where(recall_curve[1:] != recall_curve[:-1])[0]
    return float(np.sum((recall_curve[changes + 1] - recall_curve[changes]) * precision_curve[changes + 1]))


def trajectory_viou(
    left_trajectory: Sequence[Sequence[float]],
    left_duration: Sequence[int],
    right_trajectory: Sequence[Sequence[float]],
    right_duration: Sequence[int],
) -> float:
    """Official voluminal IoU for half-open durations and contiguous tubes."""

    left_start, left_end = int(left_duration[0]), int(left_duration[1])
    right_start, right_end = int(right_duration[0]), int(right_duration[1])
    overlap_start, overlap_end = max(left_start, right_start), min(left_end, right_end)
    if overlap_start >= overlap_end:
        return 0.0
    overlap = 0.0
    for frame in range(overlap_start, overlap_end):
        left = left_trajectory[frame - left_start]
        right = right_trajectory[frame - right_start]
        width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]) + 1.0)
        height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]) + 1.0)
        overlap += width * height
    left_volume = sum((box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0) for box in left_trajectory)
    right_volume = sum((box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0) for box in right_trajectory)
    union = left_volume + right_volume - overlap
    return overlap / union if union > 0 else 0.0


def _detection_scores(
    groundtruth: Sequence[Dict[str, Any]], predictions: Sequence[Dict[str, Any]], threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = sorted(predictions, key=lambda item: float(item["score"]), reverse=True)
    detected = np.zeros(len(groundtruth), dtype=bool)
    hit_scores = np.full(len(ordered), -np.inf, dtype=float)
    for prediction_index, prediction in enumerate(ordered):
        best_overlap, best_index = -1.0, -1
        for target_index, target in enumerate(groundtruth):
            if detected[target_index] or tuple(prediction["triplet"]) != tuple(target["triplet"]):
                continue
            subject_overlap = trajectory_viou(
                prediction["sub_traj"], prediction["duration"], target["sub_traj"], target["duration"]
            )
            object_overlap = trajectory_viou(
                prediction["obj_traj"], prediction["duration"], target["obj_traj"], target["duration"]
            )
            overlap = min(subject_overlap, object_overlap)
            if overlap >= threshold and overlap > best_overlap:
                best_overlap, best_index = overlap, target_index
        if best_index >= 0:
            detected[best_index] = True
            hit_scores[prediction_index] = float(prediction["score"])
    true_positive = np.isfinite(hit_scores)
    cumulative_true = np.cumsum(true_positive).astype(float)
    cumulative_false = np.cumsum(~true_positive).astype(float)
    recall = cumulative_true / max(len(groundtruth), np.finfo(float).eps)
    precision = cumulative_true / np.maximum(cumulative_true + cumulative_false, np.finfo(float).eps)
    return precision, recall, hit_scores


def _tagging_precision(groundtruth: Sequence[Dict[str, Any]], predictions: Sequence[Dict[str, Any]]) -> np.ndarray:
    target_triplets = {tuple(item["triplet"]) for item in groundtruth}
    predicted_triplets: List[Tuple[str, str, str]] = []
    hits: List[bool] = []
    for item in sorted(predictions, key=lambda value: float(value["score"]), reverse=True):
        triplet = tuple(item["triplet"])
        if triplet in predicted_triplets:
            continue
        predicted_triplets.append(triplet)
        hits.append(triplet in target_triplets)
    if not hits:
        return np.asarray([], dtype=float)
    cumulative = np.cumsum(np.asarray(hits, dtype=bool)).astype(float)
    return cumulative / np.arange(1, len(hits) + 1, dtype=float)


def evaluate_official_vidvrd(
    groundtruth: Dict[str, List[Dict[str, Any]]],
    predictions: Dict[str, List[Dict[str, Any]]],
    *,
    viou_threshold: float = 0.5,
    detection_returns: Sequence[int] = (50, 100),
    tagging_returns: Sequence[int] = (1, 5, 10),
) -> Dict[str, Any]:
    """Reproduce the official per-video AP, detection recall and tagging precision."""

    video_ap: Dict[str, float] = {}
    detection_hits: Dict[int, List[np.ndarray]] = defaultdict(list)
    tagging_values: Dict[int, List[float]] = defaultdict(list)
    relation_count = 0
    for video_id, targets in groundtruth.items():
        if not targets:
            continue
        relation_count += len(targets)
        predicted = predictions.get(video_id, [])
        precision, recall, hit_scores = _detection_scores(targets, predicted, viou_threshold)
        video_ap[video_id] = _voc_ap(recall, precision)
        hits = np.isfinite(hit_scores)
        for limit in detection_returns:
            detection_hits[limit].append(hits[: min(limit, hits.size)])
        tag_precision = _tagging_precision(targets, predicted)
        for limit in tagging_returns:
            cutoff = min(limit, tag_precision.size)
            tagging_values[limit].append(float(tag_precision[cutoff - 1]) if cutoff else 0.0)

    detection_recall = {
        str(limit): (
            sum(int(values.sum()) for values in detection_hits[limit]) / relation_count
            if relation_count else 0.0
        )
        for limit in detection_returns
    }
    tagging_precision = {
        str(limit): float(np.mean(tagging_values[limit])) if tagging_values[limit] else 0.0
        for limit in tagging_returns
    }
    return {
        "evaluator": "imagenet_vidvrd_official_2017_compatible_v1",
        "protocol": "official_helper_relation_detection_and_tagging",
        "viou_threshold": viou_threshold,
        "evaluated_videos": list(groundtruth),
        "evaluated_video_count": len(groundtruth),
        "groundtruth_relation_count": relation_count,
        "relation_detection": {
            "mean_ap": float(np.mean(list(video_ap.values()))) if video_ap else 0.0,
            "recall_at": detection_recall,
            "per_video_ap": video_ap,
        },
        "relation_tagging": {"precision_at": tagging_precision},
    }


def _continuous_runs(frames: Iterable[int]) -> List[List[int]]:
    runs: List[List[int]] = []
    for frame in sorted(set(frames)):
        if not runs or frame != runs[-1][-1] + 1:
            runs.append([frame])
        else:
            runs[-1].append(frame)
    return runs


def project_artifacts_to_official(
    relations: Dict[str, Any],
    trajectories: Dict[str, Any],
    video_ids: Sequence[str],
    *,
    prediction: bool,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    """Adapt project ID-based artifacts to the official tube relation format."""

    output: Dict[str, List[Dict[str, Any]]] = {video_id: [] for video_id in video_ids}
    skipped_missing_tracks = 0
    skipped_empty_tubes = 0
    split_on_track_gaps = 0
    for video_id in video_ids:
        track_map = {
            int(item["track_id"]): item
            for item in trajectories.get(video_id, [])
            if isinstance(item, dict) and item.get("track_id") is not None
        }
        for relation in relations.get(video_id, []):
            if not isinstance(relation, dict):
                continue
            subject = track_map.get(int(relation.get("subject_track_id", relation.get("subject_id", -1))))
            obj = track_map.get(int(relation.get("object_track_id", relation.get("object_id", -1))))
            if subject is None or obj is None:
                skipped_missing_tracks += 1
                continue
            span = relation_span(relation)
            start, end = span.start_frame, span.end_frame
            subject_boxes = subject.get("trajectory", {})
            object_boxes = obj.get("trajectory", {})
            common_frames = [
                frame for frame in range(start, end)
                if str(frame) in subject_boxes and str(frame) in object_boxes
            ]
            runs = _continuous_runs(common_frames)
            if not runs:
                skipped_empty_tubes += 1
                continue
            if len(runs) > 1:
                split_on_track_gaps += len(runs) - 1
            for run in runs:
                output[video_id].append(
                    {
                        "triplet": [str(subject.get("category", "unknown")), str(relation.get("predicate", "")), str(obj.get("category", "unknown"))],
                        "duration": [run[0], run[-1] + 1],
                        "sub_traj": [subject_boxes[str(frame)] for frame in run],
                        "obj_traj": [object_boxes[str(frame)] for frame in run],
                        "score": _score(relation) if prediction else 1.0,
                    }
                )
    return output, {
        "skipped_missing_tracks": skipped_missing_tracks,
        "skipped_empty_tubes": skipped_empty_tubes,
        "split_on_track_gaps": split_on_track_gaps,
        "relation_count": sum(len(items) for items in output.values()),
    }
