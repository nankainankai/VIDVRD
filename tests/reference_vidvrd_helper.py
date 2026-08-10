"""Pinned VidVRD-helper evaluator logic used only for differential tests.

Adapted to Python 3 from MIT-licensed commit
1b4de175ce6e7a103d5feaae66b68d32a306a877.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(precision.size - 1, 0, -1):
        precision[index - 1] = max(precision[index - 1], precision[index])
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def _viou(left: dict, right: dict, field: str) -> float:
    left_start, left_end = left["duration"]
    right_start, right_end = right["duration"]
    overlap_start, overlap_end = max(left_start, right_start), min(left_end, right_end)
    if overlap_start >= overlap_end:
        return 0.0
    left_traj, right_traj = left[field], right[field]
    overlap = 0.0
    for frame in range(overlap_start, overlap_end):
        left_box = left_traj[frame - left_start]
        right_box = right_traj[frame - right_start]
        width = max(0.0, min(left_box[2], right_box[2]) - max(left_box[0], right_box[0]) + 1.0)
        height = max(0.0, min(left_box[3], right_box[3]) - max(left_box[1], right_box[1]) + 1.0)
        overlap += width * height
    left_volume = sum((box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0) for box in left_traj)
    right_volume = sum((box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0) for box in right_traj)
    return overlap / (left_volume + right_volume - overlap)


def _detection_scores(targets: list[dict], predictions: list[dict], threshold: float):
    predictions = sorted(predictions, key=lambda item: item["score"], reverse=True)
    detected = np.zeros(len(targets), dtype=bool)
    hit_scores = np.full(len(predictions), -np.inf)
    for prediction_index, prediction in enumerate(predictions):
        best_overlap, best_index = -np.inf, -1
        for target_index, target in enumerate(targets):
            if detected[target_index] or tuple(prediction["triplet"]) != tuple(target["triplet"]):
                continue
            overlap = min(
                _viou(prediction, target, "sub_traj"),
                _viou(prediction, target, "obj_traj"),
            )
            if overlap >= threshold and overlap > best_overlap:
                best_overlap, best_index = overlap, target_index
        if best_index >= 0:
            hit_scores[prediction_index] = prediction["score"]
            detected[best_index] = True
    true_positive = np.isfinite(hit_scores)
    cumulative_true = np.cumsum(true_positive).astype(np.float32)
    cumulative_false = np.cumsum(~true_positive).astype(np.float32)
    recall = cumulative_true / max(len(targets), np.finfo(np.float32).eps)
    precision = cumulative_true / np.maximum(
        cumulative_true + cumulative_false, np.finfo(np.float32).eps
    )
    return precision, recall, hit_scores


def _tagging_precision(targets: list[dict], predictions: list[dict]) -> np.ndarray:
    target_triplets = {tuple(item["triplet"]) for item in targets}
    predicted_triplets: list[tuple[str, str, str]] = []
    hits: list[bool] = []
    for prediction in sorted(predictions, key=lambda item: item["score"], reverse=True):
        triplet = tuple(prediction["triplet"])
        if triplet in predicted_triplets:
            continue
        predicted_triplets.append(triplet)
        hits.append(triplet in target_triplets)
    if not hits:
        return np.asarray([], dtype=float)
    return np.cumsum(hits).astype(np.float32) / np.arange(1, len(hits) + 1)


def evaluate(groundtruth: dict[str, list[dict]], predictions: dict[str, list[dict]]):
    video_ap: dict[str, float] = {}
    detection_hits: dict[int, list[np.ndarray]] = defaultdict(list)
    tagging: dict[int, list[float]] = defaultdict(list)
    relation_count = 0
    for video_id, targets in groundtruth.items():
        if not targets:
            continue
        relation_count += len(targets)
        precision, recall, hit_scores = _detection_scores(
            targets, predictions.get(video_id, []), 0.5
        )
        video_ap[video_id] = _voc_ap(recall, precision)
        hits = np.isfinite(hit_scores)
        for limit in (50, 100):
            detection_hits[limit].append(hits[: min(limit, hits.size)])
        tag_precision = _tagging_precision(targets, predictions.get(video_id, []))
        for limit in (1, 5, 10):
            cutoff = min(limit, tag_precision.size)
            tagging[limit].append(float(tag_precision[cutoff - 1]) if cutoff else 0.0)
    return {
        "mean_ap": float(np.mean(list(video_ap.values()))),
        "recall": {
            limit: sum(int(values.sum()) for values in detection_hits[limit]) / relation_count
            for limit in (50, 100)
        },
        "tagging": {limit: float(np.mean(tagging[limit])) for limit in (1, 5, 10)},
    }
