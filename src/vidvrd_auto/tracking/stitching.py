from __future__ import annotations

"""Offline one-predecessor/one-successor tracklet stitching."""

import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    top_left = np.maximum(left[:2], right[:2])
    bottom_right = np.minimum(left[2:], right[2:])
    intersection = max(0.0, bottom_right[0] - top_left[0]) * max(0.0, bottom_right[1] - top_left[1])
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1e-6, left_area + right_area - intersection)


def _center_distance(left: np.ndarray, right: np.ndarray) -> float:
    center_left = (left[:2] + left[2:]) / 2.0
    center_right = (right[:2] + right[2:]) / 2.0
    scale = math.sqrt(max(1.0, (left[2] - left[0]) * (left[3] - left[1])))
    return float(np.linalg.norm(center_left - center_right) / scale)


def _class_overlap(left: dict[str, float], right: dict[str, float]) -> float:
    return sum(min(float(value), float(right.get(name, 0.0))) for name, value in left.items())


def _edge(left: dict[str, Any], right: dict[str, Any], config: dict[str, Any]) -> dict[str, float] | None:
    gap = int(right["start_frame"]) - int(left["end_frame"])
    if gap <= 0 or gap > int(config.get("stitch_max_gap_frames", 90)):
        return None
    if not bool(config.get("stitch_across_scenes", False)) and int(left["scene_id"]) != int(right["scene_id"]):
        return None

    left_box = np.asarray(left["last_bbox"], dtype=float)
    velocity = np.asarray(left.get("last_velocity", [0.0, 0.0, 0.0, 0.0]), dtype=float)
    predicted = left_box + velocity * gap
    right_box = np.asarray(right["first_bbox"], dtype=float)
    overlap = _iou(predicted, right_box)
    center = _center_distance(predicted, right_box)
    left_embedding = left.get("mean_embedding")
    right_embedding = right.get("mean_embedding")
    similarity = 0.0
    if left_embedding is not None and right_embedding is not None:
        similarity = float(np.dot(np.asarray(left_embedding), np.asarray(right_embedding)))
    class_overlap = _class_overlap(left.get("class_distribution", {}), right.get("class_distribution", {}))

    min_similarity = float(config.get("stitch_min_appearance_similarity", 0.60))
    max_center = float(config.get("stitch_max_center_distance", 4.0))
    if similarity < min_similarity and center > max_center:
        return None
    weights = np.asarray(
        [
            float(config.get("stitch_appearance_weight", 0.55)),
            float(config.get("stitch_motion_weight", 0.25)),
            float(config.get("stitch_iou_weight", 0.15)),
            float(config.get("stitch_class_weight", 0.05)),
        ],
        dtype=float,
    )
    weights /= weights.sum()
    components = np.asarray(
        [1.0 - max(0.0, similarity), min(1.0, center / max_center), 1.0 - overlap, 1.0 - class_overlap],
        dtype=float,
    )
    cost = float(np.dot(weights, components))
    return {
        "cost": cost,
        "gap_frames": gap,
        "appearance_similarity": similarity,
        "predicted_iou": overlap,
        "center_distance": center,
        "class_overlap": class_overlap,
    }


def stitch_tracklets(
    summaries: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[int, int], list[dict[str, Any]]]:
    """Return local-to-global IDs and the accepted DAG edges."""

    ordered = sorted(summaries, key=lambda item: (int(item["start_frame"]), int(item["local_tracklet_id"])))
    count = len(ordered)
    if not count:
        return {}, []
    threshold = float(config.get("stitch_max_link_cost", 0.58))
    costs = np.full((count, count * 2), threshold, dtype=float)
    details: dict[tuple[int, int], dict[str, float]] = {}
    for row, left in enumerate(ordered):
        costs[row, count:] = threshold
        for column, right in enumerate(ordered):
            evidence = _edge(left, right, config)
            if evidence is not None and evidence["cost"] < threshold:
                costs[row, column] = evidence["cost"]
                details[(row, column)] = evidence

    rows, columns = linear_sum_assignment(costs)
    successor: dict[int, int] = {}
    predecessor: dict[int, int] = {}
    links: list[dict[str, Any]] = []
    for row, column in zip(rows, columns):
        if column >= count or (row, column) not in details:
            continue
        left_id = int(ordered[row]["local_tracklet_id"])
        right_id = int(ordered[column]["local_tracklet_id"])
        successor[left_id] = right_id
        predecessor[right_id] = left_id
        links.append({"from_local_tracklet_id": left_id, "to_local_tracklet_id": right_id, **details[(row, column)]})

    mapping: dict[int, int] = {}
    global_id = 1
    starts = [item for item in ordered if int(item["local_tracklet_id"]) not in predecessor]
    for start in starts:
        local_id = int(start["local_tracklet_id"])
        while local_id not in mapping:
            mapping[local_id] = global_id
            if local_id not in successor:
                break
            local_id = successor[local_id]
        global_id += 1
    return mapping, sorted(links, key=lambda item: (item["from_local_tracklet_id"], item["to_local_tracklet_id"]))


def apply_global_ids(
    frame_outputs: dict[int, dict[int, dict[str, Any]]], mapping: dict[int, int], links: list[dict[str, Any]]
) -> dict[int, dict[int, dict[str, Any]]]:
    stitched_ids = {
        int(link[key])
        for link in links
        for key in ("from_local_tracklet_id", "to_local_tracklet_id")
    }
    link_support: dict[int, float] = {}
    for link in links:
        support = max(0.0, 1.0 - float(link["cost"]))
        link_support[int(link["from_local_tracklet_id"])] = support
        link_support[int(link["to_local_tracklet_id"])] = support

    output: dict[int, dict[int, dict[str, Any]]] = {}
    for frame, tracks in frame_outputs.items():
        for local_id, source in tracks.items():
            item = dict(source)
            global_id = mapping[int(local_id)]
            item["track_id"] = global_id
            if int(local_id) in stitched_ids:
                item["identity_source"] = "offline_stitch"
                item["identity_support"] = link_support[int(local_id)]
            output.setdefault(frame, {})[global_id] = item
    return output
