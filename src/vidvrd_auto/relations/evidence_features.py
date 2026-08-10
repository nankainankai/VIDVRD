from __future__ import annotations

"""Trajectory-pair evidence used for candidate ranking and frame selection."""

import math
from statistics import median
from typing import Any


def _bbox(item: dict[str, Any]) -> list[float]:
    return [float(value) for value in (item.get("bbox_observed") or item["bbox"])]


def _center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _area(box: list[float]) -> float:
    return max(1.0, box[2] - box[0]) * max(1.0, box[3] - box[1])


def _iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return intersection / max(1e-6, _area(left) + _area(right) - intersection)


def _edge_gap(left: list[float], right: list[float]) -> float:
    dx = max(left[0] - right[2], right[0] - left[2], 0.0)
    dy = max(left[1] - right[3], right[1] - left[3], 0.0)
    return math.hypot(dx, dy)


def trajectory_evidence(
    tracks: dict[int, list[dict[str, Any]]], frames: list[int], subject_id: int, object_id: int
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for frame in frames:
        by_id = {int(item["track_id"]): item for item in tracks.get(frame, [])}
        subject, obj = by_id[subject_id], by_id[object_id]
        subject_box, object_box = _bbox(subject), _bbox(obj)
        sx, sy = _center(subject_box)
        ox, oy = _center(object_box)
        scale = math.sqrt(max(_area(subject_box), _area(object_box)))
        samples.append(
            {
                "frame": frame,
                "dx": (sx - ox) / scale,
                "dy": (sy - oy) / scale,
                "distance": math.hypot(sx - ox, sy - oy) / scale,
                "edge_gap": _edge_gap(subject_box, object_box) / scale,
                "iou": _iou(subject_box, object_box),
                "size_ratio": _area(subject_box) / _area(object_box),
                "subject_center": (sx, sy),
                "object_center": (ox, oy),
                "scale": scale,
                "subject_source": str(subject.get("box_source", "observed")),
                "object_source": str(obj.get("box_source", "observed")),
                "identity_support": min(
                    float(subject.get("identity_support", 1.0)), float(obj.get("identity_support", 1.0))
                ),
            }
        )

    subject_speeds: list[float] = []
    object_speeds: list[float] = []
    motion_cosines: list[float] = []
    approach_rates: list[float] = []
    for previous, current in zip(samples, samples[1:]):
        frame_delta = current["frame"] - previous["frame"]
        scale = (current["scale"] + previous["scale"]) / 2.0
        subject_velocity = tuple(
            (current["subject_center"][index] - previous["subject_center"][index]) / scale / frame_delta
            for index in (0, 1)
        )
        object_velocity = tuple(
            (current["object_center"][index] - previous["object_center"][index]) / scale / frame_delta
            for index in (0, 1)
        )
        distance_rate = (previous["distance"] - current["distance"]) / frame_delta
        approach_rates.append(distance_rate)
        subject_speed = math.hypot(*subject_velocity)
        object_speed = math.hypot(*object_velocity)
        subject_speeds.append(subject_speed)
        object_speeds.append(object_speed)
        if subject_speed > 0 and object_speed > 0:
            motion_cosines.append(
                max(
                    -1.0,
                    min(
                        1.0,
                        sum(a * b for a, b in zip(subject_velocity, object_velocity))
                        / (subject_speed * object_speed),
                    ),
                )
            )

    distances = [item["distance"] for item in samples]
    ious = [item["iou"] for item in samples]
    gaps = [item["edge_gap"] for item in samples]
    event_frames = {
        samples[distances.index(min(distances))]["frame"],
        samples[ious.index(max(ious))]["frame"],
    }
    if approach_rates and max(abs(value) for value in approach_rates) > 1e-6:
        strongest_change = max(range(len(approach_rates)), key=lambda index: abs(approach_rates[index]))
        event_frames.add(samples[strongest_change + 1]["frame"])
    observed = sum(
        item["subject_source"] == item["object_source"] == "observed" for item in samples
    )
    return {
        "frame_count": len(samples),
        "observed_pair_ratio": observed / len(samples),
        "identity_support_min": min(item["identity_support"] for item in samples),
        "identity_support_mean": sum(item["identity_support"] for item in samples) / len(samples),
        "median_dx": median(item["dx"] for item in samples),
        "median_dy": median(item["dy"] for item in samples),
        "distance_start": distances[0],
        "distance_end": distances[-1],
        "distance_min": min(distances),
        "edge_gap_min": min(gaps),
        "iou_max": max(ious),
        "approach_rate": median(approach_rates) if approach_rates else 0.0,
        "subject_speed": median(subject_speeds) if subject_speeds else 0.0,
        "object_speed": median(object_speeds) if object_speeds else 0.0,
        "relative_motion": median(abs(value) for value in approach_rates) if approach_rates else 0.0,
        "motion_alignment": median(motion_cosines) if motion_cosines else 0.0,
        "size_ratio": median(item["size_ratio"] for item in samples),
        "crosses_horizontal_axis": min(item["dx"] for item in samples) < 0 < max(item["dx"] for item in samples),
        "event_frames": sorted(event_frames),
    }
