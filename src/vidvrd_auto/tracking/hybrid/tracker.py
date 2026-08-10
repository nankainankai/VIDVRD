from __future__ import annotations

"""Sparse, real-frame-time association with appearance and soft classes."""

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def _name(value: object) -> str:
    return str(value or "unknown").strip().lower() or "unknown"


def _native_score(detection: dict[str, Any]) -> float | None:
    value = detection.get("score")
    if value is None:
        value = detection.get("confidence")
    return None if value is None else float(value)


def _association_weight(detection: dict[str, Any]) -> float:
    score = _native_score(detection)
    return 1.0 if score is None else score


def _box(box: Sequence[float], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, x2 = np.clip([x1, x2], 0.0, max(0.0, width - 1.0))
    y1, y2 = np.clip([y1, y2], 0.0, max(0.0, height - 1.0))
    return np.asarray([x1, y1, max(x1 + 1.0, x2), max(y1 + 1.0, y2)], dtype=float)


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1e-6, left_area + right_area - intersection)


def _center_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_center = (left[:2] + left[2:]) / 2.0
    right_center = (right[:2] + right[2:]) / 2.0
    scale = math.sqrt(max(1.0, (left[2] - left[0]) * (left[3] - left[1])))
    return float(np.linalg.norm(left_center - right_center) / scale)


def _normalize(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 0 else None


@dataclass
class _Tracklet:
    local_id: int
    scene_id: int
    first_frame: int
    last_frame: int
    last_box: np.ndarray
    previous_frame: int | None = None
    previous_box: np.ndarray | None = None
    hits: int = 0
    class_scores: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    scores: list[float] = field(default_factory=list)
    embeddings: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=20))
    observations: list[dict[str, Any]] = field(default_factory=list)
    emitted: int = 0
    total_distance: float = 0.0

    def class_distribution(self) -> dict[str, float]:
        total = sum(self.class_scores.values())
        return {name: value / total for name, value in sorted(self.class_scores.items())} if total else {"unknown": 1.0}

    def class_name(self) -> str:
        return max(self.class_scores, key=self.class_scores.get) if self.class_scores else "unknown"

    def mean_embedding(self) -> np.ndarray | None:
        if not self.embeddings:
            return None
        return _normalize(np.mean(np.stack(self.embeddings), axis=0))

    def velocity(self) -> np.ndarray:
        if self.previous_box is None or self.previous_frame is None or self.last_frame == self.previous_frame:
            return np.zeros(4, dtype=float)
        return (self.last_box - self.previous_box) / float(self.last_frame - self.previous_frame)

    def predict(self, frame_num: int) -> np.ndarray:
        return self.last_box + self.velocity() * float(frame_num - self.last_frame)


class HybridTracker:
    """Associate sparse detections using motion, MASA appearance and soft class evidence."""

    def __init__(
        self,
        *,
        min_hits: int = 2,
        max_lost_frames: int = 30,
        min_new_track_conf: float = 0.0,
        appearance_weight: float = 0.45,
        iou_weight: float = 0.30,
        motion_weight: float = 0.20,
        class_weight: float = 0.05,
        max_match_cost: float = 0.72,
        min_iou: float = 0.01,
        min_appearance_similarity: float = 0.35,
        max_center_distance: float = 4.0,
        appearance_memory: int = 20,
    ) -> None:
        self.min_hits = max(1, int(min_hits))
        self.max_lost_frames = int(max_lost_frames)
        self.min_new_track_conf = float(min_new_track_conf)
        self.weights = np.asarray([iou_weight, motion_weight, appearance_weight, class_weight], dtype=float)
        self.weights /= self.weights.sum()
        self.max_match_cost = float(max_match_cost)
        self.min_iou = float(min_iou)
        self.min_appearance_similarity = float(min_appearance_similarity)
        self.max_center_distance = float(max_center_distance)
        self.appearance_memory = max(1, int(appearance_memory))
        self.tracklets: list[_Tracklet] = []
        self.active: list[_Tracklet] = []
        self._next_id = 1

    def _cost(
        self, track: _Tracklet, detection: dict[str, Any], box: np.ndarray, embedding: np.ndarray | None, frame_num: int
    ) -> tuple[float, dict[str, float]]:
        predicted = track.predict(frame_num)
        overlap = _iou(predicted, box)
        center = _center_distance(predicted, box)
        memory = track.mean_embedding()
        similarity = float(np.dot(memory, embedding)) if memory is not None and embedding is not None else 0.0
        class_probability = track.class_distribution().get(_name(detection.get("class_name")), 0.0)
        components = np.asarray(
            [1.0 - overlap, min(1.0, center / self.max_center_distance), 1.0 - max(0.0, similarity), 1.0 - class_probability],
            dtype=float,
        )
        cost = float(np.dot(self.weights, components))
        if overlap < self.min_iou and center > self.max_center_distance and similarity < self.min_appearance_similarity:
            cost = math.inf
        return cost, {
            "cost": cost,
            "iou": overlap,
            "center_distance": center,
            "appearance_similarity": similarity,
            "class_probability": class_probability,
        }

    def _observe(
        self,
        track: _Tracklet,
        detection: dict[str, Any],
        box: np.ndarray,
        embedding: np.ndarray | None,
        frame_num: int,
        association: dict[str, float],
    ) -> None:
        if track.hits:
            old_center = (track.last_box[:2] + track.last_box[2:]) / 2.0
            new_center = (box[:2] + box[2:]) / 2.0
            track.total_distance += float(np.linalg.norm(new_center - old_center))
            track.previous_frame, track.previous_box = track.last_frame, track.last_box.copy()
        track.last_frame, track.last_box = int(frame_num), box.copy()
        track.hits += 1
        score = _native_score(detection)
        if score is not None:
            track.scores.append(score)
        track.class_scores[_name(detection.get("class_name"))] += _association_weight(detection)
        if embedding is not None:
            track.embeddings.append(embedding)
        duration = max(1, frame_num - track.first_frame + 1)
        track.observations.append(
            {
                "frame": int(frame_num),
                "track_id": track.local_id,
                "local_tracklet_id": track.local_id,
                "bbox": box.tolist(),
                "bbox_observed": box.tolist(),
                "box_source": "observed",
                "track_status": "confirmed" if track.hits >= self.min_hits else "tentative",
                "class_name": track.class_name(),
                "class_distribution": track.class_distribution(),
                "confidence": sum(track.scores) / len(track.scores) if track.scores else None,
                "duration_frames": duration,
                "total_distance": track.total_distance,
                "instant_distance": 0.0,
                "avg_speed": track.total_distance / duration,
                "motion_state": "unknown",
                "age": frame_num - track.last_frame,
                "hits": track.hits,
                "is_predicted": False,
                "identity_source": "online",
                "identity_support": max(0.0, 1.0 - float(association.get("cost", 1.0))),
                "association": association,
                "scene_id": track.scene_id,
            }
        )

    def update(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        embeddings: np.ndarray,
        *,
        frame_num: int,
        scene_id: int,
    ) -> list[dict[str, Any]]:
        height, width = frame.shape[:2]
        valid = [d for d in detections if isinstance(d.get("bbox"), (list, tuple)) and len(d["bbox"]) == 4]
        boxes = [_box(detection["bbox"], width, height) for detection in valid]
        vectors = [_normalize(embeddings[index]) if embeddings.size else None for index in range(len(valid))]
        self.active = [
            track for track in self.active
            if frame_num - track.last_frame <= self.max_lost_frames and track.scene_id == scene_id
        ]

        details: dict[tuple[int, int], dict[str, float]] = {}
        costs = np.full((len(self.active), len(valid)), math.inf, dtype=float)
        for row, track in enumerate(self.active):
            for column, (detection, box, vector) in enumerate(zip(valid, boxes, vectors)):
                costs[row, column], details[(row, column)] = self._cost(track, detection, box, vector, frame_num)

        matches: list[tuple[int, int]] = []
        if costs.size:
            finite = np.where(np.isfinite(costs), costs, 1e6)
            rows, columns = linear_sum_assignment(finite)
            matches = [(int(row), int(column)) for row, column in zip(rows, columns) if costs[row, column] <= self.max_match_cost]

        matched_detections = {column for _, column in matches}
        changed: list[_Tracklet] = []
        for row, column in matches:
            track = self.active[row]
            self._observe(track, valid[column], boxes[column], vectors[column], frame_num, details[(row, column)])
            changed.append(track)

        for column, (detection, box, vector) in enumerate(zip(valid, boxes, vectors)):
            native_score = _native_score(detection)
            if column in matched_detections or (
                native_score is not None and native_score < self.min_new_track_conf
            ):
                continue
            track = _Tracklet(self._next_id, scene_id, frame_num, frame_num, box.copy())
            track.embeddings = deque(maxlen=self.appearance_memory)
            self._next_id += 1
            self.tracklets.append(track)
            self.active.append(track)
            self._observe(
                track, detection, box, vector, frame_num,
                {"cost": 0.5, "iou": 0.0, "center_distance": 0.0, "appearance_similarity": 0.0, "class_probability": 0.0},
            )
            changed.append(track)

        output: list[dict[str, Any]] = []
        for track in changed:
            if track.hits >= self.min_hits:
                for observation in track.observations[track.emitted:]:
                    observation["track_status"] = "confirmed"
                    output.append(observation)
                track.emitted = len(track.observations)
        return sorted(output, key=lambda item: (item["frame"], item["local_tracklet_id"]))

    def summaries(self) -> list[dict[str, Any]]:
        output = []
        for track in self.tracklets:
            if track.hits < self.min_hits:
                continue
            embedding = track.mean_embedding()
            output.append(
                {
                    "local_tracklet_id": track.local_id,
                    "scene_id": track.scene_id,
                    "start_frame": track.first_frame,
                    "end_frame": track.last_frame,
                    "observation_count": track.hits,
                    "class_name": track.class_name(),
                    "class_distribution": track.class_distribution(),
                    "first_bbox": track.observations[0]["bbox_observed"],
                    "last_bbox": track.last_box.tolist(),
                    "last_velocity": track.velocity().tolist(),
                    "mean_confidence": sum(track.scores) / len(track.scores) if track.scores else None,
                    "mean_embedding": embedding.tolist() if embedding is not None else None,
                }
            )
        return output
