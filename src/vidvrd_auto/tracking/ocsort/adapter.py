"""Thin VIDVRD adapter around an unmodified official OC-SORT core."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..third_party.oc_sort.association import iou_batch, linear_assignment
from ..third_party.oc_sort.ocsort import OCSort


def _name(value: object) -> str:
    return str(value or "unknown").strip().lower() or "unknown"


def _clip(box: Sequence[float], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = (float(value) for value in box)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, x2 = np.clip([x1, x2], 0.0, max(0.0, width - 1.0))
    y1, y2 = np.clip([y1, y2], 0.0, max(0.0, height - 1.0))
    return np.asarray([x1, y1, max(x1 + 1.0, x2), max(y1 + 1.0, y2)], dtype=float)


@dataclass
class _Metadata:
    start_frame: int
    end_frame: int
    confidence: float = 0.0
    observed_count: int = 0
    class_name: str = "unknown"
    votes: deque[tuple[str, float]] = field(default_factory=deque)
    history: list[dict[str, Any]] = field(default_factory=list)


class ObjectTracker:
    """Run exact official OC-SORT independently per compatible class group.

    Keeping class groups outside the vendored core prevents cross-class identity
    switches without modifying OC-SORT association or Kalman code.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        class_aware: bool = True,
        min_new_track_conf: float = 0.35,
        delta_t: int = 3,
        inertia: float = 0.2,
        class_vote_window: int = 12,
        max_output_age: int = 8,
        class_compatibility: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = max(1, int(min_hits))
        self.det_threshold = float(min_new_track_conf)
        self.delta_t = int(delta_t)
        self.inertia = float(inertia)
        self.class_aware = bool(class_aware)
        self.class_vote_window = max(1, int(class_vote_window))
        self.max_output_age = max(0, int(max_output_age))
        self._groups = self._compatibility_groups(class_compatibility or {})
        self._backends: dict[str, OCSort] = {}
        self._global_ids: dict[int, int] = {}
        self._metadata: dict[int, _Metadata] = {}
        self._next_id = 1

    def _new_backend(self) -> OCSort:
        return OCSort(
            det_thresh=self.det_threshold,
            max_age=self.max_age,
            min_hits=self.min_hits,
            iou_threshold=self.iou_threshold,
            delta_t=self.delta_t,
            inertia=self.inertia,
            use_byte=False,
        )

    @staticmethod
    def _compatibility_groups(mapping: Mapping[str, Sequence[str]]) -> dict[str, str]:
        parent: dict[str, str] = {}

        def find(item: str) -> str:
            parent.setdefault(item, item)
            if parent[item] != item:
                parent[item] = find(parent[item])
            return parent[item]

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for raw_name, compatible in mapping.items():
            name = _name(raw_name)
            find(name)
            for other in compatible:
                union(name, _name(other))
        return {name: find(name) for name in parent}

    def _group(self, class_name: str) -> str:
        if not self.class_aware:
            return "all"
        name = _name(class_name)
        return self._groups.get(name, name)

    def _id(self, tracker: Any) -> int:
        key = id(tracker)
        if key not in self._global_ids:
            self._global_ids[key] = self._next_id
            self._next_id += 1
        return self._global_ids[key]

    def reset(self) -> None:
        self._backends.clear()
        self._global_ids.clear()
        self._metadata.clear()
        self._next_id = 1

    def _observe(self, track_id: int, detection: dict[str, Any], box: np.ndarray, frame_num: int) -> None:
        confidence = float(detection.get("confidence", 0.0))
        class_name = _name(detection.get("class_name", "unknown"))
        meta = self._metadata.setdefault(track_id, _Metadata(frame_num, frame_num))
        meta.end_frame = frame_num
        meta.observed_count += 1
        meta.confidence = confidence if meta.observed_count == 1 else 0.7 * meta.confidence + 0.3 * confidence
        meta.votes.append((class_name, max(confidence, 1e-6)))
        while len(meta.votes) > self.class_vote_window:
            meta.votes.popleft()
        totals: dict[str, float] = defaultdict(float)
        for candidate, weight in meta.votes:
            if candidate != "unknown":
                totals[candidate] += weight
        if totals:
            meta.class_name = max(totals, key=totals.get)
        center_x, center_y = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        meta.history.append({"frame": frame_num, "bbox": box.tolist(), "center_x": float(center_x), "center_y": float(center_y)})
        meta.history = meta.history[-30:]

    def track(self, frame: np.ndarray, detections: list[dict[str, Any]], frame_num: int = 0) -> list[dict[str, Any]]:
        height, width = frame.shape[:2]
        grouped: dict[str, list[tuple[dict[str, Any], np.ndarray, float]]] = defaultdict(list)
        for detection in detections or []:
            raw_box = detection.get("bbox")
            if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                continue
            grouped[self._group(_name(detection.get("class_name")))].append(
                (detection, _clip(raw_box, width, height), float(detection.get("confidence", 0.0)))
            )

        for group in grouped:
            self._backends.setdefault(group, self._new_backend())
        observed_boxes: dict[int, np.ndarray] = {}
        for group, backend in list(self._backends.items()):
            entries = grouped.get(group, [])
            boxes = [entry[1] for entry in entries]
            scores = [entry[2] for entry in entries]
            backend.update_public(
                np.asarray(boxes, dtype=float).reshape((-1, 4)) if boxes else np.empty((0, 4), dtype=float),
                np.zeros(len(boxes), dtype=int),
                np.asarray(scores, dtype=float),
            )
            eligible = [index for index, score in enumerate(scores) if score > self.det_threshold]
            current = [tracker for tracker in backend.trackers if int(tracker.time_since_update) == 0]
            if current and eligible:
                overlaps = iou_batch(
                    np.asarray([tracker.last_observation[:4] for tracker in current], dtype=float),
                    np.asarray([boxes[index] for index in eligible], dtype=float),
                )
                for row_index, eligible_index in linear_assignment(-overlaps).tolist():
                    entry_index = eligible[eligible_index]
                    global_id = self._id(current[row_index])
                    box = boxes[entry_index]
                    observed_boxes[global_id] = box
                    self._observe(global_id, entries[entry_index][0], box, int(frame_num))

        live_keys = {
            id(tracker)
            for backend in self._backends.values()
            for tracker in backend.trackers
        }
        self._global_ids = {key: value for key, value in self._global_ids.items() if key in live_keys}
        live_global_ids = set(self._global_ids.values())
        self._metadata = {track_id: meta for track_id, meta in self._metadata.items() if track_id in live_global_ids}

        output: list[dict[str, Any]] = []
        for backend in self._backends.values():
            for tracker in backend.trackers:
                track_id = self._global_ids.get(id(tracker))
                meta = self._metadata.get(track_id) if track_id is not None else None
                if track_id is None or meta is None:
                    continue
                predicted = int(tracker.time_since_update) > 0
                if predicted and int(tracker.time_since_update) > self.max_output_age:
                    continue
                box = _clip(tracker.get_state()[0], width, height) if predicted else observed_boxes.get(track_id)
                if box is None:
                    box = _clip(tracker.last_observation[:4], width, height)
                observed_box = None if predicted else box.tolist()
                instant_distance = 0.0
                if not predicted and len(meta.history) >= 2:
                    previous, current = meta.history[-2:]
                    instant_distance = math.hypot(float(current["center_x"]) - float(previous["center_x"]), float(current["center_y"]) - float(previous["center_y"]))
                total_distance = sum(
                    math.hypot(float(current["center_x"]) - float(previous["center_x"]), float(current["center_y"]) - float(previous["center_y"]))
                    for previous, current in zip(meta.history, meta.history[1:])
                )
                duration = max(1, int(frame_num) - meta.start_frame + 1)
                avg_speed = total_distance / duration
                output.append(
                    {
                        "track_id": track_id,
                        "bbox": box.tolist(),
                        "bbox_observed": observed_box,
                        "class_name": meta.class_name,
                        "confidence": float(meta.confidence),
                        "duration_frames": duration,
                        "total_distance": total_distance,
                        "instant_distance": instant_distance,
                        "avg_speed": avg_speed,
                        "motion_state": "stationary" if avg_speed < 1.0 else ("slow" if avg_speed < 5.0 else "fast"),
                        "age": int(tracker.time_since_update),
                        "time_since_update": int(tracker.time_since_update),
                        "hits": int(meta.observed_count),
                        "is_predicted": predicted,
                        "history": list(meta.history),
                    }
                )
        return sorted(output, key=lambda item: item["track_id"])
