"""Project adapter around the unmodified official OC-SORT core."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..third_party.oc_sort.association import iou_batch
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


def _native_score(detection: Mapping[str, Any]) -> float | None:
    value = detection.get("score", detection.get("confidence"))
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


@dataclass
class _Metadata:
    start_frame: int
    end_frame: int
    observed_frames: set[int] = field(default_factory=set)
    scores: list[float] = field(default_factory=list)
    class_name: str = "unknown"
    votes: deque[tuple[str, float]] = field(default_factory=deque)
    history: list[dict[str, Any]] = field(default_factory=list)


class ObjectTracker:
    """Expose only tracks returned as confirmed by official ``update_public``.

    The caller defines the clock. The dense reference calls once per video
    frame; the production route calls once per Rex detection anchor.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        class_aware: bool = True,
        min_new_track_conf: float = 0.0,
        delta_t: int = 3,
        inertia: float = 0.2,
        class_vote_window: int = 12,
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
        self._groups = self._compatibility_groups(class_compatibility or {})
        self._backends: dict[str, OCSort] = {}
        self._steps: dict[str, list[tuple[int, list[tuple[dict[str, Any], np.ndarray, float]]]]] = defaultdict(list)
        self._global_ids: dict[tuple[str, int], int] = {}
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
        groups: dict[str, str] = {}
        for raw_name, compatible in mapping.items():
            canonical = _name(raw_name)
            groups[canonical] = canonical
            for other in compatible:
                groups[_name(other)] = canonical
        return groups

    def _group(self, class_name: str) -> str:
        if not self.class_aware:
            return "all"
        name = _name(class_name)
        return self._groups.get(name, name)

    def _id(self, group: str, local_id: int) -> int:
        key = (group, int(local_id))
        if key not in self._global_ids:
            self._global_ids[key] = self._next_id
            self._next_id += 1
        return self._global_ids[key]

    def reset(self) -> None:
        self._backends.clear()
        self._steps.clear()
        self._global_ids.clear()
        self._metadata.clear()
        self._next_id = 1

    @staticmethod
    def _matching_entry(
        box: np.ndarray, entries: list[tuple[dict[str, Any], np.ndarray, float]]
    ) -> tuple[dict[str, Any], np.ndarray] | None:
        if not entries:
            return None
        overlaps = iou_batch(box.reshape(1, 4), np.asarray([entry[1] for entry in entries], dtype=float))[0]
        detection, observed_box, _ = entries[int(np.argmax(overlaps))]
        return detection, observed_box

    def _observe(self, track_id: int, detection: Mapping[str, Any], box: np.ndarray, frame_num: int) -> _Metadata:
        meta = self._metadata.setdefault(track_id, _Metadata(frame_num, frame_num))
        if frame_num in meta.observed_frames:
            return meta
        meta.start_frame = min(meta.start_frame, frame_num)
        meta.end_frame = max(meta.end_frame, frame_num)
        meta.observed_frames.add(frame_num)
        score = _native_score(detection)
        if score is not None:
            meta.scores.append(score)
        class_name = _name(detection.get("class_name", "unknown"))
        meta.votes.append((class_name, score if score is not None else 1.0))
        while len(meta.votes) > self.class_vote_window:
            meta.votes.popleft()
        totals: dict[str, float] = defaultdict(float)
        for candidate, weight in meta.votes:
            if candidate != "unknown":
                totals[candidate] += weight
        if totals:
            meta.class_name = max(totals, key=totals.get)
        center_x, center_y = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        meta.history.append(
            {"frame": frame_num, "bbox": box.tolist(), "center_x": float(center_x), "center_y": float(center_y)}
        )
        meta.history.sort(key=lambda item: int(item["frame"]))
        return meta

    def track(self, frame: np.ndarray, detections: list[dict[str, Any]], frame_num: int = 0) -> list[dict[str, Any]]:
        height, width = frame.shape[:2]
        grouped: dict[str, list[tuple[dict[str, Any], np.ndarray, float]]] = defaultdict(list)
        for detection in detections or []:
            raw_box = detection.get("bbox")
            if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                continue
            score = _native_score(detection)
            grouped[self._group(_name(detection.get("class_name")))].append(
                (detection, _clip(raw_box, width, height), score if score is not None else 1.0)
            )

        for group in grouped:
            self._backends.setdefault(group, self._new_backend())

        output: list[dict[str, Any]] = []
        for group, backend in list(self._backends.items()):
            entries = grouped.get(group, [])
            self._steps[group].append((int(frame_num), entries))
            boxes = np.asarray([entry[1] for entry in entries], dtype=float).reshape((-1, 4))
            scores = np.asarray([entry[2] for entry in entries], dtype=float)
            confirmed = backend.update_public(boxes, np.zeros(len(entries), dtype=int), scores)

            for row in sorted(confirmed.tolist(), key=lambda item: int(item[6])):
                step_index = len(self._steps[group]) - 1 + int(row[6])
                target_frame, target_entries = self._steps[group][step_index]
                box = _clip(row[:4], width, height)
                matched = self._matching_entry(box, target_entries)
                detection, box = matched if matched is not None else ({}, box)
                track_id = self._id(group, int(row[4]))
                meta = self._observe(track_id, detection, box, target_frame)
                total_distance = sum(
                    math.hypot(
                        float(current["center_x"]) - float(previous["center_x"]),
                        float(current["center_y"]) - float(previous["center_y"]),
                    )
                    for previous, current in zip(meta.history, meta.history[1:])
                )
                duration = max(1, target_frame - meta.start_frame + 1)
                output.append(
                    {
                        "frame": target_frame,
                        "track_id": track_id,
                        "bbox": box.tolist(),
                        "bbox_observed": box.tolist(),
                        "box_source": "observed",
                        "track_status": "confirmed",
                        "class_name": meta.class_name,
                        "confidence": sum(meta.scores) / len(meta.scores) if meta.scores else None,
                        "duration_frames": duration,
                        "total_distance": total_distance,
                        "instant_distance": 0.0,
                        "avg_speed": total_distance / duration,
                        "motion_state": "unknown",
                        "age": 0,
                        "time_since_update": 0,
                        "hits": len(meta.observed_frames),
                        "is_predicted": False,
                        "history": list(meta.history),
                    }
                )
        return sorted(output, key=lambda item: (item["frame"], item["track_id"]))
