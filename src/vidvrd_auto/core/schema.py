"""Validated serialization contracts for pipeline records."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


BBox = tuple[float, float, float, float]


def _mapping(value: Any, kind: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{kind} must be a mapping")
    return value


def _text(value: Any, name: str, *, default: str = "") -> str:
    text = str(default if value is None else value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if number < minimum or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{name} must be at least {minimum}")
    return number


def _score(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def _bbox(value: Any, name: str = "bbox") -> BBox:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{name} must contain four coordinates")
    try:
        box = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} coordinates must be numbers") from exc
    if not all(math.isfinite(item) for item in box):
        raise ValueError(f"{name} coordinates must be finite")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{name} must have positive width and height")
    return box  # type: ignore[return-value]


def _extras(item: Mapping[str, Any], known: set[str]) -> dict[str, Any]:
    return {str(key): value for key, value in item.items() if key not in known}


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    class_name: str
    confidence: float
    frame: int | None = None
    source: str = "unknown"
    batch_id: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", _bbox(self.bbox))
        object.__setattr__(self, "class_name", _text(self.class_name, "class_name"))
        object.__setattr__(self, "confidence", _score(self.confidence, "confidence"))
        object.__setattr__(self, "source", _text(self.source, "source", default="unknown"))
        if self.frame is not None:
            object.__setattr__(self, "frame", _integer(self.frame, "frame"))
        if self.batch_id is not None:
            object.__setattr__(self, "batch_id", _integer(self.batch_id, "batch_id"))
        object.__setattr__(self, "extra", dict(self.extra))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Detection:
        item = _mapping(value, "detection")
        known = {
            "bbox", "class_name", "class", "confidence", "frame", "frame_idx",
            "source", "batch_id",
        }
        return cls(
            bbox=_bbox(item.get("bbox")),
            class_name=_text(item.get("class_name", item.get("class")), "class_name"),
            confidence=_score(item.get("confidence"), "confidence"),
            frame=item.get("frame", item.get("frame_idx")),
            source=str(item.get("source", "unknown") or "unknown"),
            batch_id=item.get("batch_id"),
            extra=_extras(item, known),
        )

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.extra)
        out.update({"bbox": list(self.bbox), "class_name": self.class_name, "confidence": self.confidence})
        if self.frame is not None:
            out["frame"] = self.frame
        out["source"] = self.source
        if self.batch_id is not None:
            out["batch_id"] = self.batch_id
        return out


@dataclass(frozen=True)
class Track:
    track_id: int
    bbox: BBox
    class_name: str
    confidence: float
    frame: int | None = None
    bbox_observed: BBox | None = None
    is_predicted: bool = False
    time_since_update: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _integer(self.track_id, "track_id"))
        object.__setattr__(self, "bbox", _bbox(self.bbox))
        object.__setattr__(self, "class_name", _text(self.class_name, "class_name"))
        object.__setattr__(self, "confidence", _score(self.confidence, "confidence"))
        if self.frame is not None:
            object.__setattr__(self, "frame", _integer(self.frame, "frame"))
        if self.bbox_observed is not None:
            object.__setattr__(self, "bbox_observed", _bbox(self.bbox_observed, "bbox_observed"))
        object.__setattr__(self, "is_predicted", bool(self.is_predicted))
        object.__setattr__(self, "time_since_update", _integer(self.time_since_update, "time_since_update"))
        if self.is_predicted and self.bbox_observed is not None:
            raise ValueError("predicted tracks must have bbox_observed=None")
        object.__setattr__(self, "extra", dict(self.extra))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Track:
        item = _mapping(value, "track")
        known = {
            "track_id", "bbox", "class_name", "class", "confidence", "frame", "frame_idx",
            "bbox_observed", "is_predicted", "time_since_update",
        }
        return cls(
            track_id=_integer(item.get("track_id"), "track_id"),
            bbox=_bbox(item.get("bbox")),
            class_name=_text(item.get("class_name", item.get("class")), "class_name"),
            confidence=_score(item.get("confidence", 1.0), "confidence"),
            frame=item.get("frame", item.get("frame_idx")),
            bbox_observed=None if item.get("bbox_observed") is None else _bbox(item.get("bbox_observed"), "bbox_observed"),
            is_predicted=bool(item.get("is_predicted", False)),
            time_since_update=_integer(item.get("time_since_update", item.get("age", 0)), "time_since_update"),
            extra=_extras(item, known),
        )

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.extra)
        out.update(
            {
                "track_id": self.track_id,
                "bbox": list(self.bbox),
                "bbox_observed": None if self.bbox_observed is None else list(self.bbox_observed),
                "class_name": self.class_name,
                "confidence": self.confidence,
                "is_predicted": self.is_predicted,
                "time_since_update": self.time_since_update,
            }
        )
        if self.frame is not None:
            out["frame"] = self.frame
        return out


@dataclass(frozen=True)
class Relation:
    subject_track_id: int
    predicate: str
    object_track_id: int
    start_frame: int
    end_frame: int
    confidence: float = 1.0
    source: str = "unknown"
    evidence_frames: tuple[int, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_track_id", _integer(self.subject_track_id, "subject_track_id"))
        object.__setattr__(self, "object_track_id", _integer(self.object_track_id, "object_track_id"))
        if self.subject_track_id == self.object_track_id:
            raise ValueError("relation subject and object must be different tracks")
        object.__setattr__(self, "predicate", _text(self.predicate, "predicate").lower())
        object.__setattr__(self, "start_frame", _integer(self.start_frame, "start_frame"))
        object.__setattr__(self, "end_frame", _integer(self.end_frame, "end_frame"))
        # Existing VIDVRD artifacts use inclusive frame spans.
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must not be before start_frame")
        object.__setattr__(self, "confidence", _score(self.confidence, "confidence"))
        object.__setattr__(self, "source", _text(self.source, "source", default="unknown"))
        frames = tuple(sorted({_integer(frame, "evidence_frame") for frame in self.evidence_frames}))
        if any(frame < self.start_frame or frame > self.end_frame for frame in frames):
            raise ValueError("evidence_frames must fall inside [start_frame, end_frame]")
        object.__setattr__(self, "evidence_frames", frames)
        object.__setattr__(self, "extra", dict(self.extra))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Relation:
        item = _mapping(value, "relation")
        known = {
            "subject_track_id", "subject_id", "predicate", "object_track_id", "object_id",
            "start_frame", "end_frame", "confidence", "source", "sources", "evidence_frames",
        }
        source = item.get("source")
        if source is None and isinstance(item.get("sources"), Sequence) and not isinstance(item.get("sources"), (str, bytes)):
            source = ",".join(str(part) for part in item.get("sources", []) if str(part).strip())
        return cls(
            subject_track_id=_integer(item.get("subject_track_id", item.get("subject_id")), "subject_track_id"),
            predicate=_text(item.get("predicate"), "predicate"),
            object_track_id=_integer(item.get("object_track_id", item.get("object_id")), "object_track_id"),
            start_frame=_integer(item.get("start_frame"), "start_frame"),
            end_frame=_integer(item.get("end_frame"), "end_frame"),
            confidence=_score(item.get("confidence", 1.0), "confidence"),
            source=str(source or "unknown"),
            evidence_frames=tuple(item.get("evidence_frames", ()) or ()),
            extra=_extras(item, known),
        )

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.extra)
        out.update(
            {
                "subject_track_id": self.subject_track_id,
                "predicate": self.predicate,
                "object_track_id": self.object_track_id,
                "start_frame": self.start_frame,
                "end_frame": self.end_frame,
                "confidence": self.confidence,
                "source": self.source,
                "evidence_frames": list(self.evidence_frames),
            }
        )
        return out
