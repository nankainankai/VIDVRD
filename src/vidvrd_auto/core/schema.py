"""Validated serialization contracts for pipeline records."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


BBox = tuple[float, float, float, float]
SPAN_CONVENTIONS = frozenset({"half_open", "inclusive"})
BOX_SOURCES = frozenset({"observed", "interpolated", "predicted"})
SCORE_KINDS = frozenset({"detector_probability", "detector_logit", "heuristic", "legacy_confidence", "unavailable"})


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


def _optional_score(value: Any, name: str) -> float | None:
    return None if value is None else _score(value, name)


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
class FrameSpan:
    """Canonical half-open frame span with explicit legacy conversion."""

    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_frame", _integer(self.start_frame, "start_frame"))
        object.__setattr__(self, "end_frame", _integer(self.end_frame, "end_frame"))
        if self.end_frame <= self.start_frame:
            raise ValueError("half-open end_frame must be after start_frame")

    @classmethod
    def from_values(cls, start_frame: Any, end_frame: Any, *, convention: str = "half_open") -> FrameSpan:
        normalized = str(convention).strip().lower()
        if normalized not in SPAN_CONVENTIONS:
            raise ValueError(f"unsupported span convention: {convention}")
        start = _integer(start_frame, "start_frame")
        end = _integer(end_frame, "end_frame")
        return cls(start, end + 1 if normalized == "inclusive" else end)

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    def contains(self, frame: int) -> bool:
        value = _integer(frame, "frame")
        return self.start_frame <= value < self.end_frame

    def to_dict(self, *, convention: str = "half_open") -> dict[str, Any]:
        normalized = str(convention).strip().lower()
        if normalized not in SPAN_CONVENTIONS:
            raise ValueError(f"unsupported span convention: {convention}")
        end = self.end_frame - 1 if normalized == "inclusive" else self.end_frame
        return {"start_frame": self.start_frame, "end_frame": end, "span_convention": normalized}


def relation_span(value: Mapping[str, Any]) -> FrameSpan:
    """Read a relation span, treating unmarked v1 artifacts as inclusive."""

    return FrameSpan.from_values(
        value.get("start_frame"),
        value.get("end_frame"),
        convention=str(value.get("span_convention", "inclusive") or "inclusive"),
    )


def serialize_relation_artifact(
    value: Mapping[str, Any], *, convention: str = "inclusive"
) -> dict[str, Any]:
    """Write a relation dict with an explicit, normalized span convention."""

    output = dict(value)
    output.update(relation_span(value).to_dict(convention=convention))
    return output


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    class_name: str
    confidence: float | None = None
    frame: int | None = None
    source: str = "unknown"
    batch_id: int | None = None
    score_kind: str = "legacy_confidence"
    class_id: int | None = None
    is_observation: bool = True
    anchor_reason: str = "unknown"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", _bbox(self.bbox))
        object.__setattr__(self, "class_name", _text(self.class_name, "class_name"))
        object.__setattr__(self, "confidence", _optional_score(self.confidence, "confidence"))
        object.__setattr__(self, "source", _text(self.source, "source", default="unknown"))
        score_kind = str(self.score_kind or "unavailable").strip().lower()
        if score_kind not in SCORE_KINDS:
            raise ValueError(f"unsupported score_kind: {self.score_kind}")
        if self.confidence is None:
            score_kind = "unavailable"
        object.__setattr__(self, "score_kind", score_kind)
        if self.frame is not None:
            object.__setattr__(self, "frame", _integer(self.frame, "frame"))
        if self.batch_id is not None:
            object.__setattr__(self, "batch_id", _integer(self.batch_id, "batch_id"))
        if self.class_id is not None:
            object.__setattr__(self, "class_id", _integer(self.class_id, "class_id"))
        object.__setattr__(self, "is_observation", bool(self.is_observation))
        object.__setattr__(self, "anchor_reason", _text(self.anchor_reason, "anchor_reason", default="unknown"))
        object.__setattr__(self, "extra", dict(self.extra))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Detection:
        item = _mapping(value, "detection")
        known = {
            "bbox", "class_name", "class", "class_id", "confidence", "score", "score_kind", "frame", "frame_idx",
            "source", "batch_id", "is_observation", "anchor_reason",
        }
        score = item.get("score", item.get("confidence"))
        score_kind = item.get("score_kind")
        if score_kind is None:
            score_kind = "legacy_confidence" if score is not None else "unavailable"
        return cls(
            bbox=_bbox(item.get("bbox")),
            class_name=_text(item.get("class_name", item.get("class")), "class_name"),
            confidence=_optional_score(score, "score"),
            frame=item.get("frame", item.get("frame_idx")),
            source=str(item.get("source", "unknown") or "unknown"),
            batch_id=item.get("batch_id"),
            score_kind=str(score_kind),
            class_id=item.get("class_id"),
            is_observation=bool(item.get("is_observation", True)),
            anchor_reason=str(item.get("anchor_reason", "unknown") or "unknown"),
            extra=_extras(item, known),
        )

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.extra)
        out.update(
            {
                "bbox": list(self.bbox),
                "class_name": self.class_name,
                "score": self.confidence,
                "confidence": self.confidence,
                "score_kind": self.score_kind,
                "source": self.source,
                "is_observation": self.is_observation,
                "anchor_reason": self.anchor_reason,
            }
        )
        if self.frame is not None:
            out["frame"] = self.frame
        if self.batch_id is not None:
            out["batch_id"] = self.batch_id
        if self.class_id is not None:
            out["class_id"] = self.class_id
        return out


@dataclass(frozen=True)
class Track:
    track_id: int
    bbox: BBox
    class_name: str
    confidence: float | None
    frame: int | None = None
    bbox_observed: BBox | None = None
    is_predicted: bool = False
    time_since_update: int = 0
    box_source: str = "observed"
    track_status: str = "confirmed"
    track_quality: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _integer(self.track_id, "track_id"))
        object.__setattr__(self, "bbox", _bbox(self.bbox))
        object.__setattr__(self, "class_name", _text(self.class_name, "class_name"))
        object.__setattr__(self, "confidence", _optional_score(self.confidence, "confidence"))
        if self.frame is not None:
            object.__setattr__(self, "frame", _integer(self.frame, "frame"))
        if self.bbox_observed is not None:
            object.__setattr__(self, "bbox_observed", _bbox(self.bbox_observed, "bbox_observed"))
        object.__setattr__(self, "is_predicted", bool(self.is_predicted))
        object.__setattr__(self, "time_since_update", _integer(self.time_since_update, "time_since_update"))
        box_source = str(self.box_source).strip().lower()
        if box_source not in BOX_SOURCES:
            raise ValueError(f"unsupported box_source: {self.box_source}")
        object.__setattr__(self, "box_source", box_source)
        object.__setattr__(self, "track_status", _text(self.track_status, "track_status", default="confirmed").lower())
        if self.is_predicted and self.bbox_observed is not None:
            raise ValueError("predicted tracks must have bbox_observed=None")
        if box_source == "observed" and self.bbox_observed is None:
            raise ValueError("observed tracks must include bbox_observed")
        if box_source != "observed" and self.bbox_observed is not None:
            raise ValueError("non-observed tracks must have bbox_observed=None")
        object.__setattr__(self, "track_quality", dict(self.track_quality))
        object.__setattr__(self, "extra", dict(self.extra))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Track:
        item = _mapping(value, "track")
        known = {
            "track_id", "bbox", "class_name", "class", "confidence", "frame", "frame_idx",
            "bbox_observed", "is_predicted", "time_since_update", "box_source", "track_status", "track_quality",
        }
        box = _bbox(item.get("bbox"))
        bbox_observed = None if item.get("bbox_observed") is None else _bbox(item.get("bbox_observed"), "bbox_observed")
        is_predicted = bool(item.get("is_predicted", False))
        box_source = str(item.get("box_source") or ("predicted" if is_predicted else "observed"))
        if bbox_observed is None and not is_predicted and box_source == "observed":
            # Old records often omitted bbox_observed for a real observation.
            bbox_observed = box
        return cls(
            track_id=_integer(item.get("track_id"), "track_id"),
            bbox=box,
            class_name=_text(item.get("class_name", item.get("class")), "class_name"),
            confidence=_optional_score(item.get("confidence"), "confidence"),
            frame=item.get("frame", item.get("frame_idx")),
            bbox_observed=bbox_observed,
            is_predicted=is_predicted,
            time_since_update=_integer(item.get("time_since_update", item.get("age", 0)), "time_since_update"),
            box_source=box_source,
            track_status=str(item.get("track_status", "confirmed") or "confirmed"),
            track_quality=item.get("track_quality", {}) if isinstance(item.get("track_quality", {}), Mapping) else {},
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
                "box_source": self.box_source,
                "track_status": self.track_status,
                "track_quality": dict(self.track_quality),
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
    confidence: float | None = None
    rule_support: float | None = None
    agent_score: float | None = None
    ranking_score: float | None = None
    score_kind: str = "unavailable"
    source: str = "unknown"
    evidence_frames: tuple[int, ...] = ()
    span_convention: str = "half_open"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_track_id", _integer(self.subject_track_id, "subject_track_id"))
        object.__setattr__(self, "object_track_id", _integer(self.object_track_id, "object_track_id"))
        if self.subject_track_id == self.object_track_id:
            raise ValueError("relation subject and object must be different tracks")
        object.__setattr__(self, "predicate", _text(self.predicate, "predicate").lower())
        object.__setattr__(self, "start_frame", _integer(self.start_frame, "start_frame"))
        object.__setattr__(self, "end_frame", _integer(self.end_frame, "end_frame"))
        if self.end_frame <= self.start_frame:
            raise ValueError("half-open end_frame must be after start_frame")
        object.__setattr__(self, "confidence", _optional_score(self.confidence, "confidence"))
        object.__setattr__(self, "rule_support", _optional_score(self.rule_support, "rule_support"))
        object.__setattr__(self, "agent_score", _optional_score(self.agent_score, "agent_score"))
        ranking_score = _optional_score(self.ranking_score, "ranking_score")
        if ranking_score is None:
            ranking_score = next(
                (score for score in (self.agent_score, self.rule_support, self.confidence) if score is not None),
                None,
            )
        object.__setattr__(self, "ranking_score", ranking_score)
        object.__setattr__(self, "score_kind", str(self.score_kind or "unavailable"))
        object.__setattr__(self, "source", _text(self.source, "source", default="unknown"))
        frames = tuple(sorted({_integer(frame, "evidence_frame") for frame in self.evidence_frames}))
        if any(frame < self.start_frame or frame >= self.end_frame for frame in frames):
            raise ValueError("evidence_frames must fall inside [start_frame, end_frame)")
        object.__setattr__(self, "evidence_frames", frames)
        if str(self.span_convention).strip().lower() != "half_open":
            raise ValueError("Relation instances use canonical half_open spans")
        object.__setattr__(self, "span_convention", "half_open")
        object.__setattr__(self, "extra", dict(self.extra))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Relation:
        item = _mapping(value, "relation")
        known = {
            "subject_track_id", "subject_id", "predicate", "object_track_id", "object_id",
            "start_frame", "end_frame", "span_convention", "confidence", "rule_support", "agent_score",
            "ranking_score", "score_kind", "source", "sources", "evidence_frames",
        }
        source = item.get("source")
        if source is None and isinstance(item.get("sources"), Sequence) and not isinstance(item.get("sources"), (str, bytes)):
            source = ",".join(str(part) for part in item.get("sources", []) if str(part).strip())
        span = relation_span(item)
        return cls(
            subject_track_id=_integer(item.get("subject_track_id", item.get("subject_id")), "subject_track_id"),
            predicate=_text(item.get("predicate"), "predicate"),
            object_track_id=_integer(item.get("object_track_id", item.get("object_id")), "object_track_id"),
            start_frame=span.start_frame,
            end_frame=span.end_frame,
            confidence=_optional_score(item.get("confidence"), "confidence"),
            rule_support=_optional_score(item.get("rule_support"), "rule_support"),
            agent_score=_optional_score(item.get("agent_score"), "agent_score"),
            ranking_score=_optional_score(item.get("ranking_score"), "ranking_score"),
            score_kind=str(item.get("score_kind", "legacy_confidence" if item.get("confidence") is not None else "unavailable")),
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
                "rule_support": self.rule_support,
                "agent_score": self.agent_score,
                "ranking_score": self.ranking_score,
                "score_kind": self.score_kind,
                "source": self.source,
                "evidence_frames": list(self.evidence_frames),
                "span_convention": "half_open",
            }
        )
        for field in ("confidence", "rule_support", "agent_score", "ranking_score"):
            if out[field] is None:
                del out[field]
        return out

    def to_legacy_dict(self) -> dict[str, Any]:
        """Serialize to the inclusive span used by existing v1 artifacts."""

        out = self.to_dict()
        out["end_frame"] = self.end_frame - 1
        out["span_convention"] = "inclusive"
        return out
