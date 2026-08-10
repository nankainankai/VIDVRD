"""Small, dependency-free contracts shared by pipeline stages."""

from .config import AppConfig, ConfigSection
from .context import RunContext, Secrets
from .paths import VideoPaths
from .ontology import load_ontology, normalize_object, object_names, predicate_names
from .schema import Detection, FrameSpan, Relation, Track, relation_span, serialize_relation_artifact

__all__ = [
    "AppConfig",
    "ConfigSection",
    "Detection",
    "FrameSpan",
    "Relation",
    "RunContext",
    "Secrets",
    "Track",
    "VideoPaths",
    "load_ontology",
    "normalize_object",
    "object_names",
    "predicate_names",
    "relation_span",
    "serialize_relation_artifact",
]
