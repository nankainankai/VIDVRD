"""Small, dependency-free contracts shared by pipeline stages."""

from .config import AppConfig, ConfigSection
from .context import RunContext, Secrets
from .paths import VideoPaths
from .ontology import load_ontology, normalize_object, object_names, predicate_names
from .schema import Detection, Relation, Track

__all__ = [
    "AppConfig",
    "ConfigSection",
    "Detection",
    "Relation",
    "RunContext",
    "Secrets",
    "Track",
    "VideoPaths",
    "load_ontology",
    "normalize_object",
    "object_names",
    "predicate_names",
]
