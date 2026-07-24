from __future__ import annotations

"""Open-vocabulary predicate candidates without object-pair whitelists."""

from typing import List

from vidvrd_auto.core.ontology import normalize_object, predicate_names


GEOMETRY_PREDICATES = {"left", "right", "above", "beneath", "front", "behind", "next_to"}
SYMMETRIC_PREDICATES = {"next_to", "touch", "fight", "play", "with"}


def normalize_category(category: str) -> str:
    return normalize_object(category)


def get_candidate_predicates(
    subject_class: str,
    object_class: str,
    *,
    split: str = "all",
) -> List[str]:
    """Return the complete official predicate vocabulary for any valid pair.

    Object names do not suppress candidates: visual evidence, rather than a
    category-pair whitelist, decides which predicates survive.
    """

    del subject_class, object_class
    return predicate_names(split=split)
