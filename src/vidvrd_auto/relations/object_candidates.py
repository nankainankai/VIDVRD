from __future__ import annotations

"""Open-vocabulary predicate candidates without object-pair whitelists."""

from typing import List

from vidvrd_auto.core.ontology import normalize_object, predicate_names


# Only relations that can be determined from 2-D boxes belong here.  In
# particular, front/behind require visual depth or scene semantics and must be
# left for the vision-language classifier.
GEOMETRY_PREDICATES = {"left", "right", "above", "beneath", "next_to"}
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
