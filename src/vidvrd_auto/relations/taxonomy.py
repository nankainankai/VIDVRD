from __future__ import annotations

"""Relationship metadata derived from the official 132-class ontology."""

from pathlib import Path
from typing import Any, Dict, List, Set

from vidvrd_auto.core.ontology import load_ontology, predicate_components, predicate_names


_INVERSES = {
    "left": "right",
    "right": "left",
    "above": "beneath",
    "beneath": "above",
    "front": "behind",
    "behind": "front",
    "away": "toward",
    "toward": "away",
}


def load_taxonomy(path: Path | None = None) -> Dict[str, Any]:
    del path
    return load_ontology()


def predicate_defs(path: Path | None = None) -> Dict[str, Dict[str, Any]]:
    del path
    return {
        str(item["name"]): {
            "id": int(item["id"]),
            "split": str(item["split"]),
            "components": dict(item.get("components", {})),
            "inverse": _INVERSES.get(str(item["name"]), ""),
        }
        for item in load_ontology()["predicates"]
    }


def coupling_inverse(path: Path | None = None) -> Dict[str, str]:
    del path
    return dict(_INVERSES)


def mutex_pairs(path: Path | None = None) -> Set[frozenset[str]]:
    del path
    return {
        frozenset(("left", "right")),
        frozenset(("above", "beneath")),
        frozenset(("front", "behind")),
        frozenset(("away", "toward")),
    }


def prompt_predicate_summary(path: Path | None = None) -> str:
    del path
    lines: List[str] = []
    for name in predicate_names():
        components = predicate_components(name)
        values = ", ".join(f"{key}={value}" for key, value in components.items() if value)
        lines.append(f"- {name}: {values or 'atomic'}")
    return "\n".join(lines)
