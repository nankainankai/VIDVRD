from __future__ import annotations

"""ImageNet-VidVRD ontology and open-vocabulary label normalization."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

from vidvrd_auto.utils.io import read_json
from vidvrd_auto.utils.paths import repo_root


def _path(path: Path | None = None) -> Path:
    return path or (repo_root() / "configs" / "vidvrd_ontology.json")


@lru_cache(maxsize=4)
def load_ontology(path: Path | None = None) -> Dict[str, Any]:
    value = read_json(_path(path))
    if not isinstance(value, dict):
        raise ValueError("VidVRD ontology must be a JSON object")
    objects = value.get("objects", [])
    predicates = value.get("predicates", [])
    if len(objects) != 35 or len(predicates) != 132:
        raise ValueError(f"expected VidVRD 35/132 ontology, got {len(objects)}/{len(predicates)}")
    return value


def object_names(*, split: str = "all") -> List[str]:
    return [
        str(item["name"])
        for item in load_ontology()["objects"]
        if split == "all" or str(item.get("split")) == split
    ]


def predicate_names(*, split: str = "all") -> List[str]:
    return [
        str(item["name"])
        for item in load_ontology()["predicates"]
        if split == "all" or str(item.get("split")) == split
    ]


def predicate_splits() -> Dict[str, str]:
    return {str(item["name"]): str(item["split"]) for item in load_ontology()["predicates"]}


def predicate_components(name: str) -> Dict[str, str]:
    key = str(name or "").strip().lower()
    for item in load_ontology()["predicates"]:
        if item.get("name") == key:
            return dict(item.get("components", {}))
    return {"action": key, "spatial": "", "comparative": ""}


def normalize_object(name: str, extra_aliases: Dict[str, str] | None = None) -> str:
    value = str(name or "unknown").strip().lower().replace("-", " ").replace("_", " ")
    value = " ".join(value.split())
    aliases = {str(k).lower(): str(v).lower() for k, v in load_ontology().get("object_aliases", {}).items()}
    aliases.update({str(k).lower(): str(v).lower() for k, v in (extra_aliases or {}).items()})
    if value in aliases:
        return aliases[value]
    canonical = value.replace(" ", "_")
    return canonical or "unknown"


def normalize_vocabulary(values: Iterable[str], *, aliases: Dict[str, str] | None = None) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    seen: set[str] = set()
    base = set(object_names())
    for raw in values:
        raw_name = str(raw or "").strip()
        canonical = normalize_object(raw_name, aliases)
        if not raw_name or canonical in seen:
            continue
        output.append(
            {
                "raw_label": raw_name,
                "canonical_label": canonical,
                "ontology_source": "vidvrd" if canonical in base else "discovered",
            }
        )
        seen.add(canonical)
    return output
