from __future__ import annotations

"""Checked hierarchy over the official ImageNet-VidVRD predicates."""

from functools import lru_cache
from typing import Any

from vidvrd_auto.core.ontology import load_ontology
from vidvrd_auto.utils.io import read_json
from vidvrd_auto.utils.paths import repo_root


@lru_cache(maxsize=1)
def load_hierarchy() -> dict[str, Any]:
    return read_json(repo_root() / "configs" / "predicate_hierarchy.json")


@lru_cache(maxsize=4)
def predicate_records(split: str = "all") -> tuple[dict[str, Any], ...]:
    hierarchy = load_hierarchy()
    family_defs = hierarchy["families"]
    records = []
    for item in load_ontology()["predicates"]:
        if split != "all" and str(item.get("split")) != split:
            continue
        name = str(item["name"])
        components = dict(item.get("components", {}))
        action = str(components.get("action", ""))
        spatial = str(components.get("spatial", ""))
        comparative = str(components.get("comparative", ""))
        families = []
        for family, rule in family_defs.items():
            if (
                action in rule.get("actions", [])
                or spatial in rule.get("spatials", [])
                or name in rule.get("predicates", [])
                or (family == "comparative" and bool(comparative))
            ):
                families.append(family)
        if not families:
            families.append("attention_social")
        records.append(
            {
                "name": name,
                "split": str(item.get("split", "")),
                "components": components,
                "families": families,
            }
        )
    return tuple(records)


def confusion_siblings(predicate: str) -> list[str]:
    by_name = {str(record["name"]): record for record in predicate_records()}
    record = by_name.get(predicate)
    if record is None:
        return []
    action = str(record["components"].get("action", ""))
    spatial = str(record["components"].get("spatial", ""))
    siblings: list[str] = []
    for group in load_hierarchy().get("confusion_groups", {}).values():
        if predicate in group:
            siblings.extend(value for value in group if value != predicate)
        if action and action in group:
            for candidate in predicate_records():
                candidate_action = str(candidate["components"].get("action", ""))
                candidate_spatial = str(candidate["components"].get("spatial", ""))
                if candidate_action in group and candidate_action != action and candidate_spatial == spatial:
                    siblings.append(str(candidate["name"]))
        if spatial and spatial in group:
            for candidate in predicate_records():
                candidate_action = str(candidate["components"].get("action", ""))
                candidate_spatial = str(candidate["components"].get("spatial", ""))
                if candidate_spatial in group and candidate_spatial != spatial and candidate_action == action:
                    siblings.append(str(candidate["name"]))
    return list(dict.fromkeys(siblings))


def family_descriptions() -> dict[str, str]:
    return {
        name: str(value.get("description", ""))
        for name, value in load_hierarchy()["families"].items()
    }


def neighboring_families(families: list[str]) -> list[str]:
    mapping = load_hierarchy().get("family_neighbors", {})
    values = [neighbor for family in families for neighbor in mapping.get(family, []) if neighbor not in families]
    return list(dict.fromkeys(values))
