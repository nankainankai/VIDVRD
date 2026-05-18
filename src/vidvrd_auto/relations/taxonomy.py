from __future__ import annotations

"""谓词 taxonomy。

这里集中定义谓词中文解释、互斥组、反向耦合、是否依赖接触和运动。
关系生成、合并、复核节点都应优先读取这里，而不是各自硬编码。
"""

from pathlib import Path
from typing import Any, Dict, List, Set

from vidvrd_auto.utils.io import read_json
from vidvrd_auto.utils.paths import repo_root


def load_taxonomy(path: Path | None = None) -> Dict[str, Any]:
    taxonomy_path = path or (repo_root() / "configs" / "predicate_taxonomy.json")
    if not taxonomy_path.exists():
        return {"predicates": {}}
    obj = read_json(taxonomy_path)
    return obj if isinstance(obj, dict) else {"predicates": {}}


def predicate_defs(path: Path | None = None) -> Dict[str, Dict[str, Any]]:
    obj = load_taxonomy(path)
    preds = obj.get("predicates", {})
    return {str(k): dict(v) for k, v in preds.items() if isinstance(v, dict)}


def coupling_inverse(path: Path | None = None) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for pred, meta in predicate_defs(path).items():
        inv = str(meta.get("inverse", "") or "").strip()
        if inv:
            out[pred] = inv
    return out


def mutex_pairs(path: Path | None = None) -> Set[frozenset[str]]:
    groups: Dict[str, List[str]] = {}
    for pred, meta in predicate_defs(path).items():
        group = str(meta.get("mutex_group", "") or "").strip()
        if group:
            groups.setdefault(group, []).append(pred)
    out: Set[frozenset[str]] = set()
    for items in groups.values():
        if len(items) >= 2:
            out.add(frozenset(items))
    return out


def prompt_predicate_summary(path: Path | None = None) -> str:
    lines = []
    for pred, meta in sorted(predicate_defs(path).items()):
        zh = str(meta.get("zh", "") or "")
        category = str(meta.get("category", "") or "")
        lines.append(f"- {pred}: {zh}，类别={category}")
    return "\n".join(lines)
