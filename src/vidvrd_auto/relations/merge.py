from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from vidvrd_auto.relations.ops import merge_relations as _merge_relations


def merge_relations(
    *,
    video_id: str,
    relation_jsons: Sequence[Path],
    out_json: Path,
    apply_coupling: bool = True,
) -> Dict[str, Any]:
    return _merge_relations(
        video_id=video_id,
        relation_jsons=relation_jsons,
        out_json=out_json,
        apply_coupling=apply_coupling,
    )
