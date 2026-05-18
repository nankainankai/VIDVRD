from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from vidvrd_auto.relations.ops import generate_rule_relations as _generate_rule_relations


def generate_rule_relations(
    *,
    windows_json: Path,
    tracks_jsonl: Path,
    out_json: Path,
    video_id: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    return _generate_rule_relations(
        windows_json=windows_json,
        tracks_jsonl=tracks_jsonl,
        out_json=out_json,
        video_id=video_id,
        config=config,
    )
