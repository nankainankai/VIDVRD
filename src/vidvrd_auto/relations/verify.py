from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from vidvrd_auto.relations.ops import verify_relations as _verify_relations


def verify_relations(
    *,
    video_id: str,
    relations_json: Path,
    tracks_jsonl: Path,
    out_relations_json: Path,
    out_qc_json: Path,
    config: Dict[str, Any],
    windows_json: Optional[Path] = None,
    api_key: str = "",
) -> Dict[str, Any]:
    return _verify_relations(
        video_id=video_id,
        relations_json=relations_json,
        tracks_jsonl=tracks_jsonl,
        out_relations_json=out_relations_json,
        out_qc_json=out_qc_json,
        config=config,
        windows_json=windows_json,
        api_key=api_key,
    )
