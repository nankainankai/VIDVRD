from __future__ import annotations

"""片段关系分类节点入口。

实现位于 `clip_relation.py`；本模块保持对外函数名不变，供 `relation_llm` 节点调用。
"""

from pathlib import Path
from typing import Any, Dict

from vidvrd_auto.relations.clip_relation import run_clip_relation as _run_clip_relation
from vidvrd_auto.utils.io import write_json


def run_clip_relation(
    *,
    windows_json: Path,
    tracks_jsonl: Path,
    out_json: Path,
    storyboards_dir: Path,
    config: Dict[str, Any],
    api_key: str,
    resume: bool,
    dry_run: bool,
    video_id: str,
    log_path: Path,
) -> None:
    try:
        _run_clip_relation(
            windows_json=windows_json,
            tracks_jsonl=tracks_jsonl,
            out_json=out_json,
            storyboards_dir=storyboards_dir,
            config=config,
            api_key=api_key,
            resume=resume,
            dry_run=dry_run,
            video_id=video_id,
            log_path=log_path,
        )
    except Exception:
        if dry_run and not out_json.exists():
            write_json(out_json, {video_id: []})
        raise
