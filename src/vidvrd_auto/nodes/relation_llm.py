from __future__ import annotations

"""片段关系分类节点。

输入：窗口 JSON、轨迹 JSONL、音频先验和关系模型配置。
输出：`relations_llm.json` 与 storyboard 审计图片。
当前通过新包 adapter 调用旧半自动关系脚本，后续会迁为 `clip_relation` 正式节点。
"""

from pathlib import Path
from typing import Any, Dict

from vidvrd_auto.relations.clip_classifier import run_clip_relation


def run_relation_llm(
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
    run_clip_relation(
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
