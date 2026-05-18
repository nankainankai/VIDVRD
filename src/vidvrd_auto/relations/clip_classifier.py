from __future__ import annotations

"""片段关系分类迁移层。

当前先通过旧 `semi_auto_label_relations.py` 完成 storyboard 生成、分组询问和结构化解析。
节点只调用本模块；后续可把旧脚本内部逻辑迁入这里并更名为正式 `clip_relation`。
"""

from pathlib import Path
from typing import Any, Dict

from vidvrd_auto.utils.io import write_json
from vidvrd_auto.utils.paths import repo_root
from vidvrd_auto.utils.process import python_executable, run_cmd


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
    scripts = repo_root() / "my_scripts"
    cmd = [
        python_executable(),
        str(scripts / "semi_auto_label_relations.py"),
        "--windows_json",
        str(windows_json),
        "--tracks_jsonl",
        str(tracks_jsonl),
        "--output_json",
        str(out_json),
        "--save_storyboards_dir",
        str(storyboards_dir),
        "--group_size",
        str(int(config.get("group_size", 3))),
        "--max_windows",
        str(int(config.get("max_windows", 0))),
        "--max_frames_per_window",
        str(int(config.get("max_frames_per_window", 8))),
        "--retries",
        str(int(config.get("retries", 2))),
        "--backoff_sec",
        str(float(config.get("backoff_sec", 1.5))),
        "--sleep_sec",
        str(float(config.get("sleep_sec", 0.0))),
    ]
    if api_key:
        cmd += ["--api_key", api_key]
    if str(config.get("api_model", "")).strip():
        cmd += ["--model_vl", str(config.get("api_model")).strip()]
    if resume:
        cmd += ["--resume"]
    if dry_run:
        cmd += ["--dry_run"]
    if str(config.get("relations", "")).strip():
        cmd += ["--relations", str(config.get("relations")).strip()]
    if str(config.get("vggsound_label", "")).strip():
        cmd += ["--vggsound_label", str(config.get("vggsound_label")).strip()]
    run_cmd(cmd, cwd=repo_root(), log_path=log_path)
    if dry_run and not out_json.exists():
        write_json(out_json, {video_id: []})
