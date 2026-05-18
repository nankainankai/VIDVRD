from __future__ import annotations

"""检测节点。

输入：单个视频和 detector 配置。
输出：`detections_full.jsonl` 与 `video_meta.json`。
当前通过新包 adapter 调用旧 Step1 脚本，后续会把检测算法完全迁入 `vidvrd_auto.detection`。
"""

from pathlib import Path
from typing import Any, Dict

from vidvrd_auto.detection.legacy_step1 import run_legacy_step1


def run_detect(*, video_path: Path, out_dir: Path, config: Dict[str, Any], log_path: Path) -> None:
    run_legacy_step1(video_path=video_path, out_dir=out_dir, config=config, log_path=log_path)
