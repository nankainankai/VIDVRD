from __future__ import annotations

"""轨迹生成节点。

输入：视频、检测 JSONL 和 tracking 配置。
输出：`tracks_full.jsonl`、`windows.json` 以及可选 pair 可视化。
当前通过新包 adapter 调用旧 Step2 脚本，后续会把 OC-SORT 和窗口化逻辑完全迁入 `vidvrd_auto.tracking`。
"""

from pathlib import Path
from typing import Any, Dict

from vidvrd_auto.tracking.legacy_step2 import run_legacy_step2
from vidvrd_auto.tracking.mock_track import run_mock_track


def run_track(
    *,
    video_path: Path,
    detections_jsonl: Path,
    out_dir: Path,
    config: Dict[str, Any],
    api_key: str,
    log_path: Path,
) -> None:
    backend = str(config.get("backend", "legacy")).strip().lower()
    if backend == "mock":
        run_mock_track(
            video_path=video_path,
            detections_jsonl=detections_jsonl,
            out_dir=out_dir,
            config=config,
            log_path=log_path,
        )
        return
    run_legacy_step2(
        video_path=video_path,
        detections_jsonl=detections_jsonl,
        out_dir=out_dir,
        config=config,
        api_key=api_key,
        log_path=log_path,
    )
