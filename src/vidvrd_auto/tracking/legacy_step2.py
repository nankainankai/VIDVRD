from __future__ import annotations

"""旧 Step2 追踪脚本的迁移适配器。"""

import os
from pathlib import Path
from typing import Any, Dict

from vidvrd_auto.utils.paths import repo_root
from vidvrd_auto.utils.process import python_executable, run_cmd


def tracking_env(config: Dict[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "TRACKING_IOU_THRESHOLD": str(config.get("iou_threshold", 0.5)),
            "TRACKING_MAX_AGE": str(config.get("max_age", 30)),
            "TRACKING_MIN_HITS": str(config.get("min_hits", 3)),
            "TRACKING_CLASS_AWARE": "1" if bool(config.get("class_aware", True)) else "0",
            "TRACKING_MAX_CENTER_DIST_RATIO": str(config.get("max_center_dist_ratio", 0.8)),
            "TRACKING_MIN_NEW_TRACK_CONF": str(config.get("min_new_track_conf", 0.35)),
            "TRACKING_VELOCITY_ALPHA": str(config.get("velocity_alpha", 0.8)),
            "ENABLE_STEP2_LLM_QC": "1" if bool(config.get("enable_llm_qc", True)) else "0",
            "STEP2_LLM_QC_MODEL": str(config.get("llm_qc_model", "qwen-vl-max")),
            "STEP2_QC_SAMPLE_FRAMES": str(config.get("qc_sample_frames", 8)),
            "STEP2_QC_COUNT_DIFF_THRESHOLD": str(config.get("qc_count_diff_threshold", 1)),
            "EXPORT_PAIR_VIZ_VIDEOS": "1" if bool(config.get("export_pair_viz_videos", True)) else "0",
            "PAIR_VIZ_MAX_WINDOWS": str(config.get("pair_viz_max_windows", 0)),
            "PAIR_VIZ_MAX_PAIRS_PER_WINDOW": str(config.get("pair_viz_max_pairs_per_window", 6)),
        }
    )
    return env


def run_legacy_step2(
    *,
    video_path: Path,
    detections_jsonl: Path,
    out_dir: Path,
    config: Dict[str, Any],
    api_key: str,
    log_path: Path,
) -> None:
    scripts = repo_root() / "my_scripts"
    cmd = [
        python_executable(),
        str(scripts / "step2_full_video_tracking_ocsort_qc_pairviz.py"),
        "--video",
        str(video_path),
        "--output_dir",
        str(out_dir),
        "--detections_jsonl",
        str(detections_jsonl),
        "--window_size",
        str(int(config.get("window_size", 30))),
        "--stride",
        str(int(config.get("stride", 15))),
    ]
    if api_key:
        cmd += ["--api_key", api_key]
    run_cmd(cmd, cwd=repo_root(), log_path=log_path, env=tracking_env(config))
