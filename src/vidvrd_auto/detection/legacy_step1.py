from __future__ import annotations

"""旧 Step1 检测脚本的迁移适配器。

节点通过本模块调用检测能力，而不是直接拼旧脚本命令。
迁移完成后，本文件可以替换为纯 Python 检测实现。
"""

import os
from pathlib import Path
from typing import Any, Dict

from vidvrd_auto.utils.paths import repo_root
from vidvrd_auto.utils.process import python_executable, run_cmd


def detector_env(config: Dict[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "DETECTOR_BACKEND": str(config.get("backend", "rexomni")),
            "DINOX_TEXT_PROMPT": str(config.get("dinox_text_prompt", "person")),
            "DINOX_MODEL": str(config.get("dinox_model", "DINO-X-1.0")),
            "DINOX_BBOX_THRESHOLD": str(config.get("dinox_bbox_threshold", 0.25)),
            "DINOX_IOU_THRESHOLD": str(config.get("dinox_iou_threshold", 0.8)),
            "DINOX_NMS_IOU_THRESHOLD": str(config.get("dinox_nms_iou_threshold", 0.55)),
            "DINOX_MIN_BOX_AREA": str(config.get("dinox_min_box_area", 500.0)),
            "DINOX_MAX_DETECTIONS_PER_FRAME": str(config.get("dinox_max_detections_per_frame", 60)),
            "DINOX_IMAGE_MAX_LONG_EDGE": str(config.get("dinox_image_max_long_edge", 640)),
            "DINOX_DETECTION_INTERVAL": str(config.get("dinox_detection_interval", 5)),
            "DINOX_MAX_CALLS": str(config.get("dinox_max_calls", 50)),
            "DINOX_REQUEST_RETRIES": str(config.get("dinox_request_retries", 1)),
            "DINOX_REQUEST_BACKOFF_SEC": str(config.get("dinox_request_backoff_sec", 1.2)),
            "REXOMNI_MODEL_PATH": str(config.get("rex_model_path", "")),
            "REXOMNI_BACKEND": str(config.get("rex_backend", "transformers")),
            "REXOMNI_CATEGORIES": str(config.get("rex_categories", "")),
            "REXOMNI_DETECTION_INTERVAL": str(config.get("rex_detection_interval", 1)),
            "REXOMNI_MIN_BOX_AREA": str(config.get("rex_min_box_area", config.get("dinox_min_box_area", 500.0))),
            "REXOMNI_MAX_DETECTIONS_PER_FRAME": str(config.get("rex_max_detections_per_frame", 60)),
            "REXOMNI_MAX_TOKENS": str(config.get("rex_max_tokens", 512)),
        }
    )
    return env


def run_legacy_step1(*, video_path: Path, out_dir: Path, config: Dict[str, Any], log_path: Path) -> None:
    scripts = repo_root() / "my_scripts"
    cmd = [
        python_executable(),
        str(scripts / "step1_full_video_box_detection_dinox.py"),
        "--video",
        str(video_path),
        "--output_dir",
        str(out_dir),
        "--backend",
        str(config.get("backend", "rexomni")),
        "--keyframe_interval",
        str(int(config.get("keyframe_interval", 25))),
        "--interp_iou_thresh",
        str(float(config.get("interp_iou_thresh", 0.1))),
    ]
    if str(config.get("rex_model_path", "")).strip():
        cmd += ["--rex_model_path", str(config.get("rex_model_path")).strip()]
    if str(config.get("rex_backend", "")).strip():
        cmd += ["--rex_backend", str(config.get("rex_backend")).strip()]
    if str(config.get("rex_categories", "")).strip():
        cmd += ["--rex_categories", str(config.get("rex_categories")).strip()]
    cmd += ["--save_box_video" if bool(config.get("save_box_video", False)) else "--no_save_box_video"]
    if bool(config.get("auto_install_rexomni", False)):
        cmd += ["--auto_install_rexomni", "--auto_install_torch", str(config.get("auto_install_torch", "skip"))]
    run_cmd(cmd, cwd=repo_root(), log_path=log_path, env=detector_env(config))
