from __future__ import annotations

"""检测模型统一入口。

当前阶段先提供配置规范化和后端名称解析，节点仍通过迁移适配层调用旧 Step1。
后续 Rex-Omni/DINO-X 的直接 Python 调用应收敛到这里。
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DetectorConfig:
    backend: str
    rex_model_path: str = ""
    rex_backend: str = "transformers"
    rex_categories: str = "person"
    keyframe_interval: int = 25
    interp_iou_thresh: float = 0.1
    save_box_video: bool = False


def load_detector_config(config: Dict[str, Any]) -> DetectorConfig:
    return DetectorConfig(
        backend=str(config.get("backend", "rexomni") or "rexomni").strip().lower(),
        rex_model_path=str(config.get("rex_model_path", "") or ""),
        rex_backend=str(config.get("rex_backend", "transformers") or "transformers"),
        rex_categories=str(config.get("rex_categories", "person") or "person"),
        keyframe_interval=int(config.get("keyframe_interval", 25) or 25),
        interp_iou_thresh=float(config.get("interp_iou_thresh", 0.1) or 0.1),
        save_box_video=bool(config.get("save_box_video", False)),
    )
