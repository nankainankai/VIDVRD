"""目标检测模块（可选后端）

默认后端：DINO-X (CloudAPI)

可选后端：Rex-Omni（本地模型，需额外依赖）

说明：
- 通过 dds-cloudapi-sdk 调用 /v2/task/dinox/detection。
- 输出格式与原工作流兼容：
  [{'bbox':[x1,y1,x2,y2], 'class':int, 'class_name':str, 'confidence':float}, ...]
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

# 导入配置
try:
    import config
except Exception:  # pragma: no cover
    config = None  # type: ignore


class ObjectDetector:
    """目标检测器（可选后端：dinox / rexomni）。"""

    def __init__(self, backend: str = "", **kwargs):
        if config is None:
            raise ImportError("config.py 未找到，无法读取检测器配置")

        backend_cfg = str(getattr(config, "DETECTOR_BACKEND", "dinox") or "dinox").strip().lower()
        self.backend = (backend or backend_cfg or "dinox").strip().lower()

        if self.backend == "dinox":
            from modules.dinox_detector import DINOXObjectDetector

            interval_override = kwargs.get("detection_interval")
            interval = (
                int(interval_override)
                if interval_override is not None
                else int(getattr(config, "DINOX_DETECTION_INTERVAL", 1))
            )

            self._det = DINOXObjectDetector(
                api_token=str(getattr(config, "DINOX_API_TOKEN", "")),
                text_prompt=str(getattr(config, "DINOX_TEXT_PROMPT", "")),
                model=str(getattr(config, "DINOX_MODEL", "DINO-X-1.0")),
                bbox_threshold=float(getattr(config, "DINOX_BBOX_THRESHOLD", 0.25)),
                iou_threshold=float(getattr(config, "DINOX_IOU_THRESHOLD", 0.8)),
                image_max_long_edge=int(getattr(config, "DINOX_IMAGE_MAX_LONG_EDGE", 960)),
                detection_interval=interval,
                max_calls=int(getattr(config, "DINOX_MAX_CALLS", 0)),
                nms_iou_threshold=float(getattr(config, "DINOX_NMS_IOU_THRESHOLD", 0.55)),
                min_box_area=float(getattr(config, "DINOX_MIN_BOX_AREA", 400.0)),
                max_detections_per_frame=int(getattr(config, "DINOX_MAX_DETECTIONS_PER_FRAME", 80)),
                request_retries=int(getattr(config, "DINOX_REQUEST_RETRIES", 1)),
                request_backoff_sec=float(getattr(config, "DINOX_REQUEST_BACKOFF_SEC", 1.2)),
            )
            return

        if self.backend == "rexomni":
            from modules.rexomni_detector import RexOmniObjectDetector

            interval_override = kwargs.get("detection_interval")
            interval = (
                int(interval_override)
                if interval_override is not None
                else int(getattr(config, "REXOMNI_DETECTION_INTERVAL", 1))
            )

            model_path = str(kwargs.get("rex_model_path") or getattr(config, "REXOMNI_MODEL_PATH", "IDEA-Research/Rex-Omni"))
            engine = str(kwargs.get("rex_backend") or getattr(config, "REXOMNI_BACKEND", "transformers"))
            categories = kwargs.get("rex_categories")
            if categories is None:
                categories = getattr(config, "REXOMNI_CATEGORIES", None)
            if isinstance(categories, str):
                categories = [c.strip() for c in categories.split(",") if c.strip()]
            if not categories:
                categories = [str(getattr(config, "DINOX_TEXT_PROMPT", "person"))]

            self._det = RexOmniObjectDetector(
                model_path=model_path,
                backend=engine,
                categories=categories,
                detection_interval=interval,
                min_box_area=float(getattr(config, "REXOMNI_MIN_BOX_AREA", getattr(config, "DINOX_MIN_BOX_AREA", 400.0))),
                max_detections_per_frame=int(getattr(config, "REXOMNI_MAX_DETECTIONS_PER_FRAME", getattr(config, "DINOX_MAX_DETECTIONS_PER_FRAME", 80))),
                max_tokens=int(getattr(config, "REXOMNI_MAX_TOKENS", 512)),
            )
            return

        raise ValueError(f"Unknown detector backend: {self.backend} (expected dinox|rexomni)")

    def load_model(self) -> None:
        self._det.load_model()

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        return self._det.detect(frame)

    def get_stats(self) -> Dict[str, Any]:
        getter = getattr(self._det, "get_stats", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                return {}
        return {}
