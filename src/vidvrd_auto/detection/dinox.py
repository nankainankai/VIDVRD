from __future__ import annotations

"""DINO-X cloud detection adapter."""

import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np


def _dotenv_value(key: str) -> str:
    path = Path(__file__).resolve().parents[3] / ".env"
    if not path.is_file():
        return ""
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == key:
            return value.strip().strip("'").strip('"')
    return ""


class DinoXDetector:
    """Small adapter around the DDS DINO-X V2 detection task."""

    def __init__(
        self,
        *,
        categories: Sequence[str] | str,
        category_aliases: Dict[str, str] | None = None,
        model: str = "DINO-X-1.0",
        api_key_env: str = "DDS_API_TOKEN",
        bbox_threshold: float = 0.25,
        iou_threshold: float = 0.8,
        max_detections_per_frame: int = 60,
    ) -> None:
        category_items = categories.split(",") if isinstance(categories, str) else categories
        self.categories = [str(item).strip() for item in category_items if str(item).strip()]
        self.category_aliases = {
            str(key).strip().lower(): str(value).strip().lower()
            for key, value in dict(category_aliases or {}).items()
        }
        self.model = str(model)
        self.api_key_env = str(api_key_env)
        self.bbox_threshold = float(bbox_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections_per_frame = max(1, int(max_detections_per_frame))
        self._class_ids = {name.lower(): index for index, name in enumerate(self.categories)}
        self._client: Any = None
        self._task_type: Any = None
        self._image_to_base64: Any = None
        self._calls = 0
        self._detected_objects_total = 0

    def load_model(self) -> None:
        token = (os.getenv(self.api_key_env, "") or _dotenv_value(self.api_key_env)).strip()
        if not token:
            raise RuntimeError(
                f"DINO-X API token is missing; set {self.api_key_env} in the environment or project .env"
            )
        try:
            from dds_cloudapi_sdk import Client, Config  # type: ignore
            from dds_cloudapi_sdk.image_resizer import image_to_base64  # type: ignore
            from dds_cloudapi_sdk.tasks.v2_task import V2Task  # type: ignore
        except ImportError as exc:
            raise ImportError("DINO-X requires dds-cloudapi-sdk; install the project dependencies") from exc
        self._client = Client(Config(token))
        self._task_type = V2Task
        self._image_to_base64 = image_to_base64

    def _canonical(self, raw_name: str) -> str:
        key = raw_name.strip().lower()
        return self.category_aliases.get(key, key.replace(" ", "_"))

    def _class_id(self, name: str) -> int:
        key = name.lower()
        if key not in self._class_ids:
            self._class_ids[key] = len(self._class_ids)
        return self._class_ids[key]

    def _parse(self, result: Any, *, width: int, height: int) -> List[Dict[str, Any]]:
        objects = result.get("objects", []) if isinstance(result, dict) else []
        output: List[Dict[str, Any]] = []
        for obj in objects if isinstance(objects, list) else []:
            if not isinstance(obj, dict):
                continue
            box = obj.get("bbox")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            try:
                x1, y1, x2, y2 = [float(value) for value in box]
                score = float(obj.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            x1, x2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
            y1, y2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
            if x2 <= x1 or y2 <= y1:
                continue
            raw_name = str(obj.get("category", "unknown")).strip()
            class_name = self._canonical(raw_name)
            output.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "class": self._class_id(class_name),
                    "class_name": class_name,
                    "raw_class_name": raw_name,
                    "score": score,
                    "score_kind": "native",
                    "association_weight": 1.0,
                    "source": "dinox",
                }
            )
        return output[: self.max_detections_per_frame]

    def detect_batch(self, frames_bgr: Sequence[np.ndarray]) -> List[List[Dict[str, Any]]]:
        if self._client is None or self._task_type is None or self._image_to_base64 is None:
            raise RuntimeError("DinoXDetector is not loaded; call load_model() first")
        output: List[List[Dict[str, Any]]] = []
        prompt = " . ".join(self.categories)
        for frame in frames_bgr:
            if frame is None:
                raise ValueError("detect_batch received an empty frame")
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok:
                raise RuntimeError("failed to encode frame for DINO-X")
            task = self._task_type(
                api_path="/v2/task/dinox/detection",
                api_body={
                    "model": self.model,
                    "image": self._image_to_base64(encoded.tobytes()),
                    "prompt": {"type": "text", "text": prompt},
                    "targets": ["bbox"],
                    "bbox_threshold": self.bbox_threshold,
                    "iou_threshold": self.iou_threshold,
                },
            )
            self._client.run_task(task)
            parsed = self._parse(task.result, width=int(frame.shape[1]), height=int(frame.shape[0]))
            output.append(parsed)
            self._calls += 1
            self._detected_objects_total += len(parsed)
        return output

    def get_stats(self) -> Dict[str, Any]:
        return {
            "backend": "dinox",
            "model": self.model,
            "api_path": "/v2/task/dinox/detection",
            "api_key_env": self.api_key_env,
            "categories": list(self.categories),
            "bbox_threshold": self.bbox_threshold,
            "iou_threshold": self.iou_threshold,
            "calls": self._calls,
            "detected_objects_total": self._detected_objects_total,
        }
