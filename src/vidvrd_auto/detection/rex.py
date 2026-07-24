from __future__ import annotations

"""Rex-Omni model adapter with frame-wise, auditable outputs."""

import importlib
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _add_vendored_package() -> Optional[Path]:
    for base in Path(__file__).resolve().parents:
        candidate = base / "Rex-Omni-master" / "Rex-Omni-master"
        if candidate.exists() and (candidate / "rex_omni").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    return None


def _categories(value: Sequence[str] | str) -> List[str]:
    items = value.split(",") if isinstance(value, str) else value
    return [str(item).strip() for item in items if str(item).strip()]


def _box_area(box: Sequence[float]) -> float:
    if len(box) != 4:
        return 0.0
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _attention(requested: Optional[str] = None) -> str:
    if requested and requested.strip():
        return requested.strip()
    try:
        from transformers.utils import is_flash_attn_2_available  # type: ignore

        if is_flash_attn_2_available():
            return "flash_attention_2"
    except Exception:
        pass
    return "sdpa"


def _dtype(requested: Any = None) -> Any:
    if requested is not None:
        return requested
    try:
        import torch  # type: ignore

        return torch.float16 if torch.cuda.is_available() else torch.float32
    except Exception:
        return None


class RexDetector:
    """Thin wrapper around the official Rex-Omni multi-image API."""

    def __init__(
        self,
        model_path: str,
        backend: str = "transformers",
        categories: Sequence[str] | str = ("person",),
        detection_interval: int = 1,
        min_box_area: float = 400.0,
        max_detections_per_frame: int = 80,
        **kwargs: Any,
    ) -> None:
        self.model_path = str(model_path)
        self.backend = str(backend or "transformers").strip().lower()
        self.categories = _categories(categories)
        self.detection_interval = max(1, int(detection_interval))
        self.min_box_area = float(min_box_area)
        self.max_detections_per_frame = max(1, int(max_detections_per_frame))
        self.kwargs = dict(kwargs)
        self.category_aliases = {
            str(key).strip().lower(): str(value).strip().lower()
            for key, value in dict(self.kwargs.get("category_aliases", {})).items()
        }
        self._frame_calls = 0
        self._batch_calls = 0
        self._trigger_calls = 0
        self._interval_skips = 0
        self._detected_objects_total = 0
        self._model: Any = None
        self._class_ids: Dict[str, int] = {}

    def load_model(self) -> None:
        _add_vendored_package()
        try:
            rex_omni = importlib.import_module("rex_omni")
            wrapper = getattr(rex_omni, "RexOmniWrapper")
        except Exception as exc:
            raise ImportError(
                "Rex-Omni 不可用；请安装官方 rex_omni 包，或放置仓库内置 Rex-Omni checkout。"
            ) from exc

        attention = _attention(self.kwargs.get("attn_implementation"))
        torch_dtype = _dtype(self.kwargs.get("torch_dtype"))
        self._model = wrapper(
            model_path=self.model_path,
            backend=self.backend,
            torch_dtype=torch_dtype,
            attn_implementation=attention,
            temperature=float(self.kwargs.get("temperature", 1.0)),
            top_p=float(self.kwargs.get("top_p", 1.0)),
            top_k=int(self.kwargs.get("top_k", 50)),
            repetition_penalty=float(self.kwargs.get("repetition_penalty", 1.05)),
            max_tokens=int(self.kwargs.get("max_tokens", 2048)),
            max_pixels=int(self.kwargs.get("max_pixels", 640 * 640)),
        )
        print(
            f"[RexOmni] initialized (backend={self.backend}, attn={attention}, "
            f"dtype={torch_dtype}, categories={self.categories})"
        )

    def _class_id(self, name: str) -> int:
        key = (name or "unknown").strip().lower()
        if key not in self._class_ids:
            self._class_ids[key] = len(self._class_ids)
        return self._class_ids[key]

    @staticmethod
    def _to_image(frame_bgr: np.ndarray) -> Any:
        try:
            import cv2  # type: ignore

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = frame_bgr[:, :, ::-1]
        from PIL import Image  # type: ignore

        return Image.fromarray(rgb.astype(np.uint8))

    def _parse(self, result: Any) -> List[Dict[str, Any]]:
        predictions = result.get("extracted_predictions", {}) if isinstance(result, dict) else {}
        if not isinstance(predictions, dict):
            return []
        output: List[Dict[str, Any]] = []
        for class_name, annotations in predictions.items():
            if not isinstance(annotations, list):
                continue
            for annotation in annotations:
                if not isinstance(annotation, dict):
                    continue
                coords = annotation.get("coords")
                if annotation.get("type") != "box" or not isinstance(coords, list) or len(coords) != 4:
                    continue
                try:
                    box = [float(value) for value in coords]
                except (TypeError, ValueError):
                    continue
                if not all(math.isfinite(value) for value in box) or _box_area(box) < self.min_box_area:
                    continue
                raw_name = str(class_name).strip()
                canonical_name = self.category_aliases.get(raw_name.lower(), raw_name.lower().replace(" ", "_"))
                output.append(
                    {
                        "bbox": box,
                        "class": self._class_id(canonical_name),
                        "class_name": canonical_name,
                        "raw_class_name": raw_name,
                        "confidence": 1.0,
                        "source": "rexomni",
                    }
                )
        return output[: self.max_detections_per_frame]

    def detect_batch(self, frames_bgr: Sequence[np.ndarray]) -> List[List[Dict[str, Any]]]:
        frames = list(frames_bgr)
        if not frames:
            return []
        if self._model is None:
            raise RuntimeError("RexDetector is not loaded; call load_model() first")
        if any(frame is None for frame in frames):
            raise ValueError("detect_batch received an empty frame")
        self._frame_calls += len(frames)
        self._batch_calls += 1
        results = self._model.inference(
            images=[self._to_image(frame) for frame in frames],
            task="detection",
            categories=self.categories or ["person"],
        )
        if not isinstance(results, list) or len(results) != len(frames):
            count = len(results) if isinstance(results, list) else "non-list"
            raise RuntimeError(f"Rex-Omni returned {count} results for {len(frames)} frames")
        parsed = [self._parse(result) for result in results]
        self._trigger_calls += len(frames)
        self._detected_objects_total += sum(len(items) for items in parsed)
        return parsed

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        next_call = self._frame_calls + 1
        if (next_call - 1) % self.detection_interval != 0:
            self._frame_calls += 1
            self._interval_skips += 1
            return []
        return self.detect_batch([frame_bgr])[0]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "backend": "rexomni",
            "model_path": self.model_path,
            "engine": self.backend,
            "categories": list(self.categories),
            "category_aliases": dict(self.category_aliases),
            "detection_interval": self.detection_interval,
            "frame_calls": self._frame_calls,
            "batch_calls": self._batch_calls,
            "trigger_calls": self._trigger_calls,
            "interval_skips": self._interval_skips,
            "detected_objects_total": self._detected_objects_total,
        }
