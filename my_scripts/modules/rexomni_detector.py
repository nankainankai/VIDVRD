"""Rex-Omni local detector adapter.

Adapter target: same output schema as DINO-X detector used by this repo:
[{"bbox": [x1,y1,x2,y2], "class": int, "class_name": str, "confidence": float}, ...]

This adapter is optional. It tries to import Rex-Omni from the vendored folder
`Rex-Omni-master/Rex-Omni-master/` (if present in the workspace) or from the
current Python environment if installed as a package.

Notes:
- Rex-Omni wrapper provides category -> list[{type:'box', coords:[x0,y0,x1,y1]}].
- Rex-Omni does not output confidence for boxes; we set confidence=1.0.
- Supports detection_interval (skip frames) similar to DINO-X for speed.

"""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _maybe_add_vendored_rexomni_to_syspath() -> Optional[Path]:
    """If Rex-Omni is vendored in the repo, add it to sys.path."""

    here = Path(__file__).resolve()
    for base in here.parents:
        candidate = base / "Rex-Omni-master" / "Rex-Omni-master"
        if candidate.exists() and (candidate / "rex_omni").exists():
            sys.path.insert(0, str(candidate))
            return candidate
    return None


def _split_categories(categories: Sequence[str] | str) -> List[str]:
    if isinstance(categories, str):
        parts = [p.strip() for p in categories.split(",")]
        return [p for p in parts if p]
    out: List[str] = []
    for c in categories:
        s = str(c).strip()
        if s:
            out.append(s)
    return out


def _bbox_area_xyxy(b: Sequence[float]) -> float:
    if len(b) != 4:
        return 0.0
    w = max(0.0, float(b[2]) - float(b[0]))
    h = max(0.0, float(b[3]) - float(b[1]))
    return float(w * h)


@dataclass
class _RexOmniDeps:
    RexOmniWrapper: Any
    TaskType: Any
    Image: Any


def _load_rexomni_deps() -> _RexOmniDeps:
    _maybe_add_vendored_rexomni_to_syspath()

    try:
        rex_omni = importlib.import_module("rex_omni")
        RexOmniWrapper = getattr(rex_omni, "RexOmniWrapper")
        TaskType = getattr(rex_omni, "TaskType")
    except Exception as e:
        raise ImportError(
            "Rex-Omni 未安装/不可用。\n"
            "- 如果你想用仓库内置版本：确保存在目录 Rex-Omni-master/Rex-Omni-master/rex_omni\n"
            "- 或者 pip 安装 Rex-Omni 相关依赖并保证 `import rex_omni` 可用\n"
            f"原始错误: {e}"
        )

    try:
        pil = importlib.import_module("PIL")
        Image = getattr(importlib.import_module("PIL.Image"), "Image")
    except Exception as e:
        raise ImportError(f"Pillow 不可用（需要 PIL.Image）：{e}")

    return _RexOmniDeps(RexOmniWrapper=RexOmniWrapper, TaskType=TaskType, Image=Image)


def _pick_attn_implementation(requested: Optional[str] = None) -> str:
    """Pick a safe attention implementation.

    Rex-Omni upstream defaults to FlashAttention2, but `flash_attn` is typically
    unavailable on Windows. Fall back to SDPA when FA2 isn't available.
    """

    if requested and str(requested).strip():
        return str(requested).strip()

    try:
        from transformers.utils import is_flash_attn_2_available  # type: ignore

        if is_flash_attn_2_available():
            return "flash_attention_2"
    except Exception:
        pass

    return "sdpa"


def _pick_torch_dtype(requested: Any = None) -> Any:
    """Choose a default dtype that works on CPU-only machines."""

    if requested is not None:
        return requested
    try:
        import torch  # type: ignore

        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32
    except Exception:
        return None


class RexOmniObjectDetector:
    """Local Rex-Omni detector."""

    def __init__(
        self,
        model_path: str,
        backend: str = "transformers",
        categories: Sequence[str] | str = ("person",),
        detection_interval: int = 1,
        min_box_area: float = 400.0,
        max_detections_per_frame: int = 80,
        **kwargs,
    ) -> None:
        self.model_path = str(model_path)
        self.backend = str(backend or "transformers").strip().lower()
        self.categories = _split_categories(categories)
        self.detection_interval = max(1, int(detection_interval))
        self.min_box_area = float(min_box_area)
        self.max_detections_per_frame = max(1, int(max_detections_per_frame))
        self.kwargs = dict(kwargs)

        self._frame_calls = 0
        self._trigger_calls = 0
        self._interval_skips = 0
        self._detected_objects_total = 0

        self._deps: Optional[_RexOmniDeps] = None
        self._model: Any = None
        self._class_name_to_id: Dict[str, int] = {}

    def load_model(self) -> None:
        deps = _load_rexomni_deps()
        self._deps = deps

        attn_impl = _pick_attn_implementation(self.kwargs.get("attn_implementation"))
        torch_dtype = _pick_torch_dtype(self.kwargs.get("torch_dtype"))

        # RexOmniWrapper init loads model.
        self._model = deps.RexOmniWrapper(
            model_path=self.model_path,
            backend=self.backend,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
            temperature=float(self.kwargs.get("temperature", 0.0)),
            top_p=float(self.kwargs.get("top_p", 0.8)),
            top_k=int(self.kwargs.get("top_k", 1)),
            repetition_penalty=float(self.kwargs.get("repetition_penalty", 1.05)),
            max_tokens=int(self.kwargs.get("max_tokens", 2048)),
        )

        print(
            f"[RexOmni] initialized (backend={self.backend}, attn={attn_impl}, dtype={torch_dtype}, categories={self.categories})"
        )

    def _class_id(self, name: str) -> int:
        n = (name or "unknown").strip().lower()
        if n not in self._class_name_to_id:
            self._class_name_to_id[n] = len(self._class_name_to_id)
        return self._class_name_to_id[n]

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        self._frame_calls += 1
        if (self._frame_calls - 1) % self.detection_interval != 0:
            self._interval_skips += 1
            return []

        if self._deps is None or self._model is None:
            raise RuntimeError("RexOmniObjectDetector not loaded; call load_model()")
        if frame_bgr is None:
            return []

        # BGR -> RGB PIL
        try:
            import cv2  # type: ignore

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            # fallback: assume input is BGR, swap channels
            rgb = frame_bgr[:, :, ::-1]

        from PIL import Image  # type: ignore

        image = Image.fromarray(rgb.astype(np.uint8))

        cats = self.categories if self.categories else ["person"]
        results = self._model.inference(images=image, task="detection", categories=cats)
        if not results or not isinstance(results, list):
            return []

        r0 = results[0]
        preds = r0.get("extracted_predictions", {}) if isinstance(r0, dict) else {}
        if not isinstance(preds, dict):
            return []

        out: List[Dict[str, Any]] = []
        for cls_name, anns in preds.items():
            if not isinstance(anns, list):
                continue
            for a in anns:
                if not isinstance(a, dict):
                    continue
                if str(a.get("type", "")) != "box":
                    continue
                coords = a.get("coords")
                if not (isinstance(coords, list) and len(coords) == 4):
                    continue
                try:
                    b = [float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])]
                    if not all(math.isfinite(x) for x in b):
                        continue
                except Exception:
                    continue

                if _bbox_area_xyxy(b) < self.min_box_area:
                    continue

                out.append(
                    {
                        "bbox": b,
                        "class": self._class_id(str(cls_name)),
                        "class_name": str(cls_name),
                        "confidence": 1.0,
                    }
                )

        # keep top-K
        if len(out) > self.max_detections_per_frame:
            out = out[: self.max_detections_per_frame]

        self._trigger_calls += 1
        self._detected_objects_total += len(out)
        return out

    def get_stats(self) -> Dict[str, Any]:
        return {
            "backend": "rexomni",
            "model_path": self.model_path,
            "engine": self.backend,
            "categories": list(self.categories),
            "detection_interval": int(self.detection_interval),
            "frame_calls": int(self._frame_calls),
            "trigger_calls": int(self._trigger_calls),
            "interval_skips": int(self._interval_skips),
            "detected_objects_total": int(self._detected_objects_total),
        }
