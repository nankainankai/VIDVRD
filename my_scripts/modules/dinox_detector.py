"""DINO-X 云端检测器（dds-cloudapi-sdk 适配 my_scripts 工作流）

输出格式对齐 ObjectDetector.detect():
[{bbox:[x1,y1,x2,y2], class:int, class_name:str, confidence:float}, ...]

说明：
- 该检测器会把输入帧编码为 base64 JPEG，调用 /v2/task/dinox/detection。
- 支持对长边做等比缩放以减少带宽/成本，并把 bbox 缩放回原图坐标。
- 支持 detection_interval：非触发帧返回空列表（让 tracker 自行 age）。
"""

from __future__ import annotations

import base64
import importlib
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

HAS_DINOX_SDK = False
_IMPORT_ERR: Optional[Exception] = None


def _load_dinox_sdk() -> Tuple[Any, Any, Any]:
    """动态导入 dds-cloudapi-sdk，避免在未安装环境下模块导入直接失败。"""
    global HAS_DINOX_SDK, _IMPORT_ERR
    try:
        sdk = importlib.import_module("dds_cloudapi_sdk")
        tasks = importlib.import_module("dds_cloudapi_sdk.tasks.v2_task")
        Config = getattr(sdk, "Config")
        Client = getattr(sdk, "Client")
        V2Task = getattr(tasks, "V2Task")
        HAS_DINOX_SDK = True
        _IMPORT_ERR = None
        return Client, Config, V2Task
    except Exception as e:  # pragma: no cover
        HAS_DINOX_SDK = False
        _IMPORT_ERR = e
        raise


def _encode_jpeg_b64(img_bgr: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    # dds-cloudapi-sdk v2 expects data URL format instead of raw base64.
    raw = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{raw}"


def _resize_long_edge(img_bgr: np.ndarray, max_long_edge: int) -> Tuple[np.ndarray, float, float]:
    if max_long_edge <= 0:
        return img_bgr, 1.0, 1.0
    h, w = img_bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return img_bgr, 1.0, 1.0

    scale = float(max_long_edge) / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    sx = float(w) / float(new_w)
    sy = float(h) / float(new_h)
    return resized, sx, sy


def _bbox_area_xyxy(b: List[float]) -> float:
    w = max(0.0, float(b[2]) - float(b[0]))
    h = max(0.0, float(b[3]) - float(b[1]))
    return float(w * h)


def _nms_xyxy(dets: List[Dict[str, Any]], iou_thr: float) -> List[Dict[str, Any]]:
    if not dets:
        return []
    if iou_thr <= 0.0:
        return dets

    boxes = np.array([d["bbox"] for d in dets], dtype=np.float32)
    scores = np.array([float(d.get("confidence", 0.0)) for d in dets], dtype=np.float32)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        area_i = np.maximum(1e-6, (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]))
        area_j = np.maximum(1e-6, (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1]))
        iou = inter / (area_i + area_j - inter + 1e-6)

        remain = np.where(iou <= float(iou_thr))[0]
        order = order[remain + 1]

    return [dets[i] for i in keep]


class DINOXObjectDetector:
    """DINO-X V2 detection adapter."""

    def __init__(
        self,
        api_token: str,
        text_prompt: str,
        model: str = "DINO-X-1.0",
        bbox_threshold: float = 0.25,
        iou_threshold: float = 0.8,
        targets: Optional[List[str]] = None,
        mask_format: str = "coco_rle",
        jpeg_quality: int = 90,
        image_max_long_edge: int = 960,
        detection_interval: int = 1,
        max_calls: int = 0,
        nms_iou_threshold: float = 0.55,
        min_box_area: float = 400.0,
        max_detections_per_frame: int = 80,
        request_retries: int = 1,
        request_backoff_sec: float = 1.2,
    ) -> None:
        try:
            Client, Config, _V2Task = _load_dinox_sdk()
        except Exception as e:
            raise ImportError(
                f"dds-cloudapi-sdk 未安装或不可用：{e}. "
                "请在当前环境安装：pip install dds-cloudapi-sdk==0.5.3"
            )

        api_token = (api_token or "").strip()
        if not api_token:
            raise ValueError("DINO-X API token is empty (set config.DINOX_API_TOKEN)")

        self.api_token = api_token
        self.text_prompt = (text_prompt or "").strip()
        if not self.text_prompt:
            raise ValueError("DINO-X text prompt is empty (set config.DINOX_TEXT_PROMPT)")

        self.model = model
        self.bbox_threshold = float(bbox_threshold)
        self.iou_threshold = float(iou_threshold)
        self.targets = targets or ["bbox"]
        self.mask_format = mask_format
        self.jpeg_quality = int(jpeg_quality)
        self.image_max_long_edge = int(image_max_long_edge)
        self.detection_interval = max(1, int(detection_interval))
        self.max_calls = max(0, int(max_calls))
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.min_box_area = float(min_box_area)
        self.max_detections_per_frame = max(1, int(max_detections_per_frame))
        self.request_retries = max(0, int(request_retries))
        self.request_backoff_sec = max(0.0, float(request_backoff_sec))

        self._frame_calls = 0
        self._trigger_calls = 0
        self._interval_skips = 0
        self._max_call_skips = 0
        self._raw_objects_total = 0
        self._detected_objects_total = 0
        self._cap_warned = False
        self._class_name_to_id: Dict[str, int] = {}

        self._V2Task = _V2Task

        cfg = Config(self.api_token)
        self.client = Client(cfg)

        print(
            f"🔧 DINO-X 检测器已初始化（model={self.model}, interval={self.detection_interval}, max_edge={self.image_max_long_edge}）"
        )

    def load_model(self) -> None:
        """保持与 YOLO Detector 一致的调用方式：DINO-X 无需本地加载。"""
        print("✅ DINO-X 使用云端模型：无需本地加载")

    def _class_id(self, name: str) -> int:
        n = (name or "unknown").strip().lower()
        if n not in self._class_name_to_id:
            self._class_name_to_id[n] = len(self._class_name_to_id)
        return self._class_name_to_id[n]

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        self._frame_calls += 1
        if self.max_calls > 0 and self._trigger_calls >= self.max_calls:
            self._max_call_skips += 1
            if not self._cap_warned:
                print(f"⚠️ 已达到 DINO-X 请求上限 max_calls={self.max_calls}，后续帧不再调用云端检测")
                self._cap_warned = True
            return []

        if (self._frame_calls - 1) % self.detection_interval != 0:
            self._interval_skips += 1
            return []

        if frame_bgr is None:
            return []

        resized, sx, sy = _resize_long_edge(frame_bgr, self.image_max_long_edge)
        image_b64 = _encode_jpeg_b64(resized, quality=self.jpeg_quality)

        api_path = "/v2/task/dinox/detection"
        api_body = {
            "model": self.model,
            "image": image_b64,
            "prompt": {"type": "text", "text": self.text_prompt},
            "targets": self.targets,
            "bbox_threshold": self.bbox_threshold,
            "iou_threshold": self.iou_threshold,
        }

        # only include mask settings when requested
        if any(t == "mask" for t in self.targets):
            api_body["mask_format"] = self.mask_format

        task = self._V2Task(api_path=api_path, api_body=api_body)
        self._trigger_calls += 1
        last_err: Optional[Exception] = None
        for attempt in range(self.request_retries + 1):
            try:
                self.client.run_task(task)
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt >= self.request_retries:
                    raise
                time.sleep(self.request_backoff_sec * (2 ** attempt))

        if last_err is not None:
            raise last_err
        result = task.result or {}

        objects = result.get("objects", []) if isinstance(result, dict) else []
        self._raw_objects_total += len(objects or [])
        out: List[Dict[str, Any]] = []

        for obj in objects or []:
            try:
                bbox = obj.get("bbox")
                if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                    continue

                x1, y1, x2, y2 = [float(v) for v in bbox]
                # map back to original resolution if resized
                x1, x2 = x1 * sx, x2 * sx
                y1, y2 = y1 * sy, y2 * sy

                score = float(obj.get("score", 0.0))
                if score < self.bbox_threshold:
                    continue

                cls_name = str(obj.get("category", "unknown")).strip().lower()
                cls_id = self._class_id(cls_name)

                out.append(
                    {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "class": int(cls_id),
                        "class_name": cls_name,
                        "confidence": float(score),
                    }
                )
            except Exception:
                continue

        # Post-processing: remove tiny boxes and class-wise duplicate suppression.
        out = [d for d in out if _bbox_area_xyxy(d["bbox"]) >= self.min_box_area]

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for d in out:
            grouped.setdefault(str(d.get("class_name", "unknown")), []).append(d)

        out_nms: List[Dict[str, Any]] = []
        for _, ds in grouped.items():
            out_nms.extend(_nms_xyxy(ds, iou_thr=self.nms_iou_threshold))

        out_nms.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        out = out_nms[: self.max_detections_per_frame]

        self._detected_objects_total += len(out)

        return out

    def get_stats(self) -> Dict[str, Any]:
        avg_objects = (
            float(self._detected_objects_total) / float(self._trigger_calls)
            if self._trigger_calls > 0
            else 0.0
        )
        return {
            "model": self.model,
            "text_prompt": self.text_prompt,
            "detection_interval": int(self.detection_interval),
            "max_calls": int(self.max_calls),
            "nms_iou_threshold": float(self.nms_iou_threshold),
            "min_box_area": float(self.min_box_area),
            "max_detections_per_frame": int(self.max_detections_per_frame),
            "request_retries": int(self.request_retries),
            "request_backoff_sec": float(self.request_backoff_sec),
            "frame_calls": int(self._frame_calls),
            "trigger_calls": int(self._trigger_calls),
            "interval_skips": int(self._interval_skips),
            "max_call_skips": int(self._max_call_skips),
            "raw_objects_total": int(self._raw_objects_total),
            "detected_objects_total": int(self._detected_objects_total),
            "avg_objects_per_trigger": float(avg_objects),
        }
