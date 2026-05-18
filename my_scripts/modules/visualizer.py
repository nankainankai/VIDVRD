"""可视化模块（兼容恢复版）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np


class VideoVisualizer:
    """最小可用可视化器：绘制检测框、轨迹框和轨迹历史。"""

    def __init__(self) -> None:
        pass

    def draw_tracks(self, frame: np.ndarray, tracks: List[Dict[str, Any]]) -> np.ndarray:
        out = frame.copy()
        for t in tracks or []:
            bbox = t.get("bbox")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            tid = int(t.get("track_id", -1))
            cls = str(t.get("class_name", "unknown"))
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(out, f"ID {tid} {cls}", (x1 + 4, max(20, y1 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def draw_objects(self, frame: np.ndarray, objects: List[Dict[str, Any]]) -> np.ndarray:
        out = frame.copy()
        for o in objects or []:
            bbox = o.get("bbox")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cls = str(o.get("class_name", "unknown"))
            conf = float(o.get("confidence", 0.0))
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 180, 255), 1)
            cv2.putText(out, f"{cls} {conf:.2f}", (x1 + 4, max(20, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def draw_trajectory(self, frame: np.ndarray, history_dict: Optional[Dict[int, List[Any]]]) -> np.ndarray:
        out = frame.copy()
        if not history_dict:
            return out
        for pts in history_dict.values():
            if not pts:
                continue
            for i in range(1, len(pts)):
                p0 = pts[i - 1]
                p1 = pts[i]
                cv2.line(out, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (255, 128, 0), 2)
        return out

    def draw_comprehensive(
        self,
        frame: np.ndarray,
        objects: List[Dict[str, Any]],
        tracks: List[Dict[str, Any]],
        history_dict: Optional[Dict[int, List[Any]]] = None,
    ) -> np.ndarray:
        out = self.draw_objects(frame, objects)
        out = self.draw_tracks(out, tracks)
        out = self.draw_trajectory(out, history_dict)
        return out
