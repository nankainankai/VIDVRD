"""目标追踪模块 - OC-SORT（实现版）

说明：
- 这里实现的是“OC-SORT 风格”的追踪：
  - 两阶段关联：优先匹配活跃轨迹；再匹配短暂丢失轨迹（用 last observation 更稳）
  - 匈牙利匹配：scipy.optimize.linear_sum_assignment
  - 简易运动模型：基于中心点速度的 bbox 预测

输出兼容下游：
[{'track_id', 'bbox', 'class_name', 'confidence', 'age', 'hits', 'history', ...}, ...]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment

    HAS_SCIPY = True
except Exception as e:  # pragma: no cover
    HAS_SCIPY = False
    _SCIPY_ERR = e


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: (4,) xyxy
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _center_xyxy(b: np.ndarray) -> Tuple[float, float]:
    return (float(b[0] + b[2]) / 2.0, float(b[1] + b[3]) / 2.0)


def _diag_xyxy(b: np.ndarray) -> float:
    w = max(1e-3, float(b[2] - b[0]))
    h = max(1e-3, float(b[3] - b[1]))
    return float(math.sqrt(w * w + h * h))


def _clip_bbox(b: np.ndarray) -> np.ndarray:
    # just ensure ordering; no image-size clipping here
    x1, y1, x2, y2 = [float(v) for v in b.tolist()]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _clip_bbox_to_image(b: np.ndarray, w: int, h: int) -> np.ndarray:
    b = _clip_bbox(b)
    if w <= 1 or h <= 1:
        return b
    x1, y1, x2, y2 = [float(v) for v in b.tolist()]
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))
    # ensure non-degenerate
    if x2 <= x1:
        x2 = min(float(w - 1), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(h - 1), y1 + 1.0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


@dataclass
class _Track:
    track_id: int
    bbox: np.ndarray
    class_name: str = "unknown"
    confidence: float = 0.0

    hits: int = 1
    age: int = 0  # unmatched frames since last update

    # observation-centric: keep last observed bbox
    last_observation: np.ndarray = field(default_factory=lambda: np.zeros((4,), dtype=np.float32))
    last_observed_frame: int = 0

    # velocity (center delta per frame)
    vx: float = 0.0
    vy: float = 0.0

    start_frame: int = 0
    end_frame: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def predict(self) -> np.ndarray:
        # Simple constant-velocity prediction in center domain
        x1, y1, x2, y2 = self.bbox.tolist()
        cx, cy = _center_xyxy(self.bbox)
        w = max(1e-3, float(x2 - x1))
        h = max(1e-3, float(y2 - y1))
        cx_p = cx + float(self.vx)
        cy_p = cy + float(self.vy)
        pred = np.array([cx_p - w / 2.0, cy_p - h / 2.0, cx_p + w / 2.0, cy_p + h / 2.0], dtype=np.float32)
        return _clip_bbox(pred)

    def update_velocity(self, new_bbox: np.ndarray, dt_frames: int, alpha: float = 1.0) -> None:
        cx0, cy0 = _center_xyxy(self.bbox)
        cx1, cy1 = _center_xyxy(new_bbox)
        dt = max(1, int(dt_frames))
        vx_new = float((cx1 - cx0) / float(dt))
        vy_new = float((cy1 - cy0) / float(dt))
        a = max(0.0, min(1.0, float(alpha)))
        self.vx = float((1.0 - a) * self.vx + a * vx_new)
        self.vy = float((1.0 - a) * self.vy + a * vy_new)


class ObjectTracker:
    """OC-SORT 风格追踪器（两阶段关联 + 匈牙利匹配）。"""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        class_aware: bool = True,
        max_center_dist_ratio: float = 0.8,
        min_new_track_conf: float = 0.35,
        velocity_alpha: float = 0.8,
    ) -> None:
        if not HAS_SCIPY:
            raise ImportError(f"需要 scipy（用于匈牙利匹配）：{_SCIPY_ERR}")

        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.class_aware = bool(class_aware)
        self.max_center_dist_ratio = float(max_center_dist_ratio)
        self.min_new_track_conf = float(min_new_track_conf)
        self.velocity_alpha = float(velocity_alpha)

        self._tracks: Dict[int, _Track] = {}
        self._next_id = 0

        print(
            "[Tracker] Initialized (OC-SORT, "
            f"iou={self.iou_threshold}, max_age={self.max_age}, min_hits={self.min_hits}, "
            f"class_aware={self.class_aware})"
        )

    def reset(self) -> None:
        self._tracks = {}
        self._next_id = 0
        print("[Tracker] Reset")

    def _associate(
        self,
        track_ids: List[int],
        track_boxes: List[np.ndarray],
        track_classes: List[str],
        det_boxes: List[np.ndarray],
        det_classes: List[str],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not track_ids or not det_boxes:
            return [], list(range(len(track_ids))), list(range(len(det_boxes)))

        cost = np.ones((len(track_ids), len(det_boxes)), dtype=np.float32)
        iou_mat = np.zeros((len(track_ids), len(det_boxes)), dtype=np.float32)
        norm_dist_mat = np.ones((len(track_ids), len(det_boxes)), dtype=np.float32)
        for i, tb in enumerate(track_boxes):
            tcx, tcy = _center_xyxy(tb)
            tdiag = _diag_xyxy(tb)
            for j, db in enumerate(det_boxes):
                if self.class_aware:
                    tc = str(track_classes[i]).strip().lower()
                    dc = str(det_classes[j]).strip().lower()
                    if tc and dc and tc != dc:
                        # 不同类别直接设置为高代价，减少 ID 串类。
                        cost[i, j] = 10.0
                        iou_mat[i, j] = 0.0
                        norm_dist_mat[i, j] = 99.0
                        continue

                iou = float(_iou_xyxy(tb, db))
                dcx, dcy = _center_xyxy(db)
                center_dist = math.sqrt((tcx - dcx) * (tcx - dcx) + (tcy - dcy) * (tcy - dcy))
                norm_dist = float(center_dist / max(1e-3, tdiag))

                iou_mat[i, j] = iou
                norm_dist_mat[i, j] = norm_dist
                # IoU 主导，中心距离作为辅助，降低稀疏检测场景下的 ID 抖动。
                cost[i, j] = (1.0 - iou) + 0.2 * min(norm_dist, 2.0)

        row_ind, col_ind = linear_sum_assignment(cost)

        matches: List[Tuple[int, int]] = []
        unmatched_tracks = set(range(len(track_ids)))
        unmatched_dets = set(range(len(det_boxes)))

        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            iou = float(iou_mat[r, c])
            norm_dist = float(norm_dist_mat[r, c])
            # 允许“低 IoU 但中心接近”的匹配，适配 interval>1 的场景。
            ok_iou = iou >= self.iou_threshold
            ok_dist_fallback = (iou >= 0.10) and (norm_dist <= self.max_center_dist_ratio)
            if ok_iou or ok_dist_fallback:
                matches.append((r, c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)

        return matches, sorted(list(unmatched_tracks)), sorted(list(unmatched_dets))

    def track(self, frame: np.ndarray, detections: List[Dict[str, Any]], frame_num: int = 0) -> List[Dict[str, Any]]:
        h, w = frame.shape[:2]
        # Prepare detections
        det_boxes: List[np.ndarray] = []
        det_meta: List[Dict[str, Any]] = []
        det_classes: List[str] = []
        for d in detections or []:
            bbox = d.get("bbox")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            db = _clip_bbox_to_image(
                np.array([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])], dtype=np.float32),
                w=w,
                h=h,
            )
            det_boxes.append(db)
            det_meta.append(d)
            det_classes.append(str(d.get("class_name", "unknown")))

        # Split tracks into active and lost (observation-centric re-association)
        active_ids: List[int] = []
        active_boxes: List[np.ndarray] = []
        active_classes: List[str] = []
        lost_ids: List[int] = []
        lost_boxes: List[np.ndarray] = []
        lost_classes: List[str] = []

        for tid, tr in self._tracks.items():
            if tr.age == 0:
                active_ids.append(tid)
                active_boxes.append(tr.predict())
                active_classes.append(str(tr.class_name))
            else:
                # use last observation instead of prediction for re-association (OC idea)
                lost_ids.append(tid)
                lost_boxes.append(_clip_bbox(tr.last_observation))
                lost_classes.append(str(tr.class_name))

        # Stage 1: match active tracks
        matches1, un_active, un_det = self._associate(active_ids, active_boxes, active_classes, det_boxes, det_classes)

        matched_track_ids = set()
        matched_det_ids = set()

        for r, c in matches1:
            tid = active_ids[r]
            det = det_meta[c]
            new_bbox = det_boxes[c]
            tr = self._tracks[tid]

            dt = max(1, int(frame_num) - int(getattr(tr, "last_observed_frame", frame_num)))
            tr.update_velocity(new_bbox, dt_frames=dt, alpha=self.velocity_alpha)
            tr.bbox = _clip_bbox_to_image(new_bbox, w=w, h=h)
            tr.last_observation = new_bbox.copy()
            tr.last_observed_frame = int(frame_num)
            tr.class_name = str(det.get("class_name", tr.class_name))
            det_conf = float(det.get("confidence", tr.confidence))
            tr.confidence = float(0.7 * tr.confidence + 0.3 * det_conf)
            tr.hits += 1
            tr.age = 0
            tr.end_frame = frame_num

            cx, cy = _center_xyxy(new_bbox)
            tr.history.append({"frame": frame_num, "bbox": new_bbox.tolist(), "center_x": cx, "center_y": cy})
            if len(tr.history) > 30:
                tr.history = tr.history[-30:]

            matched_track_ids.add(tid)
            matched_det_ids.add(c)

        # Stage 2: match lost tracks with remaining detections
        rem_det_boxes = [det_boxes[i] for i in un_det]
        rem_det_meta = [det_meta[i] for i in un_det]
        rem_det_classes = [det_classes[i] for i in un_det]

        matches2: List[Tuple[int, int]] = []
        un_lost = list(range(len(lost_ids)))
        un_det2 = list(range(len(rem_det_boxes)))

        if lost_ids and rem_det_boxes:
            matches2, un_lost, un_det2 = self._associate(lost_ids, lost_boxes, lost_classes, rem_det_boxes, rem_det_classes)

        for r, c in matches2:
            tid = lost_ids[r]
            det = rem_det_meta[c]
            new_bbox = rem_det_boxes[c]
            tr = self._tracks[tid]

            dt = max(1, int(frame_num) - int(getattr(tr, "last_observed_frame", frame_num)))
            tr.update_velocity(new_bbox, dt_frames=dt, alpha=self.velocity_alpha)
            tr.bbox = _clip_bbox_to_image(new_bbox, w=w, h=h)
            tr.last_observation = new_bbox.copy()
            tr.last_observed_frame = int(frame_num)
            tr.class_name = str(det.get("class_name", tr.class_name))
            det_conf = float(det.get("confidence", tr.confidence))
            tr.confidence = float(0.7 * tr.confidence + 0.3 * det_conf)
            tr.hits += 1
            tr.age = 0
            tr.end_frame = frame_num

            cx, cy = _center_xyxy(new_bbox)
            tr.history.append({"frame": frame_num, "bbox": new_bbox.tolist(), "center_x": cx, "center_y": cy})
            if len(tr.history) > 30:
                tr.history = tr.history[-30:]

            matched_track_ids.add(tid)

        # Age unmatched tracks
        for tid, tr in list(self._tracks.items()):
            if tid in matched_track_ids:
                continue
            tr.age += 1
            tr.end_frame = frame_num
            # Keep bbox as prediction to maintain smoother visualization.
            # NOTE: prediction is based on per-frame velocity; clip to image to avoid runaway values.
            tr.bbox = _clip_bbox_to_image(tr.predict(), w=w, h=h)

            if tr.age > self.max_age:
                # delete expired
                del self._tracks[tid]

        # Create new tracks for unmatched detections (remaining after stage2)
        # unmatched detections are those not matched in stage1 among det indices, plus stage2 leftover
        # Stage2 worked on rem_det list; leftovers are un_det2 indices in rem_det
        new_det_indices = []
        if rem_det_boxes:
            for idx in un_det2:
                # map back to original det index
                new_det_indices.append(un_det[idx])
        else:
            new_det_indices = un_det

        for det_idx in new_det_indices:
            det = det_meta[det_idx]
            new_bbox = det_boxes[det_idx]
            det_conf = float(det.get("confidence", 0.0))
            if det_conf < self.min_new_track_conf:
                continue

            tid = self._next_id
            self._next_id += 1

            tr = _Track(
                track_id=tid,
                bbox=new_bbox.copy(),
                class_name=str(det.get("class_name", "unknown")),
                confidence=det_conf,
                hits=1,
                age=0,
                last_observation=new_bbox.copy(),
                last_observed_frame=int(frame_num),
                vx=0.0,
                vy=0.0,
                start_frame=frame_num,
                end_frame=frame_num,
                history=[],
            )

            cx, cy = _center_xyxy(new_bbox)
            tr.history.append({"frame": frame_num, "bbox": new_bbox.tolist(), "center_x": cx, "center_y": cy})

            self._tracks[tid] = tr

        # Collect valid tracks
        out: List[Dict[str, Any]] = []
        for tid, tr in self._tracks.items():
            if tr.hits < self.min_hits:
                continue
            if tr.age > self.max_age:
                continue

            # Motion metrics for downstream UI.
            inst_dist = 0.0
            if len(tr.history) >= 2:
                p0 = tr.history[-2]
                p1 = tr.history[-1]
                dx = float(p1.get("center_x", 0.0)) - float(p0.get("center_x", 0.0))
                dy = float(p1.get("center_y", 0.0)) - float(p0.get("center_y", 0.0))
                inst_dist = math.sqrt(dx * dx + dy * dy)

            total_dist = 0.0
            if len(tr.history) >= 2:
                for i in range(1, len(tr.history)):
                    pa = tr.history[i - 1]
                    pb = tr.history[i]
                    dx = float(pb.get("center_x", 0.0)) - float(pa.get("center_x", 0.0))
                    dy = float(pb.get("center_y", 0.0)) - float(pa.get("center_y", 0.0))
                    total_dist += math.sqrt(dx * dx + dy * dy)

            duration_frames = max(1, int(tr.end_frame - tr.start_frame + 1))
            avg_speed = float(total_dist / duration_frames)

            motion_state = "基本静止" if avg_speed < 1.0 else ("缓慢移动" if avg_speed < 5.0 else "快速移动")

            out.append(
                {
                    "track_id": int(tr.track_id),
                    "bbox": [float(x) for x in tr.bbox.tolist()],
                    "bbox_observed": [float(x) for x in tr.last_observation.tolist()],
                    "class_name": tr.class_name,
                    "confidence": float(tr.confidence),
                    "duration_frames": int(duration_frames),
                    "total_distance": float(total_dist),
                    "instant_distance": float(inst_dist),
                    "avg_speed": float(avg_speed),
                    "motion_state": motion_state,
                    "age": int(tr.age),
                    "hits": int(tr.hits),
                    "is_predicted": bool(tr.age > 0),
                    "history": tr.history,
                }
            )

        return out
