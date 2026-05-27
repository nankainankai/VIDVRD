from __future__ import annotations

"""无模型追踪：由 mock 检测 JSONL 生成稳定 track_id 与 windows。"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("mock tracking requires opencv-python (pip install opencv-python)") from e


def _load_detections(path: Path) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            if not isinstance(row, dict):
                continue
            frame = int(row.get("frame", 0))
            objs = row.get("objects", [])
            out[frame] = [o for o in objs if isinstance(o, dict)] if isinstance(objs, list) else []
    return out


def _bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _assign_track_ids(detections: Dict[int, List[Dict[str, Any]]]) -> Dict[int, List[Dict[str, Any]]]:
    """按 bbox IoU 在相邻帧关联，产出稳定 track_id。"""
    next_id = 1
    active: Dict[int, Dict[str, Any]] = {}
    frame_tracks: Dict[int, List[Dict[str, Any]]] = {}

    for frame in sorted(detections.keys()):
        objs = detections[frame]
        used_active: set[int] = set()
        tracks: List[Dict[str, Any]] = []

        for obj in objs:
            bbox = obj.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            best_tid = -1
            best_iou = 0.0
            for tid, prev in active.items():
                if tid in used_active:
                    continue
                iou = _bbox_iou(bbox, prev["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid >= 0 and best_iou >= 0.3:
                tid = best_tid
            else:
                tid = next_id
                next_id += 1
            used_active.add(tid)
            tracks.append(
                {
                    "track_id": int(tid),
                    "bbox": [float(x) for x in bbox],
                    "class_name": str(obj.get("class_name", "person")),
                    "confidence": float(obj.get("confidence", 0.9)),
                }
            )

        active = {int(t["track_id"]): t for t in tracks}
        frame_tracks[frame] = tracks

    return frame_tracks


def _window_track_ids(
    frame_tracks: Dict[int, List[Dict[str, Any]]], *, start_frame: int, end_frame: int
) -> List[int]:
    ids: set[int] = set()
    for f in range(int(start_frame), int(end_frame) + 1):
        for t in frame_tracks.get(f, []):
            ids.add(int(t["track_id"]))
    return sorted(ids)


def run_mock_track(
    *,
    video_path: Path,
    detections_jsonl: Path,
    out_dir: Path,
    config: Dict[str, Any],
    log_path: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = out_dir / "tracks_full.jsonl"
    windows_path = out_dir / "windows.json"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 10
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()

    detections = _load_detections(detections_jsonl)
    if not detections:
        raise RuntimeError(f"no detections in {detections_jsonl}")

    max_det_frame = max(detections.keys())
    total_frames = max(total_frames, max_det_frame + 1)
    frame_tracks = _assign_track_ids(detections)

    with tracks_path.open("w", encoding="utf-8") as f:
        for frame in sorted(frame_tracks.keys()):
            ts = float(frame / max(1, fps))
            f.write(
                json.dumps({"frame": int(frame), "timestamp": float(ts), "tracks": frame_tracks[frame]}, ensure_ascii=False)
                + "\n"
            )

    window_size = int(config.get("window_size", 30))
    stride = int(config.get("stride", 15))
    windows: List[Dict[str, Any]] = []
    frame_count = int(total_frames)
    if window_size > 0 and stride > 0 and frame_count >= window_size:
        wid = 0
        for start in range(0, frame_count - window_size + 1, stride):
            end = start + window_size - 1
            wid += 1
            windows.append(
                {
                    "window_id": int(wid),
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "start_time": float(start / max(1, fps)),
                    "end_time": float(end / max(1, fps)),
                    "frame_count": int(window_size),
                    "track_ids": _window_track_ids(frame_tracks, start_frame=start, end_frame=end),
                }
            )
    elif frame_count > 0:
        windows.append(
            {
                "window_id": 1,
                "start_frame": 0,
                "end_frame": int(frame_count - 1),
                "start_time": 0.0,
                "end_time": float((frame_count - 1) / max(1, fps)),
                "frame_count": int(frame_count),
                "track_ids": _window_track_ids(frame_tracks, start_frame=0, end_frame=frame_count - 1),
            }
        )

    duration = float(frame_count / max(1, fps))
    windows_obj = {
        "video": {
            "path": str(video_path.resolve()),
            "fps": int(fps),
            "total_frames": int(frame_count),
            "duration": float(duration),
        },
        "window_size_frames": int(window_size),
        "stride_frames": int(stride),
        "windows": windows,
    }
    with windows_path.open("w", encoding="utf-8") as f:
        json.dump(windows_obj, f, ensure_ascii=False, indent=2)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"mock_track video={video_path} windows={len(windows)} tracks_frames={len(frame_tracks)}\n")
