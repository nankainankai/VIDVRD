from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from vidvrd_auto.models.vl_client import VLClient
from vidvrd_auto.relations.storyboard import draw_tracks, make_storyboard
from vidvrd_auto.utils.image_io import imwrite
from vidvrd_auto.utils.io import iter_jsonl, read_json

try:
    import cv2
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise ImportError("vl_frames requires opencv-python and numpy") from e


def video_path_from_meta(meta_path: Path) -> Optional[Path]:
    if not meta_path.exists():
        return None
    try:
        meta = read_json(meta_path)
        video = meta.get("video", {}) if isinstance(meta, dict) else {}
        raw = str(video.get("path", "") or "").strip()
        if not raw:
            return None
        p = Path(raw).expanduser()
        return p.resolve() if p.exists() else None
    except Exception:
        return None


def video_path_from_windows(windows_json: Path) -> Optional[Path]:
    if not windows_json.exists():
        return None
    try:
        obj = read_json(windows_json)
        video = obj.get("video", {}) if isinstance(obj, dict) else {}
        raw = str(video.get("path", "") or "").strip()
        if not raw:
            return None
        p = Path(raw).expanduser()
        return p.resolve() if p.exists() else None
    except Exception:
        return None


def read_frame(video_path: Path, frame_index: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, frame_index)))
        ok, frame = cap.read()
        return frame if ok and frame is not None else None
    finally:
        cap.release()


def draw_detection_objects(frame_bgr: np.ndarray, objects: Sequence[Dict[str, Any]]) -> np.ndarray:
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    for i, obj in enumerate(objects or []):
        if not isinstance(obj, dict):
            continue
        bbox = obj.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        x1, y1, x2, y2 = [int(round(float(x))) for x in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        color = (0, 255, 255) if i % 2 == 0 else (255, 160, 80)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img


def build_storyboard_from_video(
    *,
    video_path: Path,
    frame_indices: Sequence[int],
    tracks_by_frame: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    detections_by_frame: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    allowed_track_ids: Optional[Set[int]] = None,
    tile_h: int = 320,
) -> Optional[np.ndarray]:
    frames: List[np.ndarray] = []
    allowed = allowed_track_ids or set()
    for fi in frame_indices:
        raw = read_frame(video_path, int(fi))
        if raw is None:
            continue
        if tracks_by_frame is not None and int(fi) in tracks_by_frame:
            tracks = tracks_by_frame[int(fi)]
            if allowed:
                tracks = [t for t in tracks if int(t.get("track_id", -1)) in allowed]
            drawn = draw_tracks(raw, tracks, allowed or {int(t.get("track_id")) for t in tracks if "track_id" in t})
        elif detections_by_frame is not None and int(fi) in detections_by_frame:
            drawn = draw_detection_objects(raw, detections_by_frame[int(fi)])
        else:
            drawn = raw
        frames.append(drawn)
    if not frames:
        return None
    return make_storyboard(frames, tile_h=tile_h)


def uniform_frame_indices(*, total_frames: int, count: int) -> List[int]:
    if total_frames <= 0 or count <= 0:
        return []
    if count >= total_frames:
        return list(range(total_frames))
    if count == 1:
        return [0]
    idx = np.linspace(0, total_frames - 1, num=count)
    return sorted({int(round(x)) for x in idx})


def vl_client_from_config(config: Dict[str, Any], *, api_key: str = "") -> VLClient:
    key = (api_key or str(config.get("api_key", "") or "")).strip()
    if not key:
        key = (os.getenv(str(config.get("api_key_env", "DASHSCOPE_API_KEY") or "DASHSCOPE_API_KEY"), "") or "").strip()
    return VLClient(
        {
            "model": str(config.get("vl_model", "qwen-vl-max") or "qwen-vl-max"),
            "retries": int(config.get("vl_retries", 2) or 2),
            "backoff_sec": float(config.get("vl_backoff_sec", 1.5) or 1.5),
            "sleep_sec": float(config.get("vl_sleep_sec", 0.0) or 0.0),
            "dry_run": bool(config.get("vl_dry_run", False)),
        },
        api_key=key,
    )


def call_vl_with_storyboard(
    *,
    config: Dict[str, Any],
    prompt: str,
    storyboard_bgr: Optional[np.ndarray],
    api_key: str = "",
) -> Any:
    """有拼图则 call_bgr，否则退回纯文本 call。"""
    client = vl_client_from_config(config, api_key=api_key)
    dry = bool(config.get("vl_dry_run", False))
    if storyboard_bgr is not None:
        return client.call_bgr(prompt=prompt, image_bgr=storyboard_bgr, dry_run=dry)
    return client.call(prompt=prompt, dry_run=dry)


def write_temp_storyboard(image_bgr: np.ndarray) -> Path:
    path = Path(tempfile.gettempdir()) / f"vidvrd_vl_{uuid.uuid4().hex}.jpg"
    imwrite(path, image_bgr)
    return path
