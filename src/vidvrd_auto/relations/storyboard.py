from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from vidvrd_auto.utils.image_io import imwrite
from vidvrd_auto.utils.io import iter_jsonl

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("storyboard requires opencv-python") from e


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    h = (int(track_id) * 97) % 360
    hsv = np.uint8([[[h / 2, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def pick_bbox(track: Dict[str, Any]) -> Optional[List[float]]:
    is_predicted = bool(track.get("is_predicted", False))
    if is_predicted:
        b = track.get("bbox") if isinstance(track.get("bbox"), list) else None
        if b is None:
            b = track.get("bbox_observed") if isinstance(track.get("bbox_observed"), list) else None
    else:
        b = track.get("bbox_observed") if isinstance(track.get("bbox_observed"), list) else None
        if b is None:
            b = track.get("bbox") if isinstance(track.get("bbox"), list) else None
    if not isinstance(b, list) or len(b) != 4:
        return None
    out: List[float] = []
    for v in b:
        try:
            fv = float(v)
            if not math.isfinite(fv):
                return None
            out.append(fv)
        except Exception:
            return None
    return out


def draw_tracks(frame_bgr: np.ndarray, tracks: List[Dict[str, Any]], allowed_ids: Set[int]) -> np.ndarray:
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    for tr in tracks:
        try:
            tid = int(tr.get("track_id"))
        except Exception:
            continue
        if tid not in allowed_ids:
            continue
        bbox = pick_bbox(tr)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w - 1, int(round(x1))))
        y1 = max(0, min(h - 1, int(round(y1))))
        x2 = max(0, min(w - 1, int(round(x2))))
        y2 = max(0, min(h - 1, int(round(y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        color = color_for_id(tid)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"ID {tid}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img


def make_storyboard(frames: List[np.ndarray], tile_h: int = 360) -> np.ndarray:
    if not frames:
        raise ValueError("empty frames")
    resized: List[np.ndarray] = []
    for fr in frames:
        h, w = fr.shape[:2]
        if h <= 0 or w <= 0:
            continue
        scale = tile_h / float(h)
        tw = max(1, int(round(w * scale)))
        resized.append(cv2.resize(fr, (tw, tile_h), interpolation=cv2.INTER_AREA))
    if not resized:
        raise ValueError("no valid frames")
    n = len(resized)
    cols = 4 if n > 4 else n
    rows = int(math.ceil(n / cols))
    max_w = max(im.shape[1] for im in resized)
    canvas = np.zeros((rows * tile_h, cols * max_w, 3), dtype=np.uint8)
    for i, im in enumerate(resized):
        r, c = i // cols, i % cols
        y0, x0 = r * tile_h, c * max_w
        canvas[y0 : y0 + tile_h, x0 : x0 + im.shape[1]] = im
    return canvas


def load_tracks_for_frames(tracks_jsonl: Path, needed_frames: Set[int]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for obj in iter_jsonl(tracks_jsonl):
        try:
            fi = int(obj.get("frame"))
        except Exception:
            continue
        if fi not in needed_frames:
            continue
        tracks = obj.get("tracks")
        if isinstance(tracks, list):
            out[fi] = [t for t in tracks if isinstance(t, dict)]
    return out


def save_storyboard_image(path: Path, storyboard_bgr: np.ndarray) -> None:
    imwrite(path, storyboard_bgr)
