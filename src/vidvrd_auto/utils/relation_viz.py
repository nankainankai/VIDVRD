from __future__ import annotations

"""在视频帧上叠加轨迹框与关系箭头/谓词标签。"""

import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from vidvrd_auto.utils.io import iter_jsonl, read_json

TrackMap = Dict[int, Dict[str, Any]]
FrameTracks = Dict[int, List[Dict[str, Any]]]

_COLORS_BGR = [
    (0, 255, 255),
    (255, 128, 0),
    (0, 255, 0),
    (255, 0, 255),
    (0, 128, 255),
    (255, 255, 0),
    (128, 255, 128),
    (255, 128, 255),
]

# 可视化时视为「位置/几何」的谓词（高置信或远距离时可隐藏）
POSITIONAL_PREDICATES = frozenset(
    {
        "left",
        "right",
        "above",
        "below",
        "front",
        "behind",
        "near",
        "overlap",
    }
)


def color_for_track_id(track_id: int) -> Tuple[int, int, int]:
    return _COLORS_BGR[int(track_id) % len(_COLORS_BGR)]


def load_frame_tracks(tracks_jsonl: Path) -> FrameTracks:
    out: FrameTracks = {}
    for row in iter_jsonl(tracks_jsonl):
        try:
            frame = int(row.get("frame"))
        except Exception:
            continue
        tracks = row.get("tracks", []) or []
        if isinstance(tracks, list):
            out[frame] = [t for t in tracks if isinstance(t, dict)]
    return out


def load_relations_for_video(relations_json: Path, video_id: str) -> List[Dict[str, Any]]:
    obj = read_json(relations_json)
    if isinstance(obj, dict):
        items = obj.get(video_id, [])
        if isinstance(items, list):
            return [x for x in items if isinstance(x, dict)]
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    return []


def relations_active_at_frame(
    relations: List[Dict[str, Any]],
    frame: int,
    *,
    min_confidence: float = 0.0,
) -> List[Dict[str, Any]]:
    active: List[Dict[str, Any]] = []
    for rel in relations:
        try:
            start = int(rel.get("start_frame", 0))
            end = int(rel.get("end_frame", start))
        except Exception:
            continue
        if frame < start or frame > end:
            continue
        try:
            conf = float(rel.get("confidence", 1.0))
        except Exception:
            conf = 1.0
        if conf < min_confidence:
            continue
        active.append(rel)
    return active


def _relation_pair_key(rel: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    try:
        subj = int(rel.get("subject_track_id"))
    except Exception:
        return None
    obj_raw = rel.get("object_track_id")
    try:
        obj = int(obj_raw) if obj_raw is not None else -1
    except Exception:
        obj = -1
    return (subj, obj)


def _stable_relation_seed(subj: int, obj: int, predicate: str) -> int:
    raw = f"{subj}:{obj}:{predicate}".encode("utf-8")
    return int(zlib.adler32(raw) & 0x7FFFFFFF)


def anchor_point_in_bbox(bbox: List[float], seed: int) -> Tuple[int, int]:
    """框内锚点：相对中心做确定性偏移，避免互逆关系线重合。"""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(4.0, float(x2 - x1))
    bh = max(4.0, float(y2 - y1))
    margin_x = max(2.0, bw * 0.12)
    margin_y = max(2.0, bh * 0.12)

    rng = int(seed) & 0x7FFFFFFF
    ox = ((rng % 1000) / 1000.0 - 0.5) * 0.55 * bw
    oy = (((rng // 1000) % 1000) / 1000.0 - 0.5) * 0.55 * bh
    x = max(x1 + margin_x, min(x2 - margin_x, cx + ox))
    y = max(y1 + margin_y, min(y2 - margin_y, cy + oy))
    return int(round(x)), int(round(y))


def top_relations_by_pair(relations: List[Dict[str, Any]], *, top_k: int = 1) -> List[Dict[str, Any]]:
    """同一 (subject, object) 保留置信度最高的 top_k 条（可含不同谓词）。"""
    k = max(1, int(top_k))
    groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for rel in relations:
        key = _relation_pair_key(rel)
        if key is None:
            continue
        groups.setdefault(key, []).append(rel)

    out: List[Dict[str, Any]] = []
    for items in groups.values():
        ranked = sorted(items, key=lambda x: -float(x.get("confidence", 0.0) or 0.0))
        out.extend(ranked[:k])
    return sorted(out, key=lambda x: -float(x.get("confidence", 0.0) or 0.0))


def dedupe_relations_by_pair(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return top_relations_by_pair(relations, top_k=1)


def _relation_confidence(rel: Dict[str, Any]) -> float:
    try:
        return float(rel.get("confidence", 1.0))
    except Exception:
        return 1.0


def _normalize_predicate(predicate: str) -> str:
    return str(predicate or "").strip().lower()


def is_positional_predicate(predicate: str) -> bool:
    return _normalize_predicate(predicate) in POSITIONAL_PREDICATES


def _center_distance_for_relation(tracks: List[Dict[str, Any]], rel: Dict[str, Any]) -> Optional[float]:
    try:
        subj = int(rel.get("subject_track_id"))
        obj = int(rel.get("object_track_id"))
    except Exception:
        return None
    ta = _find_track(tracks, subj)
    tb = _find_track(tracks, obj)
    if ta is None or tb is None:
        return None
    subj_bb = _bbox_for_vis(ta)
    obj_bb = _bbox_for_vis(tb)
    if subj_bb is None or obj_bb is None:
        return None
    sx, sy = bbox_center(subj_bb)
    ox, oy = bbox_center(obj_bb)
    return float(((sx - ox) ** 2 + (sy - oy) ** 2) ** 0.5)


def filter_relations_for_visualization(
    relations: List[Dict[str, Any]],
    tracks: List[Dict[str, Any]],
    *,
    frame_width: int,
    frame_height: int,
    min_confidence: float = 0.3,
    max_confidence_spatial: float = 0.95,
    spatial_max_center_distance_ratio: float = 0.35,
) -> List[Dict[str, Any]]:
    """可视化专用过滤：去掉低置信、过高置信的位置关系、远距离位置连线。"""
    diag = max(1.0, float((max(1, frame_width) ** 2 + max(1, frame_height) ** 2) ** 0.5))
    kept: List[Dict[str, Any]] = []
    for rel in relations:
        conf = _relation_confidence(rel)
        if conf < float(min_confidence):
            continue
        pred = _normalize_predicate(str(rel.get("predicate", "") or ""))
        if is_positional_predicate(pred):
            if conf > float(max_confidence_spatial):
                continue
            if float(spatial_max_center_distance_ratio) > 0:
                dist = _center_distance_for_relation(tracks, rel)
                if dist is not None and dist / diag > float(spatial_max_center_distance_ratio):
                    continue
        kept.append(rel)
    return kept


def select_relations_for_visualization(
    relations: List[Dict[str, Any]],
    tracks: List[Dict[str, Any]],
    *,
    frame_width: int,
    frame_height: int,
    min_confidence: float = 0.3,
    max_confidence_spatial: float = 0.95,
    spatial_max_center_distance_ratio: float = 0.35,
    top_k_per_pair: int = 1,
) -> List[Dict[str, Any]]:
    """过滤后每对主体-客体保留置信度最高的 top_k 条。"""
    filtered = filter_relations_for_visualization(
        relations,
        tracks,
        frame_width=frame_width,
        frame_height=frame_height,
        min_confidence=min_confidence,
        max_confidence_spatial=max_confidence_spatial,
        spatial_max_center_distance_ratio=spatial_max_center_distance_ratio,
    )
    return top_relations_by_pair(filtered, top_k=max(1, int(top_k_per_pair)))


def _bbox_for_vis(track: Dict[str, Any]) -> Optional[List[float]]:
    if bool(track.get("is_predicted", False)):
        bb = track.get("bbox_observed", track.get("bbox"))
    else:
        bb = track.get("bbox", track.get("bbox_observed"))
    if isinstance(bb, (list, tuple)) and len(bb) == 4:
        return [float(x) for x in bb]
    return None


def bbox_center(bbox: List[float]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)


def _find_track(tracks: List[Dict[str, Any]], track_id: int) -> Optional[Dict[str, Any]]:
    for t in tracks:
        try:
            if int(t.get("track_id")) == int(track_id):
                return t
        except Exception:
            continue
    return None


def draw_tracks_on_frame(
    frame_bgr: np.ndarray,
    tracks: List[Dict[str, Any]],
    *,
    id_to_label: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    labels = id_to_label or {}
    for track in tracks:
        try:
            tid = int(track.get("track_id"))
        except Exception:
            continue
        bbox = _bbox_for_vis(track)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        x1i = max(0, min(w - 1, int(round(x1))))
        y1i = max(0, min(h - 1, int(round(y1))))
        x2i = max(0, min(w - 1, int(round(x2))))
        y2i = max(0, min(h - 1, int(round(y2))))
        if x2i <= x1i or y2i <= y1i:
            continue
        color = color_for_track_id(tid)
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), color, 2)
        label = f"ID{tid}:{labels.get(tid, track.get('class_name', 'obj'))}"
        cv2.putText(out, label, (x1i + 4, max(18, y1i + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, label, (x1i + 4, max(18, y1i + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _label_xy_for_segment(
    sx: int,
    sy: int,
    ox: int,
    oy: int,
    *,
    idx: int,
    seed: int,
    width: int,
    height: int,
) -> Tuple[int, int]:
    mx = (sx + ox) / 2.0
    my = (sy + oy) / 2.0
    dx = float(ox - sx)
    dy = float(oy - sy)
    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
    px = -dy / length
    py = dx / length
    sign = 1.0 if (seed % 2 == 0) else -1.0
    offset = 10.0 + (seed % 7) * 3.0 + idx * 8.0
    lx = int(round(mx + px * offset * sign))
    ly = int(round(my + py * offset * sign))
    return max(4, min(width - 280, lx)), max(16, min(height - 8, ly))


def _format_confidence(rel: Dict[str, Any]) -> str:
    try:
        conf = float(rel.get("confidence", 1.0))
    except Exception:
        conf = 1.0
    return f"{conf:.2f}"


def format_relation_label(rel: Dict[str, Any], *, show_confidence: bool = True) -> str:
    try:
        subj = int(rel.get("subject_track_id"))
    except Exception:
        subj = -1
    obj_raw = rel.get("object_track_id")
    predicate = str(rel.get("predicate", "") or "rel")
    conf_suffix = f" {_format_confidence(rel)}" if show_confidence else ""
    if obj_raw is None:
        return f"{subj} {predicate}{conf_suffix}".strip()
    try:
        obj = int(obj_raw)
    except Exception:
        obj = -1
    return f"{subj}->{obj}:{predicate}{conf_suffix}"


def draw_relations_on_frame(
    frame_bgr: np.ndarray,
    tracks: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    *,
    max_relations: int = 8,
    top_k_per_pair: int = 1,
    show_confidence: bool = True,
    min_confidence: float = 0.3,
    max_confidence_spatial: float = 0.95,
    spatial_max_center_distance_ratio: float = 0.35,
) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    ranked = select_relations_for_visualization(
        relations,
        tracks,
        frame_width=w,
        frame_height=h,
        min_confidence=min_confidence,
        max_confidence_spatial=max_confidence_spatial,
        spatial_max_center_distance_ratio=spatial_max_center_distance_ratio,
        top_k_per_pair=top_k_per_pair,
    )
    shown = ranked[: max(0, int(max_relations))]
    for idx, rel in enumerate(shown):
        try:
            subj = int(rel.get("subject_track_id"))
        except Exception:
            continue
        obj_raw = rel.get("object_track_id")
        predicate = str(rel.get("predicate", "") or "rel")
        seed = _stable_relation_seed(subj, int(obj_raw) if obj_raw is not None else -1, predicate)
        ta = _find_track(tracks, subj)
        if ta is None:
            continue
        subj_bb = _bbox_for_vis(ta)
        if subj_bb is None:
            continue
        sx, sy = anchor_point_in_bbox(subj_bb, seed)

        if obj_raw is None:
            label = format_relation_label(rel, show_confidence=show_confidence)
            cv2.putText(out, label, (sx, max(20, sy - 12 - idx * 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            continue

        try:
            obj = int(obj_raw)
        except Exception:
            continue
        tb = _find_track(tracks, obj)
        if tb is None:
            continue
        obj_bb = _bbox_for_vis(tb)
        if obj_bb is None:
            continue
        ox, oy = anchor_point_in_bbox(obj_bb, seed + 7919)

        color = color_for_track_id(subj)
        cv2.arrowedLine(out, (sx, sy), (ox, oy), color, 2, tipLength=0.12)
        label = format_relation_label(rel, show_confidence=show_confidence)
        lx, ly = _label_xy_for_segment(sx, sy, ox, oy, idx=idx, seed=seed, width=w, height=h)
        cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    if len(ranked) > len(shown):
        extra = len(ranked) - len(shown)
        if extra > 0:
            cv2.putText(out, f"+{extra} more relations", (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def render_frame_with_relations(
    frame_bgr: np.ndarray,
    tracks: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    *,
    id_to_label: Optional[Dict[int, str]] = None,
    max_relations: int = 8,
    top_k_per_pair: int = 1,
    show_confidence: bool = True,
    min_confidence: float = 0.3,
    max_confidence_spatial: float = 0.95,
    spatial_max_center_distance_ratio: float = 0.35,
) -> np.ndarray:
    vis = draw_tracks_on_frame(frame_bgr, tracks, id_to_label=id_to_label)
    return draw_relations_on_frame(
        vis,
        tracks,
        relations,
        max_relations=max_relations,
        top_k_per_pair=top_k_per_pair,
        show_confidence=show_confidence,
        min_confidence=min_confidence,
        max_confidence_spatial=max_confidence_spatial,
        spatial_max_center_distance_ratio=spatial_max_center_distance_ratio,
    )


def render_relation_video(
    *,
    video_path: Path,
    tracks_jsonl: Path,
    relations_json: Path,
    video_id: str,
    out_path: Path,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = dict(config or {})
    min_conf = float(cfg.get("min_confidence", 0.3) or 0.0)
    max_conf_spatial = float(cfg.get("max_confidence_spatial", 0.95))
    spatial_dist_ratio = float(cfg.get("spatial_max_center_distance_ratio", 0.35))
    max_relations = int(cfg.get("max_relations_per_frame", 8) or 8)
    top_k_per_pair = int(cfg.get("top_k_per_pair", 1) or 1)
    show_confidence = bool(cfg.get("show_confidence", True))
    out_name = str(cfg.get("output_name", "relation_box_vis.mp4") or "relation_box_vis.mp4")

    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    if not tracks_jsonl.exists():
        raise FileNotFoundError(f"tracks not found: {tracks_jsonl}")
    if not relations_json.exists():
        raise FileNotFoundError(f"relations not found: {relations_json}")

    frame_tracks = load_frame_tracks(tracks_jsonl)
    relations = load_relations_for_video(relations_json, video_id)
    id_to_label: Dict[int, str] = {}
    for tracks in frame_tracks.values():
        for t in tracks:
            try:
                tid = int(t.get("track_id"))
            except Exception:
                continue
            id_to_label.setdefault(tid, str(t.get("class_name", "") or "obj"))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_path = out_path if out_path.suffix else out_path / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps), (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"failed to create writer: {out_path}")

    frame_idx = 0
    relation_frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            tracks = frame_tracks.get(frame_idx, [])
            active = relations_active_at_frame(relations, frame_idx, min_confidence=0.0)
            if active:
                relation_frames += 1
            vis = render_frame_with_relations(
                frame,
                tracks,
                active,
                id_to_label=id_to_label,
                max_relations=max_relations,
                top_k_per_pair=top_k_per_pair,
                show_confidence=show_confidence,
                min_confidence=min_conf,
                max_confidence_spatial=max_conf_spatial,
                spatial_max_center_distance_ratio=spatial_dist_ratio,
            )
            cv2.putText(
                vis,
                f"frame {frame_idx}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(vis)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    return {
        "video_path": str(video_path),
        "tracks_jsonl": str(tracks_jsonl),
        "relations_json": str(relations_json),
        "output_video": str(out_path),
        "video_id": video_id,
        "relation_count": len(relations),
        "frames_written": frame_idx,
        "frames_with_relations": relation_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames_reported": total_frames,
    }
