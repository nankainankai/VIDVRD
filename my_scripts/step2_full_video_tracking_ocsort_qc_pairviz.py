from __future__ import annotations

import argparse
import base64
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from modules.object_tracker import ObjectTracker
import config

try:
    from utils_io import iter_jsonl as _iter_jsonl_utils
    from utils_io import safe_read_json as _safe_read_json_utils
    from utils_io import safe_write_json as _safe_write_json_utils

    HAS_UTILS_IO = True
except Exception:
    HAS_UTILS_IO = False

try:
    import dashscope
    from dashscope import MultiModalConversation

    HAS_DASHSCOPE = True
except Exception:
    dashscope = None
    MultiModalConversation = None
    HAS_DASHSCOPE = False


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _select_video_gui() -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        video_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")],
        )
    finally:
        root.destroy()
    return str(video_path or "")


def _infer_video_path_from_meta(output_dir: Path) -> str:
    meta_name = str(getattr(config, "VIDEO_META_JSON_NAME", "video_meta.json"))
    meta_path = output_dir / meta_name
    if not meta_path.exists():
        return ""

    try:
        if HAS_UTILS_IO:
            obj = _safe_read_json_utils(meta_path)
        else:
            with meta_path.open("r", encoding="utf-8-sig") as f:
                obj = json.load(f)
    except Exception:
        return ""

    if not isinstance(obj, dict):
        return ""
    video = obj.get("video", {})
    if not isinstance(video, dict):
        return ""
    p = str(video.get("path", "") or "").strip()
    return p


def _load_detections_jsonl(path: Path) -> Dict[int, List[Dict[str, Any]]]:
    rows: Dict[int, List[Dict[str, Any]]] = {}
    if HAS_UTILS_IO:
        for row in _iter_jsonl_utils(path):
            try:
                frame = int(row.get("frame"))
            except Exception:
                continue
            objs = row.get("objects", [])
            if not isinstance(objs, list):
                objs = []
            rows[frame] = objs
        return rows

    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            try:
                frame = int(row.get("frame"))
            except Exception:
                continue
            objs = row.get("objects", [])
            if not isinstance(objs, list):
                objs = []
            rows[frame] = objs
    return rows


def _compact_tracks(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in tracks or []:
        try:
            bbox = t.get("bbox")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            bbox_obs = t.get("bbox_observed")
            out.append(
                {
                    "track_id": int(t.get("track_id")),
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    "bbox_observed": (
                        [float(bbox_obs[0]), float(bbox_obs[1]), float(bbox_obs[2]), float(bbox_obs[3])]
                        if (isinstance(bbox_obs, (list, tuple)) and len(bbox_obs) == 4)
                        else [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                    ),
                    "class_name": str(t.get("class_name", "unknown")),
                    "confidence": float(t.get("confidence", 0.0)),
                    "duration_frames": int(t.get("duration_frames", 0)),
                    "total_distance": float(t.get("total_distance", 0.0)),
                    "instant_distance": float(t.get("instant_distance", 0.0)),
                    "avg_speed": float(t.get("avg_speed", 0.0)),
                    "motion_state": str(t.get("motion_state", "unknown")),
                    "age": int(t.get("age", 0)),
                    "hits": int(t.get("hits", 0)),
                    "is_predicted": bool(t.get("is_predicted", False)),
                }
            )
        except Exception:
            continue
    return out


def _update_track_stats(stats: Dict[int, Dict[str, Any]], tracks: List[Dict[str, Any]], frame_num: int, timestamp: float) -> None:
    for t in tracks or []:
        try:
            tid = int(t.get("track_id"))
        except Exception:
            continue

        item = stats.setdefault(
            tid,
            {
                "track_id": tid,
                "class_name": str(t.get("class_name", "unknown")),
                "first_frame": frame_num,
                "last_frame": frame_num,
                "first_time": float(timestamp),
                "last_time": float(timestamp),
                "frame_hits": 0,
                "sum_confidence": 0.0,
                "max_confidence": 0.0,
                "frames": [],
            },
        )

        item["class_name"] = str(t.get("class_name", item["class_name"]))
        item["first_frame"] = min(int(item["first_frame"]), int(frame_num))
        item["last_frame"] = max(int(item["last_frame"]), int(frame_num))
        item["first_time"] = min(float(item["first_time"]), float(timestamp))
        item["last_time"] = max(float(item["last_time"]), float(timestamp))
        item["frame_hits"] = int(item["frame_hits"]) + 1

        conf = float(t.get("confidence", 0.0))
        item["sum_confidence"] = float(item["sum_confidence"]) + conf
        item["max_confidence"] = max(float(item["max_confidence"]), conf)
        item["frames"].append(int(frame_num))


def _finalize_track_index(track_stats: Dict[int, Dict[str, Any]], fps: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in track_stats.values():
        frames = sorted(set(int(x) for x in item.get("frames", [])))
        if not frames:
            continue

        gap_count = 0
        max_gap = 0
        for i in range(1, len(frames)):
            gap = int(frames[i] - frames[i - 1] - 1)
            if gap > 0:
                gap_count += 1
                max_gap = max(max_gap, gap)

        frame_hits = int(item.get("frame_hits", 0))
        span_frames = int(item.get("last_frame", frames[-1])) - int(item.get("first_frame", frames[0])) + 1
        avg_conf = float(item.get("sum_confidence", 0.0)) / float(max(1, frame_hits))

        rows.append(
            {
                "track_id": int(item["track_id"]),
                "class_name": str(item.get("class_name", "unknown")),
                "first_frame": int(item.get("first_frame", frames[0])),
                "last_frame": int(item.get("last_frame", frames[-1])),
                "first_time": float(item.get("first_time", 0.0)),
                "last_time": float(item.get("last_time", 0.0)),
                "frame_hits": frame_hits,
                "span_frames": span_frames,
                "coverage_ratio": float(frame_hits) / float(max(1, span_frames)),
                "gap_count": int(gap_count),
                "max_gap_frames": int(max_gap),
                "avg_confidence": float(avg_conf),
                "max_confidence": float(item.get("max_confidence", 0.0)),
                "duration_seconds": float(span_frames / max(1, fps)),
            }
        )
    rows.sort(key=lambda x: (x["first_frame"], x["track_id"]))
    return rows


def _window_track_ids(frame_tracks_map: Dict[int, List[int]], start_frame: int, end_frame: int) -> List[int]:
    ids = set()
    for f in range(int(start_frame), int(end_frame) + 1):
        for tid in frame_tracks_map.get(f, []):
            ids.add(int(tid))
    return sorted(ids)


def _b64_jpeg_from_bgr(img_bgr: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return base64.b64encode(buf).decode("utf-8")


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            return str(first.get("text", first)).strip()
        return str(first).strip()
    if isinstance(content, dict):
        return str(content.get("text", content)).strip()
    return str(content).strip()


def _try_parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    # Handle fenced outputs: ```json\n{...}\n```
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 2 and lines[0].lstrip().startswith("```"):
            body = lines[1:]
            if body and body[-1].strip().startswith("```"):
                body = body[:-1]
            raw = "\n".join(body).strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _sample_storyboard(video_path: Path, total_frames: int, sample_frames: int = 8) -> np.ndarray:
    indices = np.linspace(0, max(0, total_frames - 1), sample_frames).round().astype(int).tolist()
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if ok and fr is not None:
            frames.append(fr)
    cap.release()

    if not frames:
        raise RuntimeError("Cannot sample storyboard frames")

    cols, rows, tile_w, tile_h, pad = 4, 2, 320, 180, 8
    canvas = np.zeros((rows * tile_h + (rows + 1) * pad, cols * tile_w + (cols + 1) * pad, 3), dtype=np.uint8)
    for i, fr in enumerate(frames[: cols * rows]):
        r = i // cols
        c = i % cols
        tile = cv2.resize(fr, (tile_w, tile_h))
        x0 = pad + c * (tile_w + pad)
        y0 = pad + r * (tile_h + pad)
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
        cv2.putText(canvas, f"t{i+1}", (x0 + 8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _llm_count_consistency(
    video_path: Path,
    total_frames: int,
    track_index: List[Dict[str, Any]],
    frame_tracks_detail: Dict[int, List[Dict[str, Any]]],
    api_key: str,
    model: str,
) -> Dict[str, Any]:
    if not HAS_DASHSCOPE or MultiModalConversation is None:
        return {
            "enabled": False,
            "error": "dashscope unavailable",
            "issues": [],
            "llm_counts": {},
        }

    sample_frames = int(getattr(config, "STEP2_QC_SAMPLE_FRAMES", 8))
    storyboard = _sample_storyboard(video_path, total_frames=total_frames, sample_frames=sample_frames)
    dashscope.api_key = api_key

    prompt = """你是视频质检助手。请根据关键帧拼图估计主要物体的数量。
输出严格 JSON，不要额外文字：
{
  "counts": [
    {"class_name": "person", "count": 2},
    {"class_name": "skateboard", "count": 1}
  ],
  "summary": "一句话概述"
}
要求：
1) class_name 用英文小写；
2) count 用非负整数；
3) 若不确定可不输出该类别。"""

    image_b64 = _b64_jpeg_from_bgr(storyboard, quality=90)
    image_data = f"data:image/jpeg;base64,{image_b64}"
    resp = MultiModalConversation.call(
        model=model,
        messages=[{"role": "user", "content": [{"image": image_data}, {"text": prompt}]}],
    )
    if getattr(resp, "status_code", None) != 200:
        return {
            "enabled": True,
            "error": f"HTTP {getattr(resp, 'status_code', 'unknown')}",
            "issues": [],
            "llm_counts": {},
        }

    text = _extract_text(resp.output.choices[0].message.content)
    obj = _try_parse_json_object(text)
    if obj is None:
        return {
            "enabled": True,
            "error": "llm_output_not_json",
            "issues": [],
            "llm_counts": {},
            "raw_output": text,
        }

    llm_counts: Dict[str, int] = {}
    for it in obj.get("counts", []) if isinstance(obj.get("counts", []), list) else []:
        if not isinstance(it, dict):
            continue
        name = str(it.get("class_name", "")).strip().lower()
        if not name:
            continue
        try:
            count = int(it.get("count", 0))
        except Exception:
            continue
        llm_counts[name] = max(0, count)

    # Reference 1) unique track count (may over-estimate what is visible in sampled frames)
    track_counts_unique: Dict[str, int] = {}
    for t in track_index:
        name = str(t.get("class_name", "unknown")).strip().lower()
        track_counts_unique[name] = track_counts_unique.get(name, 0) + 1

    # Reference 2) max simultaneous track count on the SAME sampled frames shown to LLM.
    indices = np.linspace(0, max(0, total_frames - 1), max(1, sample_frames)).round().astype(int).tolist()

    track_counts_max_observed: Dict[str, int] = {}
    track_counts_max_all: Dict[str, int] = {}
    for idx in indices:
        tracks = frame_tracks_detail.get(int(idx), []) or []
        cnt_obs: Dict[str, int] = {}
        cnt_all: Dict[str, int] = {}
        for tr in tracks:
            if not isinstance(tr, dict):
                continue
            name = str(tr.get("class_name", "unknown")).strip().lower()
            if not name:
                name = "unknown"
            cnt_all[name] = cnt_all.get(name, 0) + 1
            if not bool(tr.get("is_predicted", False)):
                cnt_obs[name] = cnt_obs.get(name, 0) + 1

        for cls, c in cnt_obs.items():
            track_counts_max_observed[cls] = max(int(track_counts_max_observed.get(cls, 0)), int(c))
        for cls, c in cnt_all.items():
            track_counts_max_all[cls] = max(int(track_counts_max_all.get(cls, 0)), int(c))

    threshold = int(getattr(config, "STEP2_QC_COUNT_DIFF_THRESHOLD", 1))
    issues: List[Dict[str, Any]] = []
    for cls, llm_count in llm_counts.items():
        # Prefer observed-only simultaneous counts to avoid penalizing predicted boxes and re-appearing tracks.
        trk_count = int(track_counts_max_observed.get(cls, 0))
        diff = abs(trk_count - llm_count)
        if diff >= threshold:
            issues.append(
                {
                    "type": "count_mismatch",
                    "class_name": cls,
                    "llm_count": llm_count,
                    "track_count_max_observed": trk_count,
                    "track_count_max_all": int(track_counts_max_all.get(cls, 0)),
                    "track_count_unique": int(track_counts_unique.get(cls, 0)),
                    "abs_diff": diff,
                    "threshold": threshold,
                    "severity": "high" if diff >= threshold + 1 else "medium",
                }
            )

    return {
        "enabled": True,
        "model": model,
        "llm_counts": llm_counts,
        "track_counts_unique": track_counts_unique,
        "track_counts_max_observed": track_counts_max_observed,
        "track_counts_max_all": track_counts_max_all,
        "sampled_frame_indices": indices,
        "issues": issues,
        "summary": str(obj.get("summary", "")),
    }


def _find_track_by_id(tracks: List[Dict[str, Any]], tid: int) -> Optional[Dict[str, Any]]:
    for t in tracks or []:
        try:
            if int(t.get("track_id")) == int(tid):
                return t
        except Exception:
            continue
    return None


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(round(float(v))))))


def _render_pair_frame(frame_bgr: np.ndarray, ta: Optional[Dict[str, Any]], tb: Optional[Dict[str, Any]], label_a: str, label_b: str) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    base = (frame_bgr.astype(np.float32) * 0.35).clip(0, 255).astype(np.uint8)

    def draw_one(track: Optional[Dict[str, Any]], color: Tuple[int, int, int], text: str) -> None:
        if not track:
            return
        bbox = track.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            return
        x1, y1, x2, y2 = bbox
        x1i = _clamp_int(x1, 0, w - 1)
        y1i = _clamp_int(y1, 0, h - 1)
        x2i = _clamp_int(x2, 0, w - 1)
        y2i = _clamp_int(y2, 0, h - 1)
        if x2i <= x1i or y2i <= y1i:
            return
        base[y1i:y2i, x1i:x2i] = frame_bgr[y1i:y2i, x1i:x2i]
        cv2.rectangle(base, (x1i, y1i), (x2i, y2i), color, 3)
        cv2.putText(base, text, (x1i + 6, max(22, y1i + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    draw_one(ta, (0, 255, 255), label_a)
    draw_one(tb, (255, 128, 0), label_b)
    return base


def _load_window_frames(video_path: Path, start_frame: int, end_frame: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    frames: List[np.ndarray] = []
    total = int(end_frame - start_frame + 1)
    for _ in range(max(0, total)):
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        frames.append(fr)
    cap.release()
    return frames


def _export_pair_videos(
    video_path: Path,
    windows: List[Dict[str, Any]],
    frame_tracks_detail: Dict[int, List[Dict[str, Any]]],
    track_label_map: Dict[int, str],
    out_dir: Path,
    fps: int,
) -> Dict[str, Any]:
    _ensure_dir(out_dir)

    max_windows = int(getattr(config, "PAIR_VIZ_MAX_WINDOWS", 0))
    max_pairs = int(getattr(config, "PAIR_VIZ_MAX_PAIRS_PER_WINDOW", 6))

    items: List[Dict[str, Any]] = []
    for i, w in enumerate(windows):
        if max_windows > 0 and i >= max_windows:
            break
        if not isinstance(w, dict):
            continue

        window_id = int(w.get("window_id", i + 1) or i + 1)
        start_frame = int(w.get("start_frame", 0) or 0)
        end_frame = int(w.get("end_frame", start_frame) or start_frame)
        tids = sorted({int(t) for t in (w.get("track_ids", []) or [])})

        pairs = list(itertools.combinations(tids, 2))
        if max_pairs > 0:
            pairs = pairs[:max_pairs]
        if not pairs:
            continue

        frames = _load_window_frames(video_path, start_frame, end_frame)
        if not frames:
            continue
        h, wv = frames[0].shape[:2]

        for (a, b) in pairs:
            out_name = f"win_{window_id:04d}_pair_{a}_{b}.mp4"
            out_path = out_dir / out_name
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), int(max(1, fps)), (wv, h))
            try:
                for local_idx, fr in enumerate(frames):
                    global_idx = start_frame + local_idx
                    tracks = frame_tracks_detail.get(global_idx, [])
                    ta = _find_track_by_id(tracks, a)
                    tb = _find_track_by_id(tracks, b)
                    la = f"ID {a}: {track_label_map.get(a, 'unknown')}"
                    lb = f"ID {b}: {track_label_map.get(b, 'unknown')}"
                    vis = _render_pair_frame(fr, ta, tb, la, lb)
                    writer.write(vis)
            finally:
                writer.release()

            items.append(
                {
                    "window_id": window_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "pair": [a, b],
                    "video": str(out_path).replace("\\", "/"),
                }
            )

    return {"count": len(items), "items": items}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step2: OC-SORT tracking + qwen-vl-max QC + pair video export")
    parser.add_argument("--video", type=str, default="", help="视频路径（不传则尝试从 video_meta.json 推断，失败再弹窗选择）")
    parser.add_argument("--output_dir", type=str, default=str(getattr(config, "OUTPUT_DIR", "C:/video_output")))
    parser.add_argument("--detections_jsonl", type=str, default="", help="Step1 输出 detections_full.jsonl")
    parser.add_argument("--window_size", type=int, default=int(getattr(config, "WINDOW_SIZE_FRAMES", 30)))
    parser.add_argument("--stride", type=int, default=int(getattr(config, "WINDOW_STRIDE_FRAMES", 15)))
    parser.add_argument("--api_key", type=str, default="", help="用于 Step2 质检的大模型 key")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    _ensure_dir(output_dir)

    video_raw = str(args.video or "").strip()
    if not video_raw:
        video_raw = _infer_video_path_from_meta(output_dir)
    if not video_raw:
        try:
            video_raw = _select_video_gui()
        except Exception:
            video_raw = ""

    if not video_raw:
        print("ERROR: 未提供视频路径，且无法自动推断。请传入 --video 或先运行 Step1 生成 video_meta.json。")
        return

    video_path = Path(video_raw).expanduser().resolve()
    if not video_path.exists():
        print(f"ERROR: 视频不存在: {video_path}")
        return

    detections_path = Path(args.detections_jsonl).expanduser().resolve() if args.detections_jsonl else (
        output_dir / str(getattr(config, "FULL_DETECTIONS_JSONL_NAME", "detections_full.jsonl"))
    )
    if not detections_path.exists():
        print(f"ERROR: detections jsonl 不存在: {detections_path}")
        return

    tracks_path = output_dir / str(getattr(config, "FULL_TRACKS_JSONL_NAME", "tracks_full.jsonl"))
    track_index_path = output_dir / str(getattr(config, "TRACK_INDEX_JSON_NAME", "track_index.json"))
    windows_path = output_dir / str(getattr(config, "WINDOWS_JSON_NAME", "windows.json"))
    qc_report_path = output_dir / str(getattr(config, "QC_REPORT_JSON_NAME", "qc_report.json"))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("ERROR: 无法打开视频")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = float(total_frames / max(1, fps))

    detections_map = _load_detections_jsonl(detections_path)

    tracker = ObjectTracker(
        iou_threshold=float(getattr(config, "TRACKING_IOU_THRESHOLD", 0.5)),
        max_age=int(getattr(config, "TRACKING_MAX_AGE", 30)),
        min_hits=int(getattr(config, "TRACKING_MIN_HITS", 3)),
        class_aware=bool(getattr(config, "TRACKING_CLASS_AWARE", True)),
        max_center_dist_ratio=float(getattr(config, "TRACKING_MAX_CENTER_DIST_RATIO", 0.8)),
        min_new_track_conf=float(getattr(config, "TRACKING_MIN_NEW_TRACK_CONF", 0.35)),
        velocity_alpha=float(getattr(config, "TRACKING_VELOCITY_ALPHA", 0.8)),
    )

    print("=" * 70)
    print("Step2: OC-SORT 逐帧轨迹追踪 + 大模型质检 + 轨迹对可视化导出")
    print(f"视频: {video_path}")
    print(f"detections: {detections_path}")
    print(f"fps={fps}, total_frames={total_frames}, duration={duration:.2f}s")
    print("=" * 70)

    frame_num = 0
    err_count = 0
    track_stats: Dict[int, Dict[str, Any]] = {}
    frame_tracks_map: Dict[int, List[int]] = {}
    frame_tracks_detail: Dict[int, List[Dict[str, Any]]] = {}

    with tracks_path.open("w", encoding="utf-8") as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = float(frame_num / max(1, fps))
            detections = detections_map.get(int(frame_num), [])

            try:
                tracks = tracker.track(frame, detections, frame_num=frame_num)
            except Exception as e:
                tracks = []
                err_count += 1
                if err_count <= 3:
                    print(f"WARN: track failed at frame={frame_num}: {e}")

            tracks_compact = _compact_tracks(tracks)
            frame_tracks_detail[int(frame_num)] = tracks_compact

            row = {"frame": int(frame_num), "timestamp": float(timestamp), "tracks": tracks_compact}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            tids: List[int] = []
            for t in tracks_compact:
                tid = int(t.get("track_id", -1))
                if tid >= 0:
                    tids.append(tid)
            frame_tracks_map[int(frame_num)] = tids
            _update_track_stats(track_stats, tracks_compact, frame_num=frame_num, timestamp=timestamp)

            if frame_num % 100 == 0:
                print(f"Processed frame {frame_num}/{max(1, total_frames - 1)}")
            frame_num += 1

    cap.release()

    track_index = _finalize_track_index(track_stats, fps=max(1, fps))

    windows: List[Dict[str, Any]] = []
    window_size = int(args.window_size)
    stride = int(args.stride)
    if window_size > 0 and stride > 0:
        if frame_num >= window_size:
            window_id = 0
            for start_frame in range(0, frame_num - window_size + 1, stride):
                end_frame = start_frame + window_size - 1
                window_id += 1
                windows.append(
                    {
                        "window_id": int(window_id),
                        "start_frame": int(start_frame),
                        "end_frame": int(end_frame),
                        "start_time": float(start_frame / max(1, fps)),
                        "end_time": float(end_frame / max(1, fps)),
                        "frame_count": int(window_size),
                        "track_ids": _window_track_ids(frame_tracks_map, start_frame=start_frame, end_frame=end_frame),
                    }
                )
        elif frame_num > 0:
            # 短视频兜底：不足一个窗口时，生成单个窗口覆盖全视频，避免下游空跑。
            start_frame = 0
            end_frame = int(frame_num - 1)
            windows.append(
                {
                    "window_id": 1,
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "start_time": float(start_frame / max(1, fps)),
                    "end_time": float(end_frame / max(1, fps)),
                    "frame_count": int(end_frame - start_frame + 1),
                    "track_ids": _window_track_ids(frame_tracks_map, start_frame=start_frame, end_frame=end_frame),
                }
            )

    track_index_obj = {
        "video": {
            "path": str(video_path),
            "fps": int(fps),
            "total_frames": int(total_frames),
            "duration": float(duration),
        },
        "tracks": track_index,
    }
    if HAS_UTILS_IO:
        _safe_write_json_utils(track_index_path, track_index_obj, indent=2)
    else:
        with track_index_path.open("w", encoding="utf-8") as f:
            json.dump(track_index_obj, f, ensure_ascii=False, indent=2)

    windows_obj = {
        "video": {
            "path": str(video_path),
            "fps": int(fps),
            "total_frames": int(total_frames),
            "duration": float(duration),
        },
        "window_size_frames": int(window_size),
        "stride_frames": int(stride),
        "windows": windows,
    }
    if HAS_UTILS_IO:
        _safe_write_json_utils(windows_path, windows_obj, indent=2)
    else:
        with windows_path.open("w", encoding="utf-8") as f:
            json.dump(windows_obj, f, ensure_ascii=False, indent=2)

    api_key = (args.api_key or "").strip()
    if not api_key:
        api_key = str(getattr(config, "API_KEY", "") or "").strip()

    llm_qc_enabled = bool(getattr(config, "ENABLE_STEP2_LLM_QC", True))
    llm_qc_model = str(getattr(config, "STEP2_LLM_QC_MODEL", "qwen-vl-max"))

    llm_qc_result: Dict[str, Any] = {"enabled": False, "issues": []}
    if llm_qc_enabled and api_key:
        try:
            llm_qc_result = _llm_count_consistency(
                video_path=video_path,
                total_frames=total_frames,
                track_index=track_index,
                frame_tracks_detail=frame_tracks_detail,
                api_key=api_key,
                model=llm_qc_model,
            )
        except Exception as e:
            llm_qc_result = {"enabled": True, "issues": [], "error": str(e)}

    pair_viz_enabled = bool(getattr(config, "EXPORT_PAIR_VIZ_VIDEOS", True))
    pair_viz_result: Dict[str, Any] = {"enabled": False, "count": 0, "items": []}
    if pair_viz_enabled:
        try:
            review_dir = output_dir / str(getattr(config, "REVIEW_BUNDLE_DIR_NAME", "review_bundle"))
            pair_dir = review_dir / "pair_videos"
            track_label_map = {int(t.get("track_id", -1)): str(t.get("class_name", "unknown")) for t in track_index}
            pair_viz_result = _export_pair_videos(
                video_path=video_path,
                windows=windows,
                frame_tracks_detail=frame_tracks_detail,
                track_label_map=track_label_map,
                out_dir=pair_dir,
                fps=max(1, fps),
            )
            pair_viz_result["enabled"] = True
            pair_videos_index_path = review_dir / "pair_videos_index.json"
            if HAS_UTILS_IO:
                _safe_write_json_utils(pair_videos_index_path, pair_viz_result, indent=2)
            else:
                with pair_videos_index_path.open("w", encoding="utf-8") as f:
                    json.dump(pair_viz_result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            pair_viz_result = {"enabled": True, "count": 0, "items": [], "error": str(e)}

    severe_count = 0
    if isinstance(llm_qc_result.get("issues", []), list):
        for it in llm_qc_result.get("issues", []):
            if str(it.get("severity", "")) == "high":
                severe_count += 1

    qc_report = {
        "video": {
            "path": str(video_path),
            "fps": int(fps),
            "total_frames": int(total_frames),
            "duration": float(duration),
        },
        "llm_qc": llm_qc_result,
        "pair_viz": {"enabled": pair_viz_result.get("enabled", False), "count": pair_viz_result.get("count", 0)},
        "risk_tag": "high" if severe_count > 0 else "normal",
        "is_pass": severe_count == 0,
    }
    if HAS_UTILS_IO:
        _safe_write_json_utils(qc_report_path, qc_report, indent=2)
    else:
        with qc_report_path.open("w", encoding="utf-8") as f:
            json.dump(qc_report, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("DONE: Step2 完成")
    print(f"tracks_jsonl={tracks_path}")
    print(f"track_index={track_index_path}")
    print(f"windows={windows_path}")
    print(f"qc_report={qc_report_path}")
    if pair_viz_result.get("enabled", False):
        print(f"pair_videos_count={pair_viz_result.get('count', 0)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
