"""Semi-auto relation candidate generation (Phase-1).

Reads Step2 outputs (windows.json + tracks_full.jsonl + original video), builds 1fps
storyboards with track-id overlays, then queries a multimodal LLM (DashScope Qwen-VL)
for relation candidates.

Outputs a JSON compatible with tools/evaluate_presence.py:

{
  "<video_id>": [
    {
      "subject_track_id": 0,
      "object_track_id": 1,
      "predicate": "left",
      "start_frame": 60,
      "end_frame": 89,
      "confidence": 0.83,
      "source": "semi_auto",
      "segment_id": 3,
      "evidence": "..."
    }
  ]
}

Notes:
- Matching/evaluation is presence-only in Phase-1; this script uses window span as
  (start_frame,end_frame) by default ("窗口覆盖").
- Implements simple coupling completion: left<->right, above<->below, front<->behind.
- Optional audio prior: provide vggsound label via --vggsound_label; this adds a few
  audio-related predicates to query list and injects label into prompt.

"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from utils_io import iter_jsonl as _iter_jsonl_utils
    from utils_io import safe_read_json as _safe_read_json_utils
    from utils_io import safe_write_json as _safe_write_json_utils

    HAS_UTILS_IO = True
except Exception:
    HAS_UTILS_IO = False

try:
    import cv2  # type: ignore

    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

try:
    import dashscope  # type: ignore
    from dashscope import MultiModalConversation  # type: ignore

    HAS_DASHSCOPE = True
except Exception:
    dashscope = None
    MultiModalConversation = None
    HAS_DASHSCOPE = False

try:
    import config  # type: ignore
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config  # type: ignore

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from vidvrd_auto.relations.object_candidates import get_candidate_predicates, normalize_category
except Exception:
    def normalize_category(category: str) -> str:
        return str(category or "").strip().lower().replace(" ", "_") or "unknown"

    def get_candidate_predicates(subject_class: str, object_class: str, audio_label: str = "") -> List[str]:
        s = normalize_category(subject_class)
        o = normalize_category(object_class)
        if s == "person" and o == "person":
            return ["near", "follow", "chase", "talk_to", "sing_with"]
        if s == "person":
            return ["near", "on", "hold", "ride"]
        return ["near", "overlap"]


DEFAULT_RELATIONS: List[str] = [
    "left",
    "right",
    "front",
    "behind",
    "above",
    "below",
    "overlap",
    "near",
    "follow",
    "toward",
]

COUPLING_INVERSE: Dict[str, str] = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
    "front": "behind",
    "behind": "front",
}


# Common predicate aliases -> canonical English key used by this script.
# 目的：允许用户/模型使用中文或同义词，但最终写入统一 canonical，方便 presence 评测对齐。
PREDICATE_ALIASES: Dict[str, str] = {
    # left/right
    "左": "left",
    "左边": "left",
    "在左": "left",
    "在左侧": "left",
    "左侧": "left",
    "right": "right",
    "右": "right",
    "右边": "right",
    "在右": "right",
    "在右侧": "right",
    "右侧": "right",
    # above/below
    "上": "above",
    "上面": "above",
    "在上": "above",
    "在上方": "above",
    "上方": "above",
    "下": "below",
    "下面": "below",
    "在下": "below",
    "在下方": "below",
    "下方": "below",
    # front/behind
    "前": "front",
    "前面": "front",
    "在前": "front",
    "在前方": "front",
    "前方": "front",
    "后": "behind",
    "后面": "behind",
    "在后": "behind",
    "在后方": "behind",
    "后方": "behind",
    # overlap/near
    "重叠": "overlap",
    "交叠": "overlap",
    "相交": "overlap",
    "近": "near",
    "靠近": "near",
    "附近": "near",
    # follow/toward
    "跟随": "follow",
    "跟着": "follow",
    "追随": "follow",
    "追": "chase",
    "追赶": "chase",
    "朝向": "toward",
    "朝": "toward",
    "面向": "toward",
    "骑": "ride",
    "乘": "ride",
    "骑乘": "ride",
    "滑滑板": "ride",
    "在上面": "on",
    "坐在": "sit_on",
    "坐在上面": "sit_on",
    "拿": "hold",
    "握": "hold",
    "拿着": "hold",
    "携带": "carry",
    "背": "carry",
    "穿戴": "wear",
    "拥抱": "hug",
    "踢": "kick",
    "推": "push",
    "对话": "talk_to",
    "交谈": "talk_to",
    "注视": "look_at",
    "看着": "look_at",
    "同行": "walk_with",
    "一起走": "walk_with",
    "玩耍": "play_with",
    "对唱": "sing_with",
    "合唱": "sing_with",
    "sing with": "sing_with",
    "speech to": "talk_to",
    "talk to": "talk_to",
}


def _canonical_predicate(p: str) -> str:
    s = str(p or "").strip()
    if not s:
        return ""
    s_low = s.lower()
    return PREDICATE_ALIASES.get(s, PREDICATE_ALIASES.get(s_low, s_low))


def _format_predicate_for_prompt(p: str) -> str:
    """Show predicate with a small hint if it's an alias."""

    raw = str(p or "").strip()
    if not raw:
        return ""
    canon = _canonical_predicate(raw)
    if canon != raw and canon:
        return f"{raw} (={canon})"
    return raw


def _audio_predicates_from_vggsound_label(label: str) -> List[str]:
    s = (label or "").lower()
    if not s or s == "unknown":
        return []

    out: List[str] = []
    if "laugh" in s:
        out.append("play_with")
    if "whisper" in s:
        out.append("talk_to")
    if "speech" in s or "speaking" in s:
        out.append("talk_to")
    if "sing" in s and "bowl" not in s:
        out.append("sing_with")
    if "growl" in s:
        out.append("chase")
    if "bark" in s or "bow-wow" in s:
        out.append("chase")
    if "chirp" in s or "tweet" in s:
        out.append("near")

    # de-dup preserving order
    seen: Set[str] = set()
    uniq: List[str] = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _safe_read_json(path: Path) -> Any:
    if HAS_UTILS_IO:
        return _safe_read_json_utils(path)
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _safe_write_json(path: Path, obj: Any) -> None:
    if HAS_UTILS_IO:
        _safe_write_json_utils(path, obj, indent=2)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if HAS_UTILS_IO:
        yield from _iter_jsonl_utils(path)
        return
    # fallback
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue


def _b64_jpeg_from_bgr(img_bgr: np.ndarray, quality: int = 90) -> str:
    if not HAS_CV2 or cv2 is None:
        raise RuntimeError("opencv-python is required")
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(".jpg", img_bgr, encode_param)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _extract_text_from_dashscope_message(message_content: Any) -> str:
    # DashScope message.content could be str or list[dict]
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        parts: List[str] = []
        for item in message_content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return str(message_content)


def _run_vl_prompt(storyboard_bgr: np.ndarray, model: str, prompt: str, api_key: str) -> str:
    if not HAS_DASHSCOPE or MultiModalConversation is None or dashscope is None:
        raise RuntimeError("dashscope multimodal API unavailable")
    if not api_key.strip():
        raise RuntimeError("missing api_key")

    dashscope.api_key = api_key
    image_b64 = _b64_jpeg_from_bgr(storyboard_bgr, quality=90)
    image_data = f"data:image/jpeg;base64,{image_b64}"

    resp = MultiModalConversation.call(
        model=model,
        messages=[{"role": "user", "content": [{"image": image_data}, {"text": prompt}]}],
    )
    return _extract_text_from_dashscope_message(resp.output.choices[0].message.content)


def _try_parse_json_object(text: str) -> Any:
    s = (text or "").strip()
    if not s:
        return None

    # unwrap fenced ```json ... ```
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2 and lines[0].strip().startswith("```"):
            # drop first fence line
            lines = lines[1:]
            # drop last fence line if present
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    # try salvage: locate first '{' and last '}'
    try:
        i = s.find("{")
        j = s.rfind("}")
        if 0 <= i < j:
            return json.loads(s[i : j + 1])
    except Exception:
        return None
    return None


def _color_for_id(track_id: int) -> Tuple[int, int, int]:
    # deterministic vivid-ish colors (BGR)
    h = (int(track_id) * 97) % 360
    # hsv->bgr
    if not HAS_CV2 or cv2 is None:
        return (0, 255, 255)
    hsv = np.uint8([[[h / 2, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _pick_bbox(track: Dict[str, Any]) -> Optional[List[float]]:
    # For predicted frames, prefer current-frame bbox (often comes from motion model).
    # For observed frames, prefer bbox_observed to keep storyboard aligned with detections.
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


def _draw_tracks(frame_bgr: np.ndarray, tracks: List[Dict[str, Any]], allowed_ids: Set[int]) -> np.ndarray:
    if not HAS_CV2 or cv2 is None:
        return frame_bgr

    img = frame_bgr.copy()
    h, w = img.shape[:2]

    for tr in tracks:
        tid_raw = tr.get("track_id")
        try:
            tid = int(tid_raw)
        except Exception:
            continue
        if tid not in allowed_ids:
            continue

        bbox = _pick_bbox(tr)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w - 1, int(round(x1))))
        y1 = max(0, min(h - 1, int(round(y1))))
        x2 = max(0, min(w - 1, int(round(x2))))
        y2 = max(0, min(h - 1, int(round(y2))))
        if x2 <= x1 or y2 <= y1:
            continue

        color = _color_for_id(tid)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"ID {tid}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return img


def _track_category(track: Dict[str, Any]) -> str:
    for key in ("class_name", "category", "label", "class"):
        value = str(track.get(key, "") or "").strip()
        if value:
            return normalize_category(value)
    return "unknown"


def _tracks_by_id(tracks: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for track in tracks:
        if not isinstance(track, dict):
            continue
        try:
            tid = int(track.get("track_id"))
        except Exception:
            continue
        out[tid] = track
    return out


def _dominant_track_categories(
    track_ids: Sequence[int],
    frames_idx: Sequence[int],
    tracks_for_frame: Dict[int, List[Dict[str, Any]]],
) -> Dict[int, str]:
    votes: Dict[int, Dict[str, int]] = {int(tid): {} for tid in track_ids}
    for fi in frames_idx:
        for tid, track in _tracks_by_id(tracks_for_frame.get(int(fi), [])).items():
            if tid not in votes:
                continue
            category = _track_category(track)
            votes[tid][category] = votes[tid].get(category, 0) + 1
    out: Dict[int, str] = {}
    for tid, counter in votes.items():
        if counter:
            out[tid] = sorted(counter.items(), key=lambda x: (-x[1], x[0]))[0][0]
        else:
            out[tid] = "unknown"
    return out


def _bbox_distance(a: Sequence[float], b: Sequence[float]) -> float:
    ax = (float(a[0]) + float(a[2])) / 2.0
    ay = (float(a[1]) + float(a[3])) / 2.0
    bx = (float(b[0]) + float(b[2])) / 2.0
    by = (float(b[1]) + float(b[3])) / 2.0
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _select_candidate_pairs(
    track_ids: Sequence[int],
    frames_idx: Sequence[int],
    tracks_for_frame: Dict[int, List[Dict[str, Any]]],
    *,
    audio_label: str,
    max_pairs: int,
    explicit_predicates: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    categories = _dominant_track_categories(track_ids, frames_idx, tracks_for_frame)
    pair_stats: Dict[Tuple[int, int], Dict[str, float]] = {}

    for fi in frames_idx:
        frame_tracks = _tracks_by_id(tracks_for_frame.get(int(fi), []))
        for sid, oid in combinations([int(x) for x in track_ids], 2):
            st = frame_tracks.get(sid)
            ot = frame_tracks.get(oid)
            if not st or not ot:
                continue
            sb = _pick_bbox(st)
            ob = _pick_bbox(ot)
            if sb is None or ob is None:
                continue
            key = (sid, oid)
            stats = pair_stats.setdefault(key, {"cooccur": 0.0, "distance": 0.0})
            stats["cooccur"] += 1.0
            stats["distance"] += _bbox_distance(sb, ob)

    candidates: List[Dict[str, Any]] = []
    for sid, oid in combinations([int(x) for x in track_ids], 2):
        stats = pair_stats.get((sid, oid))
        if not stats:
            continue
        cooccur = max(1.0, stats["cooccur"])
        avg_dist = stats["distance"] / cooccur
        for subj, obj in ((sid, oid), (oid, sid)):
            s_cls = categories.get(subj, "unknown")
            o_cls = categories.get(obj, "unknown")
            preds = get_candidate_predicates(s_cls, o_cls, audio_label=audio_label)
            if explicit_predicates:
                preds = [p for p in preds if p in explicit_predicates]
            if not preds:
                continue
            semantic_bonus = 0
            if s_cls == "person" and o_cls != "person":
                semantic_bonus += 2
            if s_cls == "person" and o_cls == "person":
                semantic_bonus += 1
            candidates.append(
                {
                    "subject_track_id": subj,
                    "object_track_id": obj,
                    "subject_category": s_cls,
                    "object_category": o_cls,
                    "candidate_predicates": preds,
                    "score": semantic_bonus + cooccur * 0.1 - avg_dist * 0.0001,
                }
            )

    candidates.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    if max_pairs > 0:
        candidates = candidates[:max_pairs]
    return candidates


def _draw_labeled_bbox(
    img: np.ndarray,
    bbox: Sequence[float],
    crop_origin: Tuple[float, float],
    label: str,
    color: Tuple[int, int, int],
) -> None:
    if not HAS_CV2 or cv2 is None:
        return
    ox, oy = crop_origin
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, int(round(float(bbox[0]) - ox))))
    y1 = max(0, min(h - 1, int(round(float(bbox[1]) - oy))))
    x2 = max(0, min(w - 1, int(round(float(bbox[2]) - ox))))
    y2 = max(0, min(h - 1, int(round(float(bbox[3]) - oy))))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def _make_pair_storyboard(
    frame_samples: List[Tuple[int, np.ndarray, List[Dict[str, Any]]]],
    *,
    subject_id: int,
    object_id: int,
    subject_category: str,
    object_category: str,
    fps: int,
    tile_h: int = 360,
) -> np.ndarray:
    if not HAS_CV2 or cv2 is None:
        raise RuntimeError("opencv-python is required")

    crops: List[np.ndarray] = []
    for frame_idx, frame, tracks in frame_samples:
        frame_tracks = _tracks_by_id(tracks)
        st = frame_tracks.get(int(subject_id))
        ot = frame_tracks.get(int(object_id))
        if not st or not ot:
            continue
        sb = _pick_bbox(st)
        ob = _pick_bbox(ot)
        if sb is None or ob is None:
            continue

        h, w = frame.shape[:2]
        x1 = min(sb[0], ob[0])
        y1 = min(sb[1], ob[1])
        x2 = max(sb[2], ob[2])
        y2 = max(sb[3], ob[3])
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        expand = 0.4
        x1 = max(0.0, x1 - bw * expand)
        y1 = max(0.0, y1 - bh * expand)
        x2 = min(float(w), x2 + bw * expand)
        y2 = min(float(h), y2 + bh * expand)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[int(y1) : int(y2), int(x1) : int(x2)].copy()
        if crop.size == 0:
            continue
        _draw_labeled_bbox(crop, sb, (x1, y1), f"A ID{subject_id} {subject_category}", (0, 0, 255))
        _draw_labeled_bbox(crop, ob, (x1, y1), f"B ID{object_id} {object_category}", (255, 128, 0))
        timestamp = f"frame={frame_idx} t={float(frame_idx) / max(1, fps):.1f}s"
        cv2.putText(crop, timestamp, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        crops.append(crop)

    if not crops:
        raise ValueError("empty pair storyboard")
    return _make_storyboard(crops, tile_h=tile_h)


def _pair_relation_prompt(
    *,
    subject_id: int,
    object_id: int,
    subject_category: str,
    object_category: str,
    candidate_predicates: Sequence[str],
    frame_count: int,
    audio_label: str,
) -> str:
    rel_hint = "\n".join(f"- {p}" for p in candidate_predicates)
    audio_hint = f"\n音频先验：{audio_label}\n" if audio_label else ""
    return f"""你是视频关系标注专家。

画面是同一对轨迹的局部放大 storyboard，按时间从左到右、从上到下排列。
红框 A 是 subject：ID {subject_id}，类别 {subject_category}。
蓝框 B 是 object：ID {object_id}，类别 {object_category}。
共抽取 {frame_count} 帧。{audio_hint}

请只判断 A -> B 是否存在下列候选关系：
{rel_hint}

只输出 JSON，不要输出额外解释，格式如下：
{{
  "relations": [
    {{"predicate": "ride", "confidence": 0.85, "evidence": "A 持续位于 B 上方且同步移动"}}
  ],
  "scene": "一句话描述 A 和 B 的交互"
}}

要求：
- predicate 必须严格从候选关系中选择；
- 不确定的关系不要输出；
- confidence 范围 0 到 1，0.7 以上表示较有把握；
- evidence 必须引用视觉证据或音频先验。
"""


def _make_storyboard(frames: List[np.ndarray], tile_h: int = 360) -> np.ndarray:
    if not HAS_CV2 or cv2 is None:
        raise RuntimeError("opencv-python is required")
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
    tile_w = max_w

    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    for i, im in enumerate(resized):
        r = i // cols
        c = i % cols
        y0 = r * tile_h
        x0 = c * tile_w
        canvas[y0 : y0 + tile_h, x0 : x0 + im.shape[1]] = im

    return canvas


def _load_tracks_for_frames(tracks_jsonl: Path, needed_frames: Set[int]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for obj in _iter_jsonl(tracks_jsonl):
        fr = obj.get("frame")
        try:
            fi = int(fr)
        except Exception:
            continue
        if fi not in needed_frames:
            continue
        tracks = obj.get("tracks")
        if isinstance(tracks, list):
            out[fi] = [t for t in tracks if isinstance(t, dict)]
    return out


def _chunk_list(items: Sequence[str], n: int) -> List[List[str]]:
    if n <= 0:
        return [list(items)]
    out: List[List[str]] = []
    buf: List[str] = []
    for x in items:
        buf.append(x)
        if len(buf) >= n:
            out.append(buf)
            buf = []
    if buf:
        out.append(buf)
    return out


def _progress_path_for_output(output_json: Path) -> Path:
    return output_json.with_suffix(output_json.suffix + ".progress.json")


def _load_progress(progress_path: Path) -> Dict[str, Any]:
    if not progress_path.exists():
        return {"processed_segments_llm": [], "processed_segments_dry_run": [], "errors": []}
    try:
        obj = _safe_read_json(progress_path)
        if isinstance(obj, dict):
            # Backward compatible: older progress only had processed_segments.
            if "processed_segments_llm" not in obj and "processed_segments_dry_run" not in obj:
                legacy = obj.get("processed_segments")
                if isinstance(legacy, list):
                    obj["processed_segments_llm"] = legacy
                else:
                    obj["processed_segments_llm"] = []
                obj["processed_segments_dry_run"] = []

            ps_llm = obj.get("processed_segments_llm")
            if not isinstance(ps_llm, list):
                obj["processed_segments_llm"] = []
            ps_dry = obj.get("processed_segments_dry_run")
            if not isinstance(ps_dry, list):
                obj["processed_segments_dry_run"] = []
            errs = obj.get("errors")
            if not isinstance(errs, list):
                obj["errors"] = []
            return obj
    except Exception:
        pass
    return {"processed_segments_llm": [], "processed_segments_dry_run": [], "errors": []}


def _save_progress(progress_path: Path, obj: Dict[str, Any]) -> None:
    _safe_write_json(progress_path, obj)


def _save_storyboard_image(path: Path, storyboard_bgr: np.ndarray) -> None:
    if not HAS_CV2 or cv2 is None:
        raise RuntimeError("opencv-python is required")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), storyboard_bgr)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {path}")


def _run_vl_prompt_with_retry(
    storyboard_bgr: np.ndarray,
    model: str,
    prompt: str,
    api_key: str,
    retries: int,
    backoff_sec: float,
    sleep_sec: float,
) -> str:
    last_err: Optional[Exception] = None
    for k in range(max(1, int(retries) + 1)):
        try:
            if sleep_sec and sleep_sec > 0:
                time.sleep(float(sleep_sec))
            return _run_vl_prompt(storyboard_bgr, model=model, prompt=prompt, api_key=api_key)
        except Exception as e:
            last_err = e
            if k >= int(retries):
                break
            time.sleep(max(0.0, float(backoff_sec)) * (2**k))
    raise RuntimeError(f"VL request failed after retries: {last_err}")


def _dedup_triples(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[int, str, int], Dict[str, Any]] = {}
    for t in triples:
        try:
            sid = int(t.get("subject_track_id"))
            oid = int(t.get("object_track_id"))
        except Exception:
            continue
        pred = str(t.get("predicate", "")).strip()
        if not pred:
            continue
        key = (sid, pred, oid)
        conf = float(t.get("confidence", 0.0) or 0.0)
        if key not in merged:
            merged[key] = dict(t)
            merged[key]["confidence"] = conf
            continue
        if conf > float(merged[key].get("confidence", 0.0) or 0.0):
            merged[key]["confidence"] = conf
        ev = str(t.get("evidence", "") or "").strip()
        if ev and not str(merged[key].get("evidence", "") or "").strip():
            merged[key]["evidence"] = ev
    out = list(merged.values())
    out.sort(key=lambda x: float(x.get("confidence", 0.0) or 0.0), reverse=True)
    return out


def _apply_coupling(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = {(int(t["subject_track_id"]), str(t["predicate"]), int(t["object_track_id"])) for t in triples if "subject_track_id" in t}
    out = list(triples)
    for t in triples:
        pred = str(t.get("predicate", "")).strip()
        inv = COUPLING_INVERSE.get(pred)
        if not inv:
            continue
        try:
            sid = int(t.get("subject_track_id"))
            oid = int(t.get("object_track_id"))
        except Exception:
            continue
        inv_key = (oid, inv, sid)
        if inv_key in seen:
            continue
        nt = dict(t)
        nt["subject_track_id"] = oid
        nt["object_track_id"] = sid
        nt["predicate"] = inv
        nt["source"] = str(t.get("source", "semi_auto")) + "+coupling"
        out.append(nt)
        seen.add(inv_key)
    return out


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Semi-auto relation labeling from windows + tracks")
    ap.add_argument("--windows_json", type=str, required=True)
    ap.add_argument("--tracks_jsonl", type=str, required=True)
    ap.add_argument("--output_json", type=str, default="pred/relations_pred.json")
    ap.add_argument("--video_id", type=str, default="", help="Override output video_id key")

    ap.add_argument("--api_key", type=str, default="")
    ap.add_argument("--model_vl", type=str, default=str(getattr(config, "API_MODEL", "qwen-vl-max")))

    ap.add_argument("--group_size", type=int, default=3, help="Ask 3 relations per query")
    ap.add_argument("--relations", type=str, default="", help="Comma-separated predicate list; default uses a small spatial set")
    ap.add_argument("--max_windows", type=int, default=0)
    ap.add_argument("--max_frames_per_window", type=int, default=8)
    ap.add_argument("--max_pairs_per_window", type=int, default=8, help="Max directed track pairs queried per window in pair mode")
    ap.add_argument("--pair_storyboard", action="store_true", help="Use pair crops and object-aware predicates instead of global storyboard prompts")
    ap.add_argument("--vggsound_label", type=str, default="")

    ap.add_argument("--resume", action="store_true", help="Resume from progress sidecar (skip processed segments)")
    ap.add_argument(
        "--reset_progress",
        action="store_true",
        help="Ignore existing progress sidecar and start fresh (overwrites the sidecar)",
    )
    ap.add_argument(
        "--save_storyboards_dir",
        type=str,
        default="",
        help="Optional dir to save storyboard images per segment (for manual review/audit)",
    )
    ap.add_argument("--dry_run", action="store_true", help="Only generate/save storyboards; do not call LLM")
    ap.add_argument("--retries", type=int, default=2, help="LLM call retries on failure")
    ap.add_argument("--backoff_sec", type=float, default=1.5, help="Base backoff seconds for retries")
    ap.add_argument("--sleep_sec", type=float, default=0.0, help="Sleep seconds before each LLM call (rate limit)")
    return ap


def main() -> None:
    args = _build_parser().parse_args()

    if not HAS_CV2 or cv2 is None:
        raise SystemExit("ERROR: opencv-python not installed")
    if (not args.dry_run) and (not HAS_DASHSCOPE):
        raise SystemExit("ERROR: dashscope not installed (or use --dry_run)")

    windows_path = Path(args.windows_json).expanduser().resolve()
    tracks_path = Path(args.tracks_jsonl).expanduser().resolve()
    out_path = Path(args.output_json).expanduser().resolve()

    if not windows_path.exists():
        raise SystemExit(f"ERROR: windows_json not found: {windows_path}")
    if not tracks_path.exists():
        raise SystemExit(f"ERROR: tracks_jsonl not found: {tracks_path}")

    api_key = (args.api_key or "").strip() or str(getattr(config, "API_KEY", "") or "").strip()
    if (not args.dry_run) and (not api_key):
        raise SystemExit("ERROR: missing api_key (pass --api_key or set DASHSCOPE_API_KEY)")

    windows_obj = _safe_read_json(windows_path)
    windows = windows_obj.get("windows", []) if isinstance(windows_obj, dict) else []
    if not isinstance(windows, list) or not windows:
        raise SystemExit("ERROR: windows list is empty")

    video_meta = windows_obj.get("video", {}) if isinstance(windows_obj, dict) else {}
    video_path = Path(str(video_meta.get("path", "") or "")).expanduser().resolve()
    if not video_path.exists():
        raise SystemExit(f"ERROR: video.path not found: {video_path}")

    fps = int(video_meta.get("fps", 30) or 30)
    fps = max(1, fps)

    video_id = str(args.video_id or "").strip() or video_path.stem

    rel_list_raw: List[str]
    explicit_relations = bool(str(args.relations or "").strip())
    if str(args.relations or "").strip():
        rel_list_raw = [s.strip() for s in str(args.relations).split(",") if s.strip()]
    else:
        rel_list_raw = list(DEFAULT_RELATIONS)

    # canonicalize + de-dup (preserve order)
    rel_list: List[str] = []
    seen_rel: Set[str] = set()
    for r in rel_list_raw:
        c = _canonical_predicate(r)
        if c and c not in seen_rel:
            rel_list.append(c)
            seen_rel.add(c)

    # audio prior
    vgg_label = str(args.vggsound_label or "").strip()
    audio_rels = _audio_predicates_from_vggsound_label(vgg_label)
    for p in audio_rels:
        c = _canonical_predicate(p)
        if c and c not in seen_rel:
            rel_list.append(c)
            seen_rel.add(c)

    # precompute needed frames (1fps per window)
    needed_frames: Set[int] = set()
    for i, w in enumerate(windows):
        if args.max_windows and i >= int(args.max_windows):
            break
        if not isinstance(w, dict):
            continue
        start = int(w.get("start_frame", 0) or 0)
        end = int(w.get("end_frame", start) or start)
        if end < start:
            start, end = end, start
        f = start
        while f <= end:
            needed_frames.add(int(f))
            f += fps
        needed_frames.add(int(end))

    tracks_for_frame = _load_tracks_for_frames(tracks_path, needed_frames)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"ERROR: failed to open video: {video_path}")

    all_relations: List[Dict[str, Any]] = []
    if args.resume and out_path.exists():
        try:
            existing = _safe_read_json(out_path)
            if isinstance(existing, dict):
                existing_rels = existing.get(video_id, [])
                if isinstance(existing_rels, list):
                    all_relations = [dict(r) for r in existing_rels if isinstance(r, dict)]
                    print(f"RESUME: loaded {len(all_relations)} existing relations from {out_path}")
        except Exception as exc:
            print(f"WARN: failed to load existing output during resume: {exc}")
    save_storyboards_dir = str(args.save_storyboards_dir or "").strip()
    storyboards_dir = Path(save_storyboards_dir).expanduser().resolve() if save_storyboards_dir else None

    progress_path = _progress_path_for_output(out_path)
    if bool(args.reset_progress):
        progress = {"processed_segments_llm": [], "processed_segments_dry_run": [], "errors": []}
        try:
            if progress_path.exists():
                progress_path.unlink()
        except Exception:
            # non-fatal; we'll overwrite later
            pass
    else:
        progress = _load_progress(progress_path)

    processed_segments: Set[int] = set()
    if args.resume:
        try:
            key = "processed_segments_dry_run" if bool(args.dry_run) else "processed_segments_llm"
            processed_segments = {int(x) for x in (progress.get(key) or [])}
        except Exception:
            processed_segments = set()
        if processed_segments and (not out_path.exists()) and (not args.dry_run):
            print("WARN: progress sidecar exists but output_json is missing; reprocessing LLM segments")
            processed_segments = set()

    for i, w in enumerate(windows):
        if args.max_windows and i >= int(args.max_windows):
            break
        if not isinstance(w, dict):
            continue

        segment_id = int(w.get("window_id", i + 1) or i + 1)
        if args.resume and segment_id in processed_segments:
            print(f"SKIP seg={segment_id} (resume)")
            continue
        start_frame = int(w.get("start_frame", 0) or 0)
        end_frame = int(w.get("end_frame", start_frame) or start_frame)
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame

        track_ids = [int(tid) for tid in (w.get("track_ids", []) or []) if str(tid).strip()]
        allowed_ids: Set[int] = set(track_ids)
        if not allowed_ids:
            continue

        # sample 1fps frames
        frames_idx: List[int] = []
        f = start_frame
        while f <= end_frame:
            frames_idx.append(int(f))
            f += fps
        if not frames_idx or frames_idx[-1] != end_frame:
            frames_idx.append(int(end_frame))

        # cap to max_frames_per_window via uniform subsample
        max_k = int(args.max_frames_per_window)
        if max_k > 0 and len(frames_idx) > max_k:
            # uniform pick
            pick = np.linspace(0, len(frames_idx) - 1, num=max_k)
            frames_idx = [frames_idx[int(round(x))] for x in pick]
            frames_idx = sorted(set(frames_idx))

        drawn_frames: List[np.ndarray] = []
        frame_samples: List[Tuple[int, np.ndarray, List[Dict[str, Any]]]] = []
        for fi in frames_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            tracks = tracks_for_frame.get(int(fi), [])
            frame_samples.append((int(fi), frame.copy(), tracks))
            drawn = _draw_tracks(frame, tracks, allowed_ids=allowed_ids)
            drawn_frames.append(drawn)

        if not drawn_frames:
            continue

        storyboard = _make_storyboard(drawn_frames, tile_h=360)

        # optional save storyboard for audit/manual review
        if storyboards_dir is not None:
            sb_path = storyboards_dir / f"seg_{segment_id:04d}_{start_frame}-{end_frame}.jpg"
            try:
                _save_storyboard_image(sb_path, storyboard)
            except Exception as e:
                # non-fatal
                progress.setdefault("errors", []).append(
                    {"segment_id": segment_id, "stage": "save_storyboard", "error": str(e)[:300]}
                )

        if bool(args.dry_run):
            if bool(args.pair_storyboard) and storyboards_dir is not None:
                pair_infos = _select_candidate_pairs(
                    track_ids,
                    frames_idx,
                    tracks_for_frame,
                    audio_label=vgg_label,
                    max_pairs=int(args.max_pairs_per_window),
                    explicit_predicates=set(rel_list) if explicit_relations else None,
                )
                for pair_info in pair_infos:
                    try:
                        pair_sb = _make_pair_storyboard(
                            frame_samples,
                            subject_id=int(pair_info["subject_track_id"]),
                            object_id=int(pair_info["object_track_id"]),
                            subject_category=str(pair_info["subject_category"]),
                            object_category=str(pair_info["object_category"]),
                            fps=fps,
                            tile_h=360,
                        )
                        pair_path = storyboards_dir / (
                            f"seg_{segment_id:04d}_{start_frame}-{end_frame}"
                            f"_A{int(pair_info['subject_track_id'])}_B{int(pair_info['object_track_id'])}.jpg"
                        )
                        _save_storyboard_image(pair_path, pair_sb)
                    except Exception as e:
                        progress.setdefault("errors", []).append(
                            {"segment_id": segment_id, "stage": "save_pair_storyboard", "error": str(e)[:300]}
                        )
            seg_list = progress.setdefault("processed_segments_dry_run", [])
            if int(segment_id) not in set(int(x) for x in seg_list if str(x).strip()):
                seg_list.append(int(segment_id))
            _save_progress(progress_path, progress)
            print(f"OK seg={segment_id} dry_run storyboards_saved={storyboards_dir is not None}")
            continue

        triples_window: List[Dict[str, Any]] = []

        if bool(args.pair_storyboard):
            pair_infos = _select_candidate_pairs(
                track_ids,
                frames_idx,
                tracks_for_frame,
                audio_label=vgg_label,
                max_pairs=int(args.max_pairs_per_window),
                explicit_predicates=set(rel_list) if explicit_relations else None,
            )
            for pair_info in pair_infos:
                sid = int(pair_info["subject_track_id"])
                oid = int(pair_info["object_track_id"])
                s_cls = str(pair_info["subject_category"])
                o_cls = str(pair_info["object_category"])
                candidate_preds = [str(p) for p in pair_info.get("candidate_predicates", []) if str(p).strip()]
                if not candidate_preds:
                    continue
                try:
                    pair_sb = _make_pair_storyboard(
                        frame_samples,
                        subject_id=sid,
                        object_id=oid,
                        subject_category=s_cls,
                        object_category=o_cls,
                        fps=fps,
                        tile_h=360,
                    )
                except Exception as e:
                    progress.setdefault("errors", []).append(
                        {"segment_id": segment_id, "stage": "make_pair_storyboard", "error": str(e)[:300]}
                    )
                    continue

                if storyboards_dir is not None:
                    try:
                        pair_path = storyboards_dir / f"seg_{segment_id:04d}_{start_frame}-{end_frame}_A{sid}_B{oid}.jpg"
                        _save_storyboard_image(pair_path, pair_sb)
                    except Exception as e:
                        progress.setdefault("errors", []).append(
                            {"segment_id": segment_id, "stage": "save_pair_storyboard", "error": str(e)[:300]}
                        )

                prompt = _pair_relation_prompt(
                    subject_id=sid,
                    object_id=oid,
                    subject_category=s_cls,
                    object_category=o_cls,
                    candidate_predicates=candidate_preds,
                    frame_count=len(frame_samples),
                    audio_label=vgg_label,
                )
                text_out = _run_vl_prompt_with_retry(
                    pair_sb,
                    model=str(args.model_vl),
                    prompt=prompt,
                    api_key=api_key,
                    retries=int(args.retries),
                    backoff_sec=float(args.backoff_sec),
                    sleep_sec=float(args.sleep_sec),
                )
                obj = _try_parse_json_object(text_out)
                if not isinstance(obj, dict):
                    continue
                rels = obj.get("relations", obj.get("triples", []))
                if not isinstance(rels, list):
                    continue
                for rel in rels:
                    if not isinstance(rel, dict):
                        continue
                    pred = _canonical_predicate(str(rel.get("predicate", "") or ""))
                    if pred not in candidate_preds:
                        continue
                    try:
                        conf_f = float(rel.get("confidence", 0.0) or 0.0)
                        if not math.isfinite(conf_f):
                            conf_f = 0.0
                    except Exception:
                        conf_f = 0.0
                    triples_window.append(
                        {
                            "subject_track_id": sid,
                            "object_track_id": oid,
                            "subject_category": s_cls,
                            "object_category": o_cls,
                            "predicate": pred,
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "confidence": max(0.0, min(1.0, conf_f)),
                            "source": "semi_auto_pair",
                            "segment_id": segment_id,
                            "evidence": str(rel.get("evidence", "") or "").strip()[:240],
                            "frames": list(frames_idx),
                        }
                    )

            triples_window = _dedup_triples(triples_window)
            triples_window = _apply_coupling(triples_window)
            all_relations.extend(triples_window)
            seg_list = progress.setdefault("processed_segments_llm", [])
            if int(segment_id) not in set(int(x) for x in seg_list if str(x).strip()):
                seg_list.append(int(segment_id))
            _save_progress(progress_path, progress)
            print(f"OK seg={segment_id} pair_relations={len(triples_window)}")
            continue

        ids_hint = "\n".join([f"- ID {tid}" for tid in sorted(allowed_ids)])
        audio_hint = f"\n音频先验(VggSound): {vgg_label}\n" if vgg_label else ""

        for group in _chunk_list(rel_list, int(args.group_size)):
            rel_hint = "\n".join([f"- {p}" for p in group])
            prompt = f"""这是一段视频窗口关键帧拼图（约 1fps 抽取，t1->tN）。图中已画出轨迹框与轨迹ID。{audio_hint}

可用轨迹 ID（仅这些）：\n{ids_hint}\n
你需要在下面这些谓词中判断是否存在关系（只在这些谓词里选，不要发明新谓词）：\n{rel_hint}\n
输出 JSON（不要额外解释），格式如下：
{{
  \"triples\": [
    {{\"subject_id\": 0, \"predicate\": \"left\", \"object_id\": 1, \"confidence\": 0.8, \"evidence\": \"一句话证据\"}}
  ]
}}
要求：
- subject_id/object_id 必须是上述可用ID；
- 对于不成立的谓词，不要输出对应 triple；
- confidence 范围 0~1；
 - predicate 必须严格从上面的列表中选择（使用英文 canonical 形式）；
"""

            text_out = _run_vl_prompt_with_retry(
                storyboard,
                model=str(args.model_vl),
                prompt=prompt,
                api_key=api_key,
                retries=int(args.retries),
                backoff_sec=float(args.backoff_sec),
                sleep_sec=float(args.sleep_sec),
            )
            obj = _try_parse_json_object(text_out)
            if not isinstance(obj, dict):
                continue
            triples = obj.get("triples", [])
            if not isinstance(triples, list):
                continue

            for t in triples:
                if not isinstance(t, dict):
                    continue
                try:
                    sid = int(t.get("subject_id"))
                    oid = int(t.get("object_id"))
                except Exception:
                    continue
                pred = str(t.get("predicate", "")).strip()
                pred = _canonical_predicate(pred)
                if pred not in group:
                    continue
                if sid not in allowed_ids or oid not in allowed_ids:
                    continue

                conf = t.get("confidence", 0.0)
                try:
                    conf_f = float(conf)
                    if not math.isfinite(conf_f):
                        conf_f = 0.0
                except Exception:
                    conf_f = 0.0

                triples_window.append(
                    {
                        "subject_track_id": sid,
                        "object_track_id": oid,
                        "predicate": pred,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "confidence": max(0.0, min(1.0, conf_f)),
                        "source": "semi_auto",
                        "segment_id": segment_id,
                        "evidence": str(t.get("evidence", "") or "").strip()[:200],
                        "frames": list(frames_idx),
                    }
                )

        triples_window = _dedup_triples(triples_window)
        triples_window = _apply_coupling(triples_window)
        all_relations.extend(triples_window)

        seg_list = progress.setdefault("processed_segments_llm", [])
        if int(segment_id) not in set(int(x) for x in seg_list if str(x).strip()):
            seg_list.append(int(segment_id))
        _save_progress(progress_path, progress)
        print(f"OK seg={segment_id} relations={len(triples_window)}")

    cap.release()

    out_obj: Dict[str, Any] = {video_id: all_relations}
    _safe_write_json(out_path, out_obj)

    print("=" * 70)
    print("DONE semi-auto")
    print(f"video_id={video_id}")
    print(f"windows={windows_path}")
    print(f"tracks={tracks_path}")
    print(f"output={out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
