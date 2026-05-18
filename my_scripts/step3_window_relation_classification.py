"""Step3: 关系分类（新主链专用：windows + tracks + full video）。

输入：
- windows.json（Step2 输出）
- tracks_full.jsonl（Step2 输出）
- 原视频（从 windows.json.video.path 读取）

输出：
- segment_descriptions/seg_XXXX.txt
- segment_relations/seg_XXXX.method2.json
- segment_relations/seg_XXXX.method3.json
- segment_relations/seg_XXXX.final.json
- relations_candidates.json
- relations_final.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import config

try:
    from utils_io import safe_read_json as _safe_read_json_utils
    from utils_io import safe_write_json as _safe_write_json_utils
    from utils_io import safe_write_text as _safe_write_text_utils

    HAS_UTILS_IO = True
except Exception:
    HAS_UTILS_IO = False

try:
    import dashscope
    from dashscope import MultiModalConversation
    try:
        from dashscope import Generation
    except Exception:
        Generation = None
    HAS_DASHSCOPE = True
except Exception:
    dashscope = None
    MultiModalConversation = None
    Generation = None
    HAS_DASHSCOPE = False


def _reconfigure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def _safe_write_text(path: Path, text: str) -> None:
    if HAS_UTILS_IO:
        _safe_write_text_utils(path, text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def _b64_jpeg_from_bgr(img_bgr: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return base64.b64encode(buf).decode("utf-8")


def _extract_text_from_dashscope_message(message_content: Any) -> str:
    if message_content is None:
        return ""
    if isinstance(message_content, list) and message_content:
        first = message_content[0]
        if isinstance(first, dict):
            return str(first.get("text", first)).strip()
        return str(first).strip()
    if isinstance(message_content, dict):
        return str(message_content.get("text", message_content)).strip()
    return str(message_content).strip()


def _try_parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    # Handle common fenced outputs: ```json\n{...}\n```
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


def _validate_triples_payload(
    payload: Dict[str, Any],
    focus_ids: Optional[Tuple[int, int]] = None,
    allowed_ids: Optional[List[int]] = None,
) -> Tuple[bool, str]:
    triples = payload.get("triples")
    if not isinstance(triples, list):
        return False, "triples_not_list"

    allowed_set = set(int(x) for x in (allowed_ids or [])) if allowed_ids is not None else None

    for t in triples:
        if not isinstance(t, dict):
            return False, "triple_not_object"
        try:
            sid = int(t.get("subject_id"))
        except Exception:
            return False, "invalid_subject_id"

        oid_raw = t.get("object_id", None)
        oid: Optional[int] = None
        if oid_raw is not None:
            try:
                oid = int(oid_raw)
            except Exception:
                oid = None

        pred = str(t.get("predicate", "")).strip()
        if not pred:
            return False, "missing_predicate"

        try:
            conf = float(t.get("confidence", 0.0))
        except Exception:
            return False, "invalid_confidence"
        if conf < 0.0 or conf > 1.0:
            return False, "confidence_out_of_range"

        ev = str(t.get("evidence", "")).strip()
        if not ev:
            return False, "missing_evidence"

        if allowed_set is not None:
            if sid not in allowed_set:
                return False, "subject_not_allowed"
            if oid is not None and oid not in allowed_set:
                return False, "object_not_allowed"

        if focus_ids is not None:
            a, b = int(focus_ids[0]), int(focus_ids[1])
            if sid not in {a, b}:
                return False, "focus_subject_violation"
            if oid is not None and oid not in {a, b}:
                return False, "focus_object_violation"

    return True, "ok"


def _color_for_id(tid: int) -> Tuple[int, int, int]:
    # deterministic bright-ish colors (BGR)
    colors = [
        (0, 255, 255),
        (255, 128, 0),
        (0, 255, 0),
        (255, 0, 255),
        (0, 128, 255),
        (255, 255, 0),
    ]
    return colors[int(tid) % len(colors)]


def _bbox_for_vis(track: Dict[str, Any]) -> Optional[List[float]]:
    if not isinstance(track, dict):
        return None
    if bool(track.get("is_predicted", False)):
        bb2 = track.get("bbox_observed")
        if isinstance(bb2, (list, tuple)) and len(bb2) == 4:
            return [float(bb2[0]), float(bb2[1]), float(bb2[2]), float(bb2[3])]
    bb = track.get("bbox")
    if isinstance(bb, (list, tuple)) and len(bb) == 4:
        return [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]
    return None


def _overlay_tracks_on_frame(frame_bgr: np.ndarray, tracks: List[Dict[str, Any]], id_to_label: Dict[int, str]) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    for t in tracks or []:
        try:
            tid = int(t.get("track_id"))
        except Exception:
            continue
        bbox = _bbox_for_vis(t)
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        x1, y1, x2, y2 = bbox
        x1i = _clamp_int(x1, 0, w - 1)
        y1i = _clamp_int(y1, 0, h - 1)
        x2i = _clamp_int(x2, 0, w - 1)
        y2i = _clamp_int(y2, 0, h - 1)
        if x2i <= x1i or y2i <= y1i:
            continue
        color = _color_for_id(tid)
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), color, 3)
        label = f"ID {tid}: {id_to_label.get(tid, str(t.get('class_name', 'unknown')))}"
        cv2.putText(out, label, (x1i + 6, max(22, y1i + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, label, (x1i + 6, max(22, y1i + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _build_window_storyboard_with_tracks(
    video_path: Path,
    meta_frames: List[Dict[str, Any]],
    start_frame: int,
    track_id_label_map: Dict[int, str],
    sample_count: int,
) -> np.ndarray:
    total = len(meta_frames)
    indices = _frame_indices(total, sample_count)
    frames: List[np.ndarray] = []
    for local_idx in indices:
        global_frame = int(start_frame) + int(local_idx)
        fr = _read_frame_at(video_path, global_frame)
        if fr is None:
            continue
        m = meta_frames[local_idx] if 0 <= local_idx < len(meta_frames) else {}
        tracks = m.get("tracks", []) or []
        frames.append(_overlay_tracks_on_frame(fr, tracks, track_id_label_map))
    if not frames:
        raise RuntimeError("window has no sampled frames")
    return build_storyboard(frames, StoryboardSpec())


def _load_tracks_jsonl(path: Path) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as f:
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
            tracks = row.get("tracks", [])
            if not isinstance(tracks, list):
                tracks = []
            out[frame] = tracks
    return out


def _is_person_label(label: str) -> bool:
    s = (label or "").strip().lower()
    return ("person" in s) or (s in {"people", "man", "woman", "boy", "girl"})


def _frame_indices(total: int, sample_count: int) -> List[int]:
    if total <= 0:
        return []
    if sample_count <= 1:
        return [max(0, total // 2)]
    return np.linspace(0, total - 1, sample_count).round().astype(int).tolist()


def _read_frame_at(video_path: Path, idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, fr = cap.read()
    cap.release()
    if not ok or fr is None:
        return None
    return fr


@dataclass
class StoryboardSpec:
    cols: int = 4
    rows: int = 2
    tile_w: int = 320
    tile_h: int = 180
    pad: int = 8


def build_storyboard(frames_bgr: List[np.ndarray], spec: StoryboardSpec) -> np.ndarray:
    if not frames_bgr:
        raise RuntimeError("No frames for storyboard")

    max_tiles = spec.cols * spec.rows
    frames_bgr = frames_bgr[:max_tiles]

    canvas_w = spec.cols * spec.tile_w + (spec.cols + 1) * spec.pad
    canvas_h = spec.rows * spec.tile_h + (spec.rows + 1) * spec.pad
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for i, fr in enumerate(frames_bgr):
        r = i // spec.cols
        c = i % spec.cols
        if r >= spec.rows:
            break

        tile = cv2.resize(fr, (spec.tile_w, spec.tile_h))
        x0 = spec.pad + c * (spec.tile_w + spec.pad)
        y0 = spec.pad + r * (spec.tile_h + spec.pad)
        canvas[y0 : y0 + spec.tile_h, x0 : x0 + spec.tile_w] = tile

        label = f"t{i + 1}"
        cv2.putText(canvas, label, (x0 + 8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (x0 + 8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas


def _build_window_meta_frames(
    frame_tracks: Dict[int, List[Dict[str, Any]]],
    start_frame: int,
    end_frame: int,
    fps: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for frame in range(int(start_frame), int(end_frame) + 1):
        out.append(
            {
                "frame": int(frame),
                "timestamp": float(frame / max(1, fps)),
                "tracks": frame_tracks.get(frame, []) or [],
            }
        )
    return out


def _infer_track_classes(meta_frames: List[Dict[str, Any]]) -> Dict[int, str]:
    classes: Dict[int, str] = {}
    for fr in meta_frames:
        for t in fr.get("tracks", []) or []:
            try:
                tid = int(t.get("track_id"))
            except Exception:
                continue
            classes[tid] = str(t.get("class_name", "unknown"))
    return classes


def _sample_window_frames(video_path: Path, start_frame: int, end_frame: int, sample_count: int) -> List[np.ndarray]:
    total = max(0, int(end_frame) - int(start_frame) + 1)
    idx_local = _frame_indices(total, sample_count)
    frames: List[np.ndarray] = []
    for li in idx_local:
        gi = int(start_frame) + int(li)
        fr = _read_frame_at(video_path, gi)
        if fr is not None:
            frames.append(fr)
    return frames


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(round(float(v))))))


def _highlight_pair_on_frame(
    frame_bgr: np.ndarray,
    track_a: Optional[Dict[str, Any]],
    track_b: Optional[Dict[str, Any]],
    label_a: str,
    label_b: str,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    base = (frame_bgr.astype(np.float32) * 0.35).clip(0, 255).astype(np.uint8)

    def draw_one(track: Optional[Dict[str, Any]], color: Tuple[int, int, int], label: str) -> None:
        if not track:
            return
        bbox = _bbox_for_vis(track)
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
        cv2.rectangle(base, (x1i, y1i), (x2i, y2i), color, 4)
        cv2.putText(base, label, (x1i + 6, max(22, y1i + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    draw_one(track_a, (0, 255, 255), label_a)
    draw_one(track_b, (255, 128, 0), label_b)
    return base


def _build_focused_storyboard(
    video_path: Path,
    meta_frames: List[Dict[str, Any]],
    start_frame: int,
    track_id_a: int,
    track_id_b: int,
    label_a: str,
    label_b: str,
    sample_count: int,
) -> np.ndarray:
    total = len(meta_frames)
    indices = _frame_indices(total, sample_count)
    frames: List[np.ndarray] = []

    for local_idx in indices:
        global_frame = int(start_frame) + int(local_idx)
        fr = _read_frame_at(video_path, global_frame)
        if fr is None:
            continue
        m = meta_frames[local_idx] if 0 <= local_idx < len(meta_frames) else {}
        tracks = m.get("tracks", []) or []
        ta = None
        tb = None
        for t in tracks:
            try:
                tid = int(t.get("track_id"))
            except Exception:
                continue
            if tid == int(track_id_a):
                ta = t
            elif tid == int(track_id_b):
                tb = t

        frames.append(
            _highlight_pair_on_frame(
                fr,
                ta,
                tb,
                label_a=f"ID {track_id_a}: {label_a}",
                label_b=f"ID {track_id_b}: {label_b}",
            )
        )

    if not frames:
        raise RuntimeError("Failed to build focused storyboard")
    return build_storyboard(frames, StoryboardSpec())


def _run_vl_prompt(storyboard_bgr: np.ndarray, model: str, prompt: str, api_key: str) -> str:
    if not HAS_DASHSCOPE or MultiModalConversation is None:
        raise RuntimeError("dashscope multimodal API unavailable")
    dashscope.api_key = api_key

    image_b64 = _b64_jpeg_from_bgr(storyboard_bgr, quality=90)
    image_data = f"data:image/jpeg;base64,{image_b64}"

    resp = MultiModalConversation.call(
        model=model,
        messages=[{"role": "user", "content": [{"image": image_data}, {"text": prompt}]}],
    )
    if getattr(resp, "status_code", None) != 200:
        raise RuntimeError(f"API error: HTTP {getattr(resp, 'status_code', 'unknown')}")

    return _extract_text_from_dashscope_message(resp.output.choices[0].message.content)


def _extract_triples_from_text(segment_id: int, description: str, model: str, api_key: str, allowed_ids: List[int]) -> Dict[str, Any]:
    if not HAS_DASHSCOPE:
        raise RuntimeError("dashscope API unavailable")

    allowed_str = ", ".join(str(int(x)) for x in (allowed_ids or []))

    prompt = f"""你是关系抽取器。给定视频片段描述，输出严格 JSON，不要额外文字。

重要约束：
1) 只允许使用这些轨迹 ID：[{allowed_str}]。
2) subject_id 必须是上述 ID 之一。
3) object_id 只能是上述 ID 之一；若关系对象不是轨迹实体（如树、草丛、道路等背景物体），则 object_id 必须为 null，并在 object_label 用中文短语描述该物体。
4) 不要发明新的 ID（如 ID1/ID2/ID3 这种）。

JSON 格式：
{{
  "segment_id": {segment_id},
  "triples": [
    {{
      "subject_id": <int>,
      "predicate": "<string>",
      "object_id": <int|null>,
      "object_label": "<string|null>",
      "confidence": <number>,
      "evidence": "<string>"
    }}
  ]
}}

描述：
{description}
"""

    dashscope.api_key = api_key

    if Generation is not None:
        resp = Generation.call(model=model, prompt=prompt)
        if getattr(resp, "status_code", None) != 200:
            raise RuntimeError(f"API error: HTTP {getattr(resp, 'status_code', 'unknown')}")
        if hasattr(resp, "output") and isinstance(resp.output, dict) and "text" in resp.output:
            text_out = str(resp.output.get("text", ""))
        elif hasattr(resp, "output") and hasattr(resp.output, "text"):
            text_out = str(resp.output.text)
        else:
            text_out = str(getattr(resp, "output", ""))
    else:
        if MultiModalConversation is None:
            raise RuntimeError("No text API available")
        resp = MultiModalConversation.call(model=model, messages=[{"role": "user", "content": [{"text": prompt}]}])
        if getattr(resp, "status_code", None) != 200:
            raise RuntimeError(f"API error: HTTP {getattr(resp, 'status_code', 'unknown')}")
        text_out = _extract_text_from_dashscope_message(resp.output.choices[0].message.content)

    parsed = _try_parse_json_object((text_out or "").strip())
    if parsed is None:
        raise RuntimeError("LLM output is not valid JSON")
    ok, reason = _validate_triples_payload(parsed, allowed_ids=allowed_ids)
    if not ok:
        raise RuntimeError(f"invalid triples payload: {reason}")
    return parsed


def _extract_pair_category_triples(
    segment_id: int,
    track_id_a: int,
    track_id_b: int,
    category: str,
    storyboard_bgr: np.ndarray,
    model_vl: str,
    api_key: str,
) -> Dict[str, Any]:
    category_prompts = {
        "static_position": "只抽取静态位置关系（left/right/overlap/in_front_of/behind）",
        "dynamic_position": "只抽取动态位置关系（toward/away/follow/cross）",
        "static_action": "只抽取静态动作关系（hold/touch/sit_on/stand_on）",
        "dynamic_action": "只抽取动态动作关系（lift/push/pull/pass_to/chase）",
    }
    c_prompt = category_prompts.get(category, "只抽取该类别关系")

    prompt = f"""你是关系分类器。图中仅关注 ID {track_id_a} 与 ID {track_id_b}。
{c_prompt}。
输出严格 JSON，不要额外文字：
{{
  "segment_id": {segment_id},
  "triples": [
    {{
      "subject_id": <int>,
      "predicate": "<string>",
      "object_id": <int|null>,
      "object_label": "<string|null>",
      "confidence": <number>,
      "evidence": "<string>"
    }}
  ]
}}
"""

    text_out = _run_vl_prompt(storyboard_bgr, model=model_vl, prompt=prompt, api_key=api_key)
    parsed = _try_parse_json_object((text_out or "").strip())
    if parsed is None:
        raise RuntimeError("pair-category output is not JSON")
    ok, reason = _validate_triples_payload(parsed, focus_ids=(track_id_a, track_id_b))
    if not ok:
        raise RuntimeError(f"invalid pair-category payload: {reason}")
    return parsed


def _pair_candidates(track_classes: Dict[int, str], track_ids: List[int]) -> List[Tuple[int, int]]:
    valid_ids = sorted({int(tid) for tid in track_ids})
    person_ids = [tid for tid in valid_ids if _is_person_label(track_classes.get(tid, ""))]
    object_ids = [tid for tid in valid_ids if tid not in set(person_ids)]

    pairs: List[Tuple[int, int]] = []
    for i in range(len(person_ids)):
        for j in range(i + 1, len(person_ids)):
            pairs.append((person_ids[i], person_ids[j]))
    for pid in person_ids:
        for oid in object_ids:
            pairs.append((pid, oid))

    if not pairs:
        for i in range(len(valid_ids)):
            for j in range(i + 1, len(valid_ids)):
                pairs.append((valid_ids[i], valid_ids[j]))

    return pairs


def _fuse_candidates(method2: List[Dict[str, Any]], method3: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[int, str, Optional[int]], Dict[str, Any]] = {}

    def add(src: str, t: Dict[str, Any]) -> None:
        try:
            sid = int(t.get("subject_id"))
        except Exception:
            return
        pred = str(t.get("predicate", "")).strip()
        if not pred:
            return

        oid_raw = t.get("object_id", None)
        oid: Optional[int] = None
        if oid_raw is not None:
            try:
                oid = int(oid_raw)
            except Exception:
                oid = None

        key = (sid, pred, oid)
        conf = float(t.get("confidence", 0.0) or 0.0)
        if key not in merged:
            merged[key] = {
                "subject_id": sid,
                "predicate": pred,
                "object_id": oid,
                "object_label": t.get("object_label"),
                "confidence": conf,
                "sources": [src],
                "evidence": [str(t.get("evidence", ""))[:200]],
            }
            return

        item = merged[key]
        item["confidence"] = max(float(item.get("confidence", 0.0)), conf)
        if src not in item["sources"]:
            item["sources"].append(src)
        ev = str(t.get("evidence", ""))[:200]
        if ev and ev not in item["evidence"]:
            item["evidence"].append(ev)

    for t in method2:
        add("method2", t)
    for t in method3:
        add("method3", t)

    out: List[Dict[str, Any]] = []
    for item in merged.values():
        bonus = 0.1 if len(item.get("sources", [])) >= 2 else 0.0
        score = min(1.0, float(item.get("confidence", 0.0)) + bonus)
        out.append(
            {
                "subject_id": item["subject_id"],
                "predicate": item["predicate"],
                "object_id": item["object_id"],
                "object_label": item.get("object_label"),
                "confidence": float(score),
                "evidence": " | ".join(item.get("evidence", [])[:2]),
                "sources": item.get("sources", []),
            }
        )

    out.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step3: relation classification from windows + tracks")
    parser.add_argument("--windows_json", type=str, default="", help="Step2输出 windows.json；不传则从OUTPUT_DIR推断")
    parser.add_argument("--tracks_jsonl", type=str, default="", help="Step2输出 tracks_full.jsonl；不传则从OUTPUT_DIR推断")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--model_vl", type=str, default="qwen-vl-max")
    parser.add_argument("--model_text", type=str, default="qwen-max")
    parser.add_argument("--sample_frames", type=int, default=8)
    parser.add_argument("--max_windows", type=int, default=0)
    parser.add_argument("--max_pairs", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    _reconfigure_stdout()
    args = _build_parser().parse_args()

    config_out_dir = Path(str(getattr(config, "OUTPUT_DIR", "C:/video_output"))).expanduser().resolve()
    default_io_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir or "").strip() else config_out_dir

    windows_path = (
        Path(args.windows_json).expanduser().resolve()
        if str(args.windows_json or "").strip()
        else (default_io_dir / str(getattr(config, "WINDOWS_JSON_NAME", "windows.json"))).resolve()
    )
    tracks_path = (
        Path(args.tracks_jsonl).expanduser().resolve()
        if str(args.tracks_jsonl or "").strip()
        else (default_io_dir / str(getattr(config, "FULL_TRACKS_JSONL_NAME", "tracks_full.jsonl"))).resolve()
    )
    if not windows_path.exists():
        print(f"ERROR: windows.json not found: {windows_path}")
        print("TIP: 传入 --windows_json，或先确认 config.OUTPUT_DIR 下存在 windows.json")
        return
    if not tracks_path.exists():
        print(f"ERROR: tracks_full.jsonl not found: {tracks_path}")
        print("TIP: 传入 --tracks_jsonl，或先确认 config.OUTPUT_DIR 下存在 tracks_full.jsonl")
        return

    windows_obj = _safe_read_json(windows_path)
    windows = windows_obj.get("windows", []) if isinstance(windows_obj, dict) else []
    if not isinstance(windows, list) or not windows:
        print("ERROR: windows list is empty")
        return

    video_meta = windows_obj.get("video", {}) if isinstance(windows_obj, dict) else {}
    video_path = Path(str(video_meta.get("path", "") or "")).expanduser().resolve()
    if not video_path.exists():
        print(f"ERROR: video.path not found: {video_path}")
        return

    fps = int(video_meta.get("fps", 30) or 30)

    api_key = (args.api_key or "").strip()
    if not api_key:
        try:
            import config as my_config  # type: ignore
            api_key = str(getattr(my_config, "API_KEY", "") or "").strip()
        except Exception:
            api_key = ""

    if not api_key:
        print("ERROR: missing API key")
        return
    if not HAS_DASHSCOPE:
        print("ERROR: dashscope not installed")
        return

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else windows_path.parent
    _ensure_dir(output_dir)
    desc_dir = output_dir / "segment_descriptions"
    rel_dir = output_dir / "segment_relations"
    _ensure_dir(desc_dir)
    _ensure_dir(rel_dir)

    frame_tracks = _load_tracks_jsonl(tracks_path)

    print("=" * 70)
    print("Step3: 关系分类")
    print(f"windows={windows_path}")
    print(f"tracks={tracks_path}")
    print(f"video={video_path}")
    print("=" * 70)

    index_candidates: List[Dict[str, Any]] = []
    index_final: List[Dict[str, Any]] = []

    for i, w in enumerate(windows):
        if args.max_windows and i >= int(args.max_windows):
            break
        if not isinstance(w, dict):
            continue

        segment_id = int(w.get("window_id", i + 1) or i + 1)
        start_frame = int(w.get("start_frame", 0) or 0)
        end_frame = int(w.get("end_frame", start_frame) or start_frame)
        track_ids = [int(tid) for tid in (w.get("track_ids", []) or [])]

        method2_path = rel_dir / f"seg_{segment_id:04d}.method2.json"
        method3_path = rel_dir / f"seg_{segment_id:04d}.method3.json"
        final_path = rel_dir / f"seg_{segment_id:04d}.final.json"
        desc_path = desc_dir / f"seg_{segment_id:04d}.txt"

        if not args.force and final_path.exists() and method2_path.exists() and method3_path.exists() and desc_path.exists():
            try:
                m2 = _safe_read_json(method2_path).get("triples", [])
                m3 = _safe_read_json(method3_path).get("triples", [])
                ff = _safe_read_json(final_path).get("triples", [])
            except Exception:
                m2, m3, ff = [], [], []
            index_candidates.append({"segment_id": segment_id, "method2": m2, "method3": m3})
            index_final.append({"segment_id": segment_id, "triples": ff})
            continue

        try:
            # Method2: window storyboard -> description -> triples
            meta_frames = _build_window_meta_frames(frame_tracks, start_frame, end_frame, fps=max(1, fps))
            track_classes = _infer_track_classes(meta_frames)
            id_label_map = {int(tid): str(track_classes.get(int(tid), "unknown")) for tid in track_ids}

            if not track_ids:
                description = "(该窗口无可用轨迹ID，跳过关系抽取)"
                _safe_write_text(desc_path, description)
                method2_triples = []
                _safe_write_json(method2_path, {"segment_id": segment_id, "triples": method2_triples, "source": "method2"})
            else:
                storyboard = _build_window_storyboard_with_tracks(
                    video_path=video_path,
                    meta_frames=meta_frames,
                    start_frame=start_frame,
                    track_id_label_map=id_label_map,
                    sample_count=int(args.sample_frames),
                )

                ids_hint = "\n".join([f"- ID {tid}: {id_label_map.get(int(tid), 'unknown')}" for tid in track_ids])
                desc_prompt = f"""这是一段视频窗口关键帧拼图（t1->tN），图中已标注轨迹框与轨迹 ID。

可用轨迹 ID（仅这些）：
{ids_hint}

请描述这些轨迹实体之间的关系变化。
要求：
1) 使用 6-10 句中文；
2) 只能使用上述数值 ID（例如“ID 0”“ID 1”），不要发明 ID1/ID2/ID3；
3) 若涉及背景物体（树、草丛、道路等），直接用中文名词描述，不要给它分配 ID；
4) 不要输出 JSON。"""
                description = _run_vl_prompt(storyboard, model=args.model_vl, prompt=desc_prompt, api_key=api_key)
                _safe_write_text(desc_path, description)

                method2 = _extract_triples_from_text(
                    segment_id=segment_id,
                    description=description,
                    model=args.model_text,
                    api_key=api_key,
                    allowed_ids=track_ids,
                )
                method2_triples = method2.get("triples", []) if isinstance(method2, dict) else []
                _safe_write_json(method2_path, {"segment_id": segment_id, "triples": method2_triples, "source": "method2"})

            # Method3: pair-focused + 4 relation categories
            pairs = _pair_candidates(track_classes, track_ids)
            if args.max_pairs and len(pairs) > int(args.max_pairs):
                pairs = pairs[: int(args.max_pairs)]

            method3_triples: List[Dict[str, Any]] = []
            for (a, b) in pairs:
                label_a = track_classes.get(a, "unknown")
                label_b = track_classes.get(b, "unknown")
                focused = _build_focused_storyboard(
                    video_path=video_path,
                    meta_frames=meta_frames,
                    start_frame=start_frame,
                    track_id_a=int(a),
                    track_id_b=int(b),
                    label_a=label_a,
                    label_b=label_b,
                    sample_count=int(args.sample_frames),
                )

                for cat in ["static_position", "dynamic_position", "static_action", "dynamic_action"]:
                    try:
                        cat_obj = _extract_pair_category_triples(
                            segment_id=segment_id,
                            track_id_a=int(a),
                            track_id_b=int(b),
                            category=cat,
                            storyboard_bgr=focused,
                            model_vl=args.model_vl,
                            api_key=api_key,
                        )
                        triples = cat_obj.get("triples", []) if isinstance(cat_obj, dict) else []
                        for t in triples:
                            if isinstance(t, dict):
                                t["category"] = cat
                                method3_triples.append(t)
                    except Exception:
                        continue

            _safe_write_json(method3_path, {"segment_id": segment_id, "triples": method3_triples, "source": "method3"})

            fused = _fuse_candidates(method2_triples, method3_triples)
            _safe_write_json(final_path, {"segment_id": segment_id, "triples": fused, "source": "fused"})

            index_candidates.append({"segment_id": segment_id, "method2": method2_triples, "method3": method3_triples})
            index_final.append({"segment_id": segment_id, "triples": fused})
            print(f"OK: seg_{segment_id:04d} (m2={len(method2_triples)}, m3={len(method3_triples)}, final={len(fused)})")
        except Exception as e:
            index_candidates.append({"segment_id": segment_id, "error": str(e)})
            index_final.append({"segment_id": segment_id, "error": str(e)})
            print(f"ERROR: seg_{segment_id:04d}: {e}")

    candidates_path = output_dir / "relations_candidates.json"
    final_index_path = output_dir / "relations_final.json"
    _safe_write_json(
        candidates_path,
        {
            "windows_json": str(windows_path).replace("\\", "/"),
            "tracks_jsonl": str(tracks_path).replace("\\", "/"),
            "items": index_candidates,
        },
    )
    _safe_write_json(
        final_index_path,
        {
            "windows_json": str(windows_path).replace("\\", "/"),
            "tracks_jsonl": str(tracks_path).replace("\\", "/"),
            "items": index_final,
        },
    )

    print("=" * 70)
    print("DONE: Step3 完成")
    print(f"candidates={candidates_path}")
    print(f"final={final_index_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
