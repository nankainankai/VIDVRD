from __future__ import annotations

"""片段关系分类（主包实现）。

由 relation_llm 节点调用：读 windows + tracks + 视频，生成 storyboard，分组询问 VL，输出 relations_llm.json。
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from vidvrd_auto.models.vl_client import VLClient
from vidvrd_auto.relations.predicate_aliases import (
    DEFAULT_RELATION_PREDICATES,
    audio_predicates_from_label,
    canonical_predicate,
)
from vidvrd_auto.relations.storyboard import (
    draw_tracks,
    load_tracks_for_frames,
    make_storyboard,
    save_storyboard_image,
)
from vidvrd_auto.relations.taxonomy import coupling_inverse
from vidvrd_auto.utils.io import read_json, write_json

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("clip_relation requires opencv-python") from e


def _log(log_path: Path | None, msg: str) -> None:
    print(msg)
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _try_parse_json_object(text: str) -> Any:
    s = (text or "").strip()
    if not s:
        return None
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2 and lines[0].strip().startswith("```"):
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        i, j = s.find("{"), s.rfind("}")
        if 0 <= i < j:
            return json.loads(s[i : j + 1])
    except Exception:
        return None
    return None


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
        obj = read_json(progress_path)
        if not isinstance(obj, dict):
            return {"processed_segments_llm": [], "processed_segments_dry_run": [], "errors": []}
        if "processed_segments_llm" not in obj and "processed_segments_dry_run" not in obj:
            legacy = obj.get("processed_segments")
            obj["processed_segments_llm"] = list(legacy) if isinstance(legacy, list) else []
            obj["processed_segments_dry_run"] = []
        for key in ("processed_segments_llm", "processed_segments_dry_run", "errors"):
            if not isinstance(obj.get(key), list):
                obj[key] = []
        return obj
    except Exception:
        return {"processed_segments_llm": [], "processed_segments_dry_run": [], "errors": []}


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
    out = list(merged.values())
    out.sort(key=lambda x: float(x.get("confidence", 0.0) or 0.0), reverse=True)
    return out


def _apply_coupling(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    inv_map = coupling_inverse()
    seen = {
        (int(t["subject_track_id"]), str(t["predicate"]), int(t["object_track_id"]))
        for t in triples
        if "subject_track_id" in t and "object_track_id" in t
    }
    out = list(triples)
    for t in triples:
        pred = str(t.get("predicate", "")).strip()
        inv = inv_map.get(pred)
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


def _build_predicate_list(config: Dict[str, Any]) -> List[str]:
    raw = str(config.get("relations", "") or "").strip()
    if raw:
        rel_list_raw = [s.strip() for s in raw.split(",") if s.strip()]
    else:
        rel_list_raw = list(DEFAULT_RELATION_PREDICATES)
    rel_list: List[str] = []
    seen: Set[str] = set()
    for r in rel_list_raw:
        c = canonical_predicate(r)
        if c and c not in seen:
            rel_list.append(c)
            seen.add(c)
    vgg = str(config.get("vggsound_label", "") or "").strip()
    for p in audio_predicates_from_label(vgg):
        if p not in seen:
            rel_list.append(p)
            seen.add(p)
    return rel_list


def run_clip_relation(
    *,
    windows_json: Path,
    tracks_jsonl: Path,
    out_json: Path,
    storyboards_dir: Path,
    config: Dict[str, Any],
    api_key: str,
    resume: bool,
    dry_run: bool,
    video_id: str,
    log_path: Path,
) -> None:
    """运行片段关系 LLM 分类（主包入口，不再 subprocess 旧脚本）。"""
    if log_path.exists():
        log_path.write_text("", encoding="utf-8")

    windows_path = windows_json.expanduser().resolve()
    tracks_path = tracks_jsonl.expanduser().resolve()
    out_path = out_json.expanduser().resolve()
    storyboards_dir.mkdir(parents=True, exist_ok=True)

    if not windows_path.exists():
        raise FileNotFoundError(f"windows_json not found: {windows_path}")
    if not tracks_path.exists():
        raise FileNotFoundError(f"tracks_jsonl not found: {tracks_path}")

    if not dry_run and not (api_key or "").strip():
        raise RuntimeError("missing api_key (pass --api_key or set DASHSCOPE_API_KEY)")

    windows_obj = read_json(windows_path)
    windows = windows_obj.get("windows", []) if isinstance(windows_obj, dict) else []
    if not isinstance(windows, list) or not windows:
        raise RuntimeError("windows list is empty")

    video_meta = windows_obj.get("video", {}) if isinstance(windows_obj, dict) else {}
    video_path = Path(str(video_meta.get("path", "") or "")).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"video.path not found: {video_path}")

    fps = max(1, int(video_meta.get("fps", 30) or 30))
    rel_list = _build_predicate_list(config)
    group_size = int(config.get("group_size", 3))
    max_windows = int(config.get("max_windows", 0) or 0)
    max_frames_per_window = int(config.get("max_frames_per_window", 8))
    reset_progress = bool(config.get("reset_progress", False))

    vl_client = VLClient(
        {
            "model": str(config.get("api_model", "qwen-vl-max")),
            "retries": int(config.get("retries", 2)),
            "backoff_sec": float(config.get("backoff_sec", 1.5)),
            "sleep_sec": float(config.get("sleep_sec", 0.0)),
        },
        api_key=api_key,
    )

    needed_frames: Set[int] = set()
    for i, w in enumerate(windows):
        if max_windows and i >= max_windows:
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

    tracks_for_frame = load_tracks_for_frames(tracks_path, needed_frames)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    all_relations: List[Dict[str, Any]] = []
    progress_path = _progress_path_for_output(out_path)
    if reset_progress:
        progress: Dict[str, Any] = {"processed_segments_llm": [], "processed_segments_dry_run": [], "errors": []}
        if progress_path.exists():
            try:
                progress_path.unlink()
            except Exception:
                pass
    else:
        progress = _load_progress(progress_path)

    processed_segments: Set[int] = set()
    if resume:
        key = "processed_segments_dry_run" if dry_run else "processed_segments_llm"
        try:
            processed_segments = {int(x) for x in (progress.get(key) or [])}
        except Exception:
            processed_segments = set()

    vgg_label = str(config.get("vggsound_label", "") or "").strip()

    for i, w in enumerate(windows):
        if max_windows and i >= max_windows:
            break
        if not isinstance(w, dict):
            continue

        segment_id = int(w.get("window_id", i + 1) or i + 1)
        if resume and segment_id in processed_segments:
            _log(log_path, f"SKIP seg={segment_id} (resume)")
            continue

        start_frame = int(w.get("start_frame", 0) or 0)
        end_frame = int(w.get("end_frame", start_frame) or start_frame)
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame

        track_ids = [int(tid) for tid in (w.get("track_ids", []) or []) if str(tid).strip()]
        allowed_ids: Set[int] = set(track_ids)
        if not allowed_ids:
            continue

        frames_idx: List[int] = []
        f = start_frame
        while f <= end_frame:
            frames_idx.append(int(f))
            f += fps
        if not frames_idx or frames_idx[-1] != end_frame:
            frames_idx.append(int(end_frame))
        if max_frames_per_window > 0 and len(frames_idx) > max_frames_per_window:
            pick = np.linspace(0, len(frames_idx) - 1, num=max_frames_per_window)
            frames_idx = sorted({frames_idx[int(round(x))] for x in pick})

        drawn_frames: List[np.ndarray] = []
        for fi in frames_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            drawn_frames.append(draw_tracks(frame, tracks_for_frame.get(int(fi), []), allowed_ids))

        if not drawn_frames:
            continue

        storyboard = make_storyboard(drawn_frames, tile_h=360)
        sb_path = storyboards_dir / f"seg_{segment_id:04d}_{start_frame}-{end_frame}.jpg"
        try:
            save_storyboard_image(sb_path, storyboard)
        except Exception as e:
            progress.setdefault("errors", []).append(
                {"segment_id": segment_id, "stage": "save_storyboard", "error": str(e)[:300]}
            )
            _log(log_path, f"WARN seg={segment_id} storyboard save failed: {e}")

        if dry_run:
            seg_list = progress.setdefault("processed_segments_dry_run", [])
            if segment_id not in seg_list:
                seg_list.append(segment_id)
            write_json(progress_path, progress)
            _log(log_path, f"OK seg={segment_id} dry_run")
            continue

        ids_hint = "\n".join([f"- ID {tid}" for tid in sorted(allowed_ids)])
        audio_hint = f"\n音频先验(VggSound): {vgg_label}\n" if vgg_label else ""
        triples_window: List[Dict[str, Any]] = []

        for group in _chunk_list(rel_list, group_size):
            rel_hint = "\n".join([f"- {p}" for p in group])
            prompt = f"""这是一段视频窗口关键帧拼图（约 1fps 抽取，t1->tN）。图中已画出轨迹框与轨迹ID。{audio_hint}

可用轨迹 ID（仅这些）：\n{ids_hint}\n
你需要在下面这些谓词中判断是否存在关系（只在这些谓词里选，不要发明新谓词）：\n{rel_hint}\n
输出 JSON（不要额外解释），格式如下：
{{
  "triples": [
    {{"subject_id": 0, "predicate": "left", "object_id": 1, "confidence": 0.8, "evidence": "一句话证据"}}
  ]
}}
要求：
- subject_id/object_id 必须是上述可用ID；
- 对于不成立的谓词，不要输出对应 triple；
- confidence 范围 0~1；
- predicate 必须严格从上面的列表中选择（使用英文 canonical 形式）；
"""
            vl_result = vl_client.call_bgr(prompt=prompt, image_bgr=storyboard)
            if not vl_result.ok:
                progress.setdefault("errors", []).append(
                    {"segment_id": segment_id, "stage": "vl_call", "error": vl_result.error[:300]}
                )
                _log(log_path, f"WARN seg={segment_id} VL failed: {vl_result.error}")
                continue

            obj = _try_parse_json_object(vl_result.text)
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
                pred = canonical_predicate(str(t.get("predicate", "")).strip())
                if pred not in group:
                    continue
                if sid not in allowed_ids or oid not in allowed_ids:
                    continue
                try:
                    conf_f = float(t.get("confidence", 0.0) or 0.0)
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

        triples_window = _apply_coupling(_dedup_triples(triples_window))
        all_relations.extend(triples_window)

        seg_list = progress.setdefault("processed_segments_llm", [])
        if segment_id not in seg_list:
            seg_list.append(segment_id)
        write_json(progress_path, progress)
        _log(log_path, f"OK seg={segment_id} relations={len(triples_window)}")

    cap.release()
    write_json(out_path, {video_id: all_relations})
    _log(log_path, f"DONE clip_relation video_id={video_id} count={len(all_relations)} output={out_path}")
