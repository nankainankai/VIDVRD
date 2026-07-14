from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from vidvrd_auto.models.vl_client import VLClient
from vidvrd_auto.prompts.templates import relation_verify_prompt
from vidvrd_auto.relations.object_candidates import GEOMETRY_PREDICATES, get_candidate_predicates, normalize_category
from vidvrd_auto.relations.taxonomy import coupling_inverse, mutex_pairs, predicate_defs
from vidvrd_auto.utils.io import iter_jsonl, read_json, write_json


COUPLING_INVERSE: Dict[str, str] = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
    "front": "behind",
    "behind": "front",
}
COUPLING_INVERSE.update(coupling_inverse())

MUTEX_PAIRS = {
    frozenset(("left", "right")),
    frozenset(("above", "below")),
    frozenset(("front", "behind")),
}
MUTEX_PAIRS.update(mutex_pairs())


def _bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(x) for x in bbox]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(x) for x in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    denom = _bbox_area(a) + _bbox_area(b) - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def _pick_bbox(track: Dict[str, Any]) -> Optional[List[float]]:
    bbox = track.get("bbox_observed", track.get("bbox"))
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return None
    try:
        return [float(x) for x in bbox]
    except Exception:
        return None


def _track_class(track: Dict[str, Any]) -> str:
    for key in ("class_name", "category", "label", "class"):
        value = str(track.get(key, "") or "").strip()
        if value:
            return normalize_category(value)
    return "unknown"


def _load_tracks_by_frame(tracks_jsonl: Path) -> Dict[int, Dict[int, Dict[str, Any]]]:
    out: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for row in iter_jsonl(tracks_jsonl):
        try:
            frame = int(row.get("frame"))
        except Exception:
            continue
        frame_tracks: Dict[int, Dict[str, Any]] = {}
        for t in row.get("tracks", []) or []:
            if not isinstance(t, dict):
                continue
            try:
                tid = int(t.get("track_id"))
            except Exception:
                continue
            frame_tracks[tid] = t
        out[frame] = frame_tracks
    return out


def generate_rule_relations(
    *,
    windows_json: Path,
    tracks_jsonl: Path,
    out_json: Path,
    video_id: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    windows_obj = read_json(windows_json)
    windows = windows_obj.get("windows", []) if isinstance(windows_obj, dict) else []
    tracks_by_frame = _load_tracks_by_frame(tracks_jsonl)

    min_votes = float(config.get("min_vote_ratio", 0.6))
    axis_margin = float(config.get("axis_margin_ratio", 0.08))
    near_ratio = float(config.get("near_distance_ratio", 0.35))
    overlap_thresh = float(config.get("overlap_iou_threshold", 0.05))
    max_pairs_per_window = int(config.get("max_pairs_per_window", 0) or 0)
    max_track_ids_per_window = int(config.get("max_track_ids_per_window", 0) or 0)
    object_aware = bool(config.get("object_aware_candidates", False))
    audio_label = str(config.get("audio_label", "") or "")

    track_class_votes: Dict[int, Counter[str]] = {}
    if object_aware:
        for frame_tracks in tracks_by_frame.values():
            for tid, track in frame_tracks.items():
                track_class_votes.setdefault(tid, Counter())[_track_class(track)] += 1
    track_main_class = {
        tid: votes.most_common(1)[0][0]
        for tid, votes in track_class_votes.items()
        if votes
    }

    relations: List[Dict[str, Any]] = []
    for wi, w in enumerate(windows):
        if not isinstance(w, dict):
            continue
        segment_id = int(w.get("window_id", wi + 1) or wi + 1)
        start = int(w.get("start_frame", 0) or 0)
        end = int(w.get("end_frame", start) or start)
        if end < start:
            start, end = end, start
        track_ids = [int(tid) for tid in (w.get("track_ids", []) or []) if str(tid).strip()]
        if len(track_ids) < 2:
            continue
        if max_track_ids_per_window > 0:
            track_ids = track_ids[:max_track_ids_per_window]

        pair_stats: Dict[Tuple[int, int], Dict[str, int]] = {}
        pair_total: Dict[Tuple[int, int], int] = {}
        for frame in range(start, end + 1):
            frame_tracks = tracks_by_frame.get(frame, {})
            for i, sid in enumerate(track_ids):
                st = frame_tracks.get(sid)
                sb = _pick_bbox(st) if st else None
                if sb is None:
                    continue
                sx, sy = _bbox_center(sb)
                s_area = max(1.0, _bbox_area(sb))
                for oid in track_ids[i + 1 :]:
                    ot = frame_tracks.get(oid)
                    ob = _pick_bbox(ot) if ot else None
                    if ob is None:
                        continue
                    ox, oy = _bbox_center(ob)
                    o_area = max(1.0, _bbox_area(ob))
                    scale = max(s_area, o_area) ** 0.5
                    margin = max(8.0, axis_margin * scale)
                    near_dist = near_ratio * scale
                    dx = sx - ox
                    dy = sy - oy
                    dist = (dx * dx + dy * dy) ** 0.5
                    iou = _bbox_iou(sb, ob)

                    for a, b, adx, ady in ((sid, oid, dx, dy), (oid, sid, -dx, -dy)):
                        key = (a, b)
                        pair_total[key] = pair_total.get(key, 0) + 1
                        stats = pair_stats.setdefault(key, {})
                        if adx < -margin:
                            stats["left"] = stats.get("left", 0) + 1
                        elif adx > margin:
                            stats["right"] = stats.get("right", 0) + 1
                        if ady < -margin:
                            stats["above"] = stats.get("above", 0) + 1
                        elif ady > margin:
                            stats["below"] = stats.get("below", 0) + 1
                        if dist <= near_dist:
                            stats["near"] = stats.get("near", 0) + 1
                        if iou >= overlap_thresh:
                            stats["overlap"] = stats.get("overlap", 0) + 1

        for (sid, oid), stats in sorted(pair_stats.items()):
            total = max(1, pair_total.get((sid, oid), 0))
            for pred, count in sorted(stats.items()):
                ratio = float(count / total)
                if ratio < min_votes:
                    continue
                relations.append(
                    {
                        "subject_track_id": int(sid),
                        "object_track_id": int(oid),
                        "predicate": pred,
                        "start_frame": int(start),
                        "end_frame": int(end),
                        "confidence": round(ratio, 4),
                        "source": "rule_geometry",
                        "segment_id": int(segment_id),
                        "evidence": f"geometry vote {count}/{total} in window {segment_id}",
                    }
                )

        if object_aware:
            candidate_count = 0
            for i, sid in enumerate(track_ids):
                for oid in track_ids[i + 1 :]:
                    for subj, obj in ((sid, oid), (oid, sid)):
                        s_cls = track_main_class.get(subj, "unknown")
                        o_cls = track_main_class.get(obj, "unknown")
                        for pred in get_candidate_predicates(s_cls, o_cls, audio_label=audio_label):
                            if pred in GEOMETRY_PREDICATES:
                                continue
                            relations.append(
                                {
                                    "subject_track_id": int(subj),
                                    "object_track_id": int(obj),
                                    "predicate": pred,
                                    "start_frame": int(start),
                                    "end_frame": int(end),
                                    "confidence": 0.15,
                                    "source": "candidate_object_aware",
                                    "segment_id": int(segment_id),
                                    "subject_category": s_cls,
                                    "object_category": o_cls,
                                    "evidence": f"candidate from {s_cls}-{o_cls} pair",
                                }
                            )
                            candidate_count += 1
                            if max_pairs_per_window > 0 and candidate_count >= max_pairs_per_window:
                                break
                        if max_pairs_per_window > 0 and candidate_count >= max_pairs_per_window:
                            break
                    if max_pairs_per_window > 0 and candidate_count >= max_pairs_per_window:
                        break
                if max_pairs_per_window > 0 and candidate_count >= max_pairs_per_window:
                    break

    out = {video_id: relations}
    write_json(out_json, out)
    return out


def _relation_items(obj: Any, video_id: str) -> List[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return []
    items = obj.get(video_id)
    if isinstance(items, list):
        return [dict(x) for x in items if isinstance(x, dict)]
    out: List[Dict[str, Any]] = []
    for v in obj.values():
        if isinstance(v, list):
            out.extend(dict(x) for x in v if isinstance(x, dict))
    return out


def _rel_key(item: Dict[str, Any]) -> Optional[Tuple[int, str, int, int, int]]:
    try:
        sid = int(item.get("subject_track_id", item.get("subject_id")))
        oid = int(item.get("object_track_id", item.get("object_id")))
        pred = str(item.get("predicate", "") or "").strip().lower()
        start = int(item.get("start_frame", 0) or 0)
        end = int(item.get("end_frame", start) or start)
    except Exception:
        return None
    if not pred:
        return None
    return sid, pred, oid, start, end


def merge_relations(
    *,
    video_id: str,
    relation_jsons: Sequence[Path],
    out_json: Path,
    apply_coupling: bool = True,
) -> Dict[str, Any]:
    merged: Dict[Tuple[int, str, int, int, int], Dict[str, Any]] = {}
    for path in relation_jsons:
        if not path.exists():
            continue
        for item in _relation_items(read_json(path), video_id):
            key = _rel_key(item)
            if key is None:
                continue
            sid, pred, oid, start, end = key
            cur = merged.get(key)
            source = str(item.get("source", "") or "unknown")
            conf = float(item.get("confidence", 0.0) or 0.0)
            if cur is None:
                cur = dict(item)
                cur["subject_track_id"] = sid
                cur["object_track_id"] = oid
                cur["predicate"] = pred
                cur["start_frame"] = start
                cur["end_frame"] = end
                cur["sources"] = [source]
                cur["confidence"] = conf
                merged[key] = cur
            else:
                cur["confidence"] = min(1.0, max(float(cur.get("confidence", 0.0) or 0.0), conf) + 0.05)
                sources = list(cur.get("sources", []))
                if source not in sources:
                    sources.append(source)
                cur["sources"] = sources

    if apply_coupling:
        for key, item in list(merged.items()):
            sid, pred, oid, start, end = key
            inv = COUPLING_INVERSE.get(pred)
            if not inv:
                continue
            inv_key = (oid, inv, sid, start, end)
            if inv_key in merged:
                continue
            coupled = dict(item)
            coupled["subject_track_id"] = oid
            coupled["object_track_id"] = sid
            coupled["predicate"] = inv
            coupled["source"] = "coupling"
            coupled["sources"] = list(dict.fromkeys(list(item.get("sources", [])) + ["coupling"]))
            coupled["evidence"] = f"coupled from {sid}-{pred}-{oid}"
            merged[inv_key] = coupled

    items = [merged[k] for k in sorted(merged.keys())]
    out = {video_id: items}
    write_json(out_json, out)
    return out


def _safe_conf(item: Dict[str, Any]) -> float:
    try:
        return float(item.get("confidence", 0.0) or 0.0)
    except Exception:
        return 0.0


def _collect_track_main_classes(tracks_jsonl: Path) -> Tuple[Dict[int, str], Dict[int, int]]:
    class_votes: Dict[int, Counter[str]] = {}
    frame_counts: Dict[int, int] = {}
    for row in iter_jsonl(tracks_jsonl):
        for t in row.get("tracks", []) or []:
            if not isinstance(t, dict):
                continue
            try:
                tid = int(t.get("track_id"))
            except Exception:
                continue
            frame_counts[tid] = frame_counts.get(tid, 0) + 1
            class_votes.setdefault(tid, Counter())[_track_class(t)] += 1
    main_classes = {
        tid: votes.most_common(1)[0][0]
        for tid, votes in class_votes.items()
        if votes
    }
    return main_classes, frame_counts


def _category_allowed(actual: str, allowed: Sequence[Any]) -> bool:
    if not allowed:
        return True
    actual_norm = normalize_category(actual)
    allowed_norm = {normalize_category(str(x)) for x in allowed if str(x).strip()}
    if not allowed_norm or "any" in allowed_norm:
        return True
    if actual_norm in allowed_norm:
        return True
    if "object" in allowed_norm and actual_norm not in {"person", "unknown"}:
        return True
    if "animal" in allowed_norm and actual_norm in {"dog", "cat", "horse"}:
        return True
    if "vehicle" in allowed_norm and actual_norm in {"bicycle", "car", "skateboard", "surfboard"}:
        return True
    return False


def _category_constraint_check(item: Dict[str, Any], track_classes: Dict[int, str], taxonomy: Dict[str, Dict[str, Any]]) -> bool:
    pred = str(item.get("predicate", "") or "").strip().lower()
    meta = taxonomy.get(pred, {})
    subj_allowed = meta.get("subject_categories", [])
    obj_allowed = meta.get("object_categories", [])
    if not subj_allowed and not obj_allowed:
        return True
    try:
        sid = int(item.get("subject_track_id", item.get("subject_id")))
        oid = int(item.get("object_track_id", item.get("object_id")))
    except Exception:
        return False
    s_cls = str(item.get("subject_category", "") or track_classes.get(sid, "unknown"))
    o_cls = str(item.get("object_category", "") or track_classes.get(oid, "unknown"))
    return _category_allowed(s_cls, subj_allowed) and _category_allowed(o_cls, obj_allowed)


def _resolve_mutex_conflicts(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_pair_span: Dict[Tuple[int, int, int, int], List[Tuple[int, str, float]]] = {}
    for idx, item in enumerate(items):
        key = _rel_key(item)
        if key is None:
            continue
        sid, pred, oid, start, end = key
        by_pair_span.setdefault((sid, oid, start, end), []).append((idx, pred, _safe_conf(item)))

    to_delete: set[int] = set()
    actions: List[Dict[str, Any]] = []
    for (sid, oid, start, end), entries in by_pair_span.items():
        pred_set = {pred for _, pred, _ in entries}
        for pair in MUTEX_PAIRS:
            if not pair.issubset(pred_set):
                continue
            pair_preds = sorted(pair)
            best_pred = max(pair_preds, key=lambda p: max((conf for _, pred, conf in entries if pred == p), default=0.0))
            for idx, pred, conf in entries:
                if pred in pair and pred != best_pred:
                    to_delete.add(idx)
                    actions.append(
                        {
                            "action": "delete",
                            "index": idx,
                            "reason": f"mutex conflict on {sid}-{oid} {start}-{end}; keep {best_pred}",
                            "predicate": pred,
                            "confidence": conf,
                        }
                    )

    return [item for idx, item in enumerate(items) if idx not in to_delete], actions


def _apply_final_actions(items: List[Dict[str, Any]], actions: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    out = [dict(item) for item in items]
    to_delete: set[int] = set()
    applied: List[Dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        try:
            idx = int(action.get("index"))
        except Exception:
            continue
        if idx < 0 or idx >= len(out):
            continue
        op = str(action.get("action", "") or "").strip().lower()
        if op == "delete":
            to_delete.add(idx)
            applied.append(dict(action))
        elif op == "change_predicate":
            new_pred = str(action.get("new_predicate", "") or action.get("predicate", "") or "").strip().lower()
            if new_pred:
                out[idx]["predicate"] = new_pred
                out[idx]["source"] = "verify_corrected"
                applied.append(dict(action))
        elif op == "adjust_span":
            if action.get("start_frame") is not None:
                out[idx]["start_frame"] = int(action["start_frame"])
            if action.get("end_frame") is not None:
                out[idx]["end_frame"] = int(action["end_frame"])
            out[idx]["source"] = "verify_corrected"
            applied.append(dict(action))
    if to_delete:
        out = [item for idx, item in enumerate(out) if idx not in to_delete]
    return out, applied


def verify_relations(
    *,
    video_id: str,
    relations_json: Path,
    tracks_jsonl: Path,
    out_relations_json: Path,
    out_qc_json: Path,
    config: Dict[str, Any],
    storyboards_dir: Path | None = None,
) -> Dict[str, Any]:
    obj = read_json(relations_json) if relations_json.exists() else {video_id: []}
    items = _relation_items(obj, video_id)
    low_conf_thresh = float(config.get("low_confidence_threshold", 0.45))
    strong_review_enabled = bool(config.get("strong_model_review_enabled", False))
    strong_review_threshold = float(config.get("strong_model_confidence_threshold", 0.35))
    strong_model = str(config.get("strong_model", "") or "").strip()
    strong_model_dry_run = bool(config.get("strong_model_dry_run", False))

    source_counts: Dict[str, int] = {}
    low_confidence: List[Dict[str, Any]] = []
    strong_model_review: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []
    by_pair_span: Dict[Tuple[int, int, int, int], List[str]] = {}

    for item in items:
        source = str(item.get("source", "") or ",".join(item.get("sources", []) or []) or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        try:
            conf = float(item.get("confidence", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        if conf < low_conf_thresh:
            low_confidence.append(item)
        if strong_review_enabled and conf < strong_review_threshold:
            strong_model_review.append(
                {
                    "subject_track_id": item.get("subject_track_id", item.get("subject_id")),
                    "predicate": item.get("predicate", ""),
                    "object_track_id": item.get("object_track_id", item.get("object_id")),
                    "start_frame": item.get("start_frame", 0),
                    "end_frame": item.get("end_frame", 0),
                    "confidence": conf,
                    "review_model": strong_model,
                    "state": "pending_external_model_review",
                }
            )
        key = _rel_key(item)
        if key is None:
            continue
        sid, pred, oid, start, end = key
        pair_key = (sid, oid, start, end)
        preds = by_pair_span.setdefault(pair_key, [])
        preds.append(pred)

    for (sid, oid, start, end), preds in by_pair_span.items():
        pred_set = set(preds)
        for pair in MUTEX_PAIRS:
            if pair.issubset(pred_set):
                conflicts.append(
                    {
                        "subject_track_id": sid,
                        "object_track_id": oid,
                        "start_frame": start,
                        "end_frame": end,
                        "predicates": sorted(pair),
                        "type": "mutual_exclusion",
                    }
                )

    final_actions: List[Dict[str, Any]] = []
    if strong_review_enabled and (strong_model_review or conflicts):
        client_cfg = {
            "model": strong_model or "qwen-vl-max",
            "api_key_env": str(config.get("api_key_env", "DASHSCOPE_API_KEY") or "DASHSCOPE_API_KEY"),
            "retries": int(config.get("strong_model_retries", 2) or 2),
            "backoff_sec": float(config.get("strong_model_backoff_sec", 2.0) or 2.0),
            "sleep_sec": float(config.get("strong_model_sleep_sec", 0.0) or 0.0),
            "dry_run": strong_model_dry_run,
        }
        issues = [{"type": "low_confidence", **x} for x in strong_model_review] + [{"type": "conflict", **x} for x in conflicts]
        indexed_items = [dict(item, index=i) for i, item in enumerate(items)]
        image_paths = sorted(storyboards_dir.glob("*.jpg"))[:4] if storyboards_dir and storyboards_dir.exists() else []
        vl_result = VLClient(client_cfg).call(
            prompt=relation_verify_prompt(indexed_items, issues),
            image_paths=image_paths,
            dry_run=strong_model_dry_run,
        )
        if vl_result.ok:
            try:
                parsed = json.loads(vl_result.text)
            except Exception:
                parsed = {}
            if isinstance(parsed.get("actions"), list):
                final_actions = [x for x in parsed["actions"] if isinstance(x, dict)]
        else:
            final_actions = [{"action": "manual_review", "reason": vl_result.error}]

    track_main_classes, track_frames = _collect_track_main_classes(tracks_jsonl)

    final_items = [dict(item) for item in items]
    applied_actions: List[Dict[str, Any]] = []
    auto_actions: List[Dict[str, Any]] = []

    if bool(config.get("apply_actions", False)) and final_actions:
        final_items, applied_actions = _apply_final_actions(final_items, final_actions)

    if bool(config.get("apply_actions", False)):
        final_items, auto_actions = _resolve_mutex_conflicts(final_items)
        applied_actions.extend(auto_actions)

    filtered_by_category = 0
    if bool(config.get("category_constraints_enabled", False)):
        taxonomy = predicate_defs()
        kept: List[Dict[str, Any]] = []
        for item in final_items:
            if _category_constraint_check(item, track_main_classes, taxonomy):
                kept.append(item)
            else:
                filtered_by_category += 1
        final_items = kept

    min_export_conf = float(config.get("min_export_confidence", 0.0) or 0.0)
    filtered_by_confidence = 0
    if min_export_conf > 0:
        kept = []
        for item in final_items:
            if _safe_conf(item) >= min_export_conf:
                kept.append(item)
            else:
                filtered_by_confidence += 1
        final_items = kept

    qc = {
        "video_id": video_id,
        "original_relation_count": len(items),
        "relation_count": len(final_items),
        "source_counts": source_counts,
        "low_confidence_count": len(low_confidence),
        "low_confidence_examples": low_confidence[:20],
        "conflict_count": len(conflicts),
        "conflicts": conflicts[:50],
        "strong_model_review_enabled": strong_review_enabled,
        "strong_model_review_count": len(strong_model_review),
        "strong_model_review_items": strong_model_review[:50],
        "final_actions": final_actions[:100],
        "applied_actions": applied_actions[:100],
        "filtered_by_category_count": filtered_by_category,
        "filtered_by_confidence_count": filtered_by_confidence,
        "track_count": len(track_frames),
        "track_frame_counts": {str(k): v for k, v in sorted(track_frames.items())},
        "track_main_classes": {str(k): v for k, v in sorted(track_main_classes.items())},
        "passed": len(conflicts) == 0 and filtered_by_category == 0,
    }
    write_json(out_relations_json, {video_id: final_items})
    write_json(out_qc_json, qc)
    return qc
