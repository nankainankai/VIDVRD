from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from vidvrd_auto.providers import DashScopeProvider
from vidvrd_auto.prompts.templates import relation_verify_prompt
from vidvrd_auto.relations.object_candidates import normalize_category
from vidvrd_auto.relations.taxonomy import coupling_inverse, mutex_pairs, predicate_defs
from vidvrd_auto.core.ontology import predicate_components
from vidvrd_auto.utils.io import iter_jsonl, read_json, write_json


COUPLING_INVERSE: Dict[str, str] = {
    "left": "right",
    "right": "left",
    "above": "beneath",
    "beneath": "above",
    "front": "behind",
    "behind": "front",
}
COUPLING_INVERSE.update(coupling_inverse())

MUTEX_PAIRS = {
    frozenset(("left", "right")),
    frozenset(("above", "beneath")),
    frozenset(("front", "behind")),
}
MUTEX_PAIRS.update(mutex_pairs())


def _bbox(track: Dict[str, Any]) -> Optional[List[float]]:
    value = track.get("bbox_observed") or track.get("bbox")
    if not isinstance(value, list) or len(value) != 4:
        return None
    return [float(item) for item in value]


def _center(box: Sequence[float]) -> Tuple[float, float]:
    return (float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0


def _area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _area(left) + _area(right) - intersection
    return intersection / union if union > 0 else 0.0


def _box_weight(track: Dict[str, Any], predicted_weight: float, interpolated_weight: float) -> float:
    source = str(track.get("box_source", "observed"))
    if source == "predicted":
        return predicted_weight
    if source == "interpolated":
        return interpolated_weight
    return 1.0


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
    overlap_threshold = float(config.get("overlap_iou_threshold", 0.05))
    predicted_weight = float(config.get("predicted_bbox_weight", 0.25))
    interpolated_weight = float(config.get("interpolated_bbox_weight", 0.75))
    max_pairs_per_window = int(config.get("max_pairs_per_window", 0) or 0)
    max_track_ids_per_window = int(config.get("max_track_ids_per_window", 0) or 0)
    min_observed_pair_frames = max(0, int(config.get("min_observed_pair_frames", 1) or 0))

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

        pair_stats: Dict[Tuple[int, int], Dict[str, float]] = {}
        pair_totals: Dict[Tuple[int, int], float] = {}
        pair_observed: Dict[Tuple[int, int], int] = {}
        pair_count = 0
        for index, sid in enumerate(track_ids):
            for oid in track_ids[index + 1 :]:
                pair_count += 1
                if max_pairs_per_window > 0 and pair_count > max_pairs_per_window:
                    break
                for frame in range(start, end + 1):
                    subject = tracks_by_frame.get(frame, {}).get(sid)
                    obj = tracks_by_frame.get(frame, {}).get(oid)
                    subject_box = _bbox(subject) if subject else None
                    object_box = _bbox(obj) if obj else None
                    if subject_box is None or object_box is None:
                        continue
                    weight = min(
                        _box_weight(subject, predicted_weight, interpolated_weight),
                        _box_weight(obj, predicted_weight, interpolated_weight),
                    )
                    if weight <= 0:
                        continue
                    sx, sy = _center(subject_box)
                    ox, oy = _center(object_box)
                    scale = max(1.0, max(_area(subject_box), _area(object_box)) ** 0.5)
                    margin = max(8.0, axis_margin * scale)
                    distance = math.hypot(sx - ox, sy - oy)
                    overlap = _iou(subject_box, object_box)
                    for left_id, right_id, dx, dy in ((sid, oid, sx - ox, sy - oy), (oid, sid, ox - sx, oy - sy)):
                        key = (left_id, right_id)
                        pair_totals[key] = pair_totals.get(key, 0.0) + weight
                        if subject.get("box_source", "observed") == "observed" and obj.get("box_source", "observed") == "observed":
                            pair_observed[key] = pair_observed.get(key, 0) + 1
                        stats = pair_stats.setdefault(key, {})
                        if dx < -margin:
                            stats["left"] = stats.get("left", 0.0) + weight
                        elif dx > margin:
                            stats["right"] = stats.get("right", 0.0) + weight
                        if dy < -margin:
                            stats["above"] = stats.get("above", 0.0) + weight
                        elif dy > margin:
                            stats["beneath"] = stats.get("beneath", 0.0) + weight
                        if distance <= near_ratio * scale or overlap >= overlap_threshold:
                            stats["next_to"] = stats.get("next_to", 0.0) + weight
            if max_pairs_per_window > 0 and pair_count >= max_pairs_per_window:
                break

        for (subject_id, object_id), stats in sorted(pair_stats.items()):
            if pair_observed.get((subject_id, object_id), 0) < min_observed_pair_frames:
                continue
            total = max(1e-9, pair_totals[(subject_id, object_id)])
            for predicate, support in sorted(stats.items()):
                ratio = support / total
                if ratio < min_votes:
                    continue
                relations.append(
                    {
                        "subject_track_id": subject_id,
                        "predicate": predicate,
                        "object_track_id": object_id,
                        "start_frame": start,
                        "end_frame": end,
                        "confidence": round(ratio, 4),
                        "source": "window_geometry",
                        "predicate_components": predicate_components(predicate),
                        "segment_id": segment_id,
                        "evidence": f"window support {support:.2f}/{total:.2f}",
                    }
                )

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
                            "relation_id": items[idx].get("relation_id", ""),
                            "reason": f"mutex conflict on {sid}-{oid} {start}-{end}; keep {best_pred}",
                            "predicate": pred,
                            "confidence": conf,
                        }
                    )

    return [item for idx, item in enumerate(items) if idx not in to_delete], actions


def _apply_final_actions(items: List[Dict[str, Any]], actions: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    out = [dict(item) for item in items]
    by_id = {str(item.get("relation_id", "")): item for item in out if item.get("relation_id")}
    to_delete: set[str] = set()
    applied: List[Dict[str, Any]] = []
    allowed_predicates = set(predicate_defs())
    for action in actions:
        if not isinstance(action, dict):
            continue
        relation_id = str(action.get("relation_id", "") or "")
        item = by_id.get(relation_id)
        if item is None:
            continue
        op = str(action.get("action", "") or "").strip().lower()
        if op == "delete":
            to_delete.add(relation_id)
            applied.append(dict(action))
        elif op == "keep":
            applied.append(dict(action))
        elif op == "change_predicate":
            new_pred = str(action.get("new_predicate", "") or action.get("predicate", "") or "").strip().lower()
            if new_pred in allowed_predicates:
                item["predicate"] = new_pred
                item["source"] = "verify_corrected"
                applied.append(dict(action))
        elif op == "adjust_span":
            try:
                start = int(action.get("start_frame", item.get("start_frame", 0)))
                end = int(action.get("end_frame", item.get("end_frame", start)))
            except (TypeError, ValueError):
                continue
            if start <= end:
                item["start_frame"], item["end_frame"] = start, end
                item["source"] = "verify_corrected"
                applied.append(dict(action))
    if to_delete:
        out = [item for item in out if str(item.get("relation_id", "")) not in to_delete]
    return out, applied


def _relevant_storyboards(directory: Path | None, items: Sequence[Dict[str, Any]], limit: int) -> List[Path]:
    if directory is None or not directory.exists() or limit <= 0:
        return []
    pairs = set()
    for item in items:
        try:
            pair = sorted((int(item.get("subject_track_id")), int(item.get("object_track_id"))))
        except (TypeError, ValueError):
            continue
        pairs.add(tuple(pair))
    selected: List[Path] = []
    for path in sorted(directory.glob("*.jpg")):
        match = re.search(r"_A(\d+)_B(\d+)\.jpg$", path.name)
        if match and tuple(sorted((int(match.group(1)), int(match.group(2))))) in pairs:
            selected.append(path)
            if len(selected) >= limit:
                break
    return selected


def _add_coupling(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = [dict(item) for item in items]
    existing = {_rel_key(item) for item in output}
    additions: List[Dict[str, Any]] = []
    for item in output:
        key = _rel_key(item)
        if key is None:
            continue
        sid, predicate, oid, start, end = key
        inverse = COUPLING_INVERSE.get(predicate)
        inverse_key = (oid, inverse, sid, start, end) if inverse else None
        if inverse_key is None or inverse_key in existing:
            continue
        coupled = dict(item)
        coupled.update(
            {
                "relation_id": f"{item.get('relation_id', 'relation')}-inv",
                "subject_track_id": oid,
                "object_track_id": sid,
                "predicate": inverse,
                "source": "coupling",
                "sources": list(dict.fromkeys(list(item.get("sources", [])) + ["coupling"])),
                "evidence": f"inverse of {item.get('relation_id', '')}",
            }
        )
        additions.append(coupled)
        existing.add(inverse_key)
    return output + additions


def verify_relations(
    *,
    video_id: str,
    relations_json: Path,
    tracks_jsonl: Path,
    out_relations_json: Path,
    out_qc_json: Path,
    config: Dict[str, Any],
    storyboards_dir: Path | None = None,
    api_key: str = "",
) -> Dict[str, Any]:
    obj = read_json(relations_json) if relations_json.exists() else {video_id: []}
    items = _relation_items(obj, video_id)
    low_conf_thresh = float(config.get("low_confidence_threshold", 0.45))
    strong_review_enabled = bool(config.get("strong_model_review_enabled", False))
    strong_model = str(config.get("strong_model", "") or "").strip()
    strong_model_dry_run = bool(config.get("strong_model_dry_run", False))

    source_counts: Dict[str, int] = {}
    low_confidence: List[Dict[str, Any]] = []
    review_items: List[Dict[str, Any]] = []
    risk_track_ids = {int(value) for value in config.get("risk_track_ids", [])}
    conflicts: List[Dict[str, Any]] = []
    by_pair_span: Dict[Tuple[int, int, int, int], List[str]] = {}

    for index, item in enumerate(items, start=1):
        item.setdefault("relation_id", f"r{index:06d}")
        source = str(item.get("source", "") or ",".join(item.get("sources", []) or []) or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        try:
            conf = float(item.get("confidence", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        if conf < low_conf_thresh:
            low_confidence.append(item)
        try:
            involves_risk = int(item.get("subject_track_id")) in risk_track_ids or int(item.get("object_track_id")) in risk_track_ids
        except (TypeError, ValueError):
            involves_risk = False
        if strong_review_enabled and (conf < low_conf_thresh or involves_risk):
            review_items.append(dict(item, review_reasons=[reason for reason, active in (("low_confidence", conf < low_conf_thresh), ("risky_track", involves_risk)) if active]))
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

    conflict_ids = set()
    for conflict in conflicts:
        for item in items:
            key = _rel_key(item)
            if key and key[0] == conflict["subject_track_id"] and key[2] == conflict["object_track_id"] and key[3] == conflict["start_frame"] and key[4] == conflict["end_frame"] and key[1] in conflict["predicates"]:
                conflict_ids.add(str(item.get("relation_id", "")))
    known_review_ids = {str(item.get("relation_id", "")) for item in review_items}
    review_items.extend(dict(item, review_reasons=["conflict"]) for item in items if str(item.get("relation_id", "")) in conflict_ids - known_review_ids)

    final_actions: List[Dict[str, Any]] = []
    review_result: Dict[str, Any] = {"state": "disabled"}
    if strong_review_enabled and review_items:
        client_cfg = {
            "model": strong_model or "qwen-vl-max",
            "api_key_env": str(config.get("api_key_env", "DASHSCOPE_API_KEY") or "DASHSCOPE_API_KEY"),
            "retries": int(config.get("strong_model_retries", 2) or 2),
            "backoff_sec": float(config.get("strong_model_backoff_sec", 2.0) or 2.0),
            "sleep_sec": float(config.get("strong_model_sleep_sec", 0.0) or 0.0),
            "dry_run": strong_model_dry_run,
        }
        image_paths = _relevant_storyboards(storyboards_dir, review_items, int(config.get("max_review_storyboards", 8) or 8))
        image_pairs = set()
        for path in image_paths:
            match = re.search(r"_A(\d+)_B(\d+)\.jpg$", path.name)
            if match:
                image_pairs.add(tuple(sorted((int(match.group(1)), int(match.group(2))))))
        evidenced_items = []
        deferred_ids = []
        for item in review_items:
            pair = tuple(sorted((int(item.get("subject_track_id")), int(item.get("object_track_id")))))
            if pair in image_pairs:
                evidenced_items.append(item)
            else:
                deferred_ids.append(str(item.get("relation_id", "")))
        issues = [{"relation_id": item["relation_id"], "reasons": item["review_reasons"]} for item in evidenced_items]
        if not evidenced_items:
            review_result = {"state": "skipped_no_visual_evidence", "image_paths": []}
        else:
            vl_result = DashScopeProvider(client_cfg, api_key=api_key).call(
                prompt=relation_verify_prompt(evidenced_items, issues),
                image_paths=image_paths,
                dry_run=strong_model_dry_run,
            )
            review_result = {**vl_result.to_dict(), "state": "succeeded" if vl_result.ok else "failed", "image_paths": [str(path) for path in image_paths], "deferred_relation_ids": deferred_ids}
            if vl_result.ok:
                try:
                    parsed = json.loads(vl_result.text)
                except Exception:
                    parsed = {}
                if isinstance(parsed.get("actions"), list):
                    final_actions = [x for x in parsed["actions"] if isinstance(x, dict)]

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

    if bool(config.get("apply_coupling", True)):
        final_items = _add_coupling(final_items)

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
        "strong_model_review_count": len(review_items),
        "strong_model_review_items": review_items[:50],
        "strong_model_review_result": review_result,
        "risk_track_ids": sorted(risk_track_ids),
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
