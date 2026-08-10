from __future__ import annotations

"""Internal trajectory-aligned diagnostics; these are not official VidVRD metrics."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from vidvrd_auto.core.ontology import normalize_object, predicate_splits
from vidvrd_auto.core.schema import relation_span
from vidvrd_auto.utils.io import read_json, write_json


def _box_iou(left: Sequence[float], right: Sequence[float]) -> Tuple[float, float]:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection, left_area + right_area - intersection


def trajectory_viou(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    left_boxes = left.get("trajectory", {})
    right_boxes = right.get("trajectory", {})
    frames = set(left_boxes) | set(right_boxes)
    intersection = 0.0
    union = 0.0
    for frame in frames:
        left_box, right_box = left_boxes.get(frame), right_boxes.get(frame)
        if left_box is not None and right_box is not None:
            current_intersection, current_union = _box_iou(left_box, right_box)
            intersection += current_intersection
            union += current_union
        else:
            box = left_box if left_box is not None else right_box
            union += max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return intersection / union if union > 0 else 0.0


def align_trajectories(
    gold: Sequence[Dict[str, Any]], pred: Sequence[Dict[str, Any]], threshold: float
) -> Tuple[Dict[int, int], List[Dict[str, Any]]]:
    if not gold or not pred:
        return {}, []
    matrix = np.zeros((len(pred), len(gold)), dtype=float)
    for pred_index, pred_track in enumerate(pred):
        pred_category = normalize_object(str(pred_track.get("category", "unknown")))
        for gold_index, gold_track in enumerate(gold):
            if pred_category != normalize_object(str(gold_track.get("category", "unknown"))):
                continue
            matrix[pred_index, gold_index] = trajectory_viou(pred_track, gold_track)
    pred_indices, gold_indices = linear_sum_assignment(-matrix)
    mapping: Dict[int, int] = {}
    matches: List[Dict[str, Any]] = []
    for pred_index, gold_index in zip(pred_indices.tolist(), gold_indices.tolist()):
        score = float(matrix[pred_index, gold_index])
        if score < threshold:
            continue
        pred_id = int(pred[pred_index]["track_id"])
        gold_id = int(gold[gold_index]["track_id"])
        mapping[pred_id] = gold_id
        matches.append({"pred_track_id": pred_id, "gold_track_id": gold_id, "viou": round(score, 6)})
    return mapping, matches


def _relation_items(obj: Any) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for video_id, values in obj.items() if isinstance(obj, dict) else []:
        for item in values if isinstance(values, list) else []:
            if isinstance(item, dict):
                output.append(dict(item, video_id=str(video_id)))
    return output


def _confidence(item: Dict[str, Any]) -> float:
    for field in ("ranking_score", "agent_score", "rule_support", "confidence"):
        if item.get(field) is not None:
            return float(item[field])
    return 0.0


def _tiou(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    left_span, right_span = relation_span(left), relation_span(right)
    left_start, left_end = left_span.start_frame, left_span.end_frame
    right_start, right_end = right_span.start_frame, right_span.end_frame
    intersection = max(0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    return intersection / union if union > 0 else 0.0


def _mapped_ids(item: Dict[str, Any], mappings: Dict[str, Dict[int, int]]) -> Tuple[int, int] | None:
    try:
        video = str(item["video_id"])
        subject = mappings.get(video, {}).get(int(item.get("subject_track_id", item.get("subject_id"))))
        obj = mappings.get(video, {}).get(int(item.get("object_track_id", item.get("object_id"))))
    except (KeyError, TypeError, ValueError):
        return None
    return (subject, obj) if subject is not None and obj is not None else None


def _relation_match(
    predictions: Sequence[Dict[str, Any]],
    gold: Sequence[Dict[str, Any]],
    mappings: Dict[str, Dict[int, int]],
    tiou_threshold: float,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    ordered = sorted(predictions, key=_confidence, reverse=True)
    used: set[int] = set()
    labels: List[int] = []
    errors: List[Dict[str, Any]] = []
    for prediction in ordered:
        mapped = _mapped_ids(prediction, mappings)
        best_index, best_tiou = -1, 0.0
        if mapped is not None:
            for index, target in enumerate(gold):
                if index in used or str(target["video_id"]) != str(prediction["video_id"]):
                    continue
                if str(target.get("predicate")) != str(prediction.get("predicate")):
                    continue
                if mapped != (int(target["subject_track_id"]), int(target["object_track_id"])):
                    continue
                score = _tiou(prediction, target)
                if score >= tiou_threshold and score > best_tiou:
                    best_index, best_tiou = index, score
        matched = best_index >= 0
        labels.append(int(matched))
        if matched:
            used.add(best_index)
        else:
            errors.append(dict(prediction, mapped_track_ids=mapped))
    false_negatives = [dict(item) for index, item in enumerate(gold) if index not in used]
    return labels, errors + [dict(item, error_type="false_negative") for item in false_negatives]


def _summary(labels: Sequence[int], gold_count: int) -> Dict[str, float | int]:
    tp = int(sum(labels))
    fp = len(labels) - tp
    fn = max(0, gold_count - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ap = sum(sum(labels[: index + 1]) / (index + 1) for index, value in enumerate(labels) if value) / gold_count if gold_count else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "ap": ap}


def _recall_at_k(
    predictions: List[Dict[str, Any]], gold: List[Dict[str, Any]], mappings: Dict[str, Dict[int, int]], threshold: float, k: int
) -> float:
    selected: List[Dict[str, Any]] = []
    by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in predictions:
        by_video[str(item["video_id"])].append(item)
    for values in by_video.values():
        selected.extend(sorted(values, key=_confidence, reverse=True)[:k])
    labels, _ = _relation_match(selected, gold, mappings, threshold)
    return sum(labels) / len(gold) if gold else 0.0


def _tagging_precision_at_k(
    predictions: List[Dict[str, Any]], gold: List[Dict[str, Any]], mappings: Dict[str, Dict[int, int]], k: int
) -> float:
    gold_tags: Dict[str, set[Tuple[int, str, int]]] = defaultdict(set)
    for item in gold:
        gold_tags[str(item["video_id"])].add((int(item["subject_track_id"]), str(item["predicate"]), int(item["object_track_id"])))
    scores: List[float] = []
    by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in predictions:
        by_video[str(item["video_id"])].append(item)
    for video, values in by_video.items():
        tags: List[Tuple[int, str, int]] = []
        for item in sorted(values, key=_confidence, reverse=True):
            mapped = _mapped_ids(item, mappings)
            if mapped is None:
                continue
            tag = (mapped[0], str(item.get("predicate", "")), mapped[1])
            if tag not in tags:
                tags.append(tag)
            if len(tags) >= k:
                break
        if tags:
            scores.append(sum(tag in gold_tags.get(video, set()) for tag in tags) / len(tags))
    return sum(scores) / len(scores) if scores else 0.0


def run_track_aligned_diagnostic(
    *,
    gold_relations: Path,
    gold_trajectories: Path,
    pred_relations: Path,
    pred_trajectories: Path,
    report_path: Path,
    metrics_path: Path,
    track_viou_threshold: float = 0.3,
    relation_tiou_threshold: float = 0.5,
    video_ids: Sequence[str] | None = None,
) -> Dict[str, Any]:
    gold_tracks_obj = read_json(gold_trajectories)
    pred_tracks_obj = read_json(pred_trajectories)
    gold_relations_obj = read_json(gold_relations)
    pred_relations_obj = read_json(pred_relations)
    gold_tracks = gold_tracks_obj if isinstance(gold_tracks_obj, dict) else {}
    pred_tracks = pred_tracks_obj if isinstance(pred_tracks_obj, dict) else {}
    prediction_videos = set(video_ids or ())
    if not prediction_videos:
        prediction_videos.update(pred_tracks)
        if isinstance(pred_relations_obj, dict):
            prediction_videos.update(str(video) for video in pred_relations_obj)
    if not prediction_videos:
        raise ValueError("prediction files do not contain any video IDs")

    # A partial benchmark must only score videos processed in this run. Full
    # evaluation is unchanged when the prediction set contains all Gold videos.
    gold_tracks = {video: values for video, values in gold_tracks.items() if video in prediction_videos}
    pred_tracks = {video: values for video, values in pred_tracks.items() if video in prediction_videos}
    mappings: Dict[str, Dict[int, int]] = {}
    match_rows: Dict[str, List[Dict[str, Any]]] = {}
    for video in sorted(set(gold_tracks) | set(pred_tracks)):
        mapping, matches = align_trajectories(
            gold_tracks.get(video, []), pred_tracks.get(video, []), track_viou_threshold
        )
        mappings[video] = mapping
        match_rows[video] = matches

    scoped_gold_relations = (
        {video: values for video, values in gold_relations_obj.items() if video in prediction_videos}
        if isinstance(gold_relations_obj, dict)
        else {}
    )
    gold = _relation_items(scoped_gold_relations)
    predictions = _relation_items(pred_relations_obj)
    labels, errors = _relation_match(predictions, gold, mappings, relation_tiou_threshold)
    overall = _summary(labels, len(gold))
    splits = predicate_splits()
    per_predicate: Dict[str, Dict[str, Any]] = {}
    for predicate in sorted({str(item["predicate"]) for item in gold + predictions}):
        predicate_gold = [item for item in gold if item["predicate"] == predicate]
        predicate_pred = [item for item in predictions if item["predicate"] == predicate]
        predicate_labels, _ = _relation_match(predicate_pred, predicate_gold, mappings, relation_tiou_threshold)
        per_predicate[predicate] = {**_summary(predicate_labels, len(predicate_gold)), "split": splits.get(predicate, "unknown")}

    evaluated_predicates = [values for values in per_predicate.values() if int(values["tp"]) + int(values["fn"]) > 0]
    mean_ap = (
        sum(float(values["ap"]) for values in evaluated_predicates) / len(evaluated_predicates)
        if evaluated_predicates
        else 0.0
    )

    split_metrics: Dict[str, Dict[str, Any]] = {}
    for split in ("base", "novel"):
        split_gold = [item for item in gold if splits.get(str(item["predicate"])) == split]
        split_pred = [item for item in predictions if splits.get(str(item["predicate"])) == split]
        split_labels, _ = _relation_match(split_pred, split_gold, mappings, relation_tiou_threshold)
        split_metrics[split] = _summary(split_labels, len(split_gold))

    matched_tracks = sum(len(values) for values in match_rows.values())
    gold_track_count = sum(len(values) for values in gold_tracks.values())
    pred_track_count = sum(len(values) for values in pred_tracks.values())
    metrics: Dict[str, Any] = {
        "evaluator": "diagnostic_track_aligned_v1",
        "metric_namespace": "diagnostic",
        "evaluated_videos": sorted(prediction_videos),
        "evaluated_video_count": len(prediction_videos),
        "thresholds": {"track_viou": track_viou_threshold, "relation_tiou": relation_tiou_threshold},
        "diagnostic_tracks": {
            "gold": gold_track_count,
            "predicted": pred_track_count,
            "matched": matched_tracks,
            "gold_recall": matched_tracks / gold_track_count if gold_track_count else 0.0,
            "mean_viou": sum(item["viou"] for values in match_rows.values() for item in values) / matched_tracks if matched_tracks else 0.0,
            "matches": match_rows,
        },
        "diagnostic_relation_summary": overall,
        "diagnostic_predicate_macro_ap": mean_ap,
        "diagnostic_splits": split_metrics,
        "diagnostic_per_predicate": per_predicate,
        "diagnostic_recall_at_50": _recall_at_k(predictions, gold, mappings, relation_tiou_threshold, 50),
        "diagnostic_recall_at_100": _recall_at_k(predictions, gold, mappings, relation_tiou_threshold, 100),
        "diagnostic_tagging_precision_at_1": _tagging_precision_at_k(predictions, gold, mappings, 1),
        "diagnostic_tagging_precision_at_5": _tagging_precision_at_k(predictions, gold, mappings, 5),
        "diagnostic_tagging_precision_at_10": _tagging_precision_at_k(predictions, gold, mappings, 10),
        "diagnostic_error_examples": errors[:100],
    }
    write_json(metrics_path, metrics)
    lines = [
        "# Track-aligned diagnostic (not official VidVRD metrics)",
        "",
        f"- Track matching: vIoU >= {track_viou_threshold:.2f}",
        f"- Relation matching: tIoU >= {relation_tiou_threshold:.2f}",
        f"- Evaluator: `diagnostic_track_aligned_v1`",
        f"- Tracks: {matched_tracks}/{gold_track_count} matched, mean vIoU={metrics['diagnostic_tracks']['mean_viou']:.4f}",
        "",
        "## Overall",
        "",
        f"TP={overall['tp']}, FP={overall['fp']}, FN={overall['fn']}, Precision={overall['precision']:.4f}, Recall={overall['recall']:.4f}, F1={overall['f1']:.4f}",
        f"Diagnostic micro AP={overall['ap']:.4f}, diagnostic predicate macro AP={mean_ap:.4f}",
        f"Diagnostic recall@50={metrics['diagnostic_recall_at_50']:.4f}, recall@100={metrics['diagnostic_recall_at_100']:.4f}",
        f"Diagnostic tagging P@1={metrics['diagnostic_tagging_precision_at_1']:.4f}, P@5={metrics['diagnostic_tagging_precision_at_5']:.4f}, P@10={metrics['diagnostic_tagging_precision_at_10']:.4f}",
        "",
        "## Base / novel",
        "",
        "| Split | TP | FP | FN | Precision | Recall | F1 | AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, values in split_metrics.items():
        lines.append(f"| {split} | {values['tp']} | {values['fp']} | {values['fn']} | {values['precision']:.4f} | {values['recall']:.4f} | {values['f1']:.4f} | {values['ap']:.4f} |")
    lines.extend(["", "## Per predicate", "", "| Predicate | Split | TP | FP | FN | P | R | F1 | AP |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for predicate, values in per_predicate.items():
        lines.append(f"| {predicate} | {values['split']} | {values['tp']} | {values['fp']} | {values['fn']} | {values['precision']:.4f} | {values['recall']:.4f} | {values['f1']:.4f} | {values['ap']:.4f} |")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics
