"""Instance-pair relation presence on fixed clips, separate from official mAP."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from vidvrd_auto.core.schema import relation_span
from vidvrd_auto.evaluation.diagnostic.track_aligned import align_trajectories


def _summary(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def evaluate_presence(
    gold_relations: Mapping, gold_trajectories: Mapping,
    pred_relations: Mapping, pred_trajectories: Mapping,
    video_ids: Sequence[str], *, clips: Sequence[dict] | None = None,
    track_iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """One directed (subject instance, predicate, object instance) per clip.

    Clips use half-open [start_frame, end_frame). Relation artifacts retain their
    declared span convention. Any nonempty temporal overlap means presence.
    Category-aware, one-to-one tube matching uses only Gold annotated frames.
    Without an independently annotated clip manifest this is a diagnostic: sparse
    Gold does not establish that unmatched predictions are incorrect.
    """
    if not 0 < track_iou_threshold <= 1:
        raise ValueError("track_iou_threshold must be in (0, 1]")
    inferred = clips is None
    if inferred:
        clips = [dict(video_id=vid, start_frame=start, end_frame=end, annotation_complete=False)
                 for vid in video_ids
                 for start, end in sorted({(relation_span(rel).start_frame, relation_span(rel).end_frame)
                                           for rel in gold_relations.get(vid, [])})]
    selected = []
    seen = set()
    for clip in clips:
        vid = str(clip["video_id"])
        if vid not in video_ids:
            continue
        start, end = int(clip["start_frame"]), int(clip["end_frame"])
        if start < 0 or end <= start:
            raise ValueError(f"invalid half-open clip span: {clip}")
        key = (vid, start, end)
        if key in seen:
            raise ValueError(f"duplicate clip: {key}")
        seen.add(key)
        selected.append((vid, start, end, clip.get("annotation_complete") is True))
    if not selected and not inferred:
        raise ValueError("no fixed clips selected for Presence evaluation")
    counts = [0, 0, 0]
    per_video = defaultdict(lambda: [0, 0, 0])
    details = []
    for vid, start, end, complete in selected:
        def crop(tracks):
            result = []
            for track in tracks:
                trajectory = {str(frame): box for frame, box in track.get("trajectory", {}).items()
                              if start <= int(frame) < end}
                if trajectory:
                    result.append(dict(track, trajectory=trajectory))
            return result
        gt_tracks = crop(gold_trajectories.get(vid, []))
        prediction_tracks = crop(pred_trajectories.get(vid, []))
        mapping, matches = align_trajectories(gt_tracks, prediction_tracks, track_iou_threshold)
        def keys(relations, predicted=False):
            result = set()
            for rel in relations:
                span = relation_span(rel)
                rel_start, rel_end = span.start_frame, span.end_frame
                if max(start, rel_start) >= min(end, rel_end):
                    continue
                sid, oid = int(rel["subject_track_id"]), int(rel["object_track_id"])
                if predicted:
                    subject = ("gold", mapping[sid]) if sid in mapping else ("unmatched", sid)
                    obj = ("gold", mapping[oid]) if oid in mapping else ("unmatched", oid)
                else:
                    subject, obj = ("gold", sid), ("gold", oid)
                result.add((subject, str(rel["predicate"]), obj))
            return result
        expected = keys(gold_relations.get(vid, []))
        actual = keys(pred_relations.get(vid, []), True)
        local = (len(expected & actual), len(actual - expected), len(expected - actual))
        counts = [total + value for total, value in zip(counts, local)]
        per_video[vid] = [total + value for total, value in zip(per_video[vid], local)]
        details.append(dict(video_id=vid, start_frame=start, end_frame=end,
                            annotation_complete=complete, **_summary(*local), track_matches=matches))
    return {
        "metric": "instance_pair_presence_v1", "span_convention": "half_open",
        "scope": "gold_annotated_segments_diagnostic" if inferred else "fixed_clip_manifest",
        "annotation_complete": bool(selected) and all(item[3] for item in selected),
        "interpretation": "Official mAP is unchanged. FP/precision are provisional when annotations are incomplete.",
        "track_iou_threshold": track_iou_threshold, "clip_count": len(details),
        "micro": _summary(*counts),
        "per_video": {vid: _summary(*values) for vid, values in per_video.items()},
        "clips": details,
    }
