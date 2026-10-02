from __future__ import annotations

"""Final relation and trajectory export."""

import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from vidvrd_auto.core.ontology import load_ontology
from vidvrd_auto.core.schema import serialize_relation_artifact
from vidvrd_auto.core.schema import relation_span
from vidvrd_auto.utils.io import iter_jsonl, read_json, write_json


def tracks_to_trajectories(tracks_path: Path, video_id: str, out_path: Path) -> None:
    tracks: Dict[int, Dict[str, Any]] = {}
    for row in iter_jsonl(tracks_path):
        try:
            frame = int(row["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        for item in row.get("tracks", []):
            if not isinstance(item, dict):
                continue
            box = item.get("bbox")
            if not isinstance(box, list) or len(box) != 4:
                continue
            track_id = int(item["track_id"])
            track = tracks.setdefault(
                track_id,
                {
                    "track_id": track_id,
                    "category": str(item.get("class_name", "unknown")),
                    "trajectory": {},
                    "box_sources": {},
                },
            )
            track["trajectory"][str(frame)] = [float(value) for value in box]
            track["box_sources"][str(frame)] = str(item.get("box_source", "observed"))
    write_json(out_path, {video_id: [tracks[key] for key in sorted(tracks)]})


def export_video_outputs(
    *,
    verified_path: Path,
    qc_path: Path,
    tracks_path: Path,
    video_id: str,
    relations_path: Path,
    trajectories_path: Path,
    config: Dict[str, Any] | None = None,
) -> None:
    relations_path.parent.mkdir(parents=True, exist_ok=True)
    verified = read_json(verified_path)
    values = verified.get(video_id, []) if isinstance(verified, dict) else []
    export_config = config or {}
    if bool(export_config.get("sliding_proposals_enabled", False)):
        values = sliding_relation_proposals(
            values,
            window_frames=int(export_config.get("window_frames", 30) or 30),
            stride_frames=int(export_config.get("stride_frames", 15) or 15),
        )
    if bool(export_config.get("action_spatial_composition_enabled", False)):
        values = compose_action_spatial_relations(
            values,
            min_overlap_frames=int(export_config.get("composition_min_overlap_frames", 15) or 15),
            score_coefficient=float(export_config.get("composition_score_coefficient", 0.55) or 0.55),
        )
    if bool(export_config.get("deduplicate_proposals_enabled", False)):
        values = deduplicate_relation_proposals(
            values,
            temporal_iou_threshold=float(export_config.get("deduplicate_temporal_iou", 0.7) or 0.7),
        )
    write_json(
        relations_path,
        {
            video_id: [
                serialize_relation_artifact(item)
                for item in values
                if isinstance(item, dict)
            ]
        },
    )
    if qc_path.exists():
        shutil.copyfile(qc_path, relations_path.parent / "qc.json")
    tracks_to_trajectories(tracks_path, video_id, trajectories_path)


def sliding_relation_proposals(
    relations: Iterable[Dict[str, Any]], *, window_frames: int = 30, stride_frames: int = 15
) -> list[Dict[str, Any]]:
    """Turn a long relation tube into bounded, overlapping prediction proposals.

    Proposal boundaries depend only on the prediction span. Ground-truth timestamps
    are never read, so the same deterministic policy can be used at test time.
    """

    if window_frames <= 0 or stride_frames <= 0:
        raise ValueError("relation proposal window and stride must be positive")
    output: list[Dict[str, Any]] = []
    for relation in relations:
        span = relation_span(relation)
        duration = span.end_frame - span.start_frame
        starts = {span.start_frame}
        aligned = ((span.start_frame + stride_frames - 1) // stride_frames) * stride_frames
        starts.update(range(aligned, span.end_frame, stride_frames))
        proposal_index = 0
        for start in sorted(starts):
            end = min(span.end_frame, start + window_frames)
            if end - start < min(stride_frames, duration):
                continue
            item = dict(relation)
            item.update(start_frame=start, end_frame=end, span_convention="half_open")
            evidence = item.get("evidence_frames")
            if isinstance(evidence, list):
                item["evidence_frames"] = [frame for frame in evidence if start <= int(frame) < end]
            item["parent_relation_id"] = str(relation.get("relation_id", ""))
            item["temporal_proposal"] = {
                "policy": "prediction_sliding_window_v1",
                "window_frames": window_frames,
                "stride_frames": stride_frames,
                "proposal_index": proposal_index,
            }
            if relation.get("relation_id"):
                item["relation_id"] = f"{relation['relation_id']}:p{proposal_index:03d}"
            output.append(item)
            proposal_index += 1
    return output


def _relation_score(item: Dict[str, Any]) -> float:
    for key in ("ranking_score", "agent_score", "rule_support", "confidence"):
        if item.get(key) is not None:
            return max(0.0, min(1.0, float(item[key])))
    return 0.0


def deduplicate_relation_proposals(
    relations: Iterable[Dict[str, Any]], *, temporal_iou_threshold: float = 0.7
) -> list[Dict[str, Any]]:
    """Remove redundant high-overlap proposals without merging distinct track pairs.

    The criterion uses only predicted identities, labels, spans, and scores.
    Annotation timestamps and categories never enter this export operation.
    """

    if not 0.0 < temporal_iou_threshold <= 1.0:
        raise ValueError("temporal IoU threshold must be in (0, 1]")
    selected: list[Dict[str, Any]] = []
    for item in sorted((dict(value) for value in relations), key=_relation_score, reverse=True):
        span = relation_span(item)
        duplicate = False
        for previous in selected:
            if (item.get("subject_track_id"), item.get("object_track_id"), item.get("predicate")) != (
                previous.get("subject_track_id"), previous.get("object_track_id"), previous.get("predicate")
            ):
                continue
            prior = relation_span(previous)
            intersection = max(0, min(span.end_frame, prior.end_frame) - max(span.start_frame, prior.start_frame))
            union = max(span.end_frame, prior.end_frame) - min(span.start_frame, prior.start_frame)
            if union > 0 and intersection / union >= temporal_iou_threshold:
                duplicate = True
                break
        if not duplicate:
            selected.append(item)
    return selected


def compose_action_spatial_relations(
    relations: Iterable[Dict[str, Any]], *, min_overlap_frames: int = 15,
    score_coefficient: float = 0.55,
) -> list[Dict[str, Any]]:
    """Compose conservative VidVRD action+spatial labels after proposal slicing.

    VidVRD contains ``sit_above`` and related composites but no atomic ``sit``
    predicate. A verified ride/drive interaction supplies the missing seated
    state; an independently generated spatial relation supplies the direction.
    Long temporal overlap is required to avoid turning brief geometry into a
    semantic composite.
    """

    if min_overlap_frames <= 0:
        raise ValueError("composition overlap must be positive")
    coefficient = max(0.0, min(1.0, float(score_coefficient)))
    definitions: list[tuple[str, str, str]] = []
    components_by_name: dict[str, dict[str, str]] = {}
    for record in load_ontology()["predicates"]:
        name = str(record["name"])
        components = {key: str(value) for key, value in record.get("components", {}).items()}
        components_by_name[name] = components
        action, spatial = components.get("action", ""), components.get("spatial", "")
        if action and spatial:
            definitions.append((name, action, spatial))

    items = [dict(item) for item in relations]
    by_pair_predicate: dict[tuple[int, int, str], list[Dict[str, Any]]] = {}
    by_pair_action: dict[tuple[int, int, str], list[Dict[str, Any]]] = {}
    known: set[tuple[int, int, str, int, int]] = set()
    for item in items:
        sid = int(item["subject_track_id"])
        oid = int(item["object_track_id"])
        predicate = str(item["predicate"])
        by_pair_predicate.setdefault((sid, oid, predicate), []).append(item)
        action = components_by_name.get(predicate, {}).get("action", "")
        if action:
            by_pair_action.setdefault((sid, oid, action), []).append(item)
        if predicate in {"ride", "drive"}:
            by_pair_action.setdefault((sid, oid, "sit"), []).append(item)
        span = relation_span(item)
        known.add((sid, oid, predicate, span.start_frame, span.end_frame))

    for composite, action, spatial in definitions:
        action_pairs = {(sid, oid) for sid, oid, value in by_pair_action if value == action}
        spatial_pairs = {(sid, oid) for sid, oid, value in by_pair_predicate if value == spatial}
        for sid, oid in action_pairs & spatial_pairs:
            for action_item in by_pair_action[(sid, oid, action)]:
                action_span = relation_span(action_item)
                for spatial_item in by_pair_predicate[(sid, oid, spatial)]:
                    spatial_span = relation_span(spatial_item)
                    start = max(action_span.start_frame, spatial_span.start_frame)
                    end = min(action_span.end_frame, spatial_span.end_frame)
                    key = (sid, oid, composite, start, end)
                    if end - start < min_overlap_frames or key in known:
                        continue
                    evidence_score = math.sqrt(_relation_score(action_item) * _relation_score(spatial_item))
                    composed = dict(action_item)
                    composed.update(
                        predicate=composite,
                        start_frame=start,
                        end_frame=end,
                        span_convention="half_open",
                        ranking_score=round(evidence_score * coefficient, 6),
                        evidence_score=round(evidence_score, 6),
                        source="action_spatial_composition",
                        source_kind="composition",
                        sources=["semantic", "rule", "composition"],
                        predicate_components={"action": action, "spatial": spatial, "comparative": ""},
                        composition={"action": action, "spatial": spatial},
                    )
                    action_evidence = {int(frame) for frame in action_item.get("evidence_frames", [])}
                    spatial_evidence = {int(frame) for frame in spatial_item.get("evidence_frames", [])}
                    composed["evidence_frames"] = sorted(
                        frame for frame in action_evidence & spatial_evidence if start <= frame < end
                    )
                    if action_item.get("relation_id"):
                        composed["relation_id"] = f"{action_item['relation_id']}:c:{composite}:{start}-{end}"
                    items.append(composed)
                    known.add(key)
    return items


def merge_relation_files(
    video_exports: Iterable[Tuple[str, Path]], out_path: Path, video_ids: Iterable[str] = ()
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for video_id, path in video_exports:
        if not path.exists():
            continue
        value = read_json(path)
        if isinstance(value, dict):
            merged[video_id] = value.get(video_id, [])
    for video_id in video_ids:
        merged.setdefault(str(video_id), [])
    write_json(out_path, merged)
    return merged


def merge_trajectory_files(
    video_exports: Iterable[Tuple[str, Path]], out_path: Path, video_ids: Iterable[str] = ()
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for video_id, path in video_exports:
        if not path.exists():
            continue
        value = read_json(path)
        if isinstance(value, dict):
            merged[video_id] = value.get(video_id, [])
    for video_id in video_ids:
        merged.setdefault(str(video_id), [])
    write_json(out_path, merged)
    return merged
