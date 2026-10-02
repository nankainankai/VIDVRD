from __future__ import annotations

"""Separate track-pair access, router recall, and downstream proposal failures."""

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from vidvrd_auto.core.schema import relation_span
from vidvrd_auto.core.ontology import normalize_object
from vidvrd_auto.evaluation.diagnostic.track_aligned import align_trajectories, gold_supported_trajectory_iou
from vidvrd_auto.relations.object_candidates import GEOMETRY_PREDICATES
from vidvrd_auto.utils.io import read_json


def diagnose_routed_candidates(
    gold_relations: Mapping[str, Any],
    gold_trajectories: Mapping[str, Any],
    pred_trajectories: Mapping[str, Any],
    video_ids: Sequence[str],
    evidence_packets_root: Path,
    *,
    track_iou_threshold: float = 0.3,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    unmatched_track_categories: Counter[str] = Counter()
    relations_missing_category: Counter[str] = Counter()
    unmatched_gold_tracks: list[dict[str, Any]] = []
    semantic_eligible_count = 0
    for video_id in video_ids:
        packet_path = evidence_packets_root / video_id / "semantic" / "evidence_packets.json"
        packet_obj = read_json(packet_path) if packet_path.is_file() else {}
        packets = packet_obj.get("packets", []) if isinstance(packet_obj, dict) else []
        mapping, _ = align_trajectories(
            gold_trajectories.get(video_id, []), pred_trajectories.get(video_id, []), track_iou_threshold
        )
        matched_gold_ids = set(mapping.values())
        gold_categories = {
            int(track["track_id"]): str(track.get("category", "unknown"))
            for track in gold_trajectories.get(video_id, []) if isinstance(track, dict) and "track_id" in track
        }
        for track_id, category in gold_categories.items():
            if track_id not in matched_gold_ids:
                unmatched_track_categories[category] += 1
                gold_track = next(track for track in gold_trajectories[video_id]
                                  if int(track["track_id"]) == track_id)
                same_category = [track for track in pred_trajectories.get(video_id, [])
                                 if normalize_object(str(track.get("category", "unknown"))) == normalize_object(category)]
                best = max(
                    ((gold_supported_trajectory_iou(track, gold_track), track) for track in same_category),
                    key=lambda item: item[0], default=None,
                )
                unmatched_gold_tracks.append({
                    "video_id": video_id,
                    "track_id": track_id,
                    "category": category,
                    "best_same_category_viou": round(best[0], 6) if best else 0.0,
                    "best_pred_track_id": int(best[1]["track_id"]) if best else None,
                })
        for index, target in enumerate(gold_relations.get(video_id, [])):
            if not isinstance(target, dict):
                continue
            subject = int(target["subject_track_id"])
            obj = int(target["object_track_id"])
            predicate = str(target["predicate"])
            semantic_eligible = predicate not in GEOMETRY_PREDICATES
            semantic_eligible_count += int(semantic_eligible)
            target_span = relation_span(target)
            pair_aligned = subject in mapping.values() and obj in mapping.values()
            temporal_pair_packets = 0
            routed = False
            for packet in packets if isinstance(packets, list) else []:
                if not isinstance(packet, dict):
                    continue
                if (int(packet.get("start_frame", 0)) >= target_span.end_frame
                        or int(packet.get("end_frame", -1)) < target_span.start_frame):
                    continue
                for direction in packet.get("candidate_directions", []):
                    try:
                        pred_subject = int(direction["subject_track_id"])
                        pred_object = int(direction["object_track_id"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if mapping.get(pred_subject) != subject or mapping.get(pred_object) != obj:
                        continue
                    temporal_pair_packets += 1
                    routed = routed or predicate in direction.get("candidate_predicates", [])
            status = (
                "track_pair_unaligned" if not pair_aligned else
                "rule_only_predicate" if not semantic_eligible else
                "pair_packet_missing" if not temporal_pair_packets else
                "candidate_routed" if routed else "router_missed_predicate"
            )
            counts[status] += 1
            if status == "track_pair_unaligned":
                for track_id in {subject, obj} - matched_gold_ids:
                    relations_missing_category[gold_categories.get(track_id, "unknown")] += 1
            rows.append({
                "video_id": video_id,
                "groundtruth_index": index,
                "predicate": predicate,
                "subject_track_id": subject,
                "object_track_id": obj,
                "temporal_pair_packet_count": temporal_pair_packets,
                "semantic_eligible": semantic_eligible,
                "status": status,
            })
    total = len(rows)
    return {
        "scope": "predicted-track-aligned evidence packets; diagnostic, not official mAP",
        "track_iou_threshold": track_iou_threshold,
        "groundtruth_count": total,
        "semantic_eligible_groundtruth_count": semantic_eligible_count,
        "counts": dict(sorted(counts.items())),
        "unmatched_gold_track_categories": dict(unmatched_track_categories.most_common()),
        "unmatched_gold_tracks": unmatched_gold_tracks,
        "relations_missing_track_category": dict(relations_missing_category.most_common()),
        "routed_candidate_recall": counts["candidate_routed"] / semantic_eligible_count if semantic_eligible_count else 0.0,
        "rows": rows,
    }
