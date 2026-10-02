from __future__ import annotations

"""Evaluation entry points separating official metrics from diagnostics."""

from pathlib import Path
from typing import Any, Dict, Sequence

from vidvrd_auto.evaluation.diagnostic import run_track_aligned_diagnostic
from vidvrd_auto.evaluation.official import evaluate_official_vidvrd, project_artifacts_to_official
from vidvrd_auto.evaluation.routing import diagnose_routed_candidates
from vidvrd_auto.evaluation.track_ceiling import diagnose_track_ceiling, summarize_track_ceiling_outcomes
from vidvrd_auto.evaluation.tagging import evaluate_known_label_coverage, write_tagging_report
from vidvrd_auto.utils.io import read_json, write_json


def _video_ids(
    gold_relations: Dict[str, Any],
    gold_manifest: Dict[str, Any],
    requested_video_ids: Sequence[str],
    *,
    dataset_split: str,
    scope: str,
) -> list[str]:
    if scope == "requested":
        return list(dict.fromkeys(str(video_id) for video_id in requested_video_ids))
    if scope != "gold_split":
        raise ValueError(f"unsupported evaluation scope: {scope}")
    split_map = gold_manifest.get("video_splits", {})
    if not isinstance(split_map, dict):
        raise ValueError("gold manifest does not contain video_splits")
    return sorted(video_id for video_id in gold_relations if str(split_map.get(video_id)) == dataset_split)


def _semantic_packet_coverage(video_ids: Sequence[str], root: Path) -> Dict[str, Any]:
    attempted = 0
    omitted = 0
    omitted_by_video: Dict[str, int] = {}
    missing_logs: list[str] = []
    for video_id in video_ids:
        log_path = root / video_id / "semantic" / "run.log"
        if not log_path.exists():
            missing_logs.append(video_id)
            continue
        log = read_json(log_path)
        batches = log.get("batch_audit", [])
        errors = log.get("errors", [])
        attempted += sum(len(batch.get("packet_ids", [])) for batch in batches)
        omitted += len(errors)
        if errors:
            omitted_by_video[video_id] = len(errors)
    return {
        "attempted_packets": attempted,
        "omitted_packets": omitted,
        "coverage_fraction": (attempted - omitted) / attempted if attempted else None,
        "omitted_by_video": omitted_by_video,
        "missing_logs": missing_logs,
    }


def _official_report(metrics: Dict[str, Any], path: Path) -> None:
    detection = metrics["relation_detection"]
    tagging = metrics["relation_tagging"]
    scope = metrics["dataset_scope"]
    lines = [
        "# Official-compatible ImageNet-VidVRD evaluation",
        "",
        "- Evaluator: `imagenet_vidvrd_official_2017_compatible_v1`",
        f"- Dataset split: `{scope['split']}`",
        f"- Evaluation scope: `{scope['scope']}` ({scope['evaluated_video_count']} videos)",
        f"- Official test annotations complete: `{str(scope['complete_test_annotations']).lower()}`",
        f"- Full-test submission complete: `{str(scope['complete_official_test']).lower()}`",
        f"- Tube vIoU threshold: {metrics['viou_threshold']:.2f}",
        "",
    ]
    if not scope["complete_official_test"]:
        lines.extend([
            "> This is a protocol-compatible partial result, not a publishable full-test benchmark.",
            "",
        ])
    partial = metrics.get("partial_run_diagnostic", {})
    if partial.get("available") and partial.get("requested_gold_video_count", 0) < scope["evaluated_video_count"]:
        lines.extend([
            "> Only a subset of the Gold videos was requested. The official-scope mAP includes every Gold video, including those with no predictions.",
            "",
        ])
    lines.extend([
        "## Relation detection",
        "",
        f"- mAP: {detection['mean_ap']:.6f}",
        f"- Recall@50: {detection['recall_at']['50']:.6f}",
        f"- Recall@100: {detection['recall_at']['100']:.6f}",
        "",
        "## Relation tagging",
        "",
        f"- Precision@1: {tagging['precision_at']['1']:.6f}",
        f"- Precision@5: {tagging['precision_at']['5']:.6f}",
        f"- Precision@10: {tagging['precision_at']['10']:.6f}",
        "",
        "## Submission coverage",
        "",
        f"- Video set exactly matches evaluation scope: `{str(scope['submission_video_set_exact']).lower()}`",
        f"- Missing prediction videos: {len(scope['missing_prediction_videos'])}",
        f"- Extra prediction videos: {len(scope['extra_prediction_videos'])}",
        f"- Explicitly empty prediction videos: {len(scope['empty_prediction_videos'])}",
        f"- Ground-truth relations: {metrics['groundtruth_relation_count']}",
        f"- Adapted predicted relations: {metrics['prediction_adapter']['relation_count']}",
    ])
    if partial.get("available"):
        lines.extend([
            "",
            "## Partial-run diagnostic (not an official benchmark score)",
            "",
            f"- Requested Gold videos: {partial['requested_gold_video_count']}/{scope['evaluated_video_count']}",
            f"- Coverage fraction: {partial['coverage_fraction']:.6f}",
            f"- mAP on requested Gold videos only: {partial['requested_mean_ap']:.6f}",
        ])
    packet_coverage = metrics.get("semantic_packet_coverage")
    if packet_coverage:
        lines.extend([
            "",
            "## Semantic API coverage",
            "",
            f"- Attempted evidence packets: {packet_coverage['attempted_packets']}",
            f"- Omitted packets after provider image moderation: {packet_coverage['omitted_packets']}",
            f"- Videos with omitted packets: {len(packet_coverage['omitted_by_video'])}",
            f"- Missing semantic logs: {len(packet_coverage['missing_logs'])}",
        ])
        if packet_coverage["omitted_packets"]:
            lines.extend([
                "",
                "> This score includes missing predictions from provider-refused images; it is not an uninterrupted semantic inference run.",
            ])
    ceiling = metrics.get("track_ceiling_diagnostic")
    if ceiling:
        ceiling_counts = ceiling["reason_counts"]
        outcomes = ceiling.get("outcomes", {})
        lines.extend([
            "",
            "## Optimistic track ceiling (diagnostic, not mAP)",
            "",
            f"- Relations even potentially matchable at tube vIoU {ceiling['viou_threshold']:.2f}: "
            f"{ceiling['maximum_matchable_relations']}/{ceiling['groundtruth_relation_count']}",
            f"- Category absent: {int(ceiling_counts.get('category_absent', 0))}",
            f"- Same-category track coverage insufficient: {int(ceiling_counts.get('insufficient_track_coverage', 0))}",
            f"- Actual true positives at any rank: {int(outcomes.get('actual_true_positive_count_any_rank', 0))}",
            f"- Potentially matchable but lacking the correct triplet: "
            f"{int(outcomes.get('potentially_matchable_failure_counts', {}).get('candidate_missing', 0))}",
            f"- Of those, predicate was routed to the semantic model: "
            f"{int(outcomes.get('potentially_matchable_candidate_missing_by_route', {}).get('candidate_routed', 0))}",
            f"- Potentially matchable but failing tube localization: "
            f"{int(outcomes.get('potentially_matchable_failure_counts', {}).get('localization_failure', 0))}",
            "",
            "> This is an optimistic upper bound from detected tracks, not achieved relation recall. "
            "It ignores prediction-only tube volume, track-pair consistency and predicate errors.",
        ])
    diagnostics = metrics.get("error_diagnostics", {})
    if diagnostics:
        failures = diagnostics.get("failure_counts", {})
        recall = diagnostics.get("candidate_recall", {})
        quality = metrics.get("quality_diagnostics", {})
        lines.extend([
            "",
            "## Error diagnosis",
            "",
            f"- Atomic predicate candidate recall: {float(recall.get('atomic', 0.0)):.6f}",
            f"- Composite predicate candidate recall: {float(recall.get('composite', 0.0)):.6f}",
            f"- Candidate missing: {int(failures.get('candidate_missing', 0))}",
            f"- Localization failures: {int(failures.get('localization_failure', 0))}",
            f"- Ranking failures: {int(failures.get('ranking_failure', 0))}",
            f"- Relations involving risk tracks: {int(quality.get('risk_track_relation_count', 0))}",
        ])
    routing = metrics.get("candidate_routing_diagnostics")
    if routing:
        routing_counts = routing.get("counts", {})
        lines.extend([
            "",
            "## Candidate routing diagnosis (predicted-track aligned; not official mAP)",
            "",
            f"- Routed candidate recall: {float(routing.get('routed_candidate_recall', 0.0)):.6f}",
            f"- Semantic-eligible ground-truth relations: {int(routing.get('semantic_eligible_groundtruth_count', 0))}",
            f"- Track pair unaligned: {int(routing_counts.get('track_pair_unaligned', 0))}",
            f"- Rule-only spatial predicates: {int(routing_counts.get('rule_only_predicate', 0))}",
            f"- No temporal pair packet: {int(routing_counts.get('pair_packet_missing', 0))}",
            f"- Predicate absent from routed packet: {int(routing_counts.get('router_missed_predicate', 0))}",
            f"- Predicate present in routed packet: {int(routing_counts.get('candidate_routed', 0))}",
        ])
        missing_categories = routing.get("relations_missing_track_category", {})
        if missing_categories:
            lines.append("- Top categories in relations with unmatched tracks: " + ", ".join(
                f"{category} ({count})" for category, count in list(missing_categories.items())[:8]
            ))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation_suite(
    *,
    gold_relations: Path,
    gold_trajectories: Path,
    gold_manifest: Path,
    pred_relations: Path,
    pred_trajectories: Path,
    requested_video_ids: Sequence[str],
    official_report_path: Path,
    official_metrics_path: Path,
    diagnostic_report_path: Path,
    diagnostic_metrics_path: Path,
    dataset_split: str = "test",
    scope: str = "gold_split",
    expected_official_video_count: int = 200,
    official_viou_threshold: float = 0.5,
    diagnostic_track_viou_threshold: float = 0.3,
    diagnostic_relation_tiou_threshold: float = 0.5,
    diagnostic_mot_iou_threshold: float = 0.5,
    evidence_packets_root: Path | None = None,
    presence_clips_path: Path | None = None,
) -> Dict[str, Any]:
    gold_relation_obj = read_json(gold_relations)
    gold_trajectory_obj = read_json(gold_trajectories)
    pred_relation_obj = read_json(pred_relations)
    pred_trajectory_obj = read_json(pred_trajectories)
    manifest_obj = read_json(gold_manifest)
    video_ids = _video_ids(
        gold_relation_obj,
        manifest_obj,
        requested_video_ids,
        dataset_split=dataset_split,
        scope=scope,
    )
    if not video_ids:
        raise ValueError(f"no Gold videos selected for split={dataset_split} scope={scope}")
    requested_set = {str(video_id) for video_id in requested_video_ids}
    requested_gold_ids = [video_id for video_id in video_ids if video_id in requested_set and video_id in gold_relation_obj]
    if scope == "gold_split" and not requested_gold_ids:
        raise ValueError(
            f"none of the requested videos belong to Gold split={dataset_split}; "
            "cannot report a meaningful benchmark mAP"
        )

    gold_official, gold_adapter = project_artifacts_to_official(
        gold_relation_obj, gold_trajectory_obj, video_ids, prediction=False
    )
    if gold_adapter["skipped_missing_tracks"] or gold_adapter["skipped_empty_tubes"] or gold_adapter["split_on_track_gaps"]:
        raise ValueError(f"Gold cannot be converted to contiguous official tubes: {gold_adapter}")
    prediction_official, prediction_adapter = project_artifacts_to_official(
        pred_relation_obj, pred_trajectory_obj, video_ids, prediction=True
    )
    official = evaluate_official_vidvrd(
        gold_official, prediction_official, viou_threshold=official_viou_threshold
    )
    if requested_gold_ids:
        requested_metrics = evaluate_official_vidvrd(
            {video_id: gold_official[video_id] for video_id in requested_gold_ids},
            {video_id: prediction_official[video_id] for video_id in requested_gold_ids},
            viou_threshold=official_viou_threshold,
        )
        partial_run = {
            "available": True,
            "requested_gold_video_ids": requested_gold_ids,
            "requested_gold_video_count": len(requested_gold_ids),
            "coverage_fraction": len(requested_gold_ids) / len(video_ids),
            "requested_mean_ap": requested_metrics["relation_detection"]["mean_ap"],
            "requested_recall_at": requested_metrics["relation_detection"]["recall_at"],
        }
    else:
        partial_run = {
            "available": False,
            "requested_gold_video_ids": [],
            "requested_gold_video_count": 0,
            "coverage_fraction": 0.0,
            "requested_mean_ap": None,
            "requested_recall_at": None,
        }
    prediction_videos = set(pred_relation_obj) | set(pred_trajectory_obj)
    expected_videos = set(video_ids)
    scoped_prediction_items = [
        item for video_id in video_ids for item in pred_relation_obj.get(video_id, [])
        if isinstance(item, dict)
    ]
    official["quality_diagnostics"] = {
        "risk_track_relation_count": sum(bool(item.get("risk_track_involved")) for item in scoped_prediction_items),
        "low_track_pair_quality_relation_count": sum(
            float(item.get("track_pair_quality", 1.0) or 0.0) < 0.5 for item in scoped_prediction_items
        ),
    }
    if evidence_packets_root is not None:
        official["candidate_routing_diagnostics"] = diagnose_routed_candidates(
            gold_relation_obj, gold_trajectory_obj, pred_trajectory_obj,
            video_ids, evidence_packets_root,
            track_iou_threshold=diagnostic_track_viou_threshold,
        )
        official["semantic_packet_coverage"] = _semantic_packet_coverage(requested_gold_ids, evidence_packets_root)
    official["track_ceiling_diagnostic"] = diagnose_track_ceiling(
        gold_official, pred_trajectory_obj, threshold=official_viou_threshold
    )
    official["track_ceiling_diagnostic"]["outcomes"] = summarize_track_ceiling_outcomes(
        official["track_ceiling_diagnostic"], official["error_diagnostics"],
        official.get("candidate_routing_diagnostics"),
    )
    complete_test_annotations = dataset_split == "test" and len(video_ids) == expected_official_video_count
    submission_video_set_exact = prediction_videos == expected_videos
    empty_prediction_videos = sorted(
        video_id for video_id in expected_videos
        if not pred_relation_obj.get(video_id) and not pred_trajectory_obj.get(video_id)
    )
    official["dataset_scope"] = {
        "dataset": "ImageNet-VidVRD",
        "split": dataset_split,
        "scope": scope,
        "evaluated_video_count": len(video_ids),
        "expected_official_video_count": expected_official_video_count,
        "complete_test_annotations": complete_test_annotations,
        "complete_official_test": (
            complete_test_annotations and expected_videos <= requested_set and submission_video_set_exact
        ),
        "requested_video_ids": list(requested_video_ids),
        "missing_prediction_videos": sorted(expected_videos - prediction_videos),
        "extra_prediction_videos": sorted(prediction_videos - expected_videos),
        "empty_prediction_videos": empty_prediction_videos,
        "submission_video_set_exact": submission_video_set_exact,
    }
    official["gold_adapter"] = gold_adapter
    official["prediction_adapter"] = prediction_adapter
    official["partial_run_diagnostic"] = partial_run
    write_json(official_metrics_path, official)
    _official_report(official, official_report_path)

    diagnostic = run_track_aligned_diagnostic(
        gold_relations=gold_relations,
        gold_trajectories=gold_trajectories,
        pred_relations=pred_relations,
        pred_trajectories=pred_trajectories,
        report_path=diagnostic_report_path,
        metrics_path=diagnostic_metrics_path,
        track_viou_threshold=diagnostic_track_viou_threshold,
        relation_tiou_threshold=diagnostic_relation_tiou_threshold,
        mot_iou_threshold=diagnostic_mot_iou_threshold,
        video_ids=video_ids,
    )
    from vidvrd_auto.evaluation.presence import evaluate_presence
    presence = evaluate_presence(
        gold_relation_obj, gold_trajectory_obj, pred_relation_obj, pred_trajectory_obj,
        video_ids, clips=read_json(presence_clips_path) if presence_clips_path else None,
    )
    write_json(diagnostic_metrics_path.parent / "presence_instance.json", presence)
    tagging = evaluate_known_label_coverage(gold_official, prediction_official, video_ids)
    if requested_gold_ids and len(requested_gold_ids) < len(video_ids):
        subset_tagging = evaluate_known_label_coverage(gold_official, prediction_official, requested_gold_ids)
        subset_tagging["scope"] = "requested_gold_subset_diagnostic"
        tagging["requested_subset"] = subset_tagging
        write_json(diagnostic_metrics_path.parent / "known_label_coverage_requested.json", subset_tagging)
        write_tagging_report(subset_tagging, diagnostic_metrics_path.parent / "known_label_coverage_requested.md")
    write_json(diagnostic_metrics_path.parent / "known_label_coverage.json", tagging)
    write_tagging_report(tagging, diagnostic_metrics_path.parent / "known_label_coverage.md")
    return {"official": official, "diagnostic": diagnostic, "presence": presence, "tagging": tagging}


__all__ = ["run_evaluation_suite", "run_track_aligned_diagnostic"]
