from __future__ import annotations

"""Evaluation entry points separating official metrics from diagnostics."""

from pathlib import Path
from typing import Any, Dict, Sequence

from vidvrd_auto.evaluation.diagnostic import run_track_aligned_diagnostic
from vidvrd_auto.evaluation.official import evaluate_official_vidvrd, project_artifacts_to_official
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
        f"- Official test set complete: `{str(scope['complete_official_test']).lower()}`",
        f"- Tube vIoU threshold: {metrics['viou_threshold']:.2f}",
        "",
    ]
    if not scope["complete_official_test"]:
        lines.extend([
            "> This is a protocol-compatible partial result, not a publishable full-test benchmark.",
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
    prediction_videos = set(pred_relation_obj) | set(pred_trajectory_obj)
    expected_videos = set(video_ids)
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
        "complete_official_test": dataset_split == "test" and len(video_ids) == expected_official_video_count,
        "requested_video_ids": list(requested_video_ids),
        "missing_prediction_videos": sorted(expected_videos - prediction_videos),
        "extra_prediction_videos": sorted(prediction_videos - expected_videos),
        "empty_prediction_videos": empty_prediction_videos,
        "submission_video_set_exact": prediction_videos == expected_videos,
    }
    official["gold_adapter"] = gold_adapter
    official["prediction_adapter"] = prediction_adapter
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
        video_ids=video_ids,
    )
    return {"official": official, "diagnostic": diagnostic}


__all__ = ["run_evaluation_suite", "run_track_aligned_diagnostic"]
