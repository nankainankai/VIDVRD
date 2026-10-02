from __future__ import annotations

"""Fixed-budget category triplet retrieval, distinct from tube detection."""

from typing import Any, Mapping, Sequence


def evaluate_known_label_coverage(
    groundtruth: Mapping[str, Sequence[dict[str, Any]]],
    predictions: Mapping[str, Sequence[dict[str, Any]]],
    video_ids: Sequence[str],
    *,
    budgets: Sequence[int] = (1, 5, 10, 20),
) -> dict[str, Any]:
    """Deduplicate category triplets; missing predictions remain zero recall.

    Inputs use the official adapter, so relations without exported subject or
    object tubes cannot receive credit. Sparse Gold supports coverage of known
    labels, not a claim of complete precision or annotation accuracy.
    """
    limits = sorted(set(int(k) for k in budgets))
    if not limits or min(limits) < 1:
        raise ValueError("budgets must contain positive integers")
    if 10 not in limits:
        raise ValueError("budgets must include the fixed primary budget of 10")
    ids = list(dict.fromkeys(video_ids))
    rows = []
    for video_id in ids:
        target = {tuple(item["triplet"]) for item in groundtruth.get(video_id, [])}
        ranked = []
        seen = set()
        for item in sorted(predictions.get(video_id, []), key=lambda item: -float(item.get("score", 0.0))):
            triplet = tuple(item["triplet"])
            if triplet not in seen:
                ranked.append(triplet)
                seen.add(triplet)
        rows.append({
            "video_id": video_id,
            "known_label_count": len(target),
            "prediction_label_count": len(ranked),
            "hits_at": {str(k): len(target.intersection(ranked[:k])) for k in limits},
            "top_labels": [list(label) for label in ranked[:max(limits)]],
            "missing_known_labels_at_primary_budget": [list(label) for label in sorted(target.difference(ranked[:10]))],
        })
    count = sum(row["known_label_count"] for row in rows)
    nonempty = [row for row in rows if row["known_label_count"]]
    return {
        "evaluator": "known_category_triplet_coverage_v1",
        "task": "video_level_category_triplet_retrieval",
        "primary_budget": 10,
        "budgets": limits,
        "annotation_complete": False,
        "evaluated_video_ids": ids,
        "evaluated_video_count": len(ids),
        "videos_without_known_labels": [row["video_id"] for row in rows if not row["known_label_count"]],
        "known_label_count": count,
        "micro_recall_at": {str(k): sum(row["hits_at"][str(k)] for row in rows) / count if count else 0.0 for k in limits},
        "macro_recall_at": {str(k): sum(row["hits_at"][str(k)] / row["known_label_count"] for row in nonempty) / len(nonempty) if nonempty else 0.0 for k in limits},
        "does_not_measure": ["instance_identity", "temporal_boundaries", "tube_localization", "complete_annotation_precision"],
        "per_video": rows,
    }


def write_tagging_report(metrics: dict[str, Any], path) -> None:
    lines = [
        "# 已标注关系标签覆盖率（辅助标注任务）", "",
        "类别三元组要求主语类别、谓词、宾语类别完全一致，并保留方向。每个视频按预测分数排序、去重后，最多提交固定数量的标签。", "",
        "此指标不要求实例 ID、时间边界或轨迹框准确，与官方关系检测 mAP 是不同任务。Gold 为稀疏标注，结果仅表示已知标签覆盖率，不能表述为完整标注准确率。", "",
        f"- 评测视频：{metrics['evaluated_video_count']}",
        f"- 已知类别三元组：{metrics['known_label_count']}",
        "- 预设主指标：已标注关系标签覆盖率 @10", "",
        "| 每视频标签预算 | 覆盖率（micro） | 覆盖率（macro） |",
        "| --- | ---: | ---: |",
    ]
    for k in metrics["budgets"]:
        lines.append(f"| {k} | {metrics['micro_recall_at'][str(k)]:.2%} | {metrics['macro_recall_at'][str(k)]:.2%} |")
    lines.extend(["", "失败或无预测的视频保留在分母。未标注标签不作为确认错误统计；需要独立完整标注才能报告精确率、F1 和人工采纳率。", ""])
    subset = metrics.get("requested_subset")
    if subset:
        lines.extend([
            "## 本次请求子集诊断", "",
            f"本报告上表包含整个 Gold 评测范围；本次实际请求其中 {subset['evaluated_video_count']} 个视频。", "",
            f"请求子集已标注标签覆盖率 @10：{subset['micro_recall_at']['10']:.2%}。失败或空预测的请求视频仍计入该子集分母。", "",
            "逐视频明细另见 `known_label_coverage_requested.json`；子集结果不代表完整测试集成绩。", "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
