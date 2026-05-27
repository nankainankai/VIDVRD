from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def _node_stats(videos: Sequence[Mapping[str, Any]], nodes: Sequence[str]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {n: {"succeeded": 0, "failed": 0, "skipped": 0, "other": 0} for n in nodes}
    for video in videos:
        node_map = video.get("nodes", {}) if isinstance(video.get("nodes"), dict) else {}
        for node in nodes:
            st = str((node_map.get(node) or {}).get("state", "") or "").strip().lower()
            if st == "succeeded":
                counts[node]["succeeded"] += 1
            elif st == "failed":
                counts[node]["failed"] += 1
            elif st in {"skipped", "partial"}:
                counts[node]["skipped"] += 1
            elif st:
                counts[node]["other"] += 1
    return counts


def _failure_reasons(videos: Sequence[Mapping[str, Any]]) -> List[str]:
    reasons: List[str] = []
    for video in videos:
        vid = str(video.get("video_id", ""))
        if str(video.get("state", "")) == "failed":
            reasons.append(f"{vid}: {video.get('error', 'unknown')}")
        node_map = video.get("nodes", {}) if isinstance(video.get("nodes"), dict) else {}
        for node, info in node_map.items():
            if isinstance(info, dict) and info.get("state") == "failed":
                reasons.append(f"{vid}/{node}: {info.get('error', 'failed')}")
    return reasons


def render_run_report(manifest: Mapping[str, Any]) -> str:
    nodes = list(manifest.get("nodes", []) or [])
    videos = list(manifest.get("videos", []) or [])
    state_counts = manifest.get("video_state_counts", {}) if isinstance(manifest.get("video_state_counts"), dict) else {}
    args = manifest.get("args", {}) if isinstance(manifest.get("args"), dict) else {}
    evaluate = manifest.get("evaluate", {}) if isinstance(manifest.get("evaluate"), dict) else {}

    succeeded = int(state_counts.get("succeeded", 0) or 0)
    failed = int(state_counts.get("failed", 0) or 0)
    skipped = int(state_counts.get("skipped", 0) or 0)
    partial = int(state_counts.get("partial", 0) or 0)
    total = len(videos)

    node_stats = _node_stats(videos, nodes)
    failures = _failure_reasons(videos)

    lines = [
        "# VIDVRD 运行报告",
        "",
        "## 基本信息",
        "",
        f"- 运行目录：`{manifest.get('run_dir', '')}`",
        f"- 开始时间：{manifest.get('started_at', '')}",
        f"- 结束时间：{manifest.get('finished_at', '')}",
        f"- 配置文件：`{manifest.get('config_path', '')}`",
        f"- 输入视频数量：{total}",
        f"- 成功：{succeeded}，失败：{failed}，跳过：{skipped}，部分完成：{partial}",
        f"- 最终关系数：{manifest.get('pred_relation_count', 0)}",
        f"- 最终关系文件：`{manifest.get('pred_relations_json', '')}`",
        f"- resume：{args.get('resume')}，force：{args.get('force')}，dry_run_relations：{args.get('dry_run_relations')}",
        f"- api_key_present：{args.get('api_key_present')}",
        "",
        "## 评测",
        "",
        f"- 状态：{evaluate.get('state', 'unknown')}",
        f"- 报告：`{evaluate.get('report', '')}`" if evaluate.get("report") else f"- 原因：{evaluate.get('reason', evaluate.get('error', ''))}",
        "",
        "## 节点状态",
        "",
        "| 节点 | 成功 | 失败 | 跳过/其他 |",
        "|---|---:|---:|---:|",
    ]
    for node in nodes:
        c = node_stats.get(node, {})
        lines.append(
            f"| {node} | {c.get('succeeded', 0)} | {c.get('failed', 0)} | {c.get('skipped', 0) + c.get('other', 0)} |"
        )

    lines.extend(["", "## 视频明细", ""])
    for video in videos:
        vid = video.get("video_id", "")
        lines.append(f"### {vid}")
        lines.append(f"- state: {video.get('state', 'unknown')}")
        if video.get("error"):
            lines.append(f"- error: {video.get('error')}")
        if video.get("skip_reason"):
            lines.append(f"- skip_reason: {video.get('skip_reason')}")
        outputs = video.get("outputs", {})
        if isinstance(outputs, dict) and outputs:
            lines.append(f"- outputs: `{outputs}`")
        lines.append("")

    if failures:
        lines.extend(["## 失败与异常", ""])
        for r in failures:
            lines.append(f"- {r}")
        lines.append("")

    lines.extend(
        [
            "## 下一步建议",
            "",
            "- 若全部 SKIP 且需重跑：使用新 `--run_dir` 或 `--force --from_node <node>`",
            "- 若部分失败：修复后原命令加 `--resume`",
            "- 正式检测：确认 Rex-Omni 模型或 `DINOX_API_TOKEN`，使用 `configs/production_full.json`",
            "",
        ]
    )
    return "\n".join(lines)


def write_run_report(manifest: Mapping[str, Any], report_path: Path) -> Path:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_run_report(manifest), encoding="utf-8")
    return report_path
