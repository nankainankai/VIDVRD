from __future__ import annotations

"""Development-only same-pair temporal proposal de-duplication ablation."""

import argparse
import json
from pathlib import Path

from vidvrd_auto.core.schema import relation_span
from vidvrd_auto.evaluation.official.vidvrd import evaluate_official_vidvrd, project_artifacts_to_official
from vidvrd_auto.utils.io import read_json


def _score(item: dict) -> float:
    return float(item.get("ranking_score", item.get("agent_score", item.get("confidence", 0.0))) or 0.0)


def _overlap(left: dict, right: dict, mode: str) -> float:
    a, b = relation_span(left), relation_span(right)
    intersection = max(0, min(a.end_frame, b.end_frame) - max(a.start_frame, b.start_frame))
    length_a, length_b = a.end_frame - a.start_frame, b.end_frame - b.start_frame
    denominator = min(length_a, length_b) if mode == "shorter" else length_a + length_b - intersection
    return intersection / denominator if denominator > 0 else 0.0


def _dedup(items: list[dict], threshold: float, mode: str) -> list[dict]:
    selected: list[dict] = []
    for item in sorted(items, key=_score, reverse=True):
        if any(
            item["subject_track_id"] == other["subject_track_id"]
            and item["object_track_id"] == other["object_track_id"]
            and item["predicate"] == other["predicate"]
            and _overlap(item, other, mode) >= threshold
            for other in selected
        ):
            continue
        selected.append(item)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    ids = [line.strip() for line in Path("experiments/dev_videos.txt").read_text(encoding="utf-8").splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
    relations = read_json(args.run_dir / "pred" / "relations.json")
    trajectories = read_json(args.run_dir / "pred" / "trajectories.json")
    gold_rel = read_json(Path("gold/vidvrd_50_relations.json"))
    gold_traj = read_json(Path("gold/vidvrd_50_trajectories.json"))
    gold, _ = project_artifacts_to_official(gold_rel, gold_traj, ids, prediction=False)
    rows = []
    for mode, threshold in [("none", 1.0)] + [(mode, value) for mode in ("union", "shorter")
                                                   for value in (0.5, 0.7, 0.9)]:
        scoped = {
            video_id: relations[video_id] if mode == "none" else _dedup(relations[video_id], threshold, mode)
            for video_id in ids
        }
        pred, adapter = project_artifacts_to_official(scoped, trajectories, ids, prediction=True)
        metrics = evaluate_official_vidvrd(gold, pred)["relation_detection"]
        rows.append({
            "mode": mode, "threshold": threshold,
            "predictions": adapter["relation_count"],
            "mean_ap": metrics["mean_ap"],
            "recall_at_50": metrics["recall_at"]["50"],
            "recall_at_100": metrics["recall_at"]["100"],
        })
    report = {"scope": "fixed development predictions only; no test data or model calls", "rows": rows}
    output = args.run_dir / "reports" / "proposal_dedup_ablation_dev.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
