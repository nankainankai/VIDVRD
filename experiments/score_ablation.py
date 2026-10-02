from __future__ import annotations

"""Compare rule-source calibration on completed development predictions only."""

import argparse
import json
from pathlib import Path

from vidvrd_auto.evaluation.official.vidvrd import evaluate_official_vidvrd, project_artifacts_to_official


def _ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def _rescore(items: dict, coefficient: float) -> dict:
    result = {}
    for video_id, relations in items.items():
        output = []
        for relation in relations:
            item = dict(relation)
            if item.get("source_kind") == "rule":
                item["ranking_score"] = round(
                    min(1.0, float(item["evidence_score"]) * float(item["temporal_quality"])
                        * float(item["track_pair_quality"]) * coefficient), 6
                )
                item["source_calibration"] = coefficient
            output.append(item)
        result[video_id] = output
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run_dir.resolve()
    ids = _ids(root / "experiments" / "dev_videos.txt")
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    states = {item["video_id"]: item["state"] for item in manifest["videos"]}
    if any(states.get(video_id) != "succeeded" for video_id in ids):
        raise ValueError("all 10 fixed development videos must finish before score calibration")
    gold_rel = json.loads((root / "gold" / "vidvrd_50_relations.json").read_text(encoding="utf-8"))
    gold_traj = json.loads((root / "gold" / "vidvrd_50_trajectories.json").read_text(encoding="utf-8"))
    pred_rel = json.loads((run / "pred" / "relations.json").read_text(encoding="utf-8"))
    pred_traj = json.loads((run / "pred" / "trajectories.json").read_text(encoding="utf-8"))
    if any("source_kind" not in item for video_id in ids for item in pred_rel.get(video_id, [])):
        raise ValueError("predictions need source_kind; rerun global/export with the current code")
    gold, _ = project_artifacts_to_official(gold_rel, gold_traj, ids, prediction=False)
    trajectories = {video_id: pred_traj.get(video_id, []) for video_id in ids}
    rows = []
    for coefficient in (0.25, 0.40, 0.55, 0.70, 0.85):
        scoped = {video_id: pred_rel.get(video_id, []) for video_id in ids}
        predictions, _ = project_artifacts_to_official(
            _rescore(scoped, coefficient), trajectories, ids, prediction=True
        )
        metrics = evaluate_official_vidvrd(gold, predictions)
        detection = metrics["relation_detection"]
        rows.append({
            "rule_coefficient": coefficient,
            "mean_ap": detection["mean_ap"],
            "recall_at_50": detection["recall_at"]["50"],
            "recall_at_100": detection["recall_at"]["100"],
        })
    baseline = next(row for row in rows if row["rule_coefficient"] == 0.55)
    best = max(rows, key=lambda row: (row["mean_ap"], row["recall_at_50"],
                                     -abs(row["rule_coefficient"] - 0.55)))
    report = {
        "scope": "completed 10-video development set only; test set unseen by this calibration",
        "baseline": baseline,
        "best_development_choice": best if best["mean_ap"] > baseline["mean_ap"] else None,
        "rows": rows,
    }
    output = run / "reports" / "score_ablation_dev.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
