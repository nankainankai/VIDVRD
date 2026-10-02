from __future__ import annotations

"""One-shot frozen-test check of the dev-selected temporal de-duplication."""

import argparse
import json
from pathlib import Path

from vidvrd_auto.evaluation.official.vidvrd import evaluate_official_vidvrd, project_artifacts_to_official
from vidvrd_auto.nodes.export import deduplicate_relation_proposals
from vidvrd_auto.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/test_v10_frozen"))
    args = parser.parse_args()
    ids = [line.strip() for line in Path("experiments/test_videos.txt").read_text(encoding="utf-8").splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
    gold_rel = read_json(Path("gold/vidvrd_50_relations.json"))
    gold_traj = read_json(Path("gold/vidvrd_50_trajectories.json"))
    relations = read_json(args.run_dir / "pred" / "relations.json")
    trajectories = read_json(args.run_dir / "pred" / "trajectories.json")
    gold, _ = project_artifacts_to_official(gold_rel, gold_traj, ids, prediction=False)
    rows = {}
    for name, scoped in (
        ("baseline", {video_id: relations[video_id] for video_id in ids}),
        ("dedup_0_7", {video_id: deduplicate_relation_proposals(relations[video_id], temporal_iou_threshold=0.7)
                       for video_id in ids}),
    ):
        prediction, adapter = project_artifacts_to_official(scoped, trajectories, ids, prediction=True)
        result = evaluate_official_vidvrd(gold, prediction)["relation_detection"]
        rows[name] = {
            "mean_ap": result["mean_ap"],
            "recall_at_50": result["recall_at"]["50"],
            "recall_at_100": result["recall_at"]["100"],
            "prediction_count": adapter["relation_count"],
        }
    report = {
        "scope": "one-shot frozen 10-video test; threshold 0.7 chosen on development set only",
        "results": rows,
        "mean_ap_delta": rows["dedup_0_7"]["mean_ap"] - rows["baseline"]["mean_ap"],
    }
    output = args.run_dir / "reports" / "dedup_test_validation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
