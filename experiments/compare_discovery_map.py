from __future__ import annotations

"""Development-only end-to-end mAP ablation for discovered object aliases."""

import argparse
import json
from pathlib import Path

from vidvrd_auto.evaluation.official.vidvrd import evaluate_official_vidvrd, project_artifacts_to_official
from vidvrd_auto.utils.io import read_json


def _ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, default=Path("runs/dev_v9_00005005"))
    parser.add_argument("--discovery-run", type=Path, required=True)
    args = parser.parse_args()
    ids = _ids(Path("experiments/dev_videos.txt"))
    changed = {Path(line).stem for line in _ids(Path("experiments/discovery_alias_dev_paths.txt"))}
    baseline_rel = read_json(args.baseline_run / "pred" / "relations.json")
    baseline_traj = read_json(args.baseline_run / "pred" / "trajectories.json")
    discovery_rel = read_json(args.discovery_run / "pred" / "relations.json")
    discovery_traj = read_json(args.discovery_run / "pred" / "trajectories.json")
    manifest = read_json(args.discovery_run / "run_manifest.json")
    states = {item["video_id"]: item["state"] for item in manifest["videos"]}
    if not changed <= set(ids) or any(states.get(video_id) != "succeeded" for video_id in changed):
        raise ValueError("all changed videos must be completed development videos")
    gold_rel = read_json(Path("gold/vidvrd_50_relations.json"))
    gold_traj = read_json(Path("gold/vidvrd_50_trajectories.json"))
    gold, _ = project_artifacts_to_official(gold_rel, gold_traj, ids, prediction=False)
    rows = {}
    for name, relations, trajectories in (
        ("baseline", baseline_rel, baseline_traj),
        ("discovery", {
            video_id: discovery_rel[video_id] if video_id in changed else baseline_rel[video_id]
            for video_id in ids
        }, {
            video_id: discovery_traj[video_id] if video_id in changed else baseline_traj[video_id]
            for video_id in ids
        }),
    ):
        pred, adapter = project_artifacts_to_official(relations, trajectories, ids, prediction=True)
        metrics = evaluate_official_vidvrd(gold, pred)["relation_detection"]
        rows[name] = {
            "mean_ap": metrics["mean_ap"],
            "recall_at_50": metrics["recall_at"]["50"],
            "recall_at_100": metrics["recall_at"]["100"],
            "prediction_count": adapter["relation_count"],
            "per_video_ap": metrics["per_video_ap"],
        }
    report = {
        "scope": "fixed 10-video development set; only four alias-discovery videos replaced; test untouched",
        "changed_video_ids": sorted(changed),
        "results": rows,
        "mean_ap_delta": rows["discovery"]["mean_ap"] - rows["baseline"]["mean_ap"],
    }
    output = args.discovery_run / "reports" / "discovery_map_ablation_dev.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
