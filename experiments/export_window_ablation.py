from __future__ import annotations

"""Development-only export-window ablation from frozen verified relations."""

import argparse
import json
from pathlib import Path

from vidvrd_auto.evaluation.official.vidvrd import evaluate_official_vidvrd, project_artifacts_to_official
from vidvrd_auto.nodes.export import compose_action_spatial_relations, sliding_relation_proposals


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run_dir.resolve()
    ids = [line.strip() for line in (root / "experiments" / "dev_videos.txt").read_text(encoding="utf-8").splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
    manifest = _load(run / "run_manifest.json")
    states = {item["video_id"]: item["state"] for item in manifest["videos"]}
    if any(states.get(video_id) != "succeeded" for video_id in ids):
        raise ValueError("all development videos must have succeeded")
    raw = {
        video_id: _load(run / "videos" / video_id / "verify" / "relations.json")[video_id]
        for video_id in ids
    }
    trajectories = {
        video_id: _load(run / "videos" / video_id / "export" / "trajectories.json")[video_id]
        for video_id in ids
    }
    gold_rel = _load(root / "gold" / "vidvrd_50_relations.json")
    gold_traj = _load(root / "gold" / "vidvrd_50_trajectories.json")
    gold, _ = project_artifacts_to_official(gold_rel, gold_traj, ids, prediction=False)
    rows = []
    for window in (0, 20, 30, 45, 60):
        stride = max(1, window // 2)
        scoped = {}
        for video_id in ids:
            relations = raw[video_id]
            if window:
                relations = sliding_relation_proposals(relations, window_frames=window, stride_frames=stride)
            scoped[video_id] = compose_action_spatial_relations(
                relations, min_overlap_frames=15, score_coefficient=0.55,
            )
        predictions, adapter = project_artifacts_to_official(scoped, trajectories, ids, prediction=True)
        metrics = evaluate_official_vidvrd(gold, predictions)["relation_detection"]
        rows.append({
            "window_frames": window,
            "stride_frames": stride if window else 0,
            "prediction_count": adapter["relation_count"],
            "mean_ap": metrics["mean_ap"],
            "recall_at_50": metrics["recall_at"]["50"],
            "recall_at_100": metrics["recall_at"]["100"],
            "per_video_ap": metrics["per_video_ap"],
        })
    report = {
        "scope": "frozen verified predictions from fixed 10-video development set; no API calls or test videos",
        "current_window_frames": 30,
        "rows": rows,
    }
    output = run / "reports" / "export_window_ablation_dev.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
