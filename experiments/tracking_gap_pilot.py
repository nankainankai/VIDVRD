from __future__ import annotations

"""Offline interpolation-gap ablation on cached detections (no Rex/API calls)."""

import argparse
import json
from pathlib import Path

from vidvrd_auto.config.loader import load_config
from vidvrd_auto.evaluation.diagnostic.track_aligned import align_trajectories, clear_mot_on_gold_frames
from vidvrd_auto.nodes.export import tracks_to_trajectories
from vidvrd_auto.tracking.video import track_video
from vidvrd_auto.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    video_id = args.video.stem
    config = dict(load_config(Path("configs/benchmark_dev.json"))["tracking"])
    gold = read_json(Path("gold/vidvrd_50_trajectories.json"))[video_id]
    rows = []
    for seconds in (1.0, 1.5, 2.0, 2.5):
        config["adaptive_interpolation_max_seconds"] = seconds
        output = args.out_root / f"gap_{seconds:g}"
        track_video(video_path=args.video, detections_path=args.detections, out_dir=output, config=config)
        trajectory_path = output / "trajectories.json"
        tracks_to_trajectories(output / "tracks.jsonl", video_id, trajectory_path)
        predicted = read_json(trajectory_path)[video_id]
        _, matches = align_trajectories(gold, predicted, 0.3)
        mot = clear_mot_on_gold_frames(gold, predicted)
        log = (output / "run.log").read_text(encoding="utf-8")
        rows.append({
            "max_seconds": seconds,
            "predicted_tracks": len(predicted),
            "matched_tracks": len(matches),
            "mota": mot["mota"],
            "false_positives": mot["false_positives"],
            "interpolated_frames": int(next(line.split("=", 1)[1] for line in log.splitlines()
                                            if line.startswith("interpolated="))),
        })
    report = {"scope": "cached detections and one dev video; tracking diagnostic only", "rows": rows}
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
