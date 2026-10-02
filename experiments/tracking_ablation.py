from __future__ import annotations

"""Offline OC-SORT ablation using cached detections (no detector/API calls)."""

import argparse
import json
from pathlib import Path
from typing import Any

from vidvrd_auto.config.loader import load_config
from vidvrd_auto.evaluation.diagnostic.track_aligned import run_track_aligned_diagnostic
from vidvrd_auto.nodes.export import merge_trajectory_files, tracks_to_trajectories
from vidvrd_auto.tracking.video import track_video


VARIANTS: dict[str, dict[str, Any]] = {
    "fixed": {},
    "iou015": {"iou_threshold": 0.15},
    "iou020_age20": {"iou_threshold": 0.20, "max_age": 20},
    "iou010_age20": {"iou_threshold": 0.10, "max_age": 20},
    "class_agnostic": {"class_aware": False, "iou_threshold": 0.20, "max_age": 20},
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark_dev.json"))
    parser.add_argument("--gold-relations", type=Path, default=Path("gold/vidvrd_50_relations.json"))
    parser.add_argument("--gold-trajectories", type=Path, default=Path("gold/vidvrd_50_trajectories.json"))
    parser.add_argument("--variants", nargs="*", choices=sorted(VARIANTS), default=list(VARIANTS))
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    video_paths = [Path(line.strip()) for line in args.videos.read_text(encoding="utf-8").splitlines() if line.strip()]
    video_ids = [path.stem for path in video_paths]
    base = load_config(args.config)
    summaries: dict[str, Any] = {}
    for name in args.variants:
        tracking = dict(base["tracking"])
        tracking.update(VARIANTS[name])
        root = args.out_root / name
        exports: list[tuple[str, Path]] = []
        for video_path, video_id in zip(video_paths, video_ids):
            out_dir = root / "videos" / video_id
            detections = args.source_run / "videos" / video_id / "detect" / "detections.jsonl"
            track_video(video_path=video_path, detections_path=detections, out_dir=out_dir, config=tracking)
            trajectory_path = out_dir / "trajectories.json"
            tracks_to_trajectories(out_dir / "tracks.jsonl", video_id, trajectory_path)
            exports.append((video_id, trajectory_path))
        trajectories = root / "trajectories.json"
        merge_trajectory_files(exports, trajectories, video_ids)
        metrics = run_track_aligned_diagnostic(
            gold_relations=args.gold_relations,
            gold_trajectories=args.gold_trajectories,
            pred_relations=args.source_run / "pred" / "relations.json",
            pred_trajectories=trajectories,
            report_path=root / "diagnostic.md",
            metrics_path=root / "diagnostic.json",
            video_ids=video_ids,
        )
        tracks = metrics["diagnostic_tracks"]
        mot = metrics["diagnostic_mot"]
        summaries[name] = {
            "matched_tracks": tracks["matched"],
            "gold_tracks": tracks["gold"],
            "track_recall": tracks["gold_recall"],
            "mean_track_iou": tracks["mean_gold_supported_iou"],
            "mota": mot["mota"],
            "motp": mot["motp"],
            "false_negatives": mot["false_negatives"],
            "false_positives": mot["false_positives"],
            "id_switches": mot["id_switches"],
        }
        print(json.dumps({name: summaries[name]}, ensure_ascii=False))
    (args.out_root / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (args.out_root / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
