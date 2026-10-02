"""Controlled regression: freeze existing detections and rerun tracking/relations.

This does not measure detection quality, GPU time, or detection cost.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from unittest.mock import patch

from vidvrd_auto.cli import build_parser
from vidvrd_auto.pipeline.run import run_pipeline
from vidvrd_auto.utils.hashing import sha256_file
from vidvrd_auto.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/main.json"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--video-id", action="append", help="Replay only these IDs from the source run")
    options = parser.parse_args()
    sources = []
    frozen = {}
    selected = set(options.video_id or [])
    for run in options.source_run:
        manifest = read_json(run / "run_manifest.json")
        for item in manifest["videos"]:
            video_id = item["video_id"]
            if selected and video_id not in selected:
                continue
            if video_id in frozen:
                raise ValueError(f"duplicate input: {video_id}")
            frozen[video_id] = run.resolve() / "videos" / video_id / "detect"
            sources.append(item["source"])
    if selected - frozen.keys():
        raise ValueError(f"video IDs absent from source runs: {sorted(selected - frozen.keys())}")
    options.run_dir.mkdir(parents=True, exist_ok=True)
    source_list = options.run_dir.resolve() / "video_paths.txt"
    source_list.write_text("\n".join(sources) + "\n", encoding="utf-8")

    def replay(*, video_path, out_dir, config, log_path):
        source = frozen[out_dir.parent.name]
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("detections.jsonl", "meta.json"):
            shutil.copy2(source / name, out_dir / name)
        write_json(log_path, {"mode": "frozen_detection_replay", "source": str(source)})

    args = build_parser().parse_args([
        "--videos", str(source_list), "--run-dir", str(options.run_dir),
        "--config", str(options.config), *(["--resume"] if options.resume else []),
    ])
    try:
        with patch("vidvrd_auto.pipeline.media.detect_video", replay):
            run_pipeline(args=args, config_path=options.config.resolve())
    finally:
        manifest_path = options.run_dir / "run_manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            manifest["validation_scope"] = {
                "mode": "frozen_detection_replay",
                "detector_time_and_cost_measured": False,
                "sources": {vid: {"path": str(path), "sha256": sha256_file(path / "detections.jsonl")}
                            for vid, path in frozen.items()},
            }
            manifest.setdefault("provenance", {}).setdefault("algorithms", {})["detector"] = {
                "name": "frozen_detection_replay", "original": "Rex-Omni",
            }
            write_json(manifest_path, manifest)
            from vidvrd_auto.evaluation.runtime import summarize_run
            summarize_run(options.run_dir)


if __name__ == "__main__":
    main()
