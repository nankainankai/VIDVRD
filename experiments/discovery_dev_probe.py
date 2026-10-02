from __future__ import annotations

"""Inspect ontology-mappable scene vocabulary on the fixed dev videos."""

import argparse
from pathlib import Path

from vidvrd_auto.config.loader import load_config
from vidvrd_auto.nodes.vocabulary import build_vocabulary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, default=Path("experiments/dev_video_paths.txt"))
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    config = dict(load_config()["vocabulary"])
    config.update(discovery_enabled=True, ontology_only_discovery=True)
    videos = [Path(line.strip()) for line in args.videos.read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    for video in videos:
        root = args.out_root / video.stem
        result = build_vocabulary(
            video_path=video, out_json=root / "objects.json",
            evidence_path=root / "evidence.jpg", config=config,
        )
        discovered = [entry["raw_label"] for entry in result["entries"] if entry["ontology_source"] == "discovered"]
        print(f"{video.stem}: state={result['discovery']['state']} aliases={discovered}", flush=True)


if __name__ == "__main__":
    main()
