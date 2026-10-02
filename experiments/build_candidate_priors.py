from __future__ import annotations

"""Build leakage-free priors from Gold train videos excluding fixed development videos."""

import argparse
import json
from pathlib import Path

from vidvrd_auto.relations.candidate_priors import learn_predicate_priors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    gold = root / "gold"
    manifest = json.loads((gold / "vidvrd_50_manifest.json").read_text(encoding="utf-8"))
    relations = json.loads((gold / "vidvrd_50_relations.json").read_text(encoding="utf-8"))
    trajectories = json.loads((gold / "vidvrd_50_trajectories.json").read_text(encoding="utf-8"))
    development = {
        line.strip() for line in (root / "experiments" / "dev_videos.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    training = {video_id for video_id, split in manifest["video_splits"].items() if split == "train"} - development
    if training & development or any(manifest["video_splits"][video_id] != "train" for video_id in training):
        raise ValueError("prior training set overlaps development or test")
    priors = learn_predicate_priors(relations, trajectories, training)
    priors["training_video_ids"] = sorted(training)
    priors["held_out_development_video_ids"] = sorted(development)
    output = root / "experiments" / "predicate_priors_train30.json"
    output.write_text(json.dumps(priors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output} from {len(training)} training videos; excluded {len(development)} development videos")


if __name__ == "__main__":
    main()
