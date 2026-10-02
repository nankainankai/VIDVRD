from __future__ import annotations

"""Compare selected COCO detections against dev annotations diagnostically."""

import argparse
import json
from pathlib import Path

from vidvrd_auto.evaluation.diagnostic.track_aligned import _box_iou
from vidvrd_auto.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args()
    probe = read_json(args.probe)
    video_id = Path(probe["video"]).stem
    gold = read_json(Path("gold/vidvrd_50_trajectories.json"))[video_id]
    rows = []
    for frame in probe["rows"]:
        index = str(frame["frame"])
        for track in gold:
            box = track["trajectory"].get(index)
            if box is None:
                continue
            candidates = [item for item in frame["objects"] if item["category"] == track["category"]]
            best = max(((_box_iou(item["bbox"], box), item) for item in candidates),
                       key=lambda pair: pair[0][0] / pair[0][1] if pair[0][1] else 0.0, default=None)
            iou = best[0][0] / best[0][1] if best and best[0][1] else 0.0
            rows.append({"frame": int(index), "gold_track_id": track["track_id"],
                         "category": track["category"], "best_iou": round(iou, 4),
                         "score": best[1]["score"] if best else None})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
