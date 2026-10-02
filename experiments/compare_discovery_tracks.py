from __future__ import annotations

"""Compare discovered-alias OC-SORT tracks with the frozen dev baseline."""

import argparse
import json
from pathlib import Path

from vidvrd_auto.evaluation.diagnostic.track_aligned import align_trajectories, clear_mot_on_gold_frames
from vidvrd_auto.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, default=Path("runs/dev_v9_00005005"))
    parser.add_argument("--videos", type=Path, default=Path("experiments/discovery_alias_dev_paths.txt"))
    args = parser.parse_args()
    ids = [Path(line.strip()).stem for line in args.videos.read_text(encoding="utf-8").splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
    gold = read_json(Path("gold/vidvrd_50_trajectories.json"))
    baseline = read_json(args.baseline_run / "pred" / "trajectories.json")
    rows = []
    for video_id in ids:
        path = args.run_dir / "videos" / video_id / "export" / "trajectories.json"
        if not path.is_file():
            continue
        candidate = read_json(path)[video_id]
        row = {"video_id": video_id, "gold_tracks": len(gold[video_id])}
        for name, tracks in (("baseline", baseline[video_id]), ("discovery", candidate)):
            _, matches = align_trajectories(gold[video_id], tracks, 0.3)
            mot = clear_mot_on_gold_frames(gold[video_id], tracks)
            row[name] = {
                "predicted_tracks": len(tracks), "matched_tracks": len(matches),
                "mota": mot["mota"], "matches": matches,
            }
        rows.append(row)
    totals = {
        "gold_tracks": sum(row["gold_tracks"] for row in rows),
        "baseline_matched": sum(row["baseline"]["matched_tracks"] for row in rows),
        "discovery_matched": sum(row["discovery"]["matched_tracks"] for row in rows),
    }
    report = {"scope": "available fixed dev videos only; tracking diagnostic, not relation mAP", "rows": rows, "totals": totals}
    output = args.run_dir / "track_comparison.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
