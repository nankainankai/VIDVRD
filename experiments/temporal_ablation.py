from __future__ import annotations

"""Development-only diagnostic for fixed-size relation tube proposals.

This changes prediction proposals, never the official evaluator or ground truth.
"""

import argparse
import json
from pathlib import Path

from vidvrd_auto.evaluation.official.vidvrd import evaluate_official_vidvrd, project_artifacts_to_official


def tile_predictions(predictions: dict, segment_frames: int) -> dict:
    tiled = {}
    for video_id, relations in predictions.items():
        output = []
        for relation in relations:
            start, end = relation["duration"]
            cursor = start
            while cursor < end:
                boundary = ((cursor // segment_frames) + 1) * segment_frames
                stop = min(end, boundary)
                offset_start, offset_stop = cursor - start, stop - start
                proposal = dict(relation)
                proposal["duration"] = [cursor, stop]
                proposal["sub_traj"] = relation["sub_traj"][offset_start:offset_stop]
                proposal["obj_traj"] = relation["obj_traj"][offset_start:offset_stop]
                output.append(proposal)
                cursor = stop
        tiled[video_id] = output
    return tiled


def window_predictions(predictions: dict, window_frames: int, stride_frames: int) -> dict:
    """Overlap windows without reading annotation timestamps or changing scores."""

    result = {}
    for video_id, relations in predictions.items():
        output = []
        for relation in relations:
            start, end = relation["duration"]
            starts = {start}
            starts.update(range(((start + stride_frames - 1) // stride_frames) * stride_frames,
                                end, stride_frames))
            for cursor in sorted(starts):
                stop = min(end, cursor + window_frames)
                if stop - cursor < min(stride_frames, end - start):
                    continue
                lo, hi = cursor - start, stop - start
                proposal = dict(relation)
                proposal["duration"] = [cursor, stop]
                proposal["sub_traj"] = relation["sub_traj"][lo:hi]
                proposal["obj_traj"] = relation["obj_traj"][lo:hi]
                output.append(proposal)
        result[video_id] = output
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--video-ids", required=True, help="Comma-separated completed development video IDs")
    parser.add_argument("--segment-frames", type=int, default=30)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.segment_frames <= 0:
        raise ValueError("segment-frames must be positive")
    root, run = args.root.resolve(), args.run_dir.resolve()
    ids = [item.strip() for item in args.video_ids.split(",") if item.strip()]
    relations, trajectories = {}, {}
    for video_id in ids:
        base = run / "videos" / video_id / "export"
        for dest, name in ((relations, "relations.json"), (trajectories, "trajectories.json")):
            source = json.loads((base / name).read_text(encoding="utf-8"))
            dest[video_id] = source[video_id]
    gold_rel = json.loads((root / "gold" / "vidvrd_50_relations.json").read_text(encoding="utf-8"))
    gold_traj = json.loads((root / "gold" / "vidvrd_50_trajectories.json").read_text(encoding="utf-8"))
    gold, _ = project_artifacts_to_official(gold_rel, gold_traj, ids, prediction=False)
    baseline, _ = project_artifacts_to_official(relations, trajectories, ids, prediction=True)
    proposals = tile_predictions(baseline, args.segment_frames)
    sliding = window_predictions(baseline, args.segment_frames, max(1, args.segment_frames // 2))
    report = {
        "scope": "completed development videos only; proposal ablation, not a test metric",
        "video_ids": ids,
        "segment_frames": args.segment_frames,
        "baseline": evaluate_official_vidvrd(gold, baseline)["relation_detection"],
        "tiled": evaluate_official_vidvrd(gold, proposals)["relation_detection"],
        "sliding": evaluate_official_vidvrd(gold, sliding)["relation_detection"],
        "baseline_proposals": sum(map(len, baseline.values())),
        "tiled_proposals": sum(map(len, proposals.values())),
        "sliding_proposals": sum(map(len, sliding.values())),
    }
    output = run / "reports" / "temporal_ablation_dev.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
