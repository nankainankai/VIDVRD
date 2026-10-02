"""Offline router coverage only; does not rerun or score semantic inference."""
from __future__ import annotations

import argparse
from pathlib import Path

from vidvrd_auto.evaluation.routing import diagnose_routed_candidates
from vidvrd_auto.relations.candidate_router import route_predicates
from vidvrd_auto.relations.object_candidates import GEOMETRY_PREDICATES
from vidvrd_auto.utils.hashing import sha256_file
from vidvrd_auto.utils.io import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--axis-motion-reserve", action="store_true")
    args = parser.parse_args()
    if args.candidate_limit < 1:
        parser.error("candidate limit must be positive")
    ids = [item["video_id"] for item in read_json(args.run / "run_manifest.json")["videos"]]
    splits = read_json(Path("gold/vidvrd_50_manifest.json"))["video_splits"]
    if any(splits.get(vid) != "train" for vid in ids):
        raise ValueError("router budget development experiment requires only Gold train videos")
    hashes = {}
    original_counts = []
    new_counts = []
    for vid in ids:
        source = args.run / "videos" / vid / "semantic" / "evidence_packets.json"
        obj = read_json(source)
        hashes[str(source.resolve())] = sha256_file(source)
        for packet in obj["packets"]:
            for direction in packet["candidate_directions"]:
                original_counts.append(len(direction["candidate_predicates"]))
                key = f"{direction['subject_track_id']}->{direction['object_track_id']}"
                direction.update(route_predicates(
                    packet["trajectory_evidence"][key], limit=args.candidate_limit,
                    exclude=GEOMETRY_PREDICATES,
                    subject_category=direction["subject_category"],
                    object_category=direction["object_category"],
                    axis_motion_reserve=args.axis_motion_reserve,
                ))
                new_counts.append(len(direction["candidate_predicates"]))
        write_json(args.out_dir / "videos" / vid / "semantic" / "evidence_packets.json", obj)
    gold = read_json(Path("gold/vidvrd_50_relations.json"))
    gold_tracks = read_json(Path("gold/vidvrd_50_trajectories.json"))
    tracks = read_json(args.run / "pred" / "trajectories.json")
    old = diagnose_routed_candidates(gold, gold_tracks, tracks, ids, args.run / "videos")
    new = diagnose_routed_candidates(gold, gold_tracks, tracks, ids, args.out_dir / "videos")
    report = {
        "task": "candidate_eligibility_diagnostic_only",
        "additional_api_calls": 0,
        "semantic_inference_rerun": False,
        "achieved_relation_recall": None,
        "candidate_limit": args.candidate_limit,
        "axis_motion_reserve": args.axis_motion_reserve,
        "source_packet_sha256": hashes,
        "mean_candidates_per_direction_before": sum(original_counts) / len(original_counts) if original_counts else 0,
        "mean_candidates_per_direction_after": sum(new_counts) / len(new_counts) if new_counts else 0,
        "before": old,
        "after": new,
    }
    write_json(args.out_dir / "routing_budget.json", report)
    print({"before": old["routed_candidate_recall"], "after": new["routed_candidate_recall"],
           "before_counts": old["counts"], "after_counts": new["counts"], "additional_api_calls": 0})


if __name__ == "__main__":
    main()
