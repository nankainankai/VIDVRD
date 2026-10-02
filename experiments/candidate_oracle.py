from __future__ import annotations

"""Candidate-recall upper bound on Gold tracks; never report as pipeline mAP."""

import argparse
from collections import Counter
import json
from pathlib import Path

from vidvrd_auto.relations.candidate_router import route_predicates
from vidvrd_auto.relations.evidence_features import trajectory_evidence
from vidvrd_auto.relations.object_candidates import GEOMETRY_PREDICATES


def _ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--limit", type=int, default=14)
    args = parser.parse_args()
    root = args.root.resolve()
    gold = root / "gold"
    relations = json.loads((gold / "vidvrd_50_relations.json").read_text(encoding="utf-8"))
    trajectories = json.loads((gold / "vidvrd_50_trajectories.json").read_text(encoding="utf-8"))
    priors = json.loads((root / "experiments" / "predicate_priors_train30.json").read_text(encoding="utf-8"))
    development = _ids(root / "experiments" / "dev_videos.txt")
    test = set(_ids(root / "experiments" / "test_videos.txt"))
    if set(development) & (set(priors["training_video_ids"]) | test):
        raise ValueError("development overlaps prior training or test")
    selected = development if args.split == "dev" else sorted(test)
    if set(selected) & set(priors["training_video_ids"]):
        raise ValueError("evaluation videos overlap prior training")

    results = {}
    for mode in ("evidence_only", "train30_prior"):
        counts: Counter[str] = Counter()
        missed: Counter[str] = Counter()
        for video_id in selected:
            tubes = {int(item["track_id"]): {int(frame): box for frame, box in item["trajectory"].items()}
                     for item in trajectories[video_id]}
            categories = {int(item["track_id"]): str(item["category"]) for item in trajectories[video_id]}
            for relation in relations[video_id]:
                subject = int(relation["subject_track_id"])
                obj = int(relation["object_track_id"])
                frames = sorted(set(tubes[subject]) & set(tubes[obj])
                                & set(range(int(relation["start_frame"]), int(relation["end_frame"]) + 1)))
                if not frames:
                    continue
                tracks = {frame: [
                    {"track_id": subject, "bbox": tubes[subject][frame]},
                    {"track_id": obj, "bbox": tubes[obj][frame]},
                ] for frame in frames}
                route = route_predicates(
                    trajectory_evidence(tracks, frames, subject, obj),
                    limit=args.limit, exclude=GEOMETRY_PREDICATES,
                    subject_category=categories[subject], object_category=categories[obj],
                    predicate_priors=priors if mode == "train30_prior" else None,
                )
                predicate = str(relation["predicate"])
                kind = "composite" if "_" in predicate else "atomic"
                counts[f"{kind}_total"] += 1
                if predicate in route["candidate_predicates"]:
                    counts[f"{kind}_hit"] += 1
                else:
                    missed[predicate] += 1
        results[mode] = {
            "counts": dict(counts),
            "atomic_candidate_recall": counts["atomic_hit"] / max(1, counts["atomic_total"]),
            "composite_candidate_recall": counts["composite_hit"] / max(1, counts["composite_total"]),
            "top_missing_predicates": missed.most_common(20),
        }
    report = {
        "scope": f"10 held-out {args.split} videos; Gold tracks/categories; candidate upper-bound diagnostic only",
        "not_official_map": True,
        "candidate_limit": args.limit,
        "prior_training_video_count": priors["training_video_count"],
        "results": results,
    }
    output = args.output or root / "experiments" / f"candidate_oracle_{args.split}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
