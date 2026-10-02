"""Offline, controlled postprocessing ablations using identical model outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

from vidvrd_auto.config.loader import load_config
from vidvrd_auto.evaluation.vidvrd import run_evaluation_suite
from vidvrd_auto.nodes.global_relation import run_global_relation
from vidvrd_auto.utils.io import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/main.json"))
    parser.add_argument("--max-merged-span", type=int, default=0)
    parser.add_argument("--rule-only", action="store_true")
    parser.add_argument("--split", default="train")
    parser.add_argument("--temporal-proposal-mode", choices=("single", "multiscale"), default="single")
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = read_json(args.run / "run_manifest.json")
    ids = [item["video_id"] for item in manifest["videos"]]
    relations = {}
    for vid in ids:
        root = args.run / "videos" / vid
        qc = read_json(root / "track_qc" / "report.json")
        gc = dict(cfg.get("global_relation", {}))
        gc.update(max_merged_span_frames=args.max_merged_span,
                  temporal_proposal_mode=args.temporal_proposal_mode,
                  source_coefficients=cfg.get("relation_scoring", {}).get("source_coefficients", {}),
                  track_quality=qc.get("track_quality", {}),
                  track_jump_frames={str(tid): value.get("jump_frames", [])
                                     for tid, value in qc.get("track_quality", {}).items() if isinstance(value, dict)})
        # No review: the exact same window outputs, no additional API calls.
        output = args.out_dir / "videos" / vid / "relations.json"
        run_global_relation(video_id=vid, relations_json=root / ("rule" if args.rule_only else "merge") / "relations.json",
                            out_json=output, config=gc)
        relations[vid] = read_json(output)[vid]
    pred_path = args.out_dir / "relations.json"
    write_json(pred_path, relations)
    write_json(args.out_dir / "ablation_manifest.json", {
        "source_run": str(args.run.resolve()), "same_semantic_outputs": not args.rule_only,
        "additional_api_calls": 0, "review_enabled": False, "rule_only": args.rule_only,
        "max_merged_span_frames": args.max_merged_span,
        "temporal_proposal_mode": args.temporal_proposal_mode,
        "secondary_proposal_decay": cfg.get("global_relation", {}).get("secondary_proposal_decay", 0.5),
    })
    reports = args.out_dir / "reports"
    result = run_evaluation_suite(
        gold_relations=Path("gold/vidvrd_50_relations.json"),
        gold_trajectories=Path("gold/vidvrd_50_trajectories.json"),
        gold_manifest=Path("gold/vidvrd_50_manifest.json"), pred_relations=pred_path,
        pred_trajectories=args.run / "pred" / "trajectories.json", requested_video_ids=ids,
        official_report_path=reports / "official_vidvrd.md", official_metrics_path=reports / "official_vidvrd.json",
        diagnostic_report_path=reports / "diagnostic_track_aligned.md", diagnostic_metrics_path=reports / "diagnostic_track_aligned.json",
        dataset_split=args.split, scope="requested", evidence_packets_root=args.run / "videos",
    )
    print(result["official"]["relation_detection"])


if __name__ == "__main__":
    main()
