from __future__ import annotations

"""Offline evaluation entry point for existing prediction artifacts."""

import argparse
from pathlib import Path
from typing import Sequence

from vidvrd_auto.evaluation.vidvrd import run_evaluation_suite
from vidvrd_auto.utils.io import read_json


def _video_ids(value: str, relations: Path, trajectories: Path) -> list[str]:
    if value.strip():
        source = Path(value).expanduser()
        if source.is_file():
            return [
                line.strip() for line in source.read_text(encoding="utf-8-sig").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        return [item.strip() for item in value.split(",") if item.strip()]
    ids: set[str] = set()
    for path in (relations, trajectories):
        obj = read_json(path)
        if isinstance(obj, dict):
            ids.update(str(key) for key in obj)
    return sorted(ids)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate existing VidVRD predictions without rerunning models")
    parser.add_argument("--pred-relations", type=Path, required=True)
    parser.add_argument("--pred-trajectories", type=Path, required=True)
    parser.add_argument("--gold-relations", type=Path, default=Path("gold/vidvrd_50_relations.json"))
    parser.add_argument("--gold-trajectories", type=Path, default=Path("gold/vidvrd_50_trajectories.json"))
    parser.add_argument("--gold-manifest", type=Path, default=Path("gold/vidvrd_50_manifest.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--video-ids", default="", help="Comma-separated IDs or a text file; inferred from predictions by default")
    parser.add_argument("--scope", choices=("requested", "gold_split"), default="gold_split")
    parser.add_argument("--split", default="test")
    parser.add_argument("--presence-clips", type=Path,
                        help="Independent fixed-clip annotation manifest; half-open spans, JSON list")
    parser.add_argument("--expected-official-video-count", type=int, default=200)
    parser.add_argument("--evidence-packets-root", type=Path, default=None,
                        help="Optional run/videos directory for candidate and API coverage diagnostics")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prediction_relations = args.pred_relations.expanduser().resolve()
    prediction_trajectories = args.pred_trajectories.expanduser().resolve()
    video_ids = _video_ids(args.video_ids, prediction_relations, prediction_trajectories)
    if not video_ids:
        raise SystemExit("prediction artifacts contain no video IDs")
    output = args.out_dir.expanduser().resolve()
    result = run_evaluation_suite(
        gold_relations=args.gold_relations.expanduser().resolve(),
        gold_trajectories=args.gold_trajectories.expanduser().resolve(),
        gold_manifest=args.gold_manifest.expanduser().resolve(),
        pred_relations=prediction_relations,
        pred_trajectories=prediction_trajectories,
        requested_video_ids=video_ids,
        official_report_path=output / "official_vidvrd.md",
        official_metrics_path=output / "official_vidvrd.json",
        diagnostic_report_path=output / "diagnostic_track_aligned.md",
        diagnostic_metrics_path=output / "diagnostic_track_aligned.json",
        dataset_split=args.split,
        scope=args.scope,
        expected_official_video_count=args.expected_official_video_count,
        evidence_packets_root=args.evidence_packets_root.expanduser().resolve() if args.evidence_packets_root else None,
        presence_clips_path=args.presence_clips.resolve() if args.presence_clips else None,
    )
    official = result["official"]["relation_detection"]
    mot = result["diagnostic"]["diagnostic_mot"]
    print(f"mAP={official['mean_ap']:.6f} R@50={official['recall_at']['50']:.6f}")
    print(f"diagnostic Gold-frame MOTA={mot['mota']:.6f} IDSW={mot['id_switches']}")
    presence = result['presence']
    print(f"Presence F1={presence['micro']['f1']:.6f} complete_annotations={presence['annotation_complete']}")
    print(f"Known-label coverage@10={result['tagging']['micro_recall_at']['10']:.6f} (category tagging, not tube mAP)")
    subset = result['tagging'].get('requested_subset')
    if subset:
        print(f"Requested {subset['evaluated_video_count']}-video known-label coverage@10={subset['micro_recall_at']['10']:.6f} (subset diagnostic)")
    print(f"reports={output}")


if __name__ == "__main__":
    main()
