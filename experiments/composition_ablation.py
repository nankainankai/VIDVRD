from __future__ import annotations

"""Offline ablation for composing action and spatial relation predictions."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from vidvrd_auto.core.ontology import load_ontology
from vidvrd_auto.core.schema import relation_span
from vidvrd_auto.evaluation.official.vidvrd import evaluate_official_vidvrd, project_artifacts_to_official


def _score(item: dict[str, Any]) -> float:
    for key in ("ranking_score", "agent_score", "rule_support", "confidence"):
        if item.get(key) is not None:
            return max(0.0, min(1.0, float(item[key])))
    return 0.0


def _compose(
    predictions: dict[str, list[dict[str, Any]]], *, coefficient: float, min_overlap: int
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    definitions = []
    for record in load_ontology()["predicates"]:
        components = record.get("components", {})
        action, spatial = str(components.get("action", "")), str(components.get("spatial", ""))
        if action and spatial:
            definitions.append((str(record["name"]), action, spatial))

    output: dict[str, list[dict[str, Any]]] = {}
    added = 0
    for video_id, source in predictions.items():
        items = [dict(item) for item in source]
        by_pair_predicate: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
        by_pair_action: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
        known = set()
        for item in items:
            pair = (int(item["subject_track_id"]), int(item["object_track_id"]), str(item["predicate"]))
            by_pair_predicate.setdefault(pair, []).append(item)
            components = next(
                (record.get("components", {}) for record in load_ontology()["predicates"]
                 if str(record["name"]) == str(item["predicate"])),
                {},
            )
            action = str(components.get("action", ""))
            if action:
                by_pair_action.setdefault((pair[0], pair[1], action), []).append(item)
            # Riding/driving visually entails a seated posture. This generic
            # implication lets a verified interaction support VidVRD's
            # composite sit_* predicates without asking the model to emit a
            # non-existent atomic "sit" label.
            if str(item["predicate"]) in {"ride", "drive"}:
                by_pair_action.setdefault((pair[0], pair[1], "sit"), []).append(item)
            span = relation_span(item)
            known.add((*pair, span.start_frame, span.end_frame))

        for composite, action, spatial in definitions:
            pairs = {
                (sid, oid) for sid, oid, candidate_action in by_pair_action if candidate_action == action
            } & {
                (sid, oid) for sid, oid, predicate in by_pair_predicate if predicate == spatial
            }
            for sid, oid in pairs:
                for action_item in by_pair_action[(sid, oid, action)]:
                    action_span = relation_span(action_item)
                    for spatial_item in by_pair_predicate[(sid, oid, spatial)]:
                        spatial_span = relation_span(spatial_item)
                        start = max(action_span.start_frame, spatial_span.start_frame)
                        end = min(action_span.end_frame, spatial_span.end_frame)
                        if end - start < min_overlap or (sid, oid, composite, start, end) in known:
                            continue
                        action_score, spatial_score = _score(action_item), _score(spatial_item)
                        item = dict(action_item)
                        item.update(
                            predicate=composite,
                            start_frame=start,
                            end_frame=end,
                            span_convention="half_open",
                            ranking_score=round(math.sqrt(action_score * spatial_score) * coefficient, 6),
                            evidence_score=round(math.sqrt(action_score * spatial_score), 6),
                            source="action_spatial_composition",
                            source_kind="composition",
                            sources=["semantic", "rule", "composition"],
                            predicate_components={"action": action, "spatial": spatial, "comparative": ""},
                            composition={"action": action, "spatial": spatial},
                        )
                        evidence = set(action_item.get("evidence_frames", [])) & set(spatial_item.get("evidence_frames", []))
                        item["evidence_frames"] = sorted(int(frame) for frame in evidence if start <= int(frame) < end)
                        items.append(item)
                        known.add((sid, oid, composite, start, end))
                        added += 1
        output[video_id] = items
    return output, added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--videos", type=Path, default=Path("experiments/dev_video_paths.txt"))
    args = parser.parse_args()
    run = args.run_dir.resolve()
    video_ids = [Path(line.strip()).stem for line in args.videos.read_text(encoding="utf-8").splitlines() if line.strip()]
    root = Path(__file__).resolve().parents[1]
    pred_rel = json.loads((run / "pred" / "relations.json").read_text(encoding="utf-8"))
    pred_traj = json.loads((run / "pred" / "trajectories.json").read_text(encoding="utf-8"))
    gold_rel = json.loads((root / "gold" / "vidvrd_50_relations.json").read_text(encoding="utf-8"))
    gold_traj = json.loads((root / "gold" / "vidvrd_50_trajectories.json").read_text(encoding="utf-8"))
    gold, _ = project_artifacts_to_official(gold_rel, gold_traj, video_ids, prediction=False)
    trajectories = {video_id: pred_traj.get(video_id, []) for video_id in video_ids}
    rows = []
    for min_overlap in (5, 10, 15):
        for coefficient in (0.25, 0.4, 0.55, 0.7):
            composed, added = _compose(
                {video_id: pred_rel.get(video_id, []) for video_id in video_ids},
                coefficient=coefficient,
                min_overlap=min_overlap,
            )
            predictions, adapter = project_artifacts_to_official(composed, trajectories, video_ids, prediction=True)
            metrics = evaluate_official_vidvrd(gold, predictions)
            detection = metrics["relation_detection"]
            rows.append(
                {
                    "min_overlap": min_overlap,
                    "coefficient": coefficient,
                    "added": added,
                    "predictions": adapter["relation_count"],
                    "mean_ap": detection["mean_ap"],
                    "recall_at_50": detection["recall_at"]["50"],
                    "recall_at_100": detection["recall_at"]["100"],
                }
            )
    baseline_predictions, adapter = project_artifacts_to_official(pred_rel, trajectories, video_ids, prediction=True)
    baseline_metrics = evaluate_official_vidvrd(gold, baseline_predictions)["relation_detection"]
    report = {
        "scope": "fixed 10-video development set; predictions only, no model calls",
        "baseline": {
            "predictions": adapter["relation_count"],
            "mean_ap": baseline_metrics["mean_ap"],
            "recall_at_50": baseline_metrics["recall_at"]["50"],
            "recall_at_100": baseline_metrics["recall_at"]["100"],
        },
        "best": max(rows, key=lambda row: (row["mean_ap"], row["recall_at_50"])),
        "rows": rows,
    }
    output = run / "reports" / "composition_ablation_dev.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
