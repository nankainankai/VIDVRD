from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vidvrd_auto.evaluation.official.vidvrd import (
    evaluate_official_vidvrd,
    project_artifacts_to_official,
    trajectory_viou,
)
from vidvrd_auto.evaluation.diagnostic.track_aligned import clear_mot_on_gold_frames
from vidvrd_auto.evaluation.vidvrd import _semantic_packet_coverage, run_evaluation_suite
from vidvrd_auto.nodes.export import (
    compose_action_spatial_relations,
    deduplicate_relation_proposals,
    export_video_outputs,
    merge_relation_files,
    merge_trajectory_files,
    sliding_relation_proposals,
)
from reference_vidvrd_helper import evaluate as evaluate_pinned_upstream


def _tube(triplet, *, score=1.0, offset=0.0):
    return {
        "triplet": list(triplet),
        "duration": [0, 2],
        "sub_traj": [[offset, 0, offset + 10, 10], [offset, 0, offset + 10, 10]],
        "obj_traj": [[20 + offset, 0, 30 + offset, 10], [20 + offset, 0, 30 + offset, 10]],
        "score": score,
    }


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class OfficialEvaluationTests(unittest.TestCase):
    def test_temporal_dedup_keeps_distinct_pairs_and_nonoverlapping_events(self) -> None:
        def relation(sid: int, start: int, end: int, score: float) -> dict:
            return {
                "subject_track_id": sid, "object_track_id": 2, "predicate": "ride",
                "start_frame": start, "end_frame": end,
                "span_convention": "half_open", "ranking_score": score,
            }

        items = [
            relation(1, 0, 30, 0.9),
            relation(1, 2, 30, 0.8),  # same event, high temporal IoU
            relation(1, 40, 70, 0.7),  # separate event
            relation(3, 2, 30, 0.6),  # different track pair
        ]
        output = deduplicate_relation_proposals(items, temporal_iou_threshold=0.7)
        self.assertEqual(len(output), 3)
        self.assertEqual([item["ranking_score"] for item in output], [0.9, 0.7, 0.6])

    def test_semantic_packet_coverage_reports_moderated_omissions(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            log_path = root / "a" / "semantic" / "run.log"
            log_path.parent.mkdir(parents=True)
            _write(log_path, {
                "batch_audit": [{"packet_ids": ["p1", "p2"]}, {"packet_ids": ["p3"]}],
                "errors": [{"error": "DataInspectionFailed"}],
            })
            coverage = _semantic_packet_coverage(["a", "b"], root)
            self.assertEqual(coverage["attempted_packets"], 3)
            self.assertEqual(coverage["omitted_packets"], 1)
            self.assertAlmostEqual(coverage["coverage_fraction"], 2 / 3)
            self.assertEqual(coverage["missing_logs"], ["b"])

    def test_gold_frame_mota_counts_false_positives_misses_and_id_switches(self) -> None:
        gold = [{
            "track_id": 1, "category": "person",
            "trajectory": {str(frame): [0, 0, 10, 10] for frame in range(3)},
        }]
        predictions = [
            {"track_id": 10, "category": "person", "trajectory": {"0": [0, 0, 10, 10]}},
            {"track_id": 11, "category": "person", "trajectory": {"1": [0, 0, 10, 10]}},
            {"track_id": 12, "category": "dog", "trajectory": {"2": [20, 20, 30, 30]}},
        ]
        metrics = clear_mot_on_gold_frames(gold, predictions)
        self.assertEqual(metrics["matches"], 2)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["id_switches"], 1)
        self.assertAlmostEqual(metrics["mota"], 0.0)
        self.assertAlmostEqual(metrics["motp"], 1.0)

    def test_metrics_match_pinned_upstream_on_diverse_tubes(self) -> None:
        groundtruth = {
            "a": [
                _tube(("person", "next_to", "dog"), offset=0),
                _tube(("person", "follow", "dog"), offset=12),
            ],
            "b": [_tube(("dog", "behind", "person"), offset=4)],
        }
        predictions = {
            "a": [
                _tube(("person", "above", "dog"), score=0.95, offset=0),
                _tube(("person", "next_to", "dog"), score=0.90, offset=1),
                _tube(("person", "follow", "dog"), score=0.80, offset=35),
                _tube(("person", "next_to", "dog"), score=0.70, offset=0),
            ],
            "b": [
                _tube(("dog", "behind", "person"), score=0.60, offset=4),
                _tube(("dog", "behind", "person"), score=0.50, offset=4),
            ],
        }
        expected = evaluate_pinned_upstream(groundtruth, predictions)
        actual = evaluate_official_vidvrd(groundtruth, predictions)
        self.assertAlmostEqual(actual["relation_detection"]["mean_ap"], expected["mean_ap"])
        for limit in (50, 100):
            self.assertAlmostEqual(
                actual["relation_detection"]["recall_at"][str(limit)],
                expected["recall"][limit],
            )
        for limit in (1, 5, 10):
            self.assertAlmostEqual(
                actual["relation_tagging"]["precision_at"][str(limit)],
                expected["tagging"][limit],
            )

    def test_export_stamps_inclusive_span_and_adapter_honors_convention(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            verified = root / "verified.json"
            tracks = root / "tracks.jsonl"
            exported_relations = root / "relations.json"
            exported_trajectories = root / "trajectories.json"
            _write(
                verified,
                {
                    "clip": [{
                        "subject_track_id": 1,
                        "predicate": "next_to",
                        "object_track_id": 2,
                        "start_frame": 0,
                        "end_frame": 3,
                        "span_convention": "half_open",
                        "ranking_score": 0.9,
                    }]
                },
            )
            tracks.write_text(
                "".join(
                    json.dumps({
                        "frame": frame,
                        "tracks": [
                            {"track_id": 1, "class_name": "person", "bbox": [0, 0, 10, 10]},
                            {"track_id": 2, "class_name": "dog", "bbox": [20, 0, 30, 10]},
                        ],
                    }) + "\n"
                    for frame in range(3)
                ),
                encoding="utf-8",
            )
            export_video_outputs(
                verified_path=verified,
                qc_path=root / "missing_qc.json",
                tracks_path=tracks,
                video_id="clip",
                relations_path=exported_relations,
                trajectories_path=exported_trajectories,
            )
            relations = json.loads(exported_relations.read_text(encoding="utf-8"))
            relation = relations["clip"][0]
            self.assertEqual((relation["start_frame"], relation["end_frame"]), (0, 2))
            self.assertEqual(relation["span_convention"], "inclusive")
            trajectories = json.loads(exported_trajectories.read_text(encoding="utf-8"))
            adapted, _ = project_artifacts_to_official(relations, trajectories, ["clip"], prediction=True)
            self.assertEqual(adapted["clip"][0]["duration"], [0, 3])
            direct, _ = project_artifacts_to_official(
                json.loads(verified.read_text(encoding="utf-8")),
                trajectories,
                ["clip"],
                prediction=True,
            )
            self.assertEqual(direct["clip"][0]["duration"], [0, 3])

    def test_export_includes_failed_video_as_empty_prediction(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            relation_output = root / "relations.json"
            trajectory_output = root / "trajectories.json"
            relations = merge_relation_files([], relation_output, ["failed_video"])
            trajectories = merge_trajectory_files([], trajectory_output, ["failed_video"])
            self.assertEqual(relations, {"failed_video": []})
            self.assertEqual(trajectories, {"failed_video": []})
            self.assertEqual(json.loads(relation_output.read_text(encoding="utf-8")), {"failed_video": []})

    def test_sliding_relation_proposals_are_prediction_only_and_bounded(self) -> None:
        source = [{
            "relation_id": "r1",
            "subject_track_id": 1,
            "predicate": "left",
            "object_track_id": 2,
            "start_frame": 0,
            "end_frame": 89,
            "span_convention": "inclusive",
            "evidence_frames": [0, 20, 40, 60, 80],
        }]
        proposals = sliding_relation_proposals(source, window_frames=30, stride_frames=15)
        self.assertEqual([(p["start_frame"], p["end_frame"]) for p in proposals],
                         [(0, 30), (15, 45), (30, 60), (45, 75), (60, 90), (75, 90)])
        self.assertTrue(all(p["span_convention"] == "half_open" for p in proposals))
        self.assertEqual(proposals[1]["evidence_frames"], [20, 40])
        self.assertEqual(source[0]["end_frame"], 89)

    def test_ride_and_spatial_evidence_compose_conservative_sit_relation(self) -> None:
        relations = [
            {
                "relation_id": "ride",
                "subject_track_id": 1,
                "predicate": "ride",
                "object_track_id": 2,
                "start_frame": 0,
                "end_frame": 30,
                "span_convention": "half_open",
                "evidence_frames": [5, 15, 25],
                "ranking_score": 0.81,
                "source_kind": "semantic",
            },
            {
                "subject_track_id": 1,
                "predicate": "above",
                "object_track_id": 2,
                "start_frame": 10,
                "end_frame": 30,
                "span_convention": "half_open",
                "evidence_frames": [15, 25],
                "ranking_score": 1.0,
                "source_kind": "rule",
            },
        ]
        composed = compose_action_spatial_relations(relations)
        generated = [item for item in composed if item["predicate"] == "sit_above"]
        self.assertEqual(len(generated), 1)
        self.assertEqual((generated[0]["start_frame"], generated[0]["end_frame"]), (10, 30))
        self.assertEqual(generated[0]["evidence_frames"], [15, 25])
        self.assertAlmostEqual(generated[0]["ranking_score"], 0.495)

    def test_composition_rejects_brief_spatial_overlap(self) -> None:
        relations = [
            {"subject_track_id": 1, "predicate": "ride", "object_track_id": 2,
             "start_frame": 0, "end_frame": 30, "span_convention": "half_open", "ranking_score": 0.8},
            {"subject_track_id": 1, "predicate": "above", "object_track_id": 2,
             "start_frame": 25, "end_frame": 30, "span_convention": "half_open", "ranking_score": 1.0},
        ]
        composed = compose_action_spatial_relations(relations)
        self.assertFalse(any(item["predicate"] == "sit_above" for item in composed))

    def test_trajectory_viou_uses_relation_tubes(self) -> None:
        trajectory = [[0, 0, 9, 9], [0, 0, 9, 9]]
        self.assertEqual(trajectory_viou(trajectory, [0, 2], trajectory, [0, 2]), 1.0)
        self.assertEqual(trajectory_viou(trajectory, [0, 2], trajectory, [2, 4]), 0.0)

    def test_fixed_official_protocol_sample(self) -> None:
        target = _tube(("person", "next_to", "dog"))
        false_prediction = _tube(("person", "above", "dog"), score=0.9)
        true_prediction = _tube(("person", "next_to", "dog"), score=0.8)
        metrics = evaluate_official_vidvrd(
            {"video_a": [target], "video_b": [target]},
            {"video_a": [false_prediction, true_prediction], "video_b": []},
        )
        self.assertAlmostEqual(metrics["relation_detection"]["mean_ap"], 0.25)
        self.assertAlmostEqual(metrics["relation_detection"]["recall_at"]["50"], 0.5)
        self.assertAlmostEqual(metrics["relation_tagging"]["precision_at"]["1"], 0.0)
        self.assertAlmostEqual(metrics["relation_tagging"]["precision_at"]["5"], 0.25)
        self.assertEqual(metrics["error_diagnostics"]["failure_counts"]["ranking_failure"], 1)
        self.assertEqual(metrics["error_diagnostics"]["failure_counts"]["candidate_missing"], 1)
        first = metrics["error_diagnostics"]["groundtruth"][0]
        self.assertEqual(first["true_positive_rank"], 2)
        self.assertEqual(first["false_positives_before_true_positive"], 1)

    def test_adapter_matches_categories_and_splits_track_gaps(self) -> None:
        relations = {"clip": [{
            "subject_track_id": 10, "predicate": "next_to", "object_track_id": 20,
            "start_frame": 0, "end_frame": 2, "ranking_score": 0.7,
        }]}
        trajectories = {"clip": [
            {"track_id": 10, "category": "person", "trajectory": {"0": [0, 0, 10, 10], "2": [0, 0, 10, 10]}},
            {"track_id": 20, "category": "dog", "trajectory": {"0": [20, 0, 30, 10], "2": [20, 0, 30, 10]}},
        ]}
        adapted, report = project_artifacts_to_official(relations, trajectories, ["clip"], prediction=True)
        self.assertEqual([item["triplet"] for item in adapted["clip"]], [["person", "next_to", "dog"]] * 2)
        self.assertEqual([item["duration"] for item in adapted["clip"]], [[0, 1], [2, 3]])
        self.assertEqual(report["split_on_track_gaps"], 1)
        self.assertEqual(report["per_video"]["clip"]["source_relation_count"], 1)
        self.assertEqual(report["per_video"]["clip"]["adapted_relation_count"], 2)
        self.assertEqual(report["per_video"]["clip"]["split_on_track_gaps"], 1)

    def test_real_gold_as_prediction_scores_perfect_map(self) -> None:
        root = Path(__file__).resolve().parents[1]
        relations = json.loads((root / "gold/vidvrd_50_relations.json").read_text(encoding="utf-8"))
        trajectories = json.loads((root / "gold/vidvrd_50_trajectories.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / "gold/vidvrd_50_manifest.json").read_text(encoding="utf-8"))
        video_ids = sorted(video_id for video_id, split in manifest["video_splits"].items() if split == "test")
        self.assertEqual(len(video_ids), 10)
        gold, gold_adapter = project_artifacts_to_official(relations, trajectories, video_ids, prediction=False)
        predictions, pred_adapter = project_artifacts_to_official(relations, trajectories, video_ids, prediction=True)
        metrics = evaluate_official_vidvrd(gold, predictions)
        self.assertEqual(gold_adapter["skipped_missing_tracks"], 0)
        self.assertEqual(gold_adapter["split_on_track_gaps"], 0)
        self.assertEqual(pred_adapter["skipped_empty_tubes"], 0)
        self.assertAlmostEqual(metrics["relation_detection"]["mean_ap"], 1.0)

    def test_suite_keeps_missing_gold_split_video_in_denominator(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {name: root / f"{name}.json" for name in (
                "gold_rel", "gold_traj", "manifest", "pred_rel", "pred_traj",
                "official_metrics", "diagnostic_metrics",
            )}
            relation = {"subject_track_id": 1, "predicate": "next_to", "object_track_id": 2, "start_frame": 0, "end_frame": 1}
            tracks = [
                {"track_id": 1, "category": "person", "trajectory": {"0": [0, 0, 10, 10], "1": [0, 0, 10, 10]}},
                {"track_id": 2, "category": "dog", "trajectory": {"0": [20, 0, 30, 10], "1": [20, 0, 30, 10]}},
            ]
            _write(paths["gold_rel"], {"video_a": [relation], "video_b": [relation], "training_video": [relation]})
            _write(paths["gold_traj"], {"video_a": tracks, "video_b": tracks, "training_video": tracks})
            _write(paths["manifest"], {"video_splits": {"video_a": "test", "video_b": "test", "training_video": "train"}})
            _write(paths["pred_rel"], {"video_a": [dict(relation, ranking_score=0.9)]})
            _write(paths["pred_traj"], {"video_a": tracks})
            result = run_evaluation_suite(
                gold_relations=paths["gold_rel"], gold_trajectories=paths["gold_traj"], gold_manifest=paths["manifest"],
                pred_relations=paths["pred_rel"], pred_trajectories=paths["pred_traj"], requested_video_ids=["video_a"],
                official_report_path=root / "official.md", official_metrics_path=paths["official_metrics"],
                diagnostic_report_path=root / "diagnostic.md", diagnostic_metrics_path=paths["diagnostic_metrics"],
                scope="gold_split", dataset_split="test", expected_official_video_count=2,
            )
            official = result["official"]
            self.assertEqual(official["evaluated_videos"], ["video_a", "video_b"])
            self.assertEqual(official["dataset_scope"]["missing_prediction_videos"], ["video_b"])
            self.assertAlmostEqual(official["relation_detection"]["mean_ap"], 0.5)
            self.assertEqual(official["partial_run_diagnostic"]["requested_gold_video_count"], 1)
            self.assertAlmostEqual(official["partial_run_diagnostic"]["coverage_fraction"], 0.5)
            self.assertAlmostEqual(official["partial_run_diagnostic"]["requested_mean_ap"], 1.0)
            self.assertTrue(official["dataset_scope"]["complete_test_annotations"])
            self.assertFalse(official["dataset_scope"]["complete_official_test"])
            self.assertIn("not a publishable full-test benchmark", (root / "official.md").read_text(encoding="utf-8"))
            self.assertIn("Only a subset of the Gold videos was requested", (root / "official.md").read_text(encoding="utf-8"))

    def test_suite_rejects_nonoverlapping_gold_split_request(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            gold_rel = root / "gold_rel.json"
            gold_traj = root / "gold_traj.json"
            manifest = root / "manifest.json"
            pred_rel = root / "pred_rel.json"
            pred_traj = root / "pred_traj.json"
            _write(gold_rel, {"test_video": []})
            _write(gold_traj, {"test_video": []})
            _write(manifest, {"video_splits": {"test_video": "test"}})
            _write(pred_rel, {"demo_video": []})
            _write(pred_traj, {"demo_video": []})
            with self.assertRaisesRegex(ValueError, "none of the requested videos belong to Gold"):
                run_evaluation_suite(
                    gold_relations=gold_rel, gold_trajectories=gold_traj, gold_manifest=manifest,
                    pred_relations=pred_rel, pred_trajectories=pred_traj,
                    requested_video_ids=["demo_video"], official_report_path=root / "official.md",
                    official_metrics_path=root / "official.json", diagnostic_report_path=root / "diagnostic.md",
                    diagnostic_metrics_path=root / "diagnostic.json", scope="gold_split", dataset_split="test",
                )


if __name__ == "__main__":
    unittest.main()
