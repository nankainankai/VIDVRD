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
from vidvrd_auto.evaluation.vidvrd import run_evaluation_suite
from vidvrd_auto.nodes.export import export_video_outputs, merge_relation_files, merge_trajectory_files
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
                pred_relations=paths["pred_rel"], pred_trajectories=paths["pred_traj"], requested_video_ids=["video_a", "video_b"],
                official_report_path=root / "official.md", official_metrics_path=paths["official_metrics"],
                diagnostic_report_path=root / "diagnostic.md", diagnostic_metrics_path=paths["diagnostic_metrics"],
                scope="gold_split", dataset_split="test", expected_official_video_count=200,
            )
            official = result["official"]
            self.assertEqual(official["evaluated_videos"], ["video_a", "video_b"])
            self.assertEqual(official["dataset_scope"]["missing_prediction_videos"], ["video_b"])
            self.assertAlmostEqual(official["relation_detection"]["mean_ap"], 0.5)
            self.assertFalse(official["dataset_scope"]["complete_official_test"])
            self.assertIn("not a publishable full-test benchmark", (root / "official.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
