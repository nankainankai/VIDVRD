from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vidvrd_auto.config.loader import load_config
from vidvrd_auto.core import Detection, FrameSpan, Relation, Track
from vidvrd_auto.nodes.ingest import source_fingerprint
from vidvrd_auto.pipeline.manifest import build_run_provenance
from vidvrd_auto.utils.hashing import sha256_file
from vidvrd_auto.utils.io import write_json
from vidvrd_auto.utils.paths import repo_root


class RunModeContractTests(unittest.TestCase):
    def test_named_run_modes_load_with_explicit_contracts(self) -> None:
        expected = {
            "reference_dense.json": "reference_dense",
            "main.json": "main",
        }
        for filename, mode in expected.items():
            with self.subTest(filename=filename):
                config = load_config(Path("configs") / filename)
                self.assertEqual(config["project"]["run_mode"], mode)
                self.assertEqual(config["project"]["artifact_span_convention"], "inclusive")
                self.assertEqual(config["project"]["canonical_span_convention"], "half_open")

    def test_unknown_run_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            write_json(path, {"project": {"run_mode": "mystery"}})
            with self.assertRaisesRegex(SystemExit, "unsupported"):
                load_config(path)


class SchemaMigrationTests(unittest.TestCase):
    def test_frame_span_converts_legacy_inclusive_to_half_open(self) -> None:
        span = FrameSpan.from_values(3, 7, convention="inclusive")
        self.assertEqual((span.start_frame, span.end_frame, span.frame_count), (3, 8, 5))
        self.assertTrue(span.contains(7))
        self.assertFalse(span.contains(8))
        self.assertEqual(span.to_dict(convention="inclusive")["end_frame"], 7)

    def test_relation_reads_legacy_and_writes_explicit_half_open(self) -> None:
        relation = Relation.from_dict(
            {
                "subject_track_id": 1,
                "predicate": "next_to",
                "object_track_id": 2,
                "start_frame": 3,
                "end_frame": 7,
                "evidence_frames": [3, 7],
            }
        )
        self.assertEqual((relation.start_frame, relation.end_frame), (3, 8))
        self.assertEqual(relation.to_dict()["span_convention"], "half_open")
        self.assertNotIn("confidence", relation.to_dict())
        self.assertEqual(relation.to_legacy_dict()["end_frame"], 7)

    def test_relation_preserves_separate_rule_and_agent_scores(self) -> None:
        relation = Relation.from_dict(
            {
                "subject_track_id": 1,
                "predicate": "next_to",
                "object_track_id": 2,
                "start_frame": 3,
                "end_frame": 7,
                "rule_support": 0.75,
                "agent_score": 0.6,
                "ranking_score": 0.75,
                "score_kind": "mixed_ranking",
            }
        )
        output = relation.to_dict()
        self.assertEqual((output["rule_support"], output["agent_score"], output["ranking_score"]), (0.75, 0.6, 0.75))
        self.assertNotIn("confidence", output)

    def test_detection_can_represent_missing_native_score(self) -> None:
        detection = Detection.from_dict({"bbox": [0, 0, 10, 10], "class_name": "person", "score": None})
        self.assertIsNone(detection.confidence)
        self.assertEqual(detection.score_kind, "unavailable")

    def test_track_reads_legacy_observation_without_bbox_observed(self) -> None:
        track = Track.from_dict(
            {"track_id": 1, "bbox": [0, 0, 10, 10], "class_name": "person", "confidence": 0.8}
        )
        self.assertEqual(track.box_source, "observed")
        self.assertEqual(track.bbox_observed, track.bbox)


class ProvenanceAndHashTests(unittest.TestCase):
    def test_default_file_hash_includes_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.bin"
            right = Path(tmp) / "right.bin"
            left.write_bytes(b"shared-prefix-a")
            right.write_bytes(b"shared-prefix-b")
            self.assertNotEqual(sha256_file(left), sha256_file(right))

    def test_same_local_path_changes_source_fingerprint_when_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clip.mp4"
            video.write_bytes(b"first")
            before = source_fingerprint(str(video), Path(tmp) / "inputs")
            video.write_bytes(b"second")
            after = source_fingerprint(str(video), Path(tmp) / "inputs")
            self.assertNotEqual(before["file_hash"], after["file_hash"])

    def test_manifest_provenance_records_mode_models_and_span_contract(self) -> None:
        root = repo_root()
        path = root / "configs" / "main.json"
        config = load_config(path)
        provenance = build_run_provenance(root=root, config=config, config_path=path)
        self.assertEqual(provenance["run_mode"], "main")
        self.assertEqual(provenance["canonical_span_convention"], "half_open")
        self.assertEqual(provenance["algorithms"]["tracker"]["name"], "hybrid_sparse_reid")
        self.assertEqual(provenance["algorithms"]["tracker"]["time_unit"], "video_frame")
        self.assertEqual(provenance["algorithms"]["detector"]["sampling"], "adaptive_sparse")
        self.assertEqual(
            provenance["algorithms"]["agent_policy"]["name"],
            "bounded_hierarchical_agent_v2",
        )
        self.assertEqual(provenance["algorithms"]["agent_policy"]["candidate_limit"], 14)
        self.assertEqual(provenance["algorithms"]["agent_policy"]["max_supplemental_calls"], 1)
        self.assertEqual(
            provenance["algorithms"]["official_evaluator"]["name"],
            "imagenet_vidvrd_official_2017_compatible_v1",
        )
        self.assertEqual(len(provenance["code_fingerprint"]), 64)
        self.assertEqual(len(provenance["effective_config_hash"]), 64)
        self.assertEqual(len(provenance["config_file_hash"]), 64)


if __name__ == "__main__":
    unittest.main()
