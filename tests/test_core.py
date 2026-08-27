from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vidvrd_auto.config.loader import load_app_config, load_config
from vidvrd_auto.core import Detection, Relation, RunContext, Secrets, Track, VideoPaths


class ConfigContractTests(unittest.TestCase):
    def test_typed_config_supports_mapping_and_attributes(self) -> None:
        config = load_app_config()
        self.assertEqual(config["detector"]["batch_size"], config.detector.batch_size)
        self.assertEqual(config.get("tracking").max_age, config.tracking["max_age"])
        detached = config.to_dict()
        detached["detector"]["batch_size"] = 99
        self.assertNotEqual(config.detector.batch_size, 99)

    def test_loader_returns_dict(self) -> None:
        self.assertIsInstance(load_config(), dict)


class ContextTests(unittest.TestCase):
    def test_paths_and_context_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = VideoPaths.for_video("runs/test", "clip_1", repo_dir=tmp)
            context = RunContext(load_app_config(), Secrets("secret"), paths, source="movie.mp4")
            self.assertEqual(context.video_id, "clip_1")
            self.assertEqual(context.api_key, "secret")
            self.assertEqual(paths.detect_dir, Path(tmp).resolve() / "runs/test/videos/clip_1/detect")
            self.assertEqual(paths.artifact("track", "tracks.jsonl").name, "tracks.jsonl")
            other = context.with_video("clip_2", source="other.mp4")
            self.assertEqual(other.paths.video_id, "clip_2")

    def test_secrets_load_from_environment_or_override(self) -> None:
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": " env-key "}):
            secrets = Secrets.from_env()
            overridden = Secrets.from_env(dashscope_api_key="cli-key")
        self.assertEqual(secrets.dashscope_api_key, "env-key")
        self.assertEqual(overridden.dashscope_api_key, "cli-key")

    def test_secrets_load_from_dotenv_when_environment_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dotenv = Path(tmp) / ".env"
            dotenv.write_text('DASHSCOPE_API_KEY="file-key"\n', encoding="utf-8")
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""}):
                from_file = Secrets.from_env(dotenv_path=dotenv)
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "env-key"}):
                env_wins = Secrets.from_env(dotenv_path=dotenv)
        self.assertEqual(from_file.dashscope_api_key, "file-key")
        self.assertEqual(env_wins.dashscope_api_key, "env-key")


class SchemaTests(unittest.TestCase):
    def test_detection_round_trip_preserves_extra_fields(self) -> None:
        raw = {
            "bbox": [1, 2, 11, 12],
            "class_name": "person",
            "confidence": 0.8,
            "batch_id": 4,
            "temporal_verified": True,
        }
        self.assertEqual(Detection.from_dict(raw).to_dict()["temporal_verified"], True)

    def test_predicted_track_cannot_claim_observation(self) -> None:
        with self.assertRaisesRegex(ValueError, "bbox_observed"):
            Track.from_dict(
                {
                    "track_id": 1,
                    "bbox": [0, 0, 10, 10],
                    "bbox_observed": [0, 0, 10, 10],
                    "class_name": "dog",
                    "confidence": 0.7,
                    "is_predicted": True,
                }
            )

    def test_relation_normalizes_legacy_inclusive_span(self) -> None:
        relation = Relation.from_dict(
            {
                "subject_id": 1,
                "predicate": "Follow",
                "object_id": 2,
                "start_frame": 5,
                "end_frame": 10,
                "evidence_frames": [10, 5],
            }
        )
        self.assertEqual(relation.predicate, "follow")
        self.assertEqual((relation.start_frame, relation.end_frame), (5, 11))
        self.assertEqual(relation.evidence_frames, (5, 10))
        self.assertEqual(relation.to_legacy_dict()["end_frame"], 10)
        with self.assertRaisesRegex(ValueError, "inside"):
            Relation(1, "follow", 2, 5, 10, evidence_frames=(11,))

    def test_invalid_bbox_and_confidence_fail_early(self) -> None:
        with self.assertRaises(ValueError):
            Detection((1, 1, 1, 4), "person", 0.5)
        with self.assertRaises(ValueError):
            Detection((1, 1, 4, 4), "person", 1.5)


if __name__ == "__main__":
    unittest.main()
