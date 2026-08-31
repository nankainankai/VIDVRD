from __future__ import annotations

import unittest
from pathlib import Path

from vidvrd_auto.cli import build_parser
from vidvrd_auto.config.loader import load_config


class ConfigTests(unittest.TestCase):
    def test_load_default_config(self) -> None:
        cfg = load_config()
        self.assertIn("video_ingest", cfg)
        self.assertIn("detector", cfg)
        self.assertIn("relations", cfg)

    def test_main_enables_cloud_review_stages(self) -> None:
        cfg = load_config(Path("configs/main.json"))
        self.assertEqual(cfg["project"]["run_mode"], "main")
        self.assertFalse(cfg["vocabulary"]["discovery_enabled"])
        self.assertEqual(cfg["vocabulary"]["discovery_model"], "qwen3.7-plus")
        self.assertEqual(cfg["relations"]["api_model"], "qwen3.7-plus")
        self.assertEqual(cfg["relation_verify"]["strong_model"], "qwen-vl-max")
        self.assertTrue(cfg["relation_verify"]["strong_model_review_enabled"])
        self.assertTrue(cfg["relation_verify"]["apply_actions"])
        self.assertTrue(cfg["relations"]["allow_request_more_frames"])
        self.assertEqual(cfg["relations"]["max_additional_frames"], 4)
        self.assertEqual(cfg["tracking"]["algorithm"], "sparse_ocsort")
        self.assertTrue(cfg["tracking"]["class_aware"])
        self.assertEqual(cfg["tracking"]["max_age"], 10)
        self.assertEqual(cfg["tracking"]["min_hits"], 2)
        self.assertEqual(cfg["detector"]["sampling_mode"], "fixed_sparse")
        self.assertEqual(cfg["detector"]["detection_interval"], 3)
        self.assertEqual(cfg["detector"]["detector_backend"], "rexomni")
        self.assertNotIn("dinox_interval", cfg["detector"])
        self.assertEqual(cfg["relations"]["batch_windows_per_call"], 6)

    def test_dinox_hybrid_is_an_explicit_experimental_config(self) -> None:
        cfg = load_config(Path("configs/experimental_hybrid_dinox.json"))
        self.assertEqual(cfg["detector"]["detector_backend"], "hybrid_rex_dinox")
        self.assertEqual(cfg["detector"]["dinox_interval"], 15)
        self.assertEqual(cfg["detector"]["dinox_model"], "DINO-X-1.0")

    def test_cli_and_compat_config_resolve_to_the_same_main_route(self) -> None:
        args = build_parser().parse_args(["--run-dir", "runs/test"])
        self.assertEqual(args.config, "configs/main.json")
        self.assertEqual(
            load_config(Path("configs/config.json")),
            load_config(Path("configs/main.json")),
        )

    def test_reference_dense_uses_every_frame(self) -> None:
        cfg = load_config(Path("configs/reference_dense.json"))
        self.assertEqual(cfg["project"]["run_mode"], "reference_dense")
        self.assertEqual(cfg["detector"]["detection_interval"], 1)
        self.assertEqual(cfg["detector"]["detector_backend"], "rexomni")
        self.assertEqual(cfg["tracking"]["algorithm"], "ocsort_reference")

    def test_cli_supports_tracking_only(self) -> None:
        args = build_parser().parse_args(["--run-dir", "runs/test", "--tracking-only"])
        self.assertTrue(args.tracking_only)

    def test_benchmark_uses_fixed_vocabulary_and_gold(self) -> None:
        cfg = load_config(Path("configs/benchmark_official.json"))
        self.assertFalse(cfg["vocabulary"]["discovery_enabled"])
        self.assertTrue(cfg["evaluate"]["enabled"])
        self.assertEqual(cfg["evaluate"]["scope"], "gold_split")
        self.assertEqual(cfg["evaluate"]["dataset_split"], "test")

    def test_dry_run_disables_cloud_review_stages(self) -> None:
        cfg = load_config(Path("configs/dry_run.json"))
        self.assertTrue(cfg["relations"]["dry_run"])
        self.assertFalse(cfg["relation_verify"]["strong_model_review_enabled"])


if __name__ == "__main__":
    unittest.main()
