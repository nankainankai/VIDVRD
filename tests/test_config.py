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
        self.assertTrue(cfg["vocabulary"]["discovery_enabled"])
        self.assertTrue(cfg["relation_verify"]["strong_model_review_enabled"])
        self.assertTrue(cfg["relation_verify"]["apply_actions"])
        self.assertTrue(cfg["relations"]["allow_request_more_frames"])
        self.assertEqual(cfg["relations"]["max_additional_frames"], 4)
        self.assertEqual(cfg["tracking"]["algorithm"], "hybrid_sparse_reid")
        self.assertEqual(cfg["tracking"]["max_lost_frames"], 30)

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
        self.assertEqual(cfg["tracking"]["algorithm"], "ocsort_reference")

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
