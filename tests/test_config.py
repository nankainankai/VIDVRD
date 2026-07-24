from __future__ import annotations

import unittest
from pathlib import Path

from vidvrd_auto.config.loader import load_config


class ConfigTests(unittest.TestCase):
    def test_load_default_config(self) -> None:
        cfg = load_config()
        self.assertIn("video_ingest", cfg)
        self.assertIn("detector", cfg)
        self.assertIn("relations", cfg)

    def test_production_enables_cloud_review_stages(self) -> None:
        cfg = load_config(Path("configs/production.json"))
        self.assertTrue(cfg["vocabulary"]["discovery_enabled"])
        self.assertTrue(cfg["relation_verify"]["strong_model_review_enabled"])
        self.assertTrue(cfg["relation_verify"]["apply_actions"])

    def test_benchmark_uses_fixed_vocabulary_and_gold(self) -> None:
        cfg = load_config(Path("configs/benchmark.json"))
        self.assertFalse(cfg["vocabulary"]["discovery_enabled"])
        self.assertTrue(cfg["evaluate"]["enabled"])

    def test_dry_run_disables_cloud_review_stages(self) -> None:
        cfg = load_config(Path("configs/dry_run.json"))
        self.assertTrue(cfg["relations"]["dry_run"])
        self.assertFalse(cfg["relation_verify"]["strong_model_review_enabled"])


if __name__ == "__main__":
    unittest.main()
