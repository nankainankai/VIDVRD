from __future__ import annotations

import unittest

from vidvrd_auto.config.loader import load_config


class ConfigTests(unittest.TestCase):
    def test_load_default_config(self) -> None:
        cfg = load_config()
        self.assertIn("video_ingest", cfg)
        self.assertIn("detector", cfg)
        self.assertIn("relations", cfg)


if __name__ == "__main__":
    unittest.main()
