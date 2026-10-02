from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from vidvrd_auto.pipeline.run import run_pipeline


class PipelineFailureTests(unittest.TestCase):
    def test_failed_video_returns_nonzero_after_writing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "run"
            args = Namespace(
                video=str(root / "missing.mp4"),
                videos="",
                run_dir=str(run_dir),
                api_key="",
                resume=False,
                force=False,
                dry_run_relations=False,
                skip_eval=True,
                tracking_only=True,
            )
            with self.assertRaises(SystemExit) as raised:
                run_pipeline(args=args, config_path=Path("configs/main.json"))
            self.assertEqual(raised.exception.code, 1)
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["video_states"], {"failed": 1})
            self.assertEqual(manifest["evaluation"]["state"], "skipped")
            self.assertEqual(manifest["evaluation"]["completed_video_count"], 0)


if __name__ == "__main__":
    unittest.main()
