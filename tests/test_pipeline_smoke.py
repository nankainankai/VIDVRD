from __future__ import annotations

import json
import subprocess
import sys
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vidvrd_auto.pipeline.runner import run_pipeline  # noqa: E402


class PipelineSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dummy = ROOT / "data" / "validation_dummy.mp4"
        if not dummy.exists():
            subprocess.check_call([sys.executable, str(ROOT / "scripts" / "make_validation_dummy.py")], cwd=str(ROOT))

    def test_full_pipeline_dry_run_mock(self) -> None:
        run_dir = ROOT / "runs" / "_unittest_smoke"
        if run_dir.exists():
            import shutil

            shutil.rmtree(run_dir, ignore_errors=True)

        args = Namespace(
            video=str(ROOT / "data" / "validation_dummy.mp4"),
            videos="",
            run_dir=str(run_dir),
            config=str(ROOT / "configs" / "dry_run.json"),
            api_key="",
            resume=False,
            force=True,
            from_node="",
            to_node="",
            dry_run_relations=True,
            skip_eval=True,
        )
        run_pipeline(args=args, config_path=ROOT / "configs" / "dry_run.json")

        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest.get("video_state_counts", {}).get("succeeded"), 1)
        pred = run_dir / "pred" / "relations_pred.json"
        self.assertTrue(pred.exists())
        export_rel = run_dir / "videos" / "validation_dummy" / "export" / "relations_pred.json"
        export_traj = run_dir / "videos" / "validation_dummy" / "export" / "trajectories_pred.json"
        self.assertTrue(export_rel.exists())
        self.assertTrue(export_traj.exists())


if __name__ == "__main__":
    unittest.main()
