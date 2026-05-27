from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vidvrd_auto.evaluation.presence import run_presence_eval  # noqa: E402
from vidvrd_auto.pipeline.report import render_run_report, write_run_report  # noqa: E402
from vidvrd_auto.utils.image_io import imwrite  # noqa: E402

try:
    import cv2
    import numpy as np

    HAS_CV = True
except ImportError:
    HAS_CV = False


class EvalAndReportTests(unittest.TestCase):
    def test_presence_eval_with_gold_sample(self) -> None:
        gold = ROOT / "gold" / "relations_gold.json"
        pred = ROOT / "runs" / "live_api" / "pred" / "relations_pred.json"
        if not gold.exists():
            self.skipTest("gold sample missing")
        if not pred.exists():
            pred = ROOT / "runs" / "_unittest_smoke" / "pred" / "relations_pred.json"
        if not pred.exists():
            self.skipTest("no pred json from prior runs")

        with TemporaryDirectory() as td:
            report = Path(td) / "presence_report.md"
            log = Path(td) / "eval.log"
            run_presence_eval(gold_json=gold, pred_json=pred, report_path=report, log_path=log)
            self.assertTrue(report.exists())
            text = report.read_text(encoding="utf-8")
            self.assertIn("Presence", text)

    def test_run_report_from_manifest(self) -> None:
        manifest = {
            "run_dir": "runs/test",
            "started_at": "2026-01-01",
            "finished_at": "2026-01-01",
            "config_path": "configs/dry_run.json",
            "nodes": ["export"],
            "videos": [{"video_id": "v1", "state": "succeeded", "nodes": {"export": {"state": "succeeded"}}}],
            "video_state_counts": {"succeeded": 1},
            "pred_relation_count": 2,
            "pred_relations_json": "runs/test/pred/relations_pred.json",
            "evaluate": {"state": "skipped"},
            "args": {},
        }
        text = render_run_report(manifest)
        self.assertIn("VIDVRD 运行报告", text)
        self.assertIn("export", text)

    @unittest.skipUnless(HAS_CV, "opencv required")
    def test_imwrite_unicode_path(self) -> None:
        with TemporaryDirectory(prefix="vidvrd测试_") as td:
            path = Path(td) / "storyboards" / "seg_0001.jpg"
            img = np.zeros((32, 32, 3), dtype=np.uint8)
            imwrite(path, img)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
