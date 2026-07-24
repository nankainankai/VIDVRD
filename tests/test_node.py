from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

from vidvrd_auto.pipeline.node import NodeJob, run_job


class NodeJobTests(unittest.TestCase):
    def test_run_and_resume(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            output = root / "out.json"
            calls = []

            def action() -> None:
                calls.append(1)
                output.write_text("{}", encoding="utf-8")

            job = NodeJob("demo", "hash-1", [output], {"out": "out.json"}, action)
            args = Namespace(resume=True, force=False)
            self.assertTrue(run_job(job=job, args=args, video_dir=root))
            self.assertFalse(run_job(job=job, args=args, video_dir=root))
            self.assertEqual(len(calls), 1)

    def test_missing_output_marks_failure(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            job = NodeJob("demo", "hash-2", [root / "missing"], {}, lambda: None)
            with self.assertRaises(RuntimeError):
                run_job(
                    job=job,
                    args=Namespace(resume=False, force=False),
                    video_dir=root,
                )


if __name__ == "__main__":
    unittest.main()
