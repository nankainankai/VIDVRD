from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vidvrd_auto.models.vl_client import VLClient
from vidvrd_auto.nodes.screen import screen_keyframes
from vidvrd_auto.nodes.track_qc import run_track_qc
from vidvrd_auto.relations.taxonomy import coupling_inverse, mutex_pairs


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class AutoNodeTests(unittest.TestCase):
    def test_vl_client_dry_run(self) -> None:
        result = VLClient({"dry_run": True, "model": "mock-vl"}).call(prompt="测试", dry_run=True)
        self.assertTrue(result.ok)
        self.assertTrue(result.dry_run)
        self.assertEqual(result.model, "mock-vl")

    def test_taxonomy_exports_coupling_and_mutex(self) -> None:
        self.assertEqual(coupling_inverse()["left"], "right")
        self.assertIn(frozenset(("left", "right")), mutex_pairs())

    def test_screen_keyframes_outputs_decision(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            detections = root / "detections.jsonl"
            out = root / "screen.json"
            _write_text(
                detections,
                json.dumps({"frame": 0, "objects": [{"bbox": [0, 0, 10, 10], "confidence": 0.9}, {"bbox": [20, 20, 40, 40], "confidence": 0.8}]})
                + "\n",
            )
            result = screen_keyframes(
                detections_jsonl=detections,
                out_json=out,
                config={"sample_frames": 1, "min_objects": 2, "vl_enabled": True, "vl_dry_run": True},
            )
            self.assertTrue(result["passed"])
            self.assertIn(result["decision"], {"keep", "crop"})
            self.assertTrue(out.exists())

    def test_track_qc_collects_risks(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            tracks = root / "tracks.jsonl"
            windows = root / "windows.json"
            out = root / "track_qc.json"
            _write_text(
                tracks,
                json.dumps({"frame": 0, "tracks": [{"track_id": 1, "bbox": [0, 0, 10, 10], "class_name": "person"}]}) + "\n",
            )
            _write_text(windows, json.dumps({"windows": []}))
            result = run_track_qc(
                tracks_jsonl=tracks,
                windows_json=windows,
                out_json=out,
                config={"min_track_frames": 2, "vl_enabled": True, "vl_dry_run": True},
            )
            self.assertEqual(result["short_track_count"], 1)
            self.assertEqual(result["vl_review"]["state"], "succeeded")


if __name__ == "__main__":
    unittest.main()
