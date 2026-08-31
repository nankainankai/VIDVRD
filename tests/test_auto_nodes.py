from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vidvrd_auto.providers import DashScopeProvider
from vidvrd_auto.nodes.track_qc import run_track_qc
from vidvrd_auto.nodes.vocabulary import build_vocabulary
from vidvrd_auto.relations.taxonomy import coupling_inverse, mutex_pairs


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class AutoNodeTests(unittest.TestCase):
    def test_vl_client_dry_run(self) -> None:
        result = DashScopeProvider({"dry_run": True, "model": "mock-vl"}).call(prompt="测试", dry_run=True)
        self.assertTrue(result.ok)
        self.assertTrue(result.dry_run)
        self.assertEqual(result.model, "mock-vl")

    def test_taxonomy_exports_coupling_and_mutex(self) -> None:
        self.assertEqual(coupling_inverse()["left"], "right")
        self.assertIn(frozenset(("left", "right")), mutex_pairs())

    def test_fixed_vocabulary_contains_complete_vidvrd_objects(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            out = root / "objects.json"
            result = build_vocabulary(
                video_path=root / "unused.mp4",
                out_json=out,
                evidence_path=root / "evidence.jpg",
                config={"discovery_enabled": False},
            )
            self.assertEqual(result["mode"], "vidvrd_base")
            self.assertEqual(len(result["categories"]), 35)
            self.assertIn("domestic_cat", result["categories"])
            self.assertEqual(result["discovery"]["state"], "disabled")
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
                config={"min_track_frames": 2},
            )
            self.assertEqual(result["short_track_count"], 1)
            self.assertEqual(result["risk_track_ids"], [1])
            self.assertTrue(result["needs_strong_review"])

    def test_track_qc_reports_internal_gaps_without_promoting_them_to_policy(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            tracks = root / "tracks.jsonl"
            windows = root / "windows.json"
            out = root / "track_qc.json"
            rows = [
                {"frame": 0, "tracks": [{"track_id": 3, "bbox": [0, 0, 10, 10], "bbox_observed": [0, 0, 10, 10], "box_source": "observed", "class_name": "bicycle"}]},
                {"frame": 1, "tracks": [{"track_id": 3, "bbox": [1, 0, 11, 10], "bbox_observed": None, "box_source": "interpolated", "class_name": "bicycle"}]},
                {"frame": 2, "tracks": []},
                {"frame": 3, "tracks": []},
                {"frame": 4, "tracks": [{"track_id": 3, "bbox": [4, 0, 14, 10], "bbox_observed": [4, 0, 14, 10], "box_source": "observed", "class_name": "bicycle"}]},
            ]
            _write_text(tracks, "".join(json.dumps(row) + "\n" for row in rows))
            _write_text(windows, json.dumps({"windows": []}))
            result = run_track_qc(
                tracks_jsonl=tracks,
                windows_json=windows,
                out_json=out,
                config={"min_track_frames": 1, "max_center_jump_ratio": 10.0},
            )
            metrics = result["track_continuity"]["3"]
            self.assertEqual(metrics["segment_count"], 2)
            self.assertEqual(metrics["max_gap_frames"], 2)
            self.assertEqual(metrics["coverage_ratio"], 0.6)
            self.assertEqual(metrics["observed_ratio"], 0.4)
            self.assertEqual(metrics["interpolation_ratio"], 0.3333)
            self.assertEqual(result["continuity_issue_count"], 1)
            self.assertEqual(result["risk_track_ids"], [])


if __name__ == "__main__":
    unittest.main()
