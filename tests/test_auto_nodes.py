from __future__ import annotations

import json
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from vidvrd_auto.providers import DashScopeProvider
from vidvrd_auto.nodes.track_qc import run_track_qc
from vidvrd_auto.nodes.vocabulary import build_vocabulary
from vidvrd_auto.core.ontology import normalize_object
from vidvrd_auto.relations.taxonomy import coupling_inverse, mutex_pairs


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class AutoNodeTests(unittest.TestCase):
    def test_discovered_alias_is_preserved_as_rex_query_and_mapped_to_vidvrd(self) -> None:
        class Response:
            ok = True
            text = '{"objects":["cougar","bird","tree"]}'

            def to_dict(self):
                return {"ok": True, "text": self.text}

        with TemporaryDirectory() as td, patch(
            "vidvrd_auto.nodes.vocabulary._storyboard", return_value=[0]
        ), patch("vidvrd_auto.nodes.vocabulary.DashScopeProvider") as provider:
            provider.return_value.call.return_value = Response()
            root = Path(td)
            result = build_vocabulary(
                video_path=root / "unused.mp4", out_json=root / "objects.json",
                evidence_path=root / "storyboard.jpg",
                config={
                    "discovery_enabled": True,
                    "ontology_only_discovery": True,
                    "discovery_aliases": {"cougar": "lion", "mountain lion": "lion"},
                },
            )
            self.assertEqual(normalize_object("mountain lion"), "mountain_lion")
            self.assertIn("cougar", result["categories"])
            self.assertEqual(result["label_map"]["cougar"], "lion")
            self.assertEqual(result["discovered_count"], 1)
            self.assertNotIn("tree", result["categories"])

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
            self.assertEqual(result["blocked_track_ids"], [1])
            self.assertLessEqual(result["track_quality"]["1"]["quality"], 0.1)
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

    def test_track_qc_separates_overlap_motion_from_teleport(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            tracks, windows, out = root / "tracks.jsonl", root / "windows.json", root / "qc.json"
            boxes = ([0, 0, 10, 10], [5, 0, 15, 10], [30, 0, 40, 10])
            _write_text(tracks, "".join(json.dumps({"frame": frame, "tracks": [
                {"track_id": 1, "bbox": box, "box_source": "observed", "class_name": "person"},
            ]}) + "\n" for frame, box in enumerate(boxes)))
            _write_text(windows, json.dumps({"windows": []}))
            report = run_track_qc(tracks_jsonl=tracks, windows_json=windows, out_json=out,
                                  config={"min_track_frames": 2, "max_center_jump_ratio": 0.45})
            self.assertEqual(report["large_jump_count"], 1)
            self.assertEqual(report["track_quality"]["1"]["jump_frames"], [2])
            self.assertGreater(report["track_quality"]["1"]["quality"], 0.4)

    def test_track_qc_does_not_call_slow_long_gap_a_teleport(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            tracks, windows, out = root / "tracks.jsonl", root / "windows.json", root / "qc.json"
            rows = [
                {"frame": 0, "tracks": [{"track_id": 1, "bbox": [0, 0, 10, 10], "box_source": "observed"}]},
                {"frame": 24, "tracks": [{"track_id": 1, "bbox": [20, 0, 30, 10], "box_source": "observed"}]},
            ]
            _write_text(tracks, "".join(json.dumps(row) + "\n" for row in rows))
            _write_text(windows, json.dumps({"windows": []}))
            report = run_track_qc(tracks_jsonl=tracks, windows_json=windows, out_json=out,
                                  config={"min_track_frames": 2, "max_center_jump_ratio": 0.45})
            self.assertEqual(report["large_jump_count"], 0)


if __name__ == "__main__":
    unittest.main()
