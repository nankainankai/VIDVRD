from __future__ import annotations

import json
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from vidvrd_auto.relations import merge_relations
from vidvrd_auto.nodes.global_relation import run_global_relation
from vidvrd_auto.providers import VLResult
from vidvrd_auto.relations.ops import _apply_final_actions, generate_rule_relations, verify_relations


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


class RelationTests(unittest.TestCase):
    def test_merge_relations_adds_coupling(self) -> None:
        with TemporaryDirectory() as td:
            tmp_path = Path(td)
            src = tmp_path / "rule.json"
            out = tmp_path / "merged.json"
            _write(
                src,
                {
                    "v1": [
                        {
                            "subject_track_id": 1,
                            "object_track_id": 2,
                            "predicate": "left",
                            "start_frame": 0,
                            "end_frame": 10,
                            "confidence": 0.8,
                            "source": "rule_geometry",
                        }
                    ]
                },
            )

            merged = merge_relations(video_id="v1", relation_jsons=[src], out_json=out, apply_coupling=True)
            items = merged["v1"]
            keys = {(x["subject_track_id"], x["predicate"], x["object_track_id"]) for x in items}
            self.assertIn((1, "left", 2), keys)
            self.assertIn((2, "right", 1), keys)

    def test_cross_window_aggregation_assigns_stable_id(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            source, output = root / "merged.json", root / "global.json"
            base = {"subject_track_id": 1, "predicate": "follow", "object_track_id": 2, "confidence": 0.8, "source": "window_semantic_vl"}
            _write(source, {"v1": [dict(base, start_frame=0, end_frame=29), dict(base, start_frame=15, end_frame=44)]})
            result = run_global_relation(video_id="v1", relations_json=source, out_json=output, config={"max_window_gap": 1})
            self.assertEqual(len(result["v1"]), 1)
            self.assertEqual((result["v1"][0]["start_frame"], result["v1"][0]["end_frame"]), (0, 44))
            self.assertEqual(result["v1"][0]["relation_id"], "r000001")
            self.assertEqual(result["v1"][0]["segment_count"], 2)

    def test_review_actions_use_relation_id_not_list_position(self) -> None:
        items = [
            {"relation_id": "r000002", "predicate": "next_to", "start_frame": 0, "end_frame": 10},
            {"relation_id": "r000001", "predicate": "left", "start_frame": 0, "end_frame": 10},
        ]
        result, applied = _apply_final_actions(items, [{"relation_id": "r000001", "action": "change_predicate", "new_predicate": "right"}])
        self.assertEqual(result[0]["predicate"], "next_to")
        self.assertEqual(result[1]["predicate"], "right")
        self.assertEqual(len(applied), 1)

    def test_rule_relation_covers_whole_window_and_requires_observation(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            windows, tracks, output = root / "windows.json", root / "tracks.jsonl", root / "relations.json"
            _write(windows, {"windows": [{"window_id": 4, "start_frame": 0, "end_frame": 2, "track_ids": [1, 2]}]})
            rows = []
            for frame in range(3):
                source = "observed" if frame == 0 else "interpolated"
                rows.append({"frame": frame, "tracks": [
                    {"track_id": 1, "bbox": [0, 0, 10, 10], "bbox_observed": [0, 0, 10, 10] if frame == 0 else None, "box_source": source},
                    {"track_id": 2, "bbox": [30, 0, 40, 10], "bbox_observed": [30, 0, 40, 10] if frame == 0 else None, "box_source": source},
                ]})
            tracks.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            result = generate_rule_relations(windows_json=windows, tracks_jsonl=tracks, out_json=output, video_id="v1", config={"min_vote_ratio": 0.6})
            left = next(item for item in result["v1"] if item["subject_track_id"] == 1 and item["predicate"] == "left")
            self.assertEqual((left["start_frame"], left["end_frame"]), (0, 2))
            self.assertEqual(left["source"], "window_geometry")

    @patch("vidvrd_auto.relations.ops.DashScopeProvider")
    def test_visual_review_action_changes_final_output(self, provider_type) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            relations, tracks = root / "relations.json", root / "tracks.jsonl"
            verified, qc = root / "verified.json", root / "qc.json"
            storyboards = root / "storyboards"
            storyboards.mkdir()
            (storyboards / "window_0001_A1_B2.jpg").write_bytes(b"visual-evidence")
            _write(relations, {"v1": [{"relation_id": "r000001", "subject_track_id": 1, "predicate": "next_to", "object_track_id": 2, "start_frame": 0, "end_frame": 29, "confidence": 0.2}]})
            tracks.write_text(json.dumps({"frame": 0, "tracks": []}) + "\n", encoding="utf-8")
            provider_type.return_value.call.return_value = VLResult(
                ok=True,
                text=json.dumps({"actions": [{"relation_id": "r000001", "action": "delete", "reason": "not visible"}]}),
                model="mock",
            )
            result = verify_relations(
                video_id="v1",
                relations_json=relations,
                tracks_jsonl=tracks,
                out_relations_json=verified,
                out_qc_json=qc,
                config={"strong_model_review_enabled": True, "apply_actions": True, "low_confidence_threshold": 0.45, "apply_coupling": False},
                storyboards_dir=storyboards,
            )
            self.assertEqual(json.loads(verified.read_text(encoding="utf-8")), {"v1": []})
            self.assertEqual(result["strong_model_review_result"]["state"], "succeeded")
            self.assertEqual(result["applied_actions"][0]["relation_id"], "r000001")


if __name__ == "__main__":
    unittest.main()
