from __future__ import annotations

import json
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from vidvrd_auto.relations import merge_relations
from vidvrd_auto.nodes.global_relation import run_global_relation
from vidvrd_auto.providers import VLResult
from vidvrd_auto.relations.object_candidates import GEOMETRY_PREDICATES
from vidvrd_auto.relations.ops import _apply_final_actions, generate_rule_relations, verify_relations
from vidvrd_auto.relations.candidate_router import route_predicates
from vidvrd_auto.relations.scoring import calibrated_score_fields
from vidvrd_auto.relations.track_risk import split_frames_on_track_jumps
from vidvrd_auto.relations.candidate_priors import learn_predicate_priors, predicate_prior_scores


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


class RelationTests(unittest.TestCase):
    def test_training_prior_uses_only_explicit_training_videos(self) -> None:
        relations = {
            "train": [{"subject_track_id": 1, "predicate": "ride", "object_track_id": 2}],
            "test": [{"subject_track_id": 1, "predicate": "watch", "object_track_id": 2}],
        }
        tubes = {video: [
            {"track_id": 1, "category": "person"}, {"track_id": 2, "category": "horse"},
        ] for video in relations}
        priors = learn_predicate_priors(relations, tubes, {"train"})
        self.assertEqual(priors["training_video_count"], 1)
        self.assertIn("ride", predicate_prior_scores(priors, "person", "horse"))
        self.assertNotIn("watch", priors["global"])

    def test_track_jump_splits_inference_frames(self) -> None:
        self.assertEqual(
            split_frames_on_track_jumps([0, 1, 2, 3, 4], [1, 2], {"1": [3]}),
            [[0, 1, 2], [3, 4]],
        )

    def test_cross_source_calibration_ranks_reliable_semantic_over_saturated_rule(self) -> None:
        rule = calibrated_score_fields({
            "subject_track_id": 1, "object_track_id": 2, "predicate": "left",
            "start_frame": 0, "end_frame": 9, "span_convention": "inclusive",
            "evidence_frames": list(range(10)), "rule_support": 1.0,
        })
        semantic = calibrated_score_fields({
            "subject_track_id": 1, "object_track_id": 2, "predicate": "ride",
            "start_frame": 0, "end_frame": 9, "span_convention": "inclusive",
            "evidence_frames": [0, 3, 6, 9], "agent_score": 0.8,
        })
        self.assertLess(rule["ranking_score"], semantic["ranking_score"])
        self.assertEqual(rule["score_kind"], "calibrated_relation_v1")
        self.assertEqual(rule["source_kind"], "rule")
        self.assertEqual(semantic["source_kind"], "semantic")

    def test_partial_observation_is_not_automatically_a_risk_track(self) -> None:
        fields = calibrated_score_fields({
            "subject_track_id": 1, "object_track_id": 2, "predicate": "ride",
            "start_frame": 0, "end_frame": 4, "agent_score": 0.8,
            "evidence_frames": [0, 2, 4],
        }, track_quality={
            "1": {"quality": 0.8, "risk_types": []},
            "2": {"quality": 0.7, "risk_types": ["large_jump"]},
        })
        self.assertTrue(fields["risk_track_involved"])
        self.assertEqual(fields["track_pair_quality"], 0.7)
        clean = calibrated_score_fields({
            "subject_track_id": 1, "object_track_id": 2, "predicate": "ride",
            "start_frame": 0, "end_frame": 4, "agent_score": 0.8,
            "evidence_frames": [0, 2, 4],
        }, track_quality={
            "1": {"quality": 0.8, "risk_types": []},
            "2": {"quality": 0.7, "risk_types": []},
        })
        self.assertFalse(clean["risk_track_involved"])

    def test_global_relation_splits_at_risky_track_jump(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            source, output = root / "merged.json", root / "global.json"
            _write(source, {"v1": [{
                "subject_track_id": 1, "predicate": "follow", "object_track_id": 2,
                "start_frame": 0, "end_frame": 9, "span_convention": "inclusive",
                "evidence_frames": [0, 2, 4, 6, 8], "agent_score": 0.8,
            }]})
            result = run_global_relation(
                video_id="v1", relations_json=source, out_json=output,
                config={"track_jump_frames": {"1": [5]}},
            )
            self.assertEqual([(item["start_frame"], item["end_frame"]) for item in result["v1"]], [(0, 4), (6, 8)])
            self.assertTrue(all(item["split_on_track_jump"] for item in result["v1"]))

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
            self.assertEqual(result["v1"][0]["span_convention"], "inclusive")

    def test_global_relation_does_not_bridge_distant_evidence(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            source, output = root / "merged.json", root / "global.json"
            base = {"subject_track_id": 1, "predicate": "follow", "object_track_id": 2, "agent_score": 0.8}
            _write(source, {"v1": [
                dict(base, start_frame=0, end_frame=10, evidence_frames=[0]),
                dict(base, start_frame=5, end_frame=15, evidence_frames=[15]),
            ]})
            result = run_global_relation(
                video_id="v1", relations_json=source, out_json=output,
                config={"max_relation_gap_frames": 1, "max_evidence_gap": 4},
            )
            self.assertEqual(len(result["v1"]), 2)

    def test_review_actions_use_relation_id_not_list_position(self) -> None:
        items = [
            {"relation_id": "r000002", "predicate": "next_to", "start_frame": 0, "end_frame": 10},
            {"relation_id": "r000001", "predicate": "left", "start_frame": 0, "end_frame": 10},
        ]
        result, applied = _apply_final_actions(items, [{"relation_id": "r000001", "action": "change_predicate", "new_predicate": "right"}])
        self.assertEqual(result[0]["predicate"], "next_to")
        self.assertEqual(result[1]["predicate"], "right")
        self.assertEqual(len(applied), 1)

    def test_rule_relation_uses_evidence_span_and_requires_observation(self) -> None:
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
            self.assertEqual(left["evidence_frames"], [0, 1, 2])
            self.assertEqual(left["rule_support"], 1.0)
            self.assertEqual(left["span_convention"], "inclusive")
            self.assertNotIn("confidence", left)
            self.assertEqual(left["source"], "window_geometry")

    def test_rule_relation_splits_disconnected_evidence(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            windows, tracks, output = root / "windows.json", root / "tracks.jsonl", root / "relations.json"
            _write(windows, {"windows": [{"window_id": 1, "start_frame": 0, "end_frame": 4, "track_ids": [1, 2]}]})
            rows = []
            for frame, object_x in enumerate((30, 30, 5, 30, 30)):
                rows.append({"frame": frame, "tracks": [
                    {"track_id": 1, "bbox": [0, 0, 10, 10], "box_source": "observed"},
                    {"track_id": 2, "bbox": [object_x, 0, object_x + 10, 10], "box_source": "observed"},
                ]})
            tracks.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            result = generate_rule_relations(
                windows_json=windows, tracks_jsonl=tracks, out_json=output, video_id="v1",
                config={"min_vote_ratio": 0.6, "min_evidence_frames": 2, "evidence_max_gap": 0},
            )
            spans = [
                (item["start_frame"], item["end_frame"])
                for item in result["v1"] if item["subject_track_id"] == 1 and item["predicate"] == "left"
            ]
            self.assertEqual(spans, [(0, 1), (3, 4)])

    def test_rule_relation_does_not_cross_qc_jump(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            windows, tracks, output = root / "windows.json", root / "tracks.jsonl", root / "relations.json"
            _write(windows, {"windows": [{"window_id": 1, "start_frame": 0, "end_frame": 5, "track_ids": [1, 2]}]})
            tracks.write_text("".join(json.dumps({"frame": frame, "tracks": [
                {"track_id": 1, "bbox": [0, 0, 10, 10], "box_source": "observed"},
                {"track_id": 2, "bbox": [30, 0, 40, 10], "box_source": "observed"},
            ]}) + "\n" for frame in range(6)), encoding="utf-8")
            result = generate_rule_relations(
                windows_json=windows, tracks_jsonl=tracks, out_json=output, video_id="v1",
                config={"track_jump_frames": {"1": [3]}},
            )
            spans = [(item["start_frame"], item["end_frame"]) for item in result["v1"]
                     if item["subject_track_id"] == 1 and item["predicate"] == "left"]
            self.assertEqual(spans, [(0, 2), (3, 5)])

    def test_depth_predicates_are_sent_to_semantic_classifier(self) -> None:
        evidence = {
            "edge_gap_min": 1.0,
            "iou_max": 0.0,
            "approach_rate": 0.0,
            "relative_motion": 0.0,
            "motion_alignment": 0.0,
            "size_ratio": 1.0,
            "subject_speed": 0.0,
            "median_dx": -1.0,
            "median_dy": 0.0,
            "distance_min": 1.0,
            "crosses_horizontal_axis": False,
        }
        candidates = route_predicates(evidence, exclude=GEOMETRY_PREDICATES)["candidate_predicates"]
        self.assertIn("front", candidates)
        self.assertIn("behind", candidates)
        self.assertNotIn("front", GEOMETRY_PREDICATES)
        self.assertNotIn("left", candidates)

    def test_merge_keeps_source_scores_separate_without_bonus(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            rule, semantic, output = root / "rule.json", root / "semantic.json", root / "merged.json"
            base = {"subject_track_id": 1, "predicate": "next_to", "object_track_id": 2, "start_frame": 1, "end_frame": 3}
            _write(rule, {"v1": [dict(base, rule_support=0.8, ranking_score=0.8, score_kind="rule_support", source="window_geometry")]})
            _write(semantic, {"v1": [dict(base, agent_score=0.6, ranking_score=0.6, score_kind="agent_ranking", source="window_semantic_vl")]})
            item = merge_relations(video_id="v1", relation_jsons=[rule, semantic], out_json=output)["v1"][0]
            self.assertEqual(item["rule_support"], 0.8)
            self.assertEqual(item["agent_score"], 0.6)
            self.assertEqual(item["ranking_score"], 0.8)
            self.assertNotIn("confidence", item)

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
                text="Visual evidence is insufficient.\n```json\n"
                + json.dumps({"actions": [{"relation_id": "r000001", "action": "reject_relation", "reason": "not visible"}]})
                + "\n```",
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

    @patch("vidvrd_auto.relations.ops.DashScopeProvider")
    def test_visual_review_invalid_response_fails_with_qc_artifact(self, provider_type) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            relations, tracks = root / "relations.json", root / "tracks.jsonl"
            verified, qc = root / "verified.json", root / "qc.json"
            storyboards = root / "storyboards"
            storyboards.mkdir()
            (storyboards / "window_0001_A1_B2.jpg").write_bytes(b"visual-evidence")
            _write(relations, {"v1": [{
                "relation_id": "r000001", "subject_track_id": 1, "predicate": "next_to",
                "object_track_id": 2, "start_frame": 0, "end_frame": 29, "confidence": 0.2,
            }]})
            tracks.write_text(json.dumps({"frame": 0, "tracks": []}) + "\n", encoding="utf-8")
            provider_type.return_value.call.return_value = VLResult(ok=True, text="no JSON", model="mock")
            with self.assertRaisesRegex(RuntimeError, "visual relation review incomplete"):
                verify_relations(
                    video_id="v1", relations_json=relations, tracks_jsonl=tracks,
                    out_relations_json=verified, out_qc_json=qc,
                    config={"strong_model_review_enabled": True, "low_confidence_threshold": 0.45},
                    storyboards_dir=storyboards,
                )
            self.assertEqual(json.loads(qc.read_text(encoding="utf-8"))["strong_model_review_result"]["state"], "invalid_response")


if __name__ == "__main__":
    unittest.main()
