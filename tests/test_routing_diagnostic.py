from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vidvrd_auto.evaluation.routing import diagnose_routed_candidates


def test_routing_diagnosis_separates_router_from_missing_pair() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        packet_file = root / "clip" / "semantic" / "evidence_packets.json"
        packet_file.parent.mkdir(parents=True)
        packet_file.write_text(json.dumps({"packets": [{
            "start_frame": 0, "end_frame": 2,
            "candidate_directions": [{
                "subject_track_id": 10, "object_track_id": 20,
                "candidate_predicates": ["ride"],
            }],
        }]}), encoding="utf-8")
        tube = {str(frame): [0, 0, 10, 10] for frame in range(3)}
        gold_tracks = {"clip": [
            {"track_id": 1, "category": "person", "trajectory": tube},
            {"track_id": 2, "category": "horse", "trajectory": tube},
            {"track_id": 3, "category": "rabbit", "trajectory": tube},
        ]}
        pred_tracks = {"clip": [
            {"track_id": 10, "category": "person", "trajectory": tube},
            {"track_id": 20, "category": "horse", "trajectory": tube},
        ]}
        relations = {"clip": [
            {"subject_track_id": 1, "object_track_id": 2, "predicate": "ride", "start_frame": 0, "end_frame": 2},
            {"subject_track_id": 1, "object_track_id": 2, "predicate": "watch", "start_frame": 0, "end_frame": 2},
            {"subject_track_id": 1, "object_track_id": 2, "predicate": "left", "start_frame": 0, "end_frame": 2},
        ]}
        report = diagnose_routed_candidates(relations, gold_tracks, pred_tracks, ["clip"], root)
        assert report["counts"] == {"candidate_routed": 1, "router_missed_predicate": 1, "rule_only_predicate": 1}
        assert report["semantic_eligible_groundtruth_count"] == 2
        assert report["unmatched_gold_track_categories"] == {"rabbit": 1}
        assert report["unmatched_gold_tracks"][0]["best_same_category_viou"] == 0.0
        assert report["routed_candidate_recall"] == 0.5


class RoutingDiagnosticTests(unittest.TestCase):
    def test_router_and_missing_pair(self):
        test_routing_diagnosis_separates_router_from_missing_pair()
