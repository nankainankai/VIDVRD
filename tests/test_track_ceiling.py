from __future__ import annotations

import unittest

from vidvrd_auto.evaluation.track_ceiling import diagnose_track_ceiling, summarize_track_ceiling_outcomes


class TrackCeilingTests(unittest.TestCase):
    def test_absent_category_and_missing_frames_limit_recall(self) -> None:
        box = [0, 0, 9, 9]
        gold = {"clip": [
            {
                "triplet": ["person", "ride", "horse"], "duration": [0, 2],
                "sub_traj": [box, box], "obj_traj": [box, box],
            },
            {
                "triplet": ["person", "watch", "dog"], "duration": [0, 2],
                "sub_traj": [box, box], "obj_traj": [box, box],
            },
        ]}
        tracks = {"clip": [
            {"track_id": 1, "category": "person", "trajectory": {"0": box, "1": box}},
            {"track_id": 2, "category": "horse", "trajectory": {"0": box, "1": box}},
            {"track_id": 3, "category": "dog", "trajectory": {"0": box}},
        ]}
        report = diagnose_track_ceiling(gold, tracks, threshold=0.6)
        self.assertEqual(report["maximum_matchable_relations"], 1)
        self.assertEqual(report["reason_counts"]["insufficient_track_coverage"], 1)
        self.assertEqual(report["rows"][1]["object_coverage_upper_bound"], 0.5)

    def test_missing_category_is_separate_from_bad_coverage(self) -> None:
        box = [0, 0, 9, 9]
        gold = {"clip": [{"triplet": ["person", "ride", "horse"], "duration": [0, 1],
                           "sub_traj": [box], "obj_traj": [box]}]}
        tracks = {"clip": [{"track_id": 1, "category": "person", "trajectory": {"0": box}}]}
        report = diagnose_track_ceiling(gold, tracks)
        self.assertEqual(report["reason_counts"], {"category_absent": 1})

    def test_reachable_outcomes_exclude_track_limited_misses(self) -> None:
        ceiling = {
            "maximum_matchable_relations": 2,
            "rows": [
                {"video_id": "clip", "groundtruth_index": 0, "reason": "track_ceiling_allows_match"},
                {"video_id": "clip", "groundtruth_index": 1, "reason": "track_ceiling_allows_match"},
                {"video_id": "clip", "groundtruth_index": 2, "reason": "category_absent"},
            ],
        }
        errors = {"groundtruth": [
            {"video_id": "clip", "groundtruth_index": 0, "true_positive_rank": 7, "failure_type": "ranking_failure"},
            {"video_id": "clip", "groundtruth_index": 1, "true_positive_rank": None, "failure_type": "candidate_missing"},
            {"video_id": "clip", "groundtruth_index": 2, "true_positive_rank": None, "failure_type": "candidate_missing"},
        ]}
        routing = {"rows": [
            {"video_id": "clip", "groundtruth_index": 1, "status": "candidate_routed"},
        ]}
        outcome = summarize_track_ceiling_outcomes(ceiling, errors, routing)
        self.assertEqual(outcome["actual_true_positive_count_any_rank"], 1)
        self.assertEqual(outcome["conditional_realized_fraction"], 0.5)
        self.assertEqual(outcome["potentially_matchable_failure_counts"], {"candidate_missing": 1})
        self.assertEqual(outcome["potentially_matchable_candidate_missing_by_route"], {"candidate_routed": 1})


if __name__ == "__main__":
    unittest.main()
