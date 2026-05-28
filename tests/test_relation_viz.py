from __future__ import annotations

import unittest

from vidvrd_auto.utils.relation_viz import (
    anchor_point_in_bbox,
    dedupe_relations_by_pair,
    filter_relations_for_visualization,
    format_relation_label,
    is_positional_predicate,
    relations_active_at_frame,
    select_relations_for_visualization,
    top_relations_by_pair,
)


class RelationVizTests(unittest.TestCase):
    def test_relations_active_at_frame(self) -> None:
        rels = [
            {"subject_track_id": 0, "object_track_id": 1, "predicate": "behind", "start_frame": 0, "end_frame": 10, "confidence": 0.9},
            {"subject_track_id": 0, "object_track_id": 1, "predicate": "away", "start_frame": 20, "end_frame": 30, "confidence": 0.8},
        ]
        active = relations_active_at_frame(rels, 5, min_confidence=0.5)
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["predicate"], "behind")
        self.assertEqual(len(relations_active_at_frame(rels, 15, min_confidence=0.5)), 0)

    def test_dedupe_relations_by_pair_keeps_highest_confidence(self) -> None:
        rels = [
            {"subject_track_id": 0, "object_track_id": 1, "predicate": "behind", "confidence": 0.7},
            {"subject_track_id": 0, "object_track_id": 1, "predicate": "below", "confidence": 0.9},
            {"subject_track_id": 1, "object_track_id": 0, "predicate": "front", "confidence": 0.6},
        ]
        out = dedupe_relations_by_pair(rels)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["predicate"], "below")
        self.assertEqual(out[1]["predicate"], "front")

    def test_top_relations_by_pair_keeps_second_highest(self) -> None:
        rels = [
            {"subject_track_id": 0, "object_track_id": 1, "predicate": "left", "confidence": 0.95},
            {"subject_track_id": 0, "object_track_id": 1, "predicate": "follow", "confidence": 0.62},
            {"subject_track_id": 0, "object_track_id": 1, "predicate": "near", "confidence": 0.40},
        ]
        out = top_relations_by_pair(rels, top_k=2)
        self.assertEqual(len(out), 2)
        preds = {r["predicate"] for r in out}
        self.assertEqual(preds, {"left", "follow"})

    def test_format_relation_label_includes_confidence(self) -> None:
        rel = {"subject_track_id": 0, "object_track_id": 1, "predicate": "follow", "confidence": 0.8567}
        self.assertEqual(format_relation_label(rel), "0->1:follow 0.86")
        self.assertEqual(format_relation_label(rel, show_confidence=False), "0->1:follow")

    def test_anchor_point_stays_inside_bbox(self) -> None:
        bbox = [100.0, 50.0, 200.0, 150.0]
        for seed in (0, 1, 99, 12345):
            x, y = anchor_point_in_bbox(bbox, seed)
            self.assertGreaterEqual(x, 100)
            self.assertLessEqual(x, 200)
            self.assertGreaterEqual(y, 50)
            self.assertLessEqual(y, 150)

    def test_filter_drops_low_and_high_confidence_spatial(self) -> None:
        tracks = [
            {"track_id": 0, "bbox": [10, 10, 50, 50], "class_name": "person"},
            {"track_id": 1, "bbox": [200, 200, 240, 240], "class_name": "person"},
        ]
        rels = [
            {"subject_track_id": 0, "object_track_id": 1, "predicate": "left", "confidence": 0.99},
            {"subject_track_id": 0, "object_track_id": 1, "predicate": "follow", "confidence": 0.55},
            {"subject_track_id": 0, "object_track_id": 1, "predicate": "away", "confidence": 0.20},
        ]
        out = filter_relations_for_visualization(rels, tracks, frame_width=640, frame_height=480)
        preds = {r["predicate"] for r in out}
        self.assertNotIn("left", preds)
        self.assertNotIn("away", preds)
        self.assertIn("follow", preds)

    def test_filter_drops_distant_positional_relation(self) -> None:
        tracks = [
            {"track_id": 0, "bbox": [0, 0, 40, 40], "class_name": "person"},
            {"track_id": 1, "bbox": [580, 420, 620, 460], "class_name": "person"},
        ]
        rels = [{"subject_track_id": 0, "object_track_id": 1, "predicate": "right", "confidence": 0.80}]
        out = filter_relations_for_visualization(
            rels,
            tracks,
            frame_width=640,
            frame_height=480,
            spatial_max_center_distance_ratio=0.35,
        )
        self.assertEqual(out, [])

    def test_select_keeps_top_confidence_after_filter(self) -> None:
        tracks = [
            {"track_id": 0, "bbox": [10, 10, 50, 50], "class_name": "person"},
            {"track_id": 1, "bbox": [60, 10, 100, 50], "class_name": "person"},
        ]
        rels = [
            {"subject_track_id": 0, "object_track_id": 1, "predicate": "left", "confidence": 0.99},
            {"subject_track_id": 0, "object_track_id": 1, "predicate": "follow", "confidence": 0.72},
            {"subject_track_id": 0, "object_track_id": 1, "predicate": "toward", "confidence": 0.65},
        ]
        out = select_relations_for_visualization(rels, tracks, frame_width=640, frame_height=480, top_k_per_pair=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["predicate"], "follow")

    def test_is_positional_predicate(self) -> None:
        self.assertTrue(is_positional_predicate("left"))
        self.assertFalse(is_positional_predicate("follow"))


if __name__ == "__main__":
    unittest.main()
