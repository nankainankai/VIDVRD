from __future__ import annotations

import unittest

import numpy as np

from vidvrd_auto.tracking.hybrid import HybridTracker
from vidvrd_auto.tracking.stitching import stitch_tracklets


FRAME = np.zeros((80, 120, 3), dtype=np.uint8)


def detection(box: list[float], name: str = "person") -> dict[str, object]:
    return {"bbox": box, "class_name": name, "score": 0.9}


class HybridTrackerTests(unittest.TestCase):
    def test_missing_native_score_is_not_exported_as_confidence(self) -> None:
        tracker = HybridTracker(min_hits=1)
        result = tracker.update(
            FRAME,
            [{"bbox": [5, 5, 25, 35], "class_name": "person"}],
            np.asarray([[1.0, 0.0]]),
            frame_num=0,
            scene_id=0,
        )
        self.assertIsNone(result[0]["confidence"])
        self.assertIsNone(tracker.summaries()[0]["mean_confidence"])

    def test_confirmation_flushes_earlier_real_observations(self) -> None:
        tracker = HybridTracker(min_hits=2)
        self.assertEqual(
            tracker.update(FRAME, [detection([5, 5, 25, 35])], np.asarray([[1.0, 0.0]]), frame_num=0, scene_id=0),
            [],
        )
        confirmed = tracker.update(
            FRAME, [detection([7, 5, 27, 35])], np.asarray([[1.0, 0.0]]), frame_num=5, scene_id=0
        )
        self.assertEqual([item["frame"] for item in confirmed], [0, 5])
        self.assertTrue(all(item["track_status"] == "confirmed" for item in confirmed))

    def test_class_is_soft_evidence_not_an_id_partition(self) -> None:
        tracker = HybridTracker(min_hits=1, max_lost_frames=20)
        first = tracker.update(FRAME, [detection([5, 5, 25, 35], "person")], np.asarray([[1.0, 0.0]]), frame_num=0, scene_id=0)
        second = tracker.update(FRAME, [detection([7, 5, 27, 35], "man")], np.asarray([[1.0, 0.0]]), frame_num=5, scene_id=0)
        self.assertEqual(first[0]["local_tracklet_id"], second[0]["local_tracklet_id"])
        self.assertIn("person", second[0]["class_distribution"])
        self.assertIn("man", second[0]["class_distribution"])

    def test_appearance_prevents_a_crossing_id_swap(self) -> None:
        tracker = HybridTracker(
            min_hits=1,
            appearance_weight=0.80,
            iou_weight=0.05,
            motion_weight=0.10,
            class_weight=0.05,
            max_match_cost=1.0,
        )
        first = tracker.update(
            FRAME,
            [detection([5, 5, 25, 35]), detection([75, 5, 95, 35])],
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            frame_num=0,
            scene_id=0,
        )
        second = tracker.update(
            FRAME,
            [detection([75, 5, 95, 35]), detection([5, 5, 25, 35])],
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            frame_num=5,
            scene_id=0,
        )
        self.assertEqual(first[0]["local_tracklet_id"], second[0]["local_tracklet_id"])
        self.assertEqual(first[1]["local_tracklet_id"], second[1]["local_tracklet_id"])

    def test_velocity_uses_real_frame_delta(self) -> None:
        tracker = HybridTracker(
            min_hits=1,
            appearance_weight=0.0,
            iou_weight=0.1,
            motion_weight=0.85,
            class_weight=0.05,
            max_match_cost=0.8,
        )
        ids = []
        for frame_num, x in ((0, 5), (5, 10), (10, 15)):
            result = tracker.update(
                FRAME,
                [detection([x, 5, x + 20, 35])],
                np.empty((0, 0)),
                frame_num=frame_num,
                scene_id=0,
            )
            ids.append(result[0]["local_tracklet_id"])
        self.assertEqual(ids, [1, 1, 1])
        self.assertEqual(tracker.summaries()[0]["last_velocity"], [1.0, 0.0, 1.0, 0.0])


class StitchingTests(unittest.TestCase):
    @staticmethod
    def summary(local_id: int, start: int, end: int, scene: int, embedding: list[float]) -> dict[str, object]:
        return {
            "local_tracklet_id": local_id,
            "scene_id": scene,
            "start_frame": start,
            "end_frame": end,
            "observation_count": 2,
            "class_distribution": {"person": 1.0},
            "first_bbox": [start, 0, start + 10, 20],
            "last_bbox": [end, 0, end + 10, 20],
            "last_velocity": [1, 0, 1, 0],
            "mean_embedding": embedding,
        }

    def test_dag_stitching_is_one_to_one_and_scene_bounded(self) -> None:
        summaries = [
            self.summary(1, 0, 10, 0, [1.0, 0.0]),
            self.summary(2, 20, 25, 0, [1.0, 0.0]),
            self.summary(3, 20, 25, 1, [1.0, 0.0]),
        ]
        mapping, links = stitch_tracklets(summaries, {"stitch_max_gap_frames": 30})
        self.assertEqual(mapping[1], mapping[2])
        self.assertNotEqual(mapping[1], mapping[3])
        self.assertEqual([(link["from_local_tracklet_id"], link["to_local_tracklet_id"]) for link in links], [(1, 2)])


if __name__ == "__main__":
    unittest.main()
