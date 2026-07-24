from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vidvrd_auto.tracking.ocsort.adapter import ObjectTracker
from vidvrd_auto.tracking.video import _compact


def detection(x: float, class_name: str = "person", confidence: float = 0.95) -> dict[str, object]:
    return {
        "bbox": [x, 20.0, x + 20.0, 50.0],
        "class_name": class_name,
        "confidence": confidence,
    }


class OfficialOCSortAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((100, 160, 3), dtype=np.uint8)

    def make_tracker(self, **overrides: object) -> ObjectTracker:
        options: dict[str, object] = {
            "iou_threshold": 0.2,
            "max_age": 3,
            "min_hits": 1,
            "min_new_track_conf": 0.3,
        }
        options.update(overrides)
        return ObjectTracker(**options)

    def test_id_continuity_for_moving_observations(self) -> None:
        tracker = self.make_tracker()
        ids = []
        for frame_num, x in enumerate((10.0, 12.0, 14.0, 16.0)):
            tracks = tracker.track(self.frame, [detection(x)], frame_num)
            self.assertEqual(len(tracks), 1)
            ids.append(tracks[0]["track_id"])
            self.assertFalse(tracks[0]["is_predicted"])
            self.assertEqual(tracks[0]["bbox_observed"], detection(x)["bbox"])
        self.assertEqual(len(set(ids)), 1)

    def test_occlusion_emits_prediction_with_no_observed_bbox_then_recovers_id(self) -> None:
        tracker = self.make_tracker()
        first = tracker.track(self.frame, [detection(10.0)], 0)[0]
        second = tracker.track(self.frame, [detection(12.0)], 1)[0]
        hidden = tracker.track(self.frame, [], 2)[0]

        self.assertEqual(first["track_id"], second["track_id"])
        self.assertEqual(hidden["track_id"], first["track_id"])
        self.assertTrue(hidden["is_predicted"])
        self.assertIsNone(hidden["bbox_observed"])
        self.assertEqual(hidden["time_since_update"], 1)
        self.assertEqual(len(hidden["bbox"]), 4)

        recovered = tracker.track(self.frame, [detection(16.0)], 3)[0]
        self.assertEqual(recovered["track_id"], first["track_id"])
        self.assertFalse(recovered["is_predicted"])
        self.assertEqual(recovered["bbox_observed"], detection(16.0)["bbox"])

    def test_incompatible_class_creates_a_distinct_track(self) -> None:
        tracker = self.make_tracker()
        person = tracker.track(self.frame, [detection(10.0, "person")], 0)[0]
        tracks = tracker.track(self.frame, [detection(10.0, "car")], 1)
        by_class = {track["class_name"]: track for track in tracks}
        self.assertEqual(by_class["person"]["track_id"], person["track_id"])
        self.assertTrue(by_class["person"]["is_predicted"])
        self.assertNotEqual(by_class["car"]["track_id"], person["track_id"])

    def test_new_class_backend_cannot_reset_project_track_ids(self) -> None:
        tracker = self.make_tracker()
        first = tracker.track(self.frame, [detection(10.0, "person")], 0)[0]
        tracker.track(self.frame, [detection(80.0, "car")], 1)
        tracks = tracker.track(self.frame, [detection(120.0, "person")], 2)
        ids = [item["track_id"] for item in tracks]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn(first["track_id"], ids)

    def test_confidence_weighted_vote_accepts_configured_alias(self) -> None:
        tracker = self.make_tracker(class_compatibility={"person": ["human"]})
        first = tracker.track(self.frame, [detection(10.0, "person", 0.95)], 0)[0]
        second = tracker.track(self.frame, [detection(12.0, "human", 0.35)], 1)[0]
        self.assertEqual(second["track_id"], first["track_id"])
        self.assertEqual(second["class_name"], "person")

    def test_low_iou_core_match_also_updates_adapter_metadata(self) -> None:
        tracker = self.make_tracker(iou_threshold=0.1)
        first = tracker.track(self.frame, [detection(10.0)], 0)[0]
        # IoU is below the adapter's former hard-coded 0.5 cutoff, but above
        # the configured OC-SORT cutoff.
        moved = tracker.track(self.frame, [detection(24.0)], 1)[0]
        self.assertEqual(moved["track_id"], first["track_id"])
        self.assertEqual(moved["hits"], 2)
        self.assertEqual(moved["bbox_observed"], detection(24.0)["bbox"])

    def test_compaction_preserves_null_observation(self) -> None:
        rows = _compact(
            [
                {
                    "track_id": 4,
                    "bbox": [1.0, 2.0, 11.0, 12.0],
                    "bbox_observed": None,
                    "is_predicted": True,
                }
            ]
        )
        self.assertIsNone(rows[0]["bbox_observed"])


if __name__ == "__main__":
    unittest.main()
