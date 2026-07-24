from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from vidvrd_auto.detection.temporal_fusion import annotate_batch_detections, make_frame_batches
from vidvrd_auto.detection.video import detect_video


class FakeDetector:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.batch_sizes = []

    def load_model(self) -> None:
        return None

    def detect_batch(self, frames):
        self.batch_sizes.append(len(frames))
        return [
            [{"bbox": [1, 2, 11, 12], "class": 0, "class_name": "person", "confidence": 1.0}]
            for _ in frames
        ]

    def detect(self, frame):
        return self.detect_batch([frame])[0]

    def get_stats(self):
        return {"backend": "rexomni", "batch_sizes": self.batch_sizes}


class DetectionBatchTests(unittest.TestCase):
    def test_batches_use_distinct_real_frames(self) -> None:
        self.assertEqual(make_frame_batches(list(range(7)), 5), [[0, 1, 2, 3, 4], [5, 6]])

    def test_provenance_is_attached_per_frame(self) -> None:
        rows = annotate_batch_detections(
            [[{"bbox": [0, 0, 10, 10]}], []],
            frame_indices=[8, 9],
            batch_id=3,
            source="rexomni",
        )
        self.assertEqual(rows[0][0]["batch_frame_indices"], [8, 9])
        self.assertEqual(rows[0][0]["source"], "rexomni")

    def test_duplicate_frame_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            annotate_batch_detections([[], []], frame_indices=[2, 2], batch_id=0, source="rexomni")

    def test_video_stage_flushes_short_final_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "seven.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (32, 24))
            self.assertTrue(writer.isOpened())
            for value in range(7):
                writer.write(np.full((24, 32, 3), value * 10, dtype=np.uint8))
            writer.release()

            out_dir = root / "detect"
            detect_video(
                video_path=video,
                out_dir=out_dir,
                config={"batch_size": 5, "rex_model_path": "fake"},
                log_path=out_dir / "run.log",
                detector_factory=FakeDetector,
            )

            rows = [json.loads(line) for line in (out_dir / "detections.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["frame"] for row in rows], list(range(7)))
            self.assertEqual(rows[0]["detection_batch"]["frame_indices"], [0, 5])
            self.assertEqual(rows[5]["detection_batch"]["frame_indices"], [0, 5])
            self.assertEqual(rows[1]["detection_batch"]["status"], "skipped")
            self.assertEqual(rows[1]["objects"], [])
            meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["detector_stats"]["batch_sizes"], [2])
            self.assertEqual(meta["sampling"]["anchor_frames"], 2)



if __name__ == "__main__":
    unittest.main()
