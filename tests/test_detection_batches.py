from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from vidvrd_auto.detection.temporal_fusion import annotate_batch_detections, make_frame_batches
from vidvrd_auto.detection.dinox import DinoXDetector
from vidvrd_auto.detection.hybrid import HybridDetector
from vidvrd_auto.detection.rex import RexDetector
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


class SourceDetector:
    def __init__(self, source, *, fail_values=()) -> None:
        self.source = source
        self.fail_values = set(fail_values)
        self.values = []

    def load_model(self) -> None:
        return None

    def detect_batch(self, frames):
        output = []
        for frame in frames:
            value = int(frame[0, 0, 0])
            if value in self.fail_values:
                raise RuntimeError("planned detector failure")
            self.values.append(value)
            output.append([
                {
                    "bbox": [1, 2, 11, 12],
                    "class": 0,
                    "class_name": "person",
                    "score": 0.9 if self.source == "dinox" else None,
                    "source": self.source,
                }
            ])
        return output

    def get_stats(self):
        return {"backend": self.source, "values": list(self.values)}


class DetectionBatchTests(unittest.TestCase):
    def test_dinox_parses_native_scores_and_canonical_labels(self) -> None:
        detector = DinoXDetector(
            categories=["person", "bicycle"],
            category_aliases={"bike": "bicycle"},
        )
        parsed = detector._parse(
            {"objects": [{"bbox": [-1, 2, 80, 50], "score": 0.83, "category": "Bike"}]},
            width=64,
            height=48,
        )
        self.assertEqual(parsed[0]["bbox"], [0.0, 2.0, 64.0, 48.0])
        self.assertEqual(parsed[0]["class_name"], "bicycle")
        self.assertEqual(parsed[0]["score"], 0.83)
        self.assertEqual(parsed[0]["score_kind"], "native")
        self.assertEqual(parsed[0]["association_weight"], 1.0)
        self.assertEqual(parsed[0]["source"], "dinox")

    def test_hybrid_uses_dinox_every_fifteen_frames(self) -> None:
        rex = SourceDetector("rexomni")
        dinox = SourceDetector("dinox")
        detector = HybridDetector(rex=rex, dinox=dinox, dinox_interval=15)
        indices = [0, 3, 6, 9, 12, 15]
        frames = [np.full((4, 4, 3), value, dtype=np.uint8) for value in indices]
        detected = detector.detect_batch_indexed(frames, frame_indices=indices)
        self.assertEqual(dinox.values, [0, 15])
        self.assertEqual(rex.values, [3, 6, 9, 12])
        self.assertEqual([items[0]["source"] for items in detected], [
            "dinox", "rexomni", "rexomni", "rexomni", "rexomni", "dinox"
        ])
        self.assertEqual(detector.source_for_frame(15), "dinox")

    def test_hybrid_falls_back_to_rex_for_failed_dinox_frame(self) -> None:
        rex = SourceDetector("rexomni")
        dinox = SourceDetector("dinox", fail_values={0})
        detector = HybridDetector(rex=rex, dinox=dinox, dinox_interval=15)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        detected = detector.detect_batch_indexed([frame], frame_indices=[0])
        self.assertEqual(detected[0][0]["source"], "rexomni")
        self.assertEqual(detector.source_for_frame(0), "rexomni_fallback")
        self.assertEqual(detector.get_stats()["dinox_fallbacks"], 1)

    def test_rex_defaults_are_deterministic_and_do_not_invent_scores(self) -> None:
        detector = RexDetector("fake", min_box_area=0)
        parsed = detector._parse(
            {"extracted_predictions": {"person": [{"type": "box", "coords": [1, 2, 11, 12]}]}}
        )
        self.assertEqual((detector.temperature, detector.top_p, detector.top_k), (0.0, 0.05, 1))
        self.assertIsNone(parsed[0]["score"])
        self.assertEqual(parsed[0]["score_kind"], "unavailable")
        self.assertNotIn("confidence", parsed[0])

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

    def test_fixed_sparse_detects_exactly_every_three_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "seven.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (32, 24))
            self.assertTrue(writer.isOpened())
            for value in range(7):
                writer.write(np.full((24, 32, 3), value * 30, dtype=np.uint8))
            writer.release()

            out_dir = root / "detect"
            detect_video(
                video_path=video,
                out_dir=out_dir,
                config={
                    "sampling_mode": "fixed_sparse",
                    "detection_interval": 3,
                    "min_detection_interval": 3,
                    "scene_change_threshold": 0.01,
                    "batch_size": 5,
                    "rex_model_path": "fake",
                },
                log_path=out_dir / "run.log",
                detector_factory=FakeDetector,
            )

            rows = [json.loads(line) for line in (out_dir / "detections.jsonl").read_text(encoding="utf-8").splitlines()]
            anchors = [row["frame"] for row in rows if row["detection_batch"]["status"] == "observed"]
            self.assertEqual(anchors, [0, 3, 6])
            self.assertEqual(json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))["sampling"]["mode"], "fixed_sparse")



if __name__ == "__main__":
    unittest.main()
