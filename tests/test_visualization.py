from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from vidvrd_auto.visualization.render import render_prediction_video


class VisualizationTests(unittest.TestCase):
    def test_render_prediction_video_creates_playable_overlay(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "clip.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()
            relations = root / "relations.json"
            trajectories = root / "trajectories.json"
            relations.write_text(json.dumps({"clip": [{
                "subject_track_id": 1, "predicate": "next_to", "object_track_id": 2,
                "start_frame": 0, "end_frame": 2, "span_convention": "inclusive",
            }]}), encoding="utf-8")
            trajectories.write_text(json.dumps({"clip": [
                {"track_id": 1, "category": "person", "trajectory": {str(i): [2, 2, 20, 30] for i in range(3)}},
                {"track_id": 2, "category": "dog", "trajectory": {str(i): [35, 10, 55, 35] for i in range(3)}},
            ]}), encoding="utf-8")
            output = root / "overlay.mp4"
            result = render_prediction_video(
                video=source, relations_path=relations, trajectories_path=trajectories, output=output
            )
            self.assertEqual(result["frames"], 3)
            self.assertGreater(output.stat().st_size, 0)
            capture = cv2.VideoCapture(str(output))
            ok, frame = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertGreater(int(frame.sum()), 0)


if __name__ == "__main__":
    unittest.main()
