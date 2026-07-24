from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from vidvrd_auto.evaluation.vidvrd import run_vidvrd_eval
from vidvrd_auto.relations.semantic import classify_relations
from vidvrd_auto.tracking.video import _smooth_short_gaps, track_video


def _video(path: Path, frames: int = 4) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("test video writer unavailable")
    for index in range(frames):
        writer.write(np.full((48, 64, 3), 30 + index * 10, dtype=np.uint8))
    writer.release()


class NativeStageTests(unittest.TestCase):
    def test_short_predicted_gap_is_interpolated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tracks.jsonl"
            rows = [
                {"frame": 0, "tracks": [{"track_id": 1, "bbox": [0, 0, 10, 10], "bbox_observed": [0, 0, 10, 10], "box_source": "observed"}]},
                {"frame": 1, "tracks": [{"track_id": 1, "bbox": [4, 0, 14, 10], "bbox_observed": None, "box_source": "predicted"}]},
                {"frame": 2, "tracks": [{"track_id": 1, "bbox": [10, 0, 20, 10], "bbox_observed": [10, 0, 20, 10], "box_source": "observed"}]},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            self.assertEqual(_smooth_short_gaps(path, 4), 1)
            middle = json.loads(path.read_text(encoding="utf-8").splitlines()[1])["tracks"][0]
            self.assertEqual(middle["bbox"], [5.0, 0.0, 15.0, 10.0])
            self.assertEqual(middle["box_source"], "interpolated")

    def test_tracking_stage_writes_tracks_and_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "clip.avi"
            detections = root / "detections.jsonl"
            out = root / "track"
            _video(video)
            detections.write_text(
                "".join(
                    json.dumps(
                        {
                            "frame": frame,
                            "objects": [
                                {"bbox": [5 + frame, 5, 25 + frame, 35], "class_name": "person", "confidence": 0.9}
                            ],
                        }
                    )
                    + "\n"
                    for frame in range(4)
                ),
                encoding="utf-8",
            )
            track_video(
                video_path=video,
                detections_path=detections,
                out_dir=out,
                config={"min_hits": 1, "window_size": 3, "stride": 2, "min_new_track_conf": 0.1},
            )
            rows = (out / "tracks.jsonl").read_text(encoding="utf-8").splitlines()
            windows = json.loads((out / "windows.json").read_text(encoding="utf-8"))["windows"]
            self.assertEqual(len(rows), 4)
            self.assertEqual([(item["start_frame"], item["end_frame"]) for item in windows], [(0, 2), (2, 3)])
            self.assertIn("mode=official_ocsort", (out / "run.log").read_text(encoding="utf-8"))

    def test_semantic_dry_run_creates_storyboard(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "clip.avi"
            tracks = root / "tracks.jsonl"
            windows = root / "windows.json"
            output = root / "semantic" / "relations.json"
            storyboards = output.parent / "storyboards"
            _video(video, frames=3)
            tracks.write_text(
                "".join(
                    json.dumps(
                        {
                            "frame": frame,
                            "tracks": [
                                {"track_id": 1, "bbox": [2, 2, 22, 32], "class_name": "person"},
                                {"track_id": 2, "bbox": [35, 4, 55, 34], "class_name": "person"},
                            ],
                        }
                    )
                    + "\n"
                    for frame in range(3)
                ),
                encoding="utf-8",
            )
            windows.write_text(
                json.dumps(
                    {
                        "video": {"path": str(video)},
                        "windows": [{"window_id": 1, "start_frame": 0, "end_frame": 2, "track_ids": [1, 2]}],
                    }
                ),
                encoding="utf-8",
            )
            classify_relations(
                windows_path=windows,
                tracks_path=tracks,
                out_path=output,
                storyboards_dir=storyboards,
                config={"model": "mock", "max_frames_per_window": 3},
                api_key="",
                dry_run=True,
                video_id="clip",
            )
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"clip": []})
            self.assertEqual(len(list(storyboards.glob("*.jpg"))), 1)

    def test_vidvrd_evaluation_aligns_track_ids_by_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gold_rel, pred_rel = root / "gold_rel.json", root / "pred_rel.json"
            gold_traj, pred_traj = root / "gold_traj.json", root / "pred_traj.json"
            gold_rel.write_text(json.dumps({
                "clip": [{"subject_track_id": 1, "predicate": "next_to", "object_track_id": 2, "start_frame": 0, "end_frame": 2}],
                "not_processed": [{"subject_track_id": 1, "predicate": "next_to", "object_track_id": 2, "start_frame": 0, "end_frame": 2}],
            }), encoding="utf-8")
            pred_rel.write_text(json.dumps({"clip": [{"subject_track_id": 10, "predicate": "next_to", "object_track_id": 20, "start_frame": 0, "end_frame": 2, "confidence": 0.9}]}), encoding="utf-8")
            gold_traj.write_text(json.dumps({
                "clip": [
                    {"track_id": 1, "category": "person", "trajectory": {"0": [0, 0, 10, 10], "1": [1, 0, 11, 10], "2": [2, 0, 12, 10]}},
                    {"track_id": 2, "category": "dog", "trajectory": {"0": [20, 0, 30, 10], "1": [21, 0, 31, 10], "2": [22, 0, 32, 10]}},
                ],
                "not_processed": [
                    {"track_id": 1, "category": "person", "trajectory": {"0": [0, 0, 10, 10]}},
                    {"track_id": 2, "category": "dog", "trajectory": {"0": [20, 0, 30, 10]}},
                ],
            }), encoding="utf-8")
            pred_traj.write_text(json.dumps({"clip": [
                {"track_id": 10, "category": "person", "trajectory": {"0": [0, 0, 10, 10], "1": [1, 0, 11, 10], "2": [2, 0, 12, 10]}},
                {"track_id": 20, "category": "dog", "trajectory": {"0": [20, 0, 30, 10], "1": [21, 0, 31, 10], "2": [22, 0, 32, 10]}},
            ]}), encoding="utf-8")
            report, metrics_path = root / "report.md", root / "metrics.json"
            metrics = run_vidvrd_eval(
                gold_relations=gold_rel, gold_trajectories=gold_traj,
                pred_relations=pred_rel, pred_trajectories=pred_traj,
                report_path=report, metrics_path=metrics_path,
            )
            text = report.read_text(encoding="utf-8")
            self.assertIn("F1=1.0000", text)
            self.assertIn("## Per predicate", text)
            self.assertIn("| next_to |", text)
            self.assertEqual(metrics["tracks"]["matched"], 2)
            self.assertEqual(metrics["mean_ap"], 1.0)
            self.assertEqual(metrics["evaluated_videos"], ["clip"])


if __name__ == "__main__":
    unittest.main()
