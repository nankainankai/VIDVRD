import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vidvrd_auto.evaluation.runtime import summarize_run
from vidvrd_auto.utils.io import write_json


class RuntimeEvaluationTests(unittest.TestCase):
    def test_successful_empty_video_is_not_counted_as_nonempty_annotation_output(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "run_manifest.json", {"videos": [
                {"video_id": "has_output", "state": "succeeded", "executed_nodes": []},
                {"video_id": "no_tracks", "state": "succeeded", "executed_nodes": []},
            ]})
            write_json(root / "pred/relations.json", {"has_output": [{"predicate": "ride"}], "no_tracks": []})
            write_json(root / "pred/trajectories.json", {"has_output": [{"track_id": 1}], "no_tracks": []})
            result = summarize_run(root)
            self.assertEqual(result["completion_rate"], 1)
            self.assertEqual(result["output_availability"]["completed_nonempty_prediction_rate"], .5)
            self.assertEqual(result["output_availability"]["empty_completed_video_ids"], ["no_tracks"])

    def test_failure_counts_in_completion_and_resume_does_not_charge_stored_calls(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "run_manifest.json", {
                "started_at": "2026-10-02 12:00:00", "finished_at": "2026-10-02 12:00:10",
                "videos": [{"video_id": "cached", "state": "succeeded", "executed_nodes": []},
                           {"video_id": "failed", "state": "failed", "executed_nodes": ["semantic"], "error": "quota"}],
            })
            write_json(root / "videos/cached/semantic/run.log", {"provider": {"calls": 5, "attempts": 5}})
            write_json(root / "videos/failed/semantic/run.log", {"provider": {"calls": 1, "attempts": 1, "failed": 1}})
            result = summarize_run(root)
            self.assertEqual(result["completion_rate"], 0.5)
            self.assertEqual(result["recorded_api"]["calls"], 6)
            self.assertEqual(result["invocation_api"]["calls"], 1)
            self.assertEqual(result["invocation_api"]["failed"], 1)
            self.assertFalse(result["recorded_api"]["usage_complete"])
            self.assertIsNone(result["actual_fee"])
            self.assertIsNone(result["recorded_api"]["elapsed_sec"])
            self.assertFalse(result["output_availability"]["available"])
            self.assertIsNone(result["output_availability"]["completed_nonempty_prediction_rate"])
