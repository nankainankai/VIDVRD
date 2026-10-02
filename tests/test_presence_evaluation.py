import unittest

from vidvrd_auto.evaluation.presence import evaluate_presence


class PresenceEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tracks = {"clip": [
            {"track_id": 1, "category": "person", "trajectory": {str(f): [0, 0, 10, 10] for f in range(10)}},
            {"track_id": 2, "category": "person", "trajectory": {str(f): [20, 0, 30, 10] for f in range(10)}},
        ]}
        self.rel = {"subject_track_id": 1, "object_track_id": 2, "predicate": "watch", "start_frame": 0, "end_frame": 9}
        self.clips = [{"video_id": "clip", "start_frame": 0, "end_frame": 10, "annotation_complete": True}]

    def evaluate(self, predictions, **kwargs):
        return evaluate_presence({"clip": [self.rel]}, self.tracks, {"clip": predictions},
                                 kwargs.pop("pred_tracks", self.tracks), ["clip"], **kwargs)

    def test_presence_ignores_boundary_accuracy_and_deduplicates(self):
        short = dict(self.rel, start_frame=4, end_frame=4)
        result = self.evaluate([short, short], clips=self.clips)
        self.assertEqual(result["micro"]["tp"], 1)
        self.assertEqual(result["micro"]["f1"], 1)
        self.assertTrue(result["annotation_complete"])

    def test_same_category_reversed_identity_is_not_a_match(self):
        result = self.evaluate([dict(self.rel, subject_track_id=2, object_track_id=1)])
        self.assertEqual((result["micro"]["tp"], result["micro"]["fp"], result["micro"]["fn"]), (0, 1, 1))
        self.assertFalse(result["annotation_complete"])

    def test_outside_clip_and_unmatched_object_are_not_true_positives(self):
        result = self.evaluate([dict(self.rel, start_frame=10, end_frame=12),
                                dict(self.rel, object_track_id=99)], clips=self.clips)
        self.assertEqual((result["micro"]["tp"], result["micro"]["fp"], result["micro"]["fn"]), (0, 1, 1))

    def test_fixed_manifest_rejects_duplicate_or_invalid_clips(self):
        with self.assertRaisesRegex(ValueError, "duplicate clip"):
            self.evaluate([], clips=self.clips * 2)
        with self.assertRaisesRegex(ValueError, "invalid half-open"):
            self.evaluate([], clips=[dict(self.clips[0], end_frame=0)])

    def test_missing_video_counts_false_negatives(self):
        result = self.evaluate([], pred_tracks={})
        self.assertEqual(result["micro"]["fn"], 1)


if __name__ == "__main__":
    unittest.main()
