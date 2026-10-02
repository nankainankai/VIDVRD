import unittest

from vidvrd_auto.evaluation.tagging import evaluate_known_label_coverage


def relation(triplet, score=1):
    return {"triplet": list(triplet), "score": score}


class KnownLabelCoverageTests(unittest.TestCase):
    def test_primary_budget_remains_fixed(self):
        with self.assertRaises(ValueError):
            evaluate_known_label_coverage({}, {}, [], budgets=(20,))

    def test_duplicate_instances_and_windows_do_not_fill_label_budget(self):
        label = ("person", "ride", "horse")
        second = ("person", "watch", "horse")
        metrics = evaluate_known_label_coverage(
            {"a": [relation(label), relation(label), relation(second)]},
            {"a": [relation(label, .9), relation(label, .8), relation(second, .7)]}, ["a"],
        )
        self.assertEqual(metrics["known_label_count"], 2)
        self.assertEqual(metrics["micro_recall_at"]["1"], .5)
        self.assertEqual(metrics["micro_recall_at"]["5"], 1)
        self.assertEqual(metrics["per_video"][0]["prediction_label_count"], 2)

    def test_failed_videos_stay_in_denominator_and_direction_matters(self):
        label = ("person", "ride", "horse")
        reversed_label = ("horse", "ride", "person")
        metrics = evaluate_known_label_coverage(
            {"a": [relation(label)], "failed": [relation(label)]},
            {"a": [relation(reversed_label, .9), relation(label, .8)]}, ["a", "failed"],
        )
        self.assertEqual(metrics["micro_recall_at"]["1"], 0)
        self.assertEqual(metrics["micro_recall_at"]["10"], .5)
        self.assertEqual(metrics["evaluated_video_count"], 2)
        self.assertFalse(metrics["annotation_complete"])

    def test_fixed_budget_cannot_receive_credit_for_late_labels(self):
        label = ("person", "ride", "horse")
        metrics = evaluate_known_label_coverage(
            {"a": [relation(label)]},
            {"a": [relation(("person", f"wrong{i}", "horse"), 1 - i / 100) for i in range(10)] + [relation(label, .1)]}, ["a"],
        )
        self.assertEqual(metrics["micro_recall_at"]["10"], 0)
        self.assertEqual(metrics["micro_recall_at"]["20"], 1)
