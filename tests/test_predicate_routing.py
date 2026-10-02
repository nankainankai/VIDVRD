from __future__ import annotations

import unittest

from vidvrd_auto.core.ontology import predicate_names
from vidvrd_auto.relations.candidate_router import expand_route, route_predicates
from vidvrd_auto.relations.evidence_features import trajectory_evidence
from vidvrd_auto.relations.object_candidates import GEOMETRY_PREDICATES
from vidvrd_auto.relations.predicate_hierarchy import confusion_siblings, predicate_records
from vidvrd_auto.relations.semantic import _event_sample


def _tracks() -> dict[int, list[dict[str, object]]]:
    return {
        frame: [
            {
                "track_id": 1,
                "bbox": [frame * 8, 0, frame * 8 + 12, 20],
                "box_source": "observed",
                "identity_support": 0.9,
            },
            {
                "track_id": 2,
                "bbox": [48, 0, 60, 20],
                "box_source": "observed",
                "identity_support": 0.8,
            },
        ]
        for frame in range(6)
    }


class PredicateRoutingTests(unittest.TestCase):
    def test_following_camera_preserves_all_supported_axis_locomotion_hypotheses(self) -> None:
        evidence = trajectory_evidence(_tracks(), list(range(6)), 1, 2)
        evidence.update(subject_speed=0.0, approach_rate=0.0, subject_vertical_speed=0.0)
        actions = {"creep", "fly", "jump", "move", "run", "swim", "walk"}
        for axis, dx, dy in [("left", -.5, 0), ("right", .5, 0), ("above", 0, -.5), ("beneath", 0, .5)]:
            with self.subTest(axis=axis):
                evidence.update(median_dx=dx, median_dy=dy)
                route = route_predicates(evidence, limit=24, exclude=GEOMETRY_PREDICATES, axis_motion_reserve=True)
                required = {r["name"] for r in predicate_records()
                            if r["components"].get("action") in actions and r["components"].get("spatial") == axis}
                self.assertTrue(required <= set(route["candidate_predicates"]))
                self.assertLessEqual(len(route["candidate_predicates"]), 24)
                self.assertEqual(route["candidate_policy"], "axis_motion_reserve_v4")
                self.assertFalse(set(route["candidate_predicates"]) & GEOMETRY_PREDICATES)

    def test_hierarchy_covers_official_ontology_without_relabeling(self) -> None:
        records = predicate_records()
        self.assertEqual({record["name"] for record in records}, set(predicate_names()))
        self.assertTrue(all(record["families"] for record in records))
        creep_toward = next(record for record in records if record["name"] == "creep_toward")
        self.assertIn("locomotion_state", creep_toward["families"])
        self.assertIn("relative_motion", creep_toward["families"])
        self.assertIn("move_toward", confusion_siblings("creep_toward"))

    def test_directional_trajectory_evidence_and_event_burst(self) -> None:
        tracks = _tracks()
        forward = trajectory_evidence(tracks, list(range(6)), 1, 2)
        reverse = trajectory_evidence(tracks, list(range(6)), 2, 1)
        self.assertLess(forward["median_dx"], 0)
        self.assertGreater(reverse["median_dx"], 0)
        self.assertGreater(forward["approach_rate"], 0)
        self.assertIn(5, forward["event_frames"])
        sampled = _event_sample(list(range(10)), [{"event_frames": [5]}], 5, 5)
        self.assertEqual(len(sampled), 5)
        self.assertTrue({0, 4, 9}.issubset(sampled))

    def test_event_sample_preserves_boundaries_for_interior_events(self) -> None:
        sampled = _event_sample(list(range(30)), [{"event_frames": [20, 25]}], 8, 5)
        self.assertEqual(len(sampled), 8)
        self.assertTrue({0, 14, 29}.issubset(sampled))
        self.assertEqual(_event_sample([], [], 8, 5), [])

    def test_route_is_small_contrastive_and_expands_only_to_neighbors(self) -> None:
        evidence = trajectory_evidence(_tracks(), list(range(6)), 1, 2)
        route = route_predicates(evidence, limit=14, exclude=GEOMETRY_PREDICATES)
        self.assertLessEqual(len(route["candidate_predicates"]), 14)
        self.assertFalse(set(route["candidate_predicates"]) & GEOMETRY_PREDICATES)
        self.assertIn("toward", route["candidate_predicates"])
        requested_family = route["expandable_families"][0]
        expanded = expand_route(route, [requested_family, "not_a_family"], limit=24)
        self.assertEqual(expanded["expanded_families"], [requested_family])
        self.assertGreaterEqual(len(expanded["candidate_predicates"]), len(route["candidate_predicates"]))
        self.assertLessEqual(len(expanded["candidate_predicates"]), 24)
        self.assertIn(requested_family, expanded["family_descriptions"])

    def test_route_reserves_composite_predicate_quota(self) -> None:
        evidence = trajectory_evidence(_tracks(), list(range(6)), 1, 2)
        evidence.update(
            subject_speed=0.0, median_dx=0.0, median_dy=-0.5,
            approach_rate=0.0, distance_min=1.0,
        )
        route = route_predicates(evidence, limit=14, exclude=GEOMETRY_PREDICATES)
        self.assertEqual(route["candidate_policy"], "quota_composite_predicate_v3_train_prior")
        self.assertIn("sit_above", route["candidate_predicates"])
        self.assertIn("sit_above", route["candidate_buckets"]["composite"])

        evidence.update(
            subject_speed=0.3, median_dy=0.5,
            subject_vertical_speed=0.2, subject_vertical_direction_changes=1,
        )
        moving_route = route_predicates(evidence, limit=14, exclude=GEOMETRY_PREDICATES)
        self.assertIn("jump_beneath", moving_route["candidate_predicates"])


if __name__ == "__main__":
    unittest.main()
