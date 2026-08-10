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
        self.assertEqual(_event_sample(list(range(10)), [{"event_frames": [5]}], 5, 5), [3, 4, 5, 6, 7])

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


if __name__ == "__main__":
    unittest.main()
