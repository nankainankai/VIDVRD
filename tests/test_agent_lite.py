from __future__ import annotations

import unittest

from vidvrd_auto.agents import EvidencePacket, validate_review_actions, validate_semantic_actions
from vidvrd_auto.prompts.templates import semantic_relation_prompt
from vidvrd_auto.relations.ops import _apply_final_actions


def _packet() -> EvidencePacket:
    return EvidencePacket(
        packet_id="clip:w1:A1:B2",
        video_id="clip",
        window_id=1,
        start_frame=0,
        end_frame=10,
        fps=30.0,
        displayed_frames=[0, 5, 10],
        available_frames=list(range(11)),
        subject_track_id=1,
        subject_category="person",
        object_track_id=2,
        object_category="dog",
        candidate_directions=[
            {
                "subject_track_id": 1,
                "object_track_id": 2,
                "candidate_predicates": ["touch", "follow"],
                "expandable_families": ["manipulation"],
            },
            {
                "subject_track_id": 2,
                "object_track_id": 1,
                "candidate_predicates": ["follow"],
                "expandable_families": ["contact"],
            },
        ],
        track_evidence={"joint_visible_frames": 11, "joint_observed_frames": 3},
        trajectory_evidence={"1->2": {}, "2->1": {}},
        candidate_policy="hierarchical_predicate_v1",
        evidence_mode="event_burst_dual_view",
        max_additional_frames=2,
    )


class AgentLiteTests(unittest.TestCase):
    def test_semantic_accept_requires_packet_evidence(self) -> None:
        valid = {
            "action": "accept_relation",
            "subject_track_id": 1,
            "predicate": "touch",
            "object_track_id": 2,
            "start_frame": 0,
            "end_frame": 5,
            "evidence_frames": [0, 5],
            "agent_score": 0.8,
            "reason": "contact is visible",
        }
        result = validate_semantic_actions([valid], _packet(), request_budget_available=True)
        self.assertEqual(len(result["accepted_relations"]), 1)
        invalid = dict(valid, predicate="next_to", evidence_frames=[3])
        result = validate_semantic_actions([invalid], _packet(), request_budget_available=True)
        self.assertEqual(result["accepted_relations"], [])
        self.assertEqual(result["rejected_actions"][0]["reason"], "relation_outside_evidence_packet")

    def test_more_frames_has_one_bounded_budget(self) -> None:
        action = {"action": "request_more_frames", "frame_ids": [1, 2, 3], "reason": "transition is ambiguous"}
        first = validate_semantic_actions([action], _packet(), request_budget_available=True)
        second = validate_semantic_actions([action], _packet(), request_budget_available=False)
        self.assertEqual(first["requested_frames"], [1, 2])
        self.assertEqual(second["requested_frames"], [])

    def test_supplemental_prompt_does_not_offer_another_request(self) -> None:
        prompt = semantic_relation_prompt(_packet(), supplemental=True)
        instructions = prompt.split("EvidencePacket:", 1)[0]
        self.assertIn("no supplemental budget remains", instructions)
        self.assertNotIn("request_more_frames requires", instructions)

    def test_candidate_expansion_is_bounded_to_one_neighbor_request(self) -> None:
        actions = [
            {
                "action": "request_candidate_expansion",
                "subject_track_id": 1,
                "object_track_id": 2,
                "candidate_families": ["manipulation", "geometry"],
                "reason": "contact may involve control",
            },
            {
                "action": "request_candidate_expansion",
                "subject_track_id": 2,
                "object_track_id": 1,
                "candidate_families": ["contact"],
                "reason": "second request",
            },
            {
                "action": "request_more_frames",
                "frame_ids": [1, 2, 3],
                "reason": "inspect the same ambiguity temporally",
            },
        ]
        result = validate_semantic_actions(actions, _packet(), request_budget_available=True)
        self.assertEqual(
            result["requested_expansions"],
            [{"subject_track_id": 1, "object_track_id": 2, "candidate_families": ["manipulation"]}],
        )
        self.assertEqual(result["requested_frames"], [1, 2])
        self.assertEqual(result["rejected_actions"][0]["reason"], "candidate_expansion_already_requested")

    def test_review_actions_are_validated_before_application(self) -> None:
        relation = {
            "relation_id": "r1", "subject_track_id": 1, "predicate": "follow", "object_track_id": 2,
            "start_frame": 0, "end_frame": 10, "evidence_frames": [0, 5, 10],
        }
        actions = [
            {"action": "change_predicate", "relation_id": "r1", "new_predicate": "touch", "reason": "contact"},
            {"action": "refine_interval", "relation_id": "r1", "start_frame": 2, "end_frame": 8, "evidence_frames": [5], "reason": "only middle frames"},
            {"action": "refine_interval", "relation_id": "r1", "start_frame": -1, "end_frame": 11, "evidence_frames": [5], "reason": "invalid expansion"},
        ]
        valid, rejected = validate_review_actions(actions, [relation])
        self.assertEqual([action["action"] for action in valid], ["change_predicate", "refine_interval"])
        self.assertEqual(rejected[0]["reason"], "interval_outside_relation_evidence")
        output, applied = _apply_final_actions([relation], valid)
        self.assertEqual(output[0]["predicate"], "touch")
        self.assertEqual((output[0]["start_frame"], output[0]["end_frame"]), (2, 8))
        self.assertIn("before", applied[0])
        self.assertIn("after", applied[0])


if __name__ == "__main__":
    unittest.main()
