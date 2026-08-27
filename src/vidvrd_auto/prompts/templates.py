from __future__ import annotations

"""Prompts for review steps that receive real visual evidence."""

import json
from typing import Any, Dict, List

from vidvrd_auto.agents.evidence import EvidencePacket


def semantic_relation_prompt(packet: EvidencePacket, *, supplemental: bool = False) -> str:
    if supplemental:
        stage = (
            "This is the single supplemental response; no supplemental budget remains. "
            "Allowed actions are accept_relation, reject_relation, and defer_for_review. "
        )
        request_rules = ""
    else:
        stage = (
            "This is the initial response. Allowed actions are accept_relation, reject_relation, "
            "request_more_frames, request_candidate_expansion, and defer_for_review. "
        )
        request_rules = (
            "request_more_frames requires frame_ids chosen from available_frames and reason. "
            "request_candidate_expansion requires subject_track_id, object_track_id, candidate_families chosen only "
            "from expandable_families, and reason. There is one shared supplemental budget, so request frames and/or "
            "family expansion only when the initial evidence is insufficient. "
        )
    return (
        "You are a bounded video-relation proposal agent. Use only the supplied EvidencePacket and attached storyboard. "
        "You cannot change tracks, inspect unlisted frames, or invent predicates. "
        f"{stage}Return JSON only with an actions array. accept_relation requires subject_track_id, predicate, "
        "object_track_id, inclusive start_frame/end_frame, evidence_frames chosen only from displayed_frames, "
        f"agent_score, and reason. {request_rules}"
        "For bite, require visible mouth/head contact; ordinary proximity supports touch at most. For kick, require visible foot impact. "
        "For hold, require sustained control rather than brief contact; for feed, require visible transfer toward a mouth. "
        "If no relation is supported, reject_relation may record why, or return an empty actions array. agent_score is only a ranking score. "
        "Example: "
        '{"actions":[{"action":"accept_relation","subject_track_id":1,"predicate":"ride",'
        '"object_track_id":2,"start_frame":3,"end_frame":18,"evidence_frames":[3,11,18],'
        '"agent_score":0.8,"reason":"visible contact and aligned motion"}]}.\n'
        f"EvidencePacket: {json.dumps(packet.to_dict(), ensure_ascii=False)}"
    )


def semantic_relation_batch_prompt(
    packets: List[EvidencePacket], *, supplemental: bool = False
) -> str:
    if supplemental:
        stage = (
            "This is the supplemental response; no supplemental budget remains. "
            "Allowed actions are accept_relation, reject_relation, and defer_for_review. "
        )
        request_rules = ""
    else:
        stage = (
            "This is the initial response. Allowed actions are accept_relation, reject_relation, "
            "request_more_frames, request_candidate_expansion, and defer_for_review. "
        )
        request_rules = (
            "request_more_frames requires frame_ids chosen from that packet's available_frames and reason. "
            "request_candidate_expansion requires subject_track_id, object_track_id, candidate_families chosen "
            "only from that packet's expandable_families, and reason. Each packet has one supplemental budget. "
        )
    packet_payloads = [
        {
            "image_index": index,
            "packet_id": packet.packet_id,
            "evidence_packet": packet.to_dict(),
        }
        for index, packet in enumerate(packets, 1)
    ]
    return (
        "You are a bounded video-relation proposal agent. The attached storyboard images and EvidencePackets all "
        "belong to the same unordered track pair in consecutive time windows. Image order matches image_index. "
        "A and B are neutral display labels, not subject/object roles. Choose direction from the semantic action: "
        "the subject_track_id must be the actor named as the subject in the reason, and object_track_id must be "
        "the acted-on entity. For example, if A is horse ID1 and B is person ID2, 'person rides horse' is "
        "subject_track_id=2 and object_track_id=1. Judge every packet independently; do not move evidence or "
        "relations between packets. You cannot change "
        "tracks, inspect unlisted frames, or invent predicates. "
        f"{stage}Return JSON only as {{\"packet_results\":[{{\"packet_id\":\"...\",\"actions\":[]}}]}} "
        "and include exactly one packet_result for every supplied packet_id. Every action object must include its "
        "action field. accept_relation requires action=accept_relation, subject_track_id, predicate, "
        "object_track_id, inclusive start_frame/end_frame, evidence_frames chosen "
        f"only from that packet's displayed_frames, agent_score, and reason. {request_rules}"
        "For bite, require visible mouth/head contact; ordinary proximity supports touch at most. For kick, require "
        "visible foot impact. For hold, require sustained control rather than brief contact; for feed, require visible "
        "transfer toward a mouth. If no relation is supported, return an empty actions array for that packet. "
        "agent_score is only a ranking score. Example action: "
        '{"action":"accept_relation","subject_track_id":1,"predicate":"ride","object_track_id":2,'
        '"start_frame":3,"end_frame":18,"evidence_frames":[3,11,18],"agent_score":0.8,'
        '"reason":"visible contact and aligned motion"}.\n'
        f"Packet list: {json.dumps(packet_payloads, ensure_ascii=False)}"
    )


def relation_verify_prompt(relations: List[Dict[str, Any]], issues: List[Dict[str, Any]]) -> str:
    return (
        "You are the final reviewer for video relations. The attached pair storyboards are the visual evidence. "
        "Review only the listed risky relations, and reference each one by its stable relation_id. "
        "Do not invent relations. Return JSON only: "
        '{"actions":[{"action":"accept_relation|reject_relation|change_predicate|refine_interval|defer_for_review",'
        '"relation_id":"r000001","new_predicate":"optional","start_frame":0,"end_frame":29,'
        '"evidence_frames":[3,11],"reason":"visual reason"}]}. '
        "change_predicate must use an official predicate. refine_interval may only narrow the existing span and must cite evidence_frames.\n"
        f"Relations: {json.dumps(relations, ensure_ascii=False)}\n"
        f"Review triggers: {json.dumps(issues, ensure_ascii=False)}"
    )
