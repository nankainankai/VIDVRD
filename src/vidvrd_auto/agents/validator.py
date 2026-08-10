from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

from vidvrd_auto.agents.actions import ACTION_TYPES, AgentAction
from vidvrd_auto.agents.evidence import EvidencePacket
from vidvrd_auto.core.ontology import predicate_names


def _action_name(raw: Dict[str, Any]) -> str:
    return str(raw.get("action", "")).strip().lower()


def validate_semantic_actions(
    actions: Iterable[Any], packet: EvidencePacket, *, request_budget_available: bool
) -> Dict[str, Any]:
    directions = {
        (int(item["subject_track_id"]), int(item["object_track_id"])): item
        for item in packet.candidate_directions
    }
    accepted: List[Dict[str, Any]] = []
    normalized: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    requested_frames: List[int] = []
    requested_expansions: List[Dict[str, Any]] = []
    for raw in actions:
        if not isinstance(raw, dict):
            rejected.append({"reason": "action_not_object"})
            continue
        name = _action_name(raw)
        reason = str(raw.get("reason", "")).strip()
        if name not in ACTION_TYPES:
            rejected.append({"action": name, "reason": "unsupported_action"})
            continue
        if not reason:
            rejected.append({"action": name, "reason": "missing_reason"})
            continue
        if name == "accept_relation":
            try:
                subject = int(raw["subject_track_id"])
                obj = int(raw["object_track_id"])
                predicate = str(raw["predicate"]).strip().lower()
                start = int(raw["start_frame"])
                end = int(raw["end_frame"])
                evidence = sorted({int(frame) for frame in raw["evidence_frames"]})
                score = float(raw["agent_score"])
            except (KeyError, TypeError, ValueError):
                rejected.append({"action": name, "reason": "invalid_relation_fields"})
                continue
            valid = (
                predicate in set(directions.get((subject, obj), {}).get("candidate_predicates", []))
                and packet.start_frame <= start <= end <= packet.end_frame
                and bool(evidence)
                and start <= evidence[0] <= evidence[-1] <= end
                and all(frame in packet.displayed_frames for frame in evidence)
                and 0.0 <= score <= 1.0
            )
            if not valid:
                rejected.append({"action": name, "reason": "relation_outside_evidence_packet"})
                continue
            action = AgentAction(
                action=name,
                reason=reason,
                subject_track_id=subject,
                predicate=predicate,
                object_track_id=obj,
                start_frame=start,
                end_frame=end,
                evidence_frames=evidence,
                agent_score=score,
            ).to_dict()
            accepted.append(action)
            normalized.append(action)
        elif name == "request_more_frames":
            try:
                frames = sorted({int(frame) for frame in raw.get("frame_ids", [])})
            except (TypeError, ValueError):
                rejected.append({"action": name, "reason": "invalid_frame_ids"})
                continue
            frames = [
                frame for frame in frames
                if frame in packet.available_frames and frame not in packet.displayed_frames
            ][: packet.max_additional_frames]
            if not request_budget_available or not frames:
                rejected.append({"action": name, "reason": "request_budget_or_frames_invalid"})
                continue
            requested_frames.extend(frames)
            normalized.append(AgentAction(action=name, reason=reason, frame_ids=frames).to_dict())
        elif name == "request_candidate_expansion":
            if requested_expansions:
                rejected.append({"action": name, "reason": "candidate_expansion_already_requested"})
                continue
            try:
                subject = int(raw["subject_track_id"])
                obj = int(raw["object_track_id"])
                families = list(dict.fromkeys(str(value) for value in raw.get("candidate_families", [])))
            except (KeyError, TypeError, ValueError):
                rejected.append({"action": name, "reason": "invalid_candidate_expansion"})
                continue
            available = set(directions.get((subject, obj), {}).get("expandable_families", []))
            families = [family for family in families if family in available]
            if not request_budget_available or not families:
                rejected.append({"action": name, "reason": "request_budget_or_families_invalid"})
                continue
            request = {
                "subject_track_id": subject,
                "object_track_id": obj,
                "candidate_families": families,
            }
            requested_expansions.append(request)
            normalized.append(
                AgentAction(
                    action=name,
                    reason=reason,
                    subject_track_id=subject,
                    object_track_id=obj,
                    candidate_families=families,
                ).to_dict()
            )
        elif name in {"reject_relation", "defer_for_review"}:
            normalized.append(AgentAction(action=name, reason=reason).to_dict())
        else:
            rejected.append({"action": name, "reason": "action_not_valid_during_proposal"})
    return {
        "accepted_relations": accepted,
        "actions": normalized,
        "requested_frames": sorted(set(requested_frames))[: packet.max_additional_frames],
        "requested_expansions": requested_expansions,
        "rejected_actions": rejected,
    }


def validate_review_actions(
    actions: Iterable[Any], relations: Sequence[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_id = {str(item.get("relation_id")): item for item in relations if item.get("relation_id")}
    predicates = set(predicate_names())
    valid: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for raw in actions:
        if not isinstance(raw, dict):
            rejected.append({"reason": "action_not_object"})
            continue
        name = _action_name(raw)
        relation_id = str(raw.get("relation_id", ""))
        relation = by_id.get(relation_id)
        if name not in ACTION_TYPES or relation is None:
            rejected.append({"action": name, "relation_id": relation_id, "reason": "unsupported_action_or_relation"})
            continue
        reason = str(raw.get("reason", "")).strip()
        if not reason:
            rejected.append({"action": name, "relation_id": relation_id, "reason": "missing_reason"})
            continue
        if name in {"accept_relation", "reject_relation", "defer_for_review"}:
            valid.append(AgentAction(action=name, relation_id=relation_id, reason=reason).to_dict())
        elif name == "change_predicate":
            predicate = str(raw.get("new_predicate", "")).strip().lower()
            if predicate not in predicates:
                rejected.append({"action": name, "relation_id": relation_id, "reason": "invalid_predicate"})
                continue
            valid.append(AgentAction(action=name, relation_id=relation_id, new_predicate=predicate, reason=reason).to_dict())
        elif name == "refine_interval":
            try:
                start = int(raw["start_frame"])
                end = int(raw["end_frame"])
                evidence = sorted({int(frame) for frame in raw["evidence_frames"]})
            except (KeyError, TypeError, ValueError):
                rejected.append({"action": name, "relation_id": relation_id, "reason": "invalid_interval"})
                continue
            original_start = int(relation["start_frame"])
            original_end = int(relation["end_frame"])
            relation_evidence = {int(frame) for frame in relation.get("evidence_frames", [])}
            if (
                not evidence
                or not set(evidence).issubset(relation_evidence)
                or not (original_start <= start <= evidence[0] <= evidence[-1] <= end <= original_end)
            ):
                rejected.append({"action": name, "relation_id": relation_id, "reason": "interval_outside_relation_evidence"})
                continue
            valid.append(
                AgentAction(
                    action=name,
                    relation_id=relation_id,
                    start_frame=start,
                    end_frame=end,
                    evidence_frames=evidence,
                    reason=reason,
                ).to_dict()
            )
        else:
            rejected.append({"action": name, "relation_id": relation_id, "reason": "supplemental_request_not_available_in_review"})
    return valid, rejected
