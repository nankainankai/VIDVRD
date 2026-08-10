from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


ACTION_TYPES = frozenset({
    "accept_relation",
    "reject_relation",
    "change_predicate",
    "refine_interval",
    "request_more_frames",
    "request_candidate_expansion",
    "defer_for_review",
})


@dataclass(frozen=True)
class AgentAction:
    action: str
    reason: str
    relation_id: str = ""
    subject_track_id: int | None = None
    predicate: str = ""
    object_track_id: int | None = None
    new_predicate: str = ""
    start_frame: int | None = None
    end_frame: int | None = None
    evidence_frames: List[int] = field(default_factory=list)
    frame_ids: List[int] = field(default_factory=list)
    candidate_families: List[str] = field(default_factory=list)
    agent_score: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value not in ("", None, [])}
