from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class EvidencePacket:
    packet_id: str
    video_id: str
    window_id: int
    start_frame: int
    end_frame: int
    fps: float
    displayed_frames: List[int]
    available_frames: List[int]
    subject_track_id: int
    subject_category: str
    object_track_id: int
    object_category: str
    candidate_directions: List[Dict[str, Any]]
    track_evidence: Dict[str, Any]
    trajectory_evidence: Dict[str, Any]
    candidate_policy: str
    evidence_mode: str
    max_additional_frames: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
