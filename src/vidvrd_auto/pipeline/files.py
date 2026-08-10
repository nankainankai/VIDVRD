from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vidvrd_auto.core import VideoPaths


@dataclass(frozen=True)
class Artifacts:
    source: Path
    vocabulary: Path
    detections: Path
    detect_meta: Path
    tracks: Path
    tracklets: Path
    stitch_links: Path
    windows: Path
    track_report: Path
    rules: Path
    semantics: Path
    semantic_evidence: Path
    merged: Path
    global_relations: Path
    verified: Path
    verify_qc: Path
    relations: Path
    trajectories: Path

    @classmethod
    def for_video(cls, paths: VideoPaths) -> Artifacts:
        def file(stage: str, name: str) -> Path:
            return paths.artifact(stage, name)

        return cls(
            file("inputs", "source.json"),
            file("vocabulary", "objects.json"),
            file("detect", "detections.jsonl"),
            file("detect", "meta.json"),
            file("track", "tracks.jsonl"),
            file("track", "tracklets.json"),
            file("track", "stitch_links.json"),
            file("track", "windows.json"),
            file("track_qc", "report.json"),
            file("rule", "relations.json"),
            file("semantic", "relations.json"),
            file("semantic", "evidence_packets.json"),
            file("merge", "relations.json"),
            file("global", "relations.json"),
            file("verify", "relations.json"),
            file("verify", "qc.json"),
            file("export", "relations.json"),
            file("export", "trajectories.json"),
        )
