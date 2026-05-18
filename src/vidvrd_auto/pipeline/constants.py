from __future__ import annotations

NODE_ORDER = [
    "video_ingest",
    "audio_prior",
    "step1_detect",
    "keyframe_screen",
    "step2_track",
    "track_qc",
    "relation_rule",
    "relation_llm",
    "relation_merge",
    "global_relation",
    "relation_verify",
    "export",
]
