from __future__ import annotations

from vidvrd_auto.core import AppConfig, VideoPaths
from vidvrd_auto.nodes.export import export_video_outputs
from vidvrd_auto.nodes.global_relation import run_global_relation
from vidvrd_auto.nodes.semantic import classify_relations
from vidvrd_auto.pipeline.files import Artifacts
from vidvrd_auto.pipeline.stage import StageRunner
from vidvrd_auto.relations import generate_rule_relations, merge_relations, verify_relations
from vidvrd_auto.utils.hashing import sha256_file
from vidvrd_auto.utils.io import read_json
from pathlib import Path


def run_relations(
    *,
    video_id: str,
    config: AppConfig,
    api_key: str,
    paths: VideoPaths,
    files: Artifacts,
    stages: StageRunner,
) -> None:
    """Run rule, semantic, merge, global, verify, and export stages."""

    track_qc = read_json(files.track_report) if files.track_report.exists() else {}
    blocked_track_ids = list(track_qc.get("blocked_track_ids", []))
    track_quality = dict(track_qc.get("track_quality", {}))
    track_jump_frames = {
        str(track_id): list(values.get("jump_frames", []))
        for track_id, values in track_quality.items() if isinstance(values, dict)
    }

    rule_cfg = config.section("relation_rule").to_dict()
    rule_cfg["blocked_track_ids"] = blocked_track_ids
    rule_cfg["track_jump_frames"] = track_jump_frames
    stages.run(
        "rule",
        {"tracks": sha256_file(files.tracks), "windows": sha256_file(files.windows), "config": rule_cfg},
        [files.rules],
        lambda: generate_rule_relations(
            windows_json=files.windows,
            tracks_jsonl=files.tracks,
            out_json=files.rules,
            video_id=video_id,
            config=rule_cfg,
        ),
    )

    semantic_cfg = config.section("relations").to_dict()
    semantic_cfg["blocked_track_ids"] = blocked_track_ids
    semantic_cfg["track_jump_frames"] = track_jump_frames
    prior_path = str(semantic_cfg.get("candidate_prior_path", "") or "").strip()
    if prior_path:
        prior_file = Path(prior_path)
        if not prior_file.is_absolute():
            prior_file = paths.repo_dir / prior_file
        semantic_cfg["candidate_prior_path"] = str(prior_file)
        semantic_cfg["candidate_prior_sha256"] = sha256_file(prior_file)
    semantic_cfg["prompt_version"] = config.section("project").get(
        "prompt_version", "main-v8-calibrated-composite-routing"
    )
    dry_run = bool(stages.args.dry_run_relations) or bool(semantic_cfg.get("dry_run", False))
    stages.run(
        "semantic",
        {
            "tracks": sha256_file(files.tracks),
            "windows": sha256_file(files.windows),
            "config": semantic_cfg,
            "dry_run": dry_run,
        },
        [files.semantics, files.semantic_evidence],
        lambda: classify_relations(
            windows_path=files.windows,
            tracks_path=files.tracks,
            out_path=files.semantics,
            storyboards_dir=paths.semantic_dir / "storyboards",
            config=semantic_cfg,
            api_key=api_key,
            dry_run=dry_run,
            video_id=video_id,
        ),
    )

    merge_cfg = config.section("relation_merge").to_dict()
    stages.run(
        "merge",
        {"rule": sha256_file(files.rules), "semantic": sha256_file(files.semantics), "config": merge_cfg},
        [files.merged],
        lambda: merge_relations(
            video_id=video_id,
            relation_jsons=[files.rules, files.semantics],
            out_json=files.merged,
            apply_coupling=bool(merge_cfg.get("apply_coupling", False)),
        ),
    )

    global_cfg = config.section("global_relation").to_dict()
    scoring_cfg = config.section("relation_scoring").to_dict()
    global_cfg["track_quality"] = track_quality
    global_cfg["track_jump_frames"] = track_jump_frames
    global_cfg["source_coefficients"] = scoring_cfg.get("source_coefficients", {})
    global_cfg["score_version"] = scoring_cfg.get("version", "explicit_risk_flag_v2")
    stages.run(
        "global",
        {"relations": sha256_file(files.merged), "config": global_cfg},
        [files.global_relations],
        lambda: run_global_relation(
            video_id=video_id,
            relations_json=files.merged,
            out_json=files.global_relations,
            config=global_cfg,
        ),
    )

    verify_cfg = config.section("relation_verify").to_dict()
    verify_cfg["prompt_version"] = config.section("project").get(
        "prompt_version", "main-v8-calibrated-composite-routing"
    )
    if files.track_report.exists():
        verify_cfg["risk_track_ids"] = track_qc.get("risk_track_ids", [])
    stages.run(
        "verify",
        {
            "relations": sha256_file(files.global_relations),
            "tracks": sha256_file(files.tracks),
            "config": verify_cfg,
        },
        [files.verified, files.verify_qc],
        lambda: verify_relations(
            video_id=video_id,
            relations_json=files.global_relations,
            tracks_jsonl=files.tracks,
            out_relations_json=files.verified,
            out_qc_json=files.verify_qc,
            config=verify_cfg,
            storyboards_dir=paths.semantic_dir / "storyboards",
            api_key=api_key,
        ),
    )

    export_cfg = config.section("relation_export").to_dict()
    stages.run(
        "export",
        {"relations": sha256_file(files.verified), "tracks": sha256_file(files.tracks), "config": export_cfg},
        [files.relations, files.trajectories],
        lambda: export_video_outputs(
            verified_path=files.verified,
            qc_path=files.verify_qc,
            tracks_path=files.tracks,
            video_id=video_id,
            relations_path=files.relations,
            trajectories_path=files.trajectories,
            config=export_cfg,
        ),
    )
