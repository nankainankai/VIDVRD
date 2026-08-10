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

    rule_cfg = config.section("relation_rule").to_dict()
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
    semantic_cfg["prompt_version"] = config.section("project").get(
        "prompt_version", "main-v4-hierarchical-agent"
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
        "prompt_version", "main-v4-hierarchical-agent"
    )
    if files.track_report.exists():
        verify_cfg["risk_track_ids"] = read_json(files.track_report).get("risk_track_ids", [])
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

    stages.run(
        "export",
        {"relations": sha256_file(files.verified), "tracks": sha256_file(files.tracks)},
        [files.relations, files.trajectories],
        lambda: export_video_outputs(
            verified_path=files.verified,
            qc_path=files.verify_qc,
            tracks_path=files.tracks,
            video_id=video_id,
            relations_path=files.relations,
            trajectories_path=files.trajectories,
        ),
    )
