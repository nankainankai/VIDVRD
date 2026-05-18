from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from vidvrd_auto.config.loader import load_config
from vidvrd_auto.evaluation.presence import run_presence_eval
from vidvrd_auto.nodes.audio_prior import build_audio_prior
from vidvrd_auto.nodes.detect import run_detect
from vidvrd_auto.nodes.export import export_video_outputs, merge_relation_files
from vidvrd_auto.nodes.global_relation import run_global_relation
from vidvrd_auto.nodes.ingest import (
    is_url,
    materialize_video,
    planned_video_path,
    video_id_for_source,
)
from vidvrd_auto.nodes.relation_llm import run_relation_llm
from vidvrd_auto.nodes.screen import screen_keyframes
from vidvrd_auto.nodes.track import run_track
from vidvrd_auto.nodes.track_qc import run_track_qc
from vidvrd_auto.pipeline.constants import NODE_ORDER
from vidvrd_auto.pipeline.manifest import (
    collect_node_statuses,
    mark_failed,
    mark_running,
    mark_succeeded,
    now_text,
    should_skip,
)
from vidvrd_auto.relations.merge import merge_relations
from vidvrd_auto.relations.rules import generate_rule_relations
from vidvrd_auto.relations.verify import verify_relations
from vidvrd_auto.utils.hashing import sha256_file, stable_hash
from vidvrd_auto.utils.io import read_json, write_json
from vidvrd_auto.utils.paths import repo_root, safe_rel


def parse_video_sources(args: Namespace) -> List[str]:
    items: List[str] = []
    if str(args.video or "").strip():
        items.append(str(args.video).strip())
    if str(args.videos or "").strip():
        videos_arg = Path(str(args.videos)).expanduser().resolve()
        if videos_arg.exists() and videos_arg.is_file():
            for line in videos_arg.read_text(encoding="utf-8-sig").splitlines():
                s = line.strip()
                if s and not s.startswith("#"):
                    items.append(s)
        else:
            for s in str(args.videos).split(","):
                s = s.strip()
                if s:
                    items.append(s)

    out: List[str] = []
    seen: set[str] = set()
    for raw in items:
        key = raw.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(raw)
    if not out:
        raise SystemExit("ERROR: pass --video <path|url> or --videos <list.txt|comma-separated>")
    return out


def node_enabled(node: str, from_node: str, to_node: str) -> bool:
    start = NODE_ORDER.index(from_node) if from_node else 0
    end = NODE_ORDER.index(to_node) if to_node else len(NODE_ORDER) - 1
    idx = NODE_ORDER.index(node)
    return start <= idx <= end


def run_node(
    *,
    args: Namespace,
    video_dir: Path,
    node: str,
    input_hash: str,
    required_outputs: Sequence[Path],
    outputs: Dict[str, str],
    fn,
) -> None:
    if should_skip(
        resume=bool(args.resume),
        force=bool(args.force),
        video_dir=video_dir,
        node=node,
        input_hash=input_hash,
        required_outputs=required_outputs,
    ):
        print(f"SKIP {video_dir.name}/{node} (resume cache hit)")
        return
    print(f"RUN {video_dir.name}/{node}")
    mark_running(video_dir, node, input_hash)
    try:
        fn()
    except Exception as e:
        mark_failed(video_dir, node, input_hash, str(e))
        raise
    for p in required_outputs:
        if not p.exists():
            err = f"required output missing: {p}"
            mark_failed(video_dir, node, input_hash, err)
            raise RuntimeError(err)
    mark_succeeded(video_dir, node, input_hash, outputs)


def run_pipeline(*, args: Namespace, config_path: Path | None) -> None:
    root = repo_root()
    cfg = load_config(config_path)
    video_sources = parse_video_sources(args)
    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    api_key = str(args.api_key or "").strip() or (os.getenv("DASHSCOPE_API_KEY", "") or "").strip()
    used_video_ids: set[str] = set()
    video_items: List[Dict[str, Any]] = []
    relation_exports: List[Tuple[str, Path]] = []
    config_path_for_manifest = config_path or (root / "configs" / "default.json")

    manifest: Dict[str, Any] = {
        "run_dir": safe_rel(run_dir),
        "started_at": now_text(),
        "config_path": safe_rel(config_path_for_manifest),
        "config_hash": stable_hash(cfg),
        "nodes": NODE_ORDER,
        "args": {
            "video": str(getattr(args, "video", "") or ""),
            "videos": str(getattr(args, "videos", "") or ""),
            "from_node": str(getattr(args, "from_node", "") or ""),
            "to_node": str(getattr(args, "to_node", "") or ""),
            "resume": bool(getattr(args, "resume", False)),
            "force": bool(getattr(args, "force", False)),
            "dry_run_relations": bool(getattr(args, "dry_run_relations", False)),
            "skip_eval": bool(getattr(args, "skip_eval", False)),
            "api_key_present": bool(api_key),
        },
        "videos": [],
    }
    write_json(run_dir / "run_manifest.json", manifest)

    export_enabled = node_enabled("export", args.from_node, args.to_node)
    for source in video_sources:
        vid = video_id_for_source(source, used_video_ids)
        video_dir = run_dir / "videos" / vid
        video_dir.mkdir(parents=True, exist_ok=True)
        inputs_dir = video_dir / "inputs"
        input_meta_json = inputs_dir / "source.json"
        video_path = planned_video_path(source, inputs_dir)

        item: Dict[str, Any] = {
            "video_id": vid,
            "source": source,
            "source_type": "url" if is_url(source) else "local",
            "nodes": {},
        }
        video_items.append(item)

        audio_dir = video_dir / "audio_prior"
        step1_dir = video_dir / "step1_detect"
        screen_dir = video_dir / "keyframe_screen"
        step2_dir = video_dir / "step2_track"
        track_qc_dir = video_dir / "track_qc"
        rule_dir = video_dir / "relation_rule"
        rel_dir = video_dir / "relation_llm"
        merge_dir = video_dir / "relation_merge"
        global_dir = video_dir / "global_relation"
        verify_dir = video_dir / "relation_verify"
        export_dir = video_dir / "export"

        audio_prior_json = audio_dir / "audio_prior.json"
        detections_jsonl = step1_dir / "detections_full.jsonl"
        video_meta_json = step1_dir / "video_meta.json"
        screen_json = screen_dir / "screen_result.json"
        tracks_jsonl = step2_dir / "tracks_full.jsonl"
        windows_json = step2_dir / "windows.json"
        track_qc_json = track_qc_dir / "track_qc.json"
        relations_rule_json = rule_dir / "relations_rule.json"
        relations_llm_json = rel_dir / "relations_llm.json"
        relations_merged_json = merge_dir / "relations_merged.json"
        relations_global_json = global_dir / "relations_global.json"
        relations_verified_json = verify_dir / "relations_verified.json"
        relation_qc_json = verify_dir / "relation_qc.json"
        relations_pred_json = export_dir / "relations_pred.json"
        trajectories_pred_json = export_dir / "trajectories_pred.json"

        try:
            ingest_cfg = cfg.get("video_ingest", {}) if isinstance(cfg.get("video_ingest"), dict) else {}
            ingest_hash = stable_hash({"node": "video_ingest", "source": source, "config": ingest_cfg})

            if node_enabled("video_ingest", args.from_node, args.to_node):
                run_node(
                    args=args,
                    video_dir=video_dir,
                    node="video_ingest",
                    input_hash=ingest_hash,
                    required_outputs=[input_meta_json, video_path],
                    outputs={"source_json": safe_rel(input_meta_json), "video_path": safe_rel(video_path)},
                    fn=lambda: materialize_video(source, inputs_dir, ingest_cfg),
                )
            if not input_meta_json.exists():
                materialize_video(source, inputs_dir, ingest_cfg)
            input_meta = read_json(input_meta_json)
            item["path"] = str(video_path)
            item["path_rel"] = safe_rel(video_path)
            item["exists"] = video_path.exists()
            item["file_hash"] = str(input_meta.get("file_hash", "") or sha256_file(video_path))

            audio_cfg = cfg.get("audio_prior", {}) if isinstance(cfg.get("audio_prior"), dict) else {}
            rel_base_cfg = cfg.get("relations", {}) if isinstance(cfg.get("relations"), dict) else {}
            audio_cfg = dict(audio_cfg)
            audio_cfg.setdefault("fallback_label", str(rel_base_cfg.get("vggsound_label", "") or ""))
            audio_hash = stable_hash({"node": "audio_prior", "source": source, "video_id": vid, "config": audio_cfg})
            if node_enabled("audio_prior", args.from_node, args.to_node):
                run_node(
                    args=args,
                    video_dir=video_dir,
                    node="audio_prior",
                    input_hash=audio_hash,
                    required_outputs=[audio_prior_json],
                    outputs={"audio_prior_json": safe_rel(audio_prior_json)},
                    fn=lambda: build_audio_prior(video_id=vid, source=source, out_json=audio_prior_json, config=audio_cfg),
                )

            detector_cfg = cfg.get("detector", {}) if isinstance(cfg.get("detector"), dict) else {}
            step1_hash = stable_hash(
                {"node": "step1_detect", "video": str(video_path), "video_hash": item["file_hash"], "config": detector_cfg}
            )
            if node_enabled("step1_detect", args.from_node, args.to_node):
                run_node(
                    args=args,
                    video_dir=video_dir,
                    node="step1_detect",
                    input_hash=step1_hash,
                    required_outputs=[detections_jsonl, video_meta_json],
                    outputs={"detections_jsonl": safe_rel(detections_jsonl), "video_meta_json": safe_rel(video_meta_json)},
                    fn=lambda: run_detect(video_path=video_path, out_dir=step1_dir, config=detector_cfg, log_path=step1_dir / "run.log"),
                )

            screen_cfg = cfg.get("keyframe_screen", {}) if isinstance(cfg.get("keyframe_screen"), dict) else {}
            screen_hash = stable_hash(
                {"node": "keyframe_screen", "detections_hash": sha256_file(detections_jsonl), "config": screen_cfg}
            )
            if node_enabled("keyframe_screen", args.from_node, args.to_node):
                run_node(
                    args=args,
                    video_dir=video_dir,
                    node="keyframe_screen",
                    input_hash=screen_hash,
                    required_outputs=[screen_json],
                    outputs={"screen_result_json": safe_rel(screen_json)},
                    fn=lambda: screen_keyframes(detections_jsonl=detections_jsonl, out_json=screen_json, config=screen_cfg),
                )
            if screen_json.exists():
                screen_result = read_json(screen_json)
                if isinstance(screen_result, dict) and not bool(screen_result.get("passed", True)):
                    item["state"] = "skipped"
                    item["skip_reason"] = str(screen_result.get("reason", "keyframe_screen_failed"))
                    print(f"SKIP {vid}: {item['skip_reason']}")
                    continue

            tracking_cfg = cfg.get("tracking", {}) if isinstance(cfg.get("tracking"), dict) else {}
            step2_hash = stable_hash(
                {
                    "node": "step2_track",
                    "video": str(video_path),
                    "detections_hash": sha256_file(detections_jsonl),
                    "config": tracking_cfg,
                    "api_key_present": bool(api_key),
                }
            )
            if node_enabled("step2_track", args.from_node, args.to_node):
                run_node(
                    args=args,
                    video_dir=video_dir,
                    node="step2_track",
                    input_hash=step2_hash,
                    required_outputs=[tracks_jsonl, windows_json],
                    outputs={"tracks_jsonl": safe_rel(tracks_jsonl), "windows_json": safe_rel(windows_json)},
                    fn=lambda: run_track(
                        video_path=video_path,
                        detections_jsonl=detections_jsonl,
                        out_dir=step2_dir,
                        config=tracking_cfg,
                        api_key=api_key,
                        log_path=step2_dir / "run.log",
                    ),
                )

            track_qc_cfg = cfg.get("track_qc", {}) if isinstance(cfg.get("track_qc"), dict) else {}
            track_qc_hash = stable_hash(
                {"node": "track_qc", "tracks_hash": sha256_file(tracks_jsonl), "windows_hash": sha256_file(windows_json), "config": track_qc_cfg}
            )
            if node_enabled("track_qc", args.from_node, args.to_node):
                run_node(
                    args=args,
                    video_dir=video_dir,
                    node="track_qc",
                    input_hash=track_qc_hash,
                    required_outputs=[track_qc_json],
                    outputs={"track_qc_json": safe_rel(track_qc_json)},
                    fn=lambda: run_track_qc(tracks_jsonl=tracks_jsonl, windows_json=windows_json, out_json=track_qc_json, config=track_qc_cfg),
                )

            rule_cfg = cfg.get("relation_rule", {}) if isinstance(cfg.get("relation_rule"), dict) else {}
            rule_hash = stable_hash(
                {"node": "relation_rule", "windows_hash": sha256_file(windows_json), "tracks_hash": sha256_file(tracks_jsonl), "config": rule_cfg}
            )
            if node_enabled("relation_rule", args.from_node, args.to_node):
                run_node(
                    args=args,
                    video_dir=video_dir,
                    node="relation_rule",
                    input_hash=rule_hash,
                    required_outputs=[relations_rule_json],
                    outputs={"relations_rule_json": safe_rel(relations_rule_json)},
                    fn=lambda: generate_rule_relations(
                        windows_json=windows_json,
                        tracks_jsonl=tracks_jsonl,
                        out_json=relations_rule_json,
                        video_id=vid,
                        config=rule_cfg,
                    ),
                )

            rel_cfg = cfg.get("relations", {}) if isinstance(cfg.get("relations"), dict) else {}
            rel_cfg = dict(rel_cfg)
            if audio_prior_json.exists():
                audio_prior = read_json(audio_prior_json)
                if isinstance(audio_prior, dict) and str(audio_prior.get("label", "") or "").strip():
                    rel_cfg["vggsound_label"] = str(audio_prior.get("label")).strip()
            rel_dry_run = bool(args.dry_run_relations) or bool(rel_cfg.get("dry_run", False))
            rel_hash = stable_hash(
                {
                    "node": "relation_llm",
                    "windows_hash": sha256_file(windows_json),
                    "tracks_hash": sha256_file(tracks_jsonl),
                    "audio_prior_hash": sha256_file(audio_prior_json) if audio_prior_json.exists() else "",
                    "config": rel_cfg,
                    "dry_run": rel_dry_run,
                    "api_key_present": bool(api_key),
                }
            )
            if node_enabled("relation_llm", args.from_node, args.to_node):
                run_node(
                    args=args,
                    video_dir=video_dir,
                    node="relation_llm",
                    input_hash=rel_hash,
                    required_outputs=[relations_llm_json],
                    outputs={"relations_llm_json": safe_rel(relations_llm_json)},
                    fn=lambda: run_relation_llm(
                        windows_json=windows_json,
                        tracks_jsonl=tracks_jsonl,
                        out_json=relations_llm_json,
                        storyboards_dir=rel_dir / "storyboards",
                        config=rel_cfg,
                        api_key=api_key,
                        resume=bool(args.resume),
                        dry_run=rel_dry_run,
                        video_id=vid,
                        log_path=rel_dir / "run.log",
                    ),
                )

            merge_cfg = cfg.get("relation_merge", {}) if isinstance(cfg.get("relation_merge"), dict) else {}
            merge_hash = stable_hash(
                {"node": "relation_merge", "rule_hash": sha256_file(relations_rule_json), "llm_hash": sha256_file(relations_llm_json), "config": merge_cfg}
            )
            if node_enabled("relation_merge", args.from_node, args.to_node):
                run_node(
                    args=args,
                    video_dir=video_dir,
                    node="relation_merge",
                    input_hash=merge_hash,
                    required_outputs=[relations_merged_json],
                    outputs={"relations_merged_json": safe_rel(relations_merged_json)},
                    fn=lambda: merge_relations(
                        video_id=vid,
                        relation_jsons=[relations_rule_json, relations_llm_json],
                        out_json=relations_merged_json,
                        apply_coupling=bool(merge_cfg.get("apply_coupling", True)),
                    ),
                )

            global_cfg = cfg.get("global_relation", {}) if isinstance(cfg.get("global_relation"), dict) else {}
            global_hash = stable_hash(
                {"node": "global_relation", "merged_hash": sha256_file(relations_merged_json), "tracks_hash": sha256_file(tracks_jsonl), "config": global_cfg}
            )
            if node_enabled("global_relation", args.from_node, args.to_node):
                run_node(
                    args=args,
                    video_dir=video_dir,
                    node="global_relation",
                    input_hash=global_hash,
                    required_outputs=[relations_global_json],
                    outputs={"relations_global_json": safe_rel(relations_global_json)},
                    fn=lambda: run_global_relation(
                        video_id=vid,
                        relations_json=relations_merged_json,
                        out_json=relations_global_json,
                        config=global_cfg,
                    ),
                )

            verify_cfg = cfg.get("relation_verify", {}) if isinstance(cfg.get("relation_verify"), dict) else {}
            verify_hash = stable_hash(
                {
                    "node": "relation_verify",
                    "global_hash": sha256_file(relations_global_json),
                    "tracks_hash": sha256_file(tracks_jsonl),
                    "config": verify_cfg,
                }
            )
            if node_enabled("relation_verify", args.from_node, args.to_node):
                run_node(
                    args=args,
                    video_dir=video_dir,
                    node="relation_verify",
                    input_hash=verify_hash,
                    required_outputs=[relations_verified_json, relation_qc_json],
                    outputs={"relations_verified_json": safe_rel(relations_verified_json), "relation_qc_json": safe_rel(relation_qc_json)},
                    fn=lambda: verify_relations(
                        video_id=vid,
                        relations_json=relations_global_json,
                        tracks_jsonl=tracks_jsonl,
                        out_relations_json=relations_verified_json,
                        out_qc_json=relation_qc_json,
                        config=verify_cfg,
                    ),
                )

            export_hash = stable_hash(
                {"node": "export", "relations_hash": sha256_file(relations_verified_json), "tracks_hash": sha256_file(tracks_jsonl)}
            )
            if node_enabled("export", args.from_node, args.to_node):
                run_node(
                    args=args,
                    video_dir=video_dir,
                    node="export",
                    input_hash=export_hash,
                    required_outputs=[relations_pred_json, trajectories_pred_json],
                    outputs={
                        "relations_pred_json": safe_rel(relations_pred_json),
                        "trajectories_pred_json": safe_rel(trajectories_pred_json),
                        "relation_qc_json": safe_rel(export_dir / "relation_qc.json"),
                    },
                    fn=lambda: export_video_outputs(
                        verified_relations_json=relations_verified_json,
                        relation_qc_json=relation_qc_json,
                        tracks_jsonl=tracks_jsonl,
                        video_id=vid,
                        relations_pred_json=relations_pred_json,
                        trajectories_pred_json=trajectories_pred_json,
                    ),
                )

            if relations_pred_json.exists() and trajectories_pred_json.exists():
                relation_exports.append((vid, relations_pred_json))
                item["state"] = "succeeded"
                item["outputs"] = {
                    "relations_pred_json": safe_rel(relations_pred_json),
                    "trajectories_pred_json": safe_rel(trajectories_pred_json),
                }
            else:
                # Partial run (e.g., --to_node stops before export).
                item["state"] = "partial" if not export_enabled else "failed"
                item["partial_reason"] = "export_not_run" if not export_enabled else "export_missing_outputs"
        except Exception as e:
            item["state"] = "failed"
            item["error"] = str(e)
            print(f"ERROR {vid}: {e}")
        finally:
            item["nodes"] = collect_node_statuses(video_dir, NODE_ORDER)

    pred_dir = run_dir / "pred"
    pred_relations = pred_dir / "relations_pred.json"
    merged_relations = merge_relation_files(relation_exports, pred_relations)

    report_path = run_dir / "reports" / "presence_report.md"
    eval_cfg = cfg.get("evaluate", {}) if isinstance(cfg.get("evaluate"), dict) else {}
    gold_path = Path(str(eval_cfg.get("gold_json", "gold/relations_gold.json"))).expanduser()
    if not gold_path.is_absolute():
        gold_path = (root / gold_path).resolve()

    eval_state: Dict[str, Any] = {"enabled": bool(eval_cfg.get("enabled", True)) and not bool(args.skip_eval)}
    if eval_state["enabled"] and (not relation_exports):
        eval_state.update({"state": "skipped", "reason": "no_exported_preds", "gold_json": safe_rel(gold_path)})
    elif eval_state["enabled"] and gold_path.exists() and pred_relations.exists():
        try:
            run_presence_eval(gold_json=gold_path, pred_json=pred_relations, report_path=report_path, log_path=run_dir / "reports" / "evaluate_presence.log")
            eval_state.update({"state": "succeeded", "report": safe_rel(report_path), "gold_json": safe_rel(gold_path)})
        except Exception as e:
            eval_state.update({"state": "failed", "error": str(e), "gold_json": safe_rel(gold_path)})
    elif eval_state["enabled"]:
        eval_state.update({"state": "skipped", "reason": "gold or pred not found", "gold_json": safe_rel(gold_path)})
    else:
        eval_state.update({"state": "skipped", "reason": "disabled"})

    state_counts: Dict[str, int] = {}
    for it in video_items:
        st = str(it.get("state", "")) or "unknown"
        state_counts[st] = state_counts.get(st, 0) + 1

    manifest.update(
        {
            "finished_at": now_text(),
            "videos": video_items,
            "pred_relations_json": safe_rel(pred_relations),
            "pred_relation_count": sum(len(v) for v in merged_relations.values() if isinstance(v, list)),
            "video_state_counts": state_counts,
            "evaluate": eval_state,
        }
    )
    write_json(run_dir / "run_manifest.json", manifest)
    print("=" * 80)
    print("DONE VIDVRD auto labeling")
    print(f"run_dir={run_dir}")
    print(f"pred={pred_relations}")
    if eval_state.get("report"):
        print(f"report={report_path}")
    print("=" * 80)
