from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from vidvrd_auto.config.loader import load_app_config
from vidvrd_auto.core import Secrets, VideoPaths
from vidvrd_auto.evaluation.vidvrd import run_evaluation_suite
from vidvrd_auto.nodes.export import merge_relation_files, merge_trajectory_files, tracks_to_trajectories
from vidvrd_auto.nodes.ingest import video_id_for_source
from vidvrd_auto.pipeline.constants import NODE_ORDER
from vidvrd_auto.pipeline.files import Artifacts
from vidvrd_auto.pipeline.manifest import build_run_provenance, collect_node_statuses, now_text
from vidvrd_auto.pipeline.media import run_media
from vidvrd_auto.pipeline.relation_flow import run_relations
from vidvrd_auto.pipeline.stage import StageRunner
from vidvrd_auto.utils.hashing import stable_hash
from vidvrd_auto.utils.io import write_json
from vidvrd_auto.utils.paths import repo_root, safe_rel


def parse_sources(args: Namespace) -> List[str]:
    values: List[str] = []
    if str(args.video or "").strip():
        values.append(str(args.video).strip())
    if str(args.videos or "").strip():
        source = Path(str(args.videos)).expanduser()
        if source.is_file():
            values.extend(
                line.strip()
                for line in source.read_text(encoding="utf-8-sig").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        else:
            values.extend(item.strip() for item in str(args.videos).split(",") if item.strip())
    unique = list(dict.fromkeys(values))
    if not unique:
        raise SystemExit("请通过 --video 或 --videos 提供输入")
    return unique


def run_pipeline(*, args: Namespace, config_path: Path | None) -> None:
    root = repo_root()
    config = load_app_config(config_path)
    run_dir = Path(args.run_dir).expanduser()
    run_dir = (root / run_dir).resolve() if not run_dir.is_absolute() else run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    api_key = Secrets.from_env(dashscope_api_key=str(args.api_key or "")).dashscope_api_key
    items: List[Dict[str, Any]] = []
    exports: List[Tuple[str, Path]] = []
    trajectory_exports: List[Tuple[str, Path]] = []
    used_ids: set[str] = set()
    resolved_config_path = (config_path or root / "configs/base.json").resolve()
    effective_config = config.to_dict()
    tracking_only = bool(getattr(args, "tracking_only", False))
    active_nodes = NODE_ORDER[:5] if tracking_only else NODE_ORDER

    manifest: Dict[str, Any] = {
        "run_dir": safe_rel(run_dir),
        "started_at": now_text(),
        "config": safe_rel(resolved_config_path),
        "config_hash": stable_hash(effective_config),
        "provenance": build_run_provenance(root=root, config=effective_config, config_path=resolved_config_path),
        "nodes": active_nodes,
        "options": {
            "resume": bool(args.resume),
            "force": bool(args.force),
            "dry_run": bool(args.dry_run_relations),
            "tracking_only": tracking_only,
            "api_key_present": bool(api_key),
        },
        "videos": items,
    }
    write_json(run_dir / "run_manifest.json", manifest)

    for source in parse_sources(args):
        video_id = video_id_for_source(source, used_ids)
        paths = VideoPaths.for_video(run_dir, video_id, repo_dir=root)
        paths.video_dir.mkdir(parents=True, exist_ok=True)
        files = Artifacts.for_video(paths)
        stages = StageRunner(args, paths)
        item: Dict[str, Any] = {"video_id": video_id, "source": source, "state": "running"}
        items.append(item)
        try:
            video_path, video_hash = run_media(
                source=source,
                video_id=video_id,
                config=config,
                api_key=api_key,
                paths=paths,
                files=files,
                stages=stages,
            )
            item["input"] = {
                "video_path": safe_rel(video_path),
                "video_hash": video_hash,
                "hash_algorithm": "sha256_full_file",
            }
            if tracking_only:
                files.trajectories.parent.mkdir(parents=True, exist_ok=True)
                tracks_to_trajectories(files.tracks, video_id, files.trajectories)
                trajectory_exports.append((video_id, files.trajectories))
                item.update(
                    state="succeeded",
                    outputs={"trajectories": safe_rel(files.trajectories), "track_qc": safe_rel(files.track_report)},
                )
            else:
                run_relations(
                    video_id=video_id,
                    config=config,
                    api_key=api_key,
                    paths=paths,
                    files=files,
                    stages=stages,
                )
            if not tracking_only and files.relations.exists() and files.trajectories.exists():
                exports.append((video_id, files.relations))
                trajectory_exports.append((video_id, files.trajectories))
                item.update(
                    state="succeeded",
                    outputs={"relations": safe_rel(files.relations), "trajectories": safe_rel(files.trajectories)},
                )
            elif not tracking_only:
                item.update(state="partial", reason="export not selected")
        except Exception as exc:
            item.update(state="failed", error=str(exc))
            print(f"ERROR {video_id}: {exc}")
        finally:
            item["nodes"] = collect_node_statuses(paths.video_dir, active_nodes)

    all_relations = run_dir / "pred" / "relations.json"
    all_trajectories = run_dir / "pred" / "trajectories.json"
    requested_video_ids = [str(item["video_id"]) for item in items]
    merged = merge_relation_files(exports, all_relations, requested_video_ids)
    merge_trajectory_files(trajectory_exports, all_trajectories, requested_video_ids)
    evaluation = _evaluate(
        config=config.section("evaluate").to_dict(),
        disabled=bool(args.skip_eval) or tracking_only,
        requested_video_ids=requested_video_ids,
        relations=all_relations,
        trajectories=all_trajectories,
        run_dir=run_dir,
        root=root,
    )
    states: Dict[str, int] = {}
    for item in items:
        states[item["state"]] = states.get(item["state"], 0) + 1
    manifest.update(
        finished_at=now_text(),
        videos=items,
        relations=safe_rel(all_relations),
        trajectories=safe_rel(all_trajectories),
        relation_count=sum(len(value) for value in merged.values() if isinstance(value, list)),
        video_states=states,
        evaluation=evaluation,
    )
    write_json(run_dir / "run_manifest.json", manifest)
    print(f"DONE run={run_dir}")
    print(f"trajectories={all_trajectories}")
    if not tracking_only:
        print(f"relations={all_relations}")


def _evaluate(
    *,
    config: Dict[str, Any],
    disabled: bool,
    requested_video_ids: Sequence[str],
    relations: Path,
    trajectories: Path,
    run_dir: Path,
    root: Path,
) -> Dict[str, Any]:
    enabled = bool(config.get("enabled", True)) and not disabled
    result: Dict[str, Any] = {"enabled": enabled}
    gold_relations = Path(str(config.get("gold_relations", "gold/vidvrd_50_relations.json"))).expanduser()
    gold_trajectories = Path(str(config.get("gold_trajectories", "gold/vidvrd_50_trajectories.json"))).expanduser()
    gold_relations = (root / gold_relations).resolve() if not gold_relations.is_absolute() else gold_relations.resolve()
    gold_trajectories = (root / gold_trajectories).resolve() if not gold_trajectories.is_absolute() else gold_trajectories.resolve()
    if not enabled:
        result["state"] = "skipped"
        return result
    if not gold_relations.exists() or not gold_trajectories.exists():
        result.update(state="failed", error="Gold relations or trajectories are missing")
        return result
    gold_manifest = Path(str(config.get("gold_manifest", "gold/vidvrd_50_manifest.json"))).expanduser()
    gold_manifest = (root / gold_manifest).resolve() if not gold_manifest.is_absolute() else gold_manifest.resolve()
    if not gold_manifest.exists():
        result.update(state="failed", error="Gold manifest is missing")
        return result
    official_report = run_dir / "reports" / "official_vidvrd.md"
    official_metrics = run_dir / "reports" / "official_vidvrd.json"
    diagnostic_report = run_dir / "reports" / "diagnostic_track_aligned.md"
    diagnostic_metrics = run_dir / "reports" / "diagnostic_track_aligned.json"
    metrics = run_evaluation_suite(
        gold_relations=gold_relations,
        gold_trajectories=gold_trajectories,
        gold_manifest=gold_manifest,
        pred_relations=relations,
        pred_trajectories=trajectories,
        requested_video_ids=requested_video_ids,
        official_report_path=official_report,
        official_metrics_path=official_metrics,
        diagnostic_report_path=diagnostic_report,
        diagnostic_metrics_path=diagnostic_metrics,
        dataset_split=str(config.get("dataset_split", "test")),
        scope=str(config.get("scope", "gold_split")),
        expected_official_video_count=int(config.get("expected_official_video_count", 200)),
        official_viou_threshold=float(config.get("official_viou_threshold", 0.5) or 0.5),
        diagnostic_track_viou_threshold=float(config.get("diagnostic_track_viou_threshold", 0.3) or 0.3),
        diagnostic_relation_tiou_threshold=float(config.get("diagnostic_relation_tiou_threshold", 0.5) or 0.5),
    )
    result.update(
        state="succeeded",
        official={
            "report": safe_rel(official_report),
            "metrics": safe_rel(official_metrics),
            "relation_detection": metrics["official"]["relation_detection"],
            "relation_tagging": metrics["official"]["relation_tagging"],
        },
        diagnostic={"report": safe_rel(diagnostic_report), "metrics": safe_rel(diagnostic_metrics)},
    )
    return result
