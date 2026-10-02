"""Recorded API usage and completion, including failed requested videos."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from vidvrd_auto.utils.io import read_json, write_json


def summarize_run(run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "run_manifest.json")
    videos = manifest.get("videos", [])
    requested = len(videos)
    succeeded = sum(item.get("state") == "succeeded" for item in videos)
    relation_path = run_dir / "pred" / "relations.json"
    trajectory_path = run_dir / "pred" / "trajectories.json"
    outputs_available = relation_path.exists() and trajectory_path.exists()
    relations = read_json(relation_path) if outputs_available else {}
    trajectories = read_json(trajectory_path) if outputs_available else {}
    records = []
    for item in videos:
        root = run_dir / "videos" / item["video_id"]
        semantic_path, review_path = root / "semantic" / "run.log", root / "verify" / "qc.json"
        semantic = read_json(semantic_path) if semantic_path.exists() else {}
        review = read_json(review_path).get("strong_model_review_result", {}) if review_path.exists() else {}
        stats = semantic.get("provider", {})
        review_stats = review.get("provider", {})
        if not review_stats and "attempts" in review:
            # Old artifacts have result-level attempts but no usage accounting.
            review_stats = {"calls": 1, "attempts": review["attempts"],
                            "retries": max(0, review["attempts"] - 1),
                            "failed": int(not review.get("ok", False))}
        records.append({
            "video_id": item["video_id"], "state": item.get("state"),
            "executed_nodes": item.get("executed_nodes"),
            "error": item.get("error", ""), "semantic": stats, "review": review_stats,
            "cached_batches": sum(bool(batch.get("cached")) for batch in semantic.get("batch_audit", [])),
            "semantic_packet_errors": len(semantic.get("errors", [])),
            "review_state": review.get("state", "unrecorded"),
            "rejected_review_actions": len(read_json(review_path).get("rejected_agent_actions", [])) if review_path.exists() else 0,
            "exported_relation_count": len(relations.get(item["video_id"], [])) if outputs_available else None,
            "exported_track_count": len(trajectories.get(item["video_id"], [])) if outputs_available else None,
        })
    providers = [record[stage] for record in records for stage in ("semantic", "review") if record[stage]]
    totals = {key: sum(stats.get(key, 0) for stats in providers) for key in
              ("calls", "attempts", "retries", "failed", "images", "input_tokens", "output_tokens", "usage_reported_calls")}
    timed = [stats["elapsed_sec"] for stats in providers if "elapsed_sec" in stats]
    totals["elapsed_sec"] = sum(timed) if timed else None
    totals["usage_complete"] = totals["usage_reported_calls"] == totals["calls"] - sum(stats.get("dry_runs", 0) for stats in providers)
    invocation_api = None
    if all(record["executed_nodes"] is not None for record in records):
        current = [record[stage] for record in records for stage, node in (("semantic", "semantic"), ("review", "verify"))
                   if node in record["executed_nodes"]]
        invocation_api = {key: sum(stats.get(key, 0) for stats in current)
                          for key in ("calls", "attempts", "retries", "failed", "input_tokens", "output_tokens")}
    wall_sec = None
    try:
        wall_sec = (datetime.fromisoformat(manifest["finished_at"]) - datetime.fromisoformat(manifest["started_at"])).total_seconds()
    except (KeyError, ValueError):
        pass
    nonempty = [record["video_id"] for record in records
                if record["state"] == "succeeded" and record["exported_relation_count"]
                and record["exported_track_count"]]
    empty_completed = [record["video_id"] for record in records
                       if record["state"] == "succeeded" and outputs_available
                       and (not record["exported_relation_count"] or not record["exported_track_count"])]
    result = {
        "metric": "recorded_runtime_v1", "requested_videos": requested,
        "completed_videos": succeeded, "completion_rate": succeeded / requested if requested else 0.0,
        "wall_elapsed_sec": wall_sec, "recorded_api": totals,
        "invocation_api": invocation_api,
        "actual_fee": None, "per_video": records,
        "output_availability": {
            "available": outputs_available,
            "completed_nonempty_prediction_video_count": len(nonempty) if outputs_available else None,
            "completed_nonempty_prediction_rate": len(nonempty) / requested if outputs_available and requested else None,
            "empty_completed_video_ids": empty_completed if outputs_available else None,
            "interpretation": "Both tracks and relations must be nonempty on a completed video. This measures output availability, not correctness; an empty video may legitimately have no relations.",
        },
        "validation_scope": manifest.get("validation_scope", {}),
        "interpretation": "API totals are from the latest stored stage execution. Resume wall time is separate. Missing billing/usage is not zero cost.",
    }
    write_json(run_dir / "reports" / "runtime.json", result)
    return result
