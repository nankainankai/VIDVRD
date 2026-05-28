from __future__ import annotations

"""导出节点。

输入：复核后的关系、轨迹 JSONL 和关系 QC。
输出：最终 `relations_pred.json`、`trajectories_pred.json` 和 `relation_qc.json`。
该节点不调用模型，负责把内部结果转换为稳定交付 schema。
"""

import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from vidvrd_auto.utils.io import iter_jsonl, read_json, write_json
from vidvrd_auto.utils.relation_viz import render_relation_video


def tracks_to_trajectories(tracks_jsonl: Path, video_id: str, out_json: Path) -> None:
    tracks: Dict[int, Dict[str, Any]] = {}
    for row in iter_jsonl(tracks_jsonl):
        try:
            frame = int(row.get("frame"))
        except Exception:
            continue
        for item in row.get("tracks", []) or []:
            if not isinstance(item, dict):
                continue
            try:
                tid = int(item.get("track_id"))
            except Exception:
                continue
            bbox = item.get("bbox_observed", item.get("bbox"))
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            if tid not in tracks:
                tracks[tid] = {
                    "track_id": tid,
                    "category": str(item.get("class_name", "") or "unknown"),
                    "trajectory": {},
                }
            tracks[tid]["trajectory"][str(frame)] = [float(x) for x in bbox]
    write_json(out_json, {video_id: [tracks[k] for k in sorted(tracks.keys())]})


def export_video_outputs(
    *,
    verified_relations_json: Path,
    relation_qc_json: Path,
    tracks_jsonl: Path,
    video_id: str,
    relations_pred_json: Path,
    trajectories_pred_json: Path,
    video_path: Path | None = None,
    export_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    relations_pred_json.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(verified_relations_json, relations_pred_json)
    if relation_qc_json.exists():
        shutil.copyfile(relation_qc_json, relations_pred_json.parent / "relation_qc.json")
    tracks_to_trajectories(tracks_jsonl, video_id, trajectories_pred_json)

    out_meta: Dict[str, Any] = {}
    cfg = export_config if isinstance(export_config, dict) else {}
    if bool(cfg.get("relation_viz_video", False)) and video_path is not None and video_path.exists():
        viz_path = relations_pred_json.parent / str(cfg.get("relation_viz_name", "relation_box_vis.mp4") or "relation_box_vis.mp4")
        out_meta["relation_viz"] = render_relation_video(
            video_path=video_path,
            tracks_jsonl=tracks_jsonl,
            relations_json=relations_pred_json,
            video_id=video_id,
            out_path=viz_path,
            config={
                "min_confidence": cfg.get("relation_viz_min_confidence", 0.3),
                "max_confidence_spatial": cfg.get("relation_viz_max_confidence_spatial", 0.95),
                "spatial_max_center_distance_ratio": cfg.get("relation_viz_spatial_max_distance_ratio", 0.35),
                "max_relations_per_frame": cfg.get("relation_viz_max_per_frame", 8),
                "top_k_per_pair": cfg.get("relation_viz_top_k_per_pair", 1),
                "show_confidence": cfg.get("relation_viz_show_confidence", True),
                "output_name": viz_path.name,
            },
        )
    return out_meta


def merge_relation_files(video_exports: Iterable[Tuple[str, Path]], out_json: Path) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for video_id, path in video_exports:
        if not path.exists():
            continue
        obj = read_json(path)
        if isinstance(obj, dict):
            if video_id in obj:
                merged[video_id] = obj.get(video_id, [])
            else:
                merged.update(obj)
    write_json(out_json, merged)
    return merged
