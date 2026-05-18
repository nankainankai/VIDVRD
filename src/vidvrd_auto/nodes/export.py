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
) -> None:
    relations_pred_json.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(verified_relations_json, relations_pred_json)
    if relation_qc_json.exists():
        shutil.copyfile(relation_qc_json, relations_pred_json.parent / "relation_qc.json")
    tracks_to_trajectories(tracks_jsonl, video_id, trajectories_pred_json)


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
