from __future__ import annotations

"""Final relation and trajectory export."""

import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from vidvrd_auto.utils.io import iter_jsonl, read_json, write_json


def tracks_to_trajectories(tracks_path: Path, video_id: str, out_path: Path) -> None:
    tracks: Dict[int, Dict[str, Any]] = {}
    for row in iter_jsonl(tracks_path):
        try:
            frame = int(row["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        for item in row.get("tracks", []):
            if not isinstance(item, dict):
                continue
            box = item.get("bbox")
            if not isinstance(box, list) or len(box) != 4:
                continue
            track_id = int(item["track_id"])
            track = tracks.setdefault(
                track_id,
                {
                    "track_id": track_id,
                    "category": str(item.get("class_name", "unknown")),
                    "trajectory": {},
                    "box_sources": {},
                },
            )
            track["trajectory"][str(frame)] = [float(value) for value in box]
            track["box_sources"][str(frame)] = str(item.get("box_source", "observed"))
    write_json(out_path, {video_id: [tracks[key] for key in sorted(tracks)]})


def export_video_outputs(
    *,
    verified_path: Path,
    qc_path: Path,
    tracks_path: Path,
    video_id: str,
    relations_path: Path,
    trajectories_path: Path,
) -> None:
    relations_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(verified_path, relations_path)
    if qc_path.exists():
        shutil.copyfile(qc_path, relations_path.parent / "qc.json")
    tracks_to_trajectories(tracks_path, video_id, trajectories_path)


def merge_relation_files(video_exports: Iterable[Tuple[str, Path]], out_path: Path) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for video_id, path in video_exports:
        if not path.exists():
            continue
        value = read_json(path)
        if isinstance(value, dict):
            merged[video_id] = value.get(video_id, [])
    write_json(out_path, merged)
    return merged


def merge_trajectory_files(video_exports: Iterable[Tuple[str, Path]], out_path: Path) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for video_id, path in video_exports:
        if not path.exists():
            continue
        value = read_json(path)
        if isinstance(value, dict):
            merged[video_id] = value.get(video_id, [])
    write_json(out_path, merged)
    return merged
