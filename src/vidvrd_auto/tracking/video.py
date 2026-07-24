from __future__ import annotations

"""Video-level OC-SORT execution."""

import json
from pathlib import Path
from typing import Any, Dict, List

import cv2

from vidvrd_auto.tracking.ocsort import ObjectTracker
from vidvrd_auto.utils.io import iter_jsonl, write_json


def _detections(path: Path) -> Dict[int, List[Dict[str, Any]]]:
    output: Dict[int, List[Dict[str, Any]]] = {}
    for row in iter_jsonl(path):
        try:
            frame = int(row["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        objects = row.get("objects", [])
        output[frame] = [dict(item) for item in objects if isinstance(item, dict)] if isinstance(objects, list) else []
    return output


def _compact(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for track in tracks:
        box = track.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        observed = track.get("bbox_observed")
        output.append(
            {
                "track_id": int(track["track_id"]),
                "bbox": [float(value) for value in box],
                "bbox_observed": [float(value) for value in observed]
                if isinstance(observed, (list, tuple)) and len(observed) == 4
                else None,
                "class_name": str(track.get("class_name", "unknown")),
                "confidence": float(track.get("confidence", 0.0)),
                "duration_frames": int(track.get("duration_frames", 0)),
                "total_distance": float(track.get("total_distance", 0.0)),
                "instant_distance": float(track.get("instant_distance", 0.0)),
                "avg_speed": float(track.get("avg_speed", 0.0)),
                "motion_state": str(track.get("motion_state", "unknown")),
                "age": int(track.get("age", 0)),
                "hits": int(track.get("hits", 0)),
                "is_predicted": bool(track.get("is_predicted", False)),
                "box_source": "predicted" if bool(track.get("is_predicted", False)) else "observed",
            }
        )
    return output


def _smooth_short_gaps(path: Path, max_gap: int) -> int:
    """Interpolate only bounded gaps enclosed by real observations."""

    rows = list(iter_jsonl(path))
    by_track: Dict[int, List[tuple[int, Dict[str, Any]]]] = {}
    for row in rows:
        frame = int(row.get("frame", 0))
        for track in row.get("tracks", []) or []:
            if not isinstance(track, dict):
                continue
            track_id = int(track.get("track_id", -1))
            by_track.setdefault(track_id, []).append((frame, track))

    changed = 0
    for items in by_track.values():
        observed = [(frame, item) for frame, item in items if isinstance(item.get("bbox_observed"), list)]
        lookup = {frame: item for frame, item in items}
        for (start, left), (end, right) in zip(observed, observed[1:]):
            gap = end - start - 1
            if gap <= 0 or gap > max_gap:
                continue
            left_box = [float(value) for value in left["bbox_observed"]]
            right_box = [float(value) for value in right["bbox_observed"]]
            for frame in range(start + 1, end):
                item = lookup.get(frame)
                if item is None or isinstance(item.get("bbox_observed"), list):
                    continue
                alpha = (frame - start) / (end - start)
                item["bbox"] = [a + (b - a) * alpha for a, b in zip(left_box, right_box)]
                item["box_source"] = "interpolated"
                item["is_predicted"] = True
                item["interpolation"] = {"left_frame": start, "right_frame": end}
                changed += 1

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return changed


def _windows(frame_tracks: Dict[int, List[int]], frame_count: int, fps: int, size: int, stride: int) -> List[Dict[str, Any]]:
    if frame_count <= 0:
        return []
    size = max(1, size)
    stride = max(1, stride)
    starts = list(range(0, max(1, frame_count - size + 1), stride))
    if not starts:
        starts = [0]
    if starts[-1] + size < frame_count:
        next_start = starts[-1] + stride
        if next_start < frame_count:
            starts.append(next_start)

    output: List[Dict[str, Any]] = []
    for window_id, start in enumerate(starts, 1):
        end = min(frame_count - 1, start + size - 1)
        track_ids = sorted({tid for frame in range(start, end + 1) for tid in frame_tracks.get(frame, [])})
        output.append(
            {
                "window_id": window_id,
                "start_frame": start,
                "end_frame": end,
                "start_time": float(start / max(1, fps)),
                "end_time": float(end / max(1, fps)),
                "frame_count": end - start + 1,
                "track_ids": track_ids,
            }
        )
    return output


def track_video(
    *,
    video_path: Path,
    detections_path: Path,
    out_dir: Path,
    config: Dict[str, Any],
) -> None:
    """Run OC-SORT and write frame tracks plus temporal windows."""

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detector_rows = _detections(detections_path)
    tracker = ObjectTracker(
        iou_threshold=float(config.get("iou_threshold", 0.5)),
        max_age=int(config.get("max_age", 30)),
        min_hits=int(config.get("min_hits", 3)),
        class_aware=bool(config.get("class_aware", True)),
        min_new_track_conf=float(config.get("min_new_track_conf", 0.35)),
        delta_t=int(config.get("delta_t", 3)),
        inertia=float(config.get("inertia", 0.2)),
        class_vote_window=int(config.get("class_vote_window", 12)),
        max_output_age=int(config.get("max_output_age", 8)),
    )

    tracks_path = out_dir / "tracks.jsonl"
    frame_tracks: Dict[int, List[int]] = {}
    frame_count = 0
    errors: List[str] = []
    try:
        with tracks_path.open("w", encoding="utf-8") as handle:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                try:
                    tracks = _compact(tracker.track(frame, detector_rows.get(frame_count, []), frame_num=frame_count))
                except Exception as exc:
                    tracks = []
                    errors.append(f"frame {frame_count}: {exc}")
                frame_tracks[frame_count] = [int(track["track_id"]) for track in tracks]
                handle.write(
                    json.dumps(
                        {
                            "frame": frame_count,
                            "timestamp": float(frame_count / max(1, fps)),
                            "tracks": tracks,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                frame_count += 1
    finally:
        cap.release()

    interpolated = _smooth_short_gaps(tracks_path, max(0, int(config.get("max_interpolation_gap", 8))))

    size = int(config.get("window_size", 30))
    stride = int(config.get("stride", 15))
    write_json(
        out_dir / "windows.json",
        {
            "video": {
                "path": str(video_path),
                "fps": fps,
                "total_frames": total_frames,
                "duration": float(total_frames / max(1, fps)),
            },
            "window_size_frames": max(1, size),
            "stride_frames": max(1, stride),
            "windows": _windows(frame_tracks, frame_count, fps, size, stride),
        },
    )
    (out_dir / "run.log").write_text(
        f"mode=official_ocsort\nframes={frame_count}\ninterpolated={interpolated}\nerrors={len(errors)}\n"
        + "\n".join(errors[:20]),
        encoding="utf-8",
    )
