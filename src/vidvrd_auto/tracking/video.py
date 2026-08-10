from __future__ import annotations

"""Video-level OC-SORT execution."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import cv2

from vidvrd_auto.tracking.ocsort import ObjectTracker
from vidvrd_auto.tracking.stitching import apply_global_ids, stitch_tracklets
from vidvrd_auto.utils.io import iter_jsonl, write_json


def _detections(path: Path) -> tuple[Dict[int, List[Dict[str, Any]]], set[int], Dict[int, int]]:
    output: Dict[int, List[Dict[str, Any]]] = {}
    anchors: set[int] = set()
    scenes: Dict[int, int] = {}
    scene_id = 0
    for row in iter_jsonl(path):
        try:
            frame = int(row["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        objects = row.get("objects", [])
        output[frame] = [dict(item) for item in objects if isinstance(item, dict)] if isinstance(objects, list) else []
        batch = row.get("detection_batch", {})
        if isinstance(batch, dict) and batch.get("reason") == "scene_change":
            scene_id += 1
        scenes[frame] = scene_id
        if output[frame] or (isinstance(batch, dict) and batch.get("status") == "observed"):
            anchors.add(frame)
    return output, anchors, scenes


def _compact(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for track in tracks:
        box = track.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        observed = track.get("bbox_observed")
        raw_confidence = track.get("confidence")
        item = {
                "track_id": int(track["track_id"]),
                "bbox": [float(value) for value in box],
                "bbox_observed": [float(value) for value in observed]
                if isinstance(observed, (list, tuple)) and len(observed) == 4
                else None,
                "class_name": str(track.get("class_name", "unknown")),
                "confidence": None if raw_confidence is None else float(raw_confidence),
                "duration_frames": int(track.get("duration_frames", 0)),
                "total_distance": float(track.get("total_distance", 0.0)),
                "instant_distance": float(track.get("instant_distance", 0.0)),
                "avg_speed": float(track.get("avg_speed", 0.0)),
                "motion_state": str(track.get("motion_state", "unknown")),
                "age": int(track.get("age", 0)),
                "hits": int(track.get("hits", 0)),
                "is_predicted": bool(track.get("is_predicted", False)),
                "box_source": str(track.get("box_source", "predicted" if bool(track.get("is_predicted", False)) else "observed")),
                "track_status": str(track.get("track_status", "confirmed")),
            }
        for key in (
            "local_tracklet_id",
            "class_distribution",
            "identity_source",
            "identity_support",
            "association",
            "scene_id",
        ):
            if key in track:
                item[key] = track[key]
        if "frame" in track:
            item["frame"] = int(track["frame"])
        output.append(item)
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
        row_lookup = {int(row.get("frame", 0)): row for row in rows}
        for (start, left), (end, right) in zip(observed, observed[1:]):
            gap = end - start - 1
            if gap <= 0 or gap > max_gap:
                continue
            left_box = [float(value) for value in left["bbox_observed"]]
            right_box = [float(value) for value in right["bbox_observed"]]
            for frame in range(start + 1, end):
                item = lookup.get(frame)
                if item is not None and isinstance(item.get("bbox_observed"), list):
                    continue
                if item is None:
                    item = {
                        "track_id": int(left["track_id"]),
                        "local_tracklet_id": int(left.get("local_tracklet_id", left["track_id"])),
                        "class_name": str(left.get("class_name", "unknown")),
                        "class_distribution": left.get("class_distribution", {}),
                        "confidence": left.get("confidence"),
                        "track_status": "confirmed",
                        "hits": int(left.get("hits", 0)),
                        "identity_source": left.get("identity_source", "online"),
                        "identity_support": left.get("identity_support", 0.0),
                        "scene_id": left.get("scene_id", 0),
                    }
                    row_lookup[frame].setdefault("tracks", []).append(item)
                    lookup[frame] = item
                alpha = (frame - start) / (end - start)
                item["bbox"] = [a + (b - a) * alpha for a, b in zip(left_box, right_box)]
                item["box_source"] = "interpolated"
                item["is_predicted"] = True
                item["bbox_observed"] = None
                item["interpolation"] = {"left_frame": start, "right_frame": end}
                changed += 1

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            row["tracks"] = sorted(row.get("tracks", []), key=lambda item: int(item.get("track_id", -1)))
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
    appearance_encoder_factory: Callable[[Dict[str, Any]], Any] | None = None,
) -> None:
    """Run the selected reference or hybrid tracker and write frame tracks."""

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detector_rows, anchor_frames, scene_by_frame = _detections(detections_path)
    algorithm = str(config.get("algorithm", "sparse_ocsort"))
    if algorithm not in {"ocsort_reference", "sparse_ocsort", "hybrid_sparse_reid"}:
        raise ValueError(f"unknown tracking algorithm: {algorithm}")
    if algorithm == "hybrid_sparse_reid":
        from vidvrd_auto.tracking.hybrid import HybridTracker

        tracker = HybridTracker(
            min_hits=int(config.get("min_hits", 2)),
            max_lost_frames=int(config.get("max_lost_frames", 30)),
            min_new_track_conf=float(config.get("min_new_track_conf", 0.0)),
            appearance_weight=float(config.get("appearance_weight", 0.45)),
            iou_weight=float(config.get("iou_weight", 0.30)),
            motion_weight=float(config.get("motion_weight", 0.20)),
            class_weight=float(config.get("class_weight", 0.05)),
            max_match_cost=float(config.get("max_match_cost", 0.72)),
            min_iou=float(config.get("min_iou", 0.01)),
            min_appearance_similarity=float(config.get("min_appearance_similarity", 0.35)),
            max_center_distance=float(config.get("max_center_distance", 4.0)),
            appearance_memory=int(config.get("appearance_memory", 20)),
        )
        if appearance_encoder_factory is not None:
            appearance_encoder = appearance_encoder_factory(config)
        else:
            from vidvrd_auto.tracking.appearance import MasaAppearanceEncoder

            appearance_encoder = MasaAppearanceEncoder(
                config_path=str(config["masa_config"]),
                checkpoint_path=str(config["masa_checkpoint"]),
                device=str(config.get("appearance_device", "cuda:0")),
                fp16=bool(config.get("appearance_fp16", True)),
            )
    else:
        tracker = ObjectTracker(
            iou_threshold=float(config.get("iou_threshold", 0.5)),
            max_age=int(config.get("max_age", 30)),
            min_hits=int(config.get("min_hits", 3)),
            class_aware=bool(config.get("class_aware", True)),
            min_new_track_conf=float(config.get("min_new_track_conf", 0.35)),
            delta_t=int(config.get("delta_t", 3)),
            inertia=float(config.get("inertia", 0.2)),
            class_vote_window=int(config.get("class_vote_window", 12)),
            class_compatibility=config.get("class_compatibility", {}),
        )

    tracks_path = out_dir / "tracks.jsonl"
    frame_outputs: Dict[int, Dict[int, Dict[str, Any]]] = {}
    frame_count = 0
    errors: List[str] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            should_update = algorithm == "ocsort_reference" or frame_count in anchor_frames
            if should_update:
                if algorithm == "hybrid_sparse_reid":
                    detections = detector_rows.get(frame_count, [])
                    embeddings = appearance_encoder.encode(
                        frame, detections, frame_num=frame_count, video_len=total_frames
                    )
                    tracks = _compact(
                        tracker.update(
                            frame,
                            detections,
                            embeddings,
                            frame_num=frame_count,
                            scene_id=scene_by_frame.get(frame_count, 0),
                        )
                    )
                else:
                    try:
                        tracks = _compact(tracker.track(frame, detector_rows.get(frame_count, []), frame_num=frame_count))
                    except Exception as exc:
                        tracks = []
                        errors.append(f"frame {frame_count}: {exc}")
                for track in tracks:
                    target_frame = int(track.pop("frame", frame_count))
                    frame_outputs.setdefault(target_frame, {})[int(track["track_id"])] = track
            frame_count += 1
    finally:
        cap.release()

    if algorithm == "hybrid_sparse_reid":
        summaries = tracker.summaries()
        write_json(
            out_dir / "tracklets.json",
            {"schema": "tracklet-summary-v1", "time_unit": "video_frame", "tracklets": summaries},
        )
        global_ids, links = stitch_tracklets(summaries, config)
        frame_outputs = apply_global_ids(frame_outputs, global_ids, links)
        write_json(
            out_dir / "stitch_links.json",
            {
                "enabled": True,
                "method": "minimum_cost_dag_path_cover",
                "local_to_global": {str(key): value for key, value in sorted(global_ids.items())},
                "links": links,
            },
        )
    else:
        write_json(out_dir / "tracklets.json", {"schema": "tracklet-summary-v1", "tracklets": []})
        write_json(out_dir / "stitch_links.json", {"enabled": False, "links": []})

    with tracks_path.open("w", encoding="utf-8") as handle:
        for frame in range(frame_count):
            tracks = [frame_outputs.get(frame, {})[key] for key in sorted(frame_outputs.get(frame, {}))]
            handle.write(
                json.dumps(
                    {"frame": frame, "timestamp": float(frame / max(1, fps)), "tracks": tracks},
                    ensure_ascii=False,
                )
                + "\n"
            )

    interpolated = _smooth_short_gaps(tracks_path, max(0, int(config.get("max_interpolation_gap", 8))))
    frame_tracks = {
        int(row.get("frame", 0)): [int(track["track_id"]) for track in row.get("tracks", [])]
        for row in iter_jsonl(tracks_path)
    }

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
        f"mode={algorithm}\nframes={frame_count}\nanchors={len(anchor_frames)}\n"
        f"interpolated={interpolated}\nerrors={len(errors)}\n"
        + "\n".join(errors[:20]),
        encoding="utf-8",
    )
