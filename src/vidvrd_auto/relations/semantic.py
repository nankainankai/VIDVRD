"""Window-level pair relation classification with visual evidence."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np

from vidvrd_auto.providers import DashScopeProvider
from vidvrd_auto.core.ontology import predicate_components
from vidvrd_auto.relations.object_candidates import GEOMETRY_PREDICATES, get_candidate_predicates, normalize_category
from vidvrd_auto.utils.io import iter_jsonl, read_json, write_json


def _tracks(path: Path) -> Dict[int, List[Dict[str, Any]]]:
    output: Dict[int, List[Dict[str, Any]]] = {}
    for row in iter_jsonl(path):
        try:
            frame = int(row["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        items = row.get("tracks", [])
        output[frame] = [dict(item) for item in items if isinstance(item, dict)] if isinstance(items, list) else []
    return output


def _by_id(items: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    output: Dict[int, Dict[str, Any]] = {}
    for item in items:
        try:
            output[int(item["track_id"])] = item
        except (KeyError, TypeError, ValueError):
            continue
    return output


def _classes(rows: Iterable[List[Dict[str, Any]]]) -> Dict[int, str]:
    votes: Dict[int, Counter[str]] = {}
    for items in rows:
        for track_id, item in _by_id(items).items():
            votes.setdefault(track_id, Counter())[normalize_category(str(item.get("class_name", "unknown")))] += 1
    return {track_id: counts.most_common(1)[0][0] for track_id, counts in votes.items() if counts}


def _box(item: Dict[str, Any] | None) -> List[float] | None:
    value = (item or {}).get("bbox_observed") or (item or {}).get("bbox")
    if not isinstance(value, list) or len(value) != 4:
        return None
    return [float(part) for part in value]


def _sample(values: List[int], count: int) -> List[int]:
    if not values:
        return []
    size = min(max(1, count), len(values))
    return sorted({values[int(round(position))] for position in np.linspace(0, len(values) - 1, size)})


def _save_jpeg(path: Path, image: Any) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("storyboard JPEG encoding failed")
    path.write_bytes(encoded.tobytes())


def _pair_storyboard(
    video_path: Path,
    frame_indices: List[int],
    tracks: Dict[int, List[Dict[str, Any]]],
    subject_id: int,
    object_id: int,
    classes: Dict[int, str],
) -> Any:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    tiles: List[Any] = []
    try:
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            frame_tracks = _by_id(tracks.get(frame_index, []))
            left, right = _box(frame_tracks.get(subject_id)), _box(frame_tracks.get(object_id))
            if not ok or left is None or right is None:
                continue
            x1, y1 = min(left[0], right[0]), min(left[1], right[1])
            x2, y2 = max(left[2], right[2]), max(left[3], right[3])
            width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
            x1, y1 = max(0, int(x1 - 0.4 * width)), max(0, int(y1 - 0.4 * height))
            x2 = min(frame.shape[1], int(x2 + 0.4 * width))
            y2 = min(frame.shape[0], int(y2 + 0.4 * height))
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue
            for box, label, color in (
                (left, f"A ID{subject_id} {classes.get(subject_id, 'unknown')}", (0, 0, 255)),
                (right, f"B ID{object_id} {classes.get(object_id, 'unknown')}", (255, 128, 0)),
            ):
                bx1, by1, bx2, by2 = [int(value) for value in box]
                cv2.rectangle(crop, (bx1 - x1, by1 - y1), (bx2 - x1, by2 - y1), color, 3)
                cv2.putText(crop, label, (max(2, bx1 - x1), max(20, by1 - y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(crop, f"frame {frame_index}", (8, crop.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            tile_height = 280
            tile_width = max(1, int(crop.shape[1] * tile_height / crop.shape[0]))
            tiles.append(cv2.resize(crop, (tile_width, tile_height)))
    finally:
        capture.release()
    if not tiles:
        raise RuntimeError("track pair has no usable visual evidence")
    columns = min(4, len(tiles))
    target_width = max(tile.shape[1] for tile in tiles)
    blank = np.zeros((280, target_width, 3), dtype=np.uint8)
    rows: List[Any] = []
    for offset in range(0, len(tiles), columns):
        row: List[Any] = []
        for tile in tiles[offset : offset + columns]:
            if tile.shape[1] < target_width:
                tile = cv2.copyMakeBorder(tile, 0, 0, 0, target_width - tile.shape[1], cv2.BORDER_CONSTANT)
            row.append(tile)
        while len(row) < columns:
            row.append(blank.copy())
        rows.append(cv2.hconcat(row))
    return cv2.vconcat(rows)


def _parse_json(text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        value = json.loads(cleaned[start : end + 1]) if start >= 0 and end > start else {}
    return value if isinstance(value, dict) else {}


def _directions(
    left: int, right: int, classes: Dict[int, str], predicate_split: str
) -> Dict[tuple[int, int], List[str]]:
    output: Dict[tuple[int, int], List[str]] = {}
    for subject, obj in ((left, right), (right, left)):
        candidates = [
            predicate
            for predicate in get_candidate_predicates(
                classes.get(subject, "unknown"), classes.get(obj, "unknown"), split=predicate_split
            )
            if predicate not in GEOMETRY_PREDICATES
        ]
        if candidates:
            output[(subject, obj)] = candidates
    return output


def _prompt(start: int, end: int, directions: Dict[tuple[int, int], List[str]], classes: Dict[int, str]) -> str:
    candidates = [
        {
            "subject_track_id": subject,
            "subject_category": classes.get(subject, "unknown"),
            "object_track_id": obj,
            "object_category": classes.get(obj, "unknown"),
            "candidate_predicates": predicates,
        }
        for (subject, obj), predicates in directions.items()
    ]
    return (
        "You are annotating relations for one track pair in a 30-frame window. The attached crops use stable A/B labels.\n"
        "Return only relations from the supplied directed candidate lists that have clear visual evidence across the window. "
        "Do not infer unlisted relations. Return JSON only: "
        '{"relations":[{"subject_track_id":1,"predicate":"ride","object_track_id":2,'
        '"confidence":0.8,"reason":"visual evidence"}]}.\n'
        f"Window: [{start},{end + 1}); candidates: {json.dumps(candidates, ensure_ascii=False)}"
    )


def classify_relations(
    *,
    windows_path: Path,
    tracks_path: Path,
    out_path: Path,
    storyboards_dir: Path,
    config: Dict[str, Any],
    api_key: str,
    dry_run: bool,
    video_id: str,
) -> None:
    """Classify semantic relations once per valid track pair and window."""

    windows_obj = read_json(windows_path)
    windows = windows_obj.get("windows", []) if isinstance(windows_obj, dict) else []
    video_path = Path(str((windows_obj.get("video", {}) if isinstance(windows_obj, dict) else {}).get("path", ""))).expanduser()
    tracks = _tracks(tracks_path)
    classes = _classes(tracks.values())
    storyboards_dir.mkdir(parents=True, exist_ok=True)
    provider = DashScopeProvider(config, api_key=api_key)
    output: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    max_windows = int(config.get("max_windows", 0) or 0)
    max_pairs = max(1, int(config.get("max_pairs_per_window", 6) or 6))
    min_pair_frames = max(1, int(config.get("min_pair_frames", 2) or 2))
    predicate_split = str(config.get("predicate_split", "all") or "all")
    if max_windows > 0:
        windows = windows[:max_windows]

    for window_index, window in enumerate(windows, 1):
        if not isinstance(window, dict):
            continue
        start = int(window.get("start_frame", 0))
        end = int(window.get("end_frame", start))
        track_ids = sorted({int(value) for value in window.get("track_ids", [])})
        pairs: List[tuple[int, int, List[int], Dict[tuple[int, int], List[str]]]] = []
        for position, left in enumerate(track_ids):
            for right in track_ids[position + 1 :]:
                visible = [
                    frame
                    for frame in range(start, end + 1)
                    if _box(_by_id(tracks.get(frame, [])).get(left)) is not None
                    and _box(_by_id(tracks.get(frame, [])).get(right)) is not None
                ]
                directions = _directions(left, right, classes, predicate_split)
                if len(visible) >= min_pair_frames:
                    pairs.append((left, right, visible, directions))
        pairs.sort(key=lambda item: len(item[2]), reverse=True)

        for left, right, visible, directions in pairs[:max_pairs]:
            frames = _sample(visible, int(config.get("max_frames_per_window", 8)))
            image = _pair_storyboard(video_path, frames, tracks, left, right, classes)
            image_path = storyboards_dir / f"window_{window_index:04d}_A{left}_B{right}.jpg"
            _save_jpeg(image_path, image)
            if dry_run or not directions:
                continue
            result = provider.call(prompt=_prompt(start, end, directions, classes), image_paths=[image_path])
            if not result.ok:
                errors.append({"window": window_index, "pair": [left, right], "error": result.error})
                continue
            try:
                relations = _parse_json(result.text).get("relations", [])
            except Exception as exc:
                errors.append({"window": window_index, "pair": [left, right], "error": f"invalid JSON: {exc}"})
                continue
            for relation in relations if isinstance(relations, list) else []:
                if not isinstance(relation, dict):
                    continue
                try:
                    subject = int(relation["subject_track_id"])
                    obj = int(relation["object_track_id"])
                    predicate = str(relation["predicate"]).strip().lower()
                    confidence = float(relation.get("confidence", 0.7))
                except (KeyError, TypeError, ValueError):
                    continue
                if predicate not in directions.get((subject, obj), []):
                    continue
                output.append(
                    {
                        "subject_track_id": subject,
                        "predicate": predicate,
                        "object_track_id": obj,
                        "start_frame": start,
                        "end_frame": end,
                        "confidence": min(1.0, max(0.0, confidence)),
                        "source": "window_semantic_vl",
                        "predicate_components": predicate_components(predicate),
                        "segment_id": int(window.get("window_id", window_index)),
                        "subject_category": classes.get(subject, "unknown"),
                        "object_category": classes.get(obj, "unknown"),
                        "evidence": str(relation.get("reason", "")),
                    }
                )

    write_json(out_path, {video_id: output})
    (out_path.parent / "run.log").write_text(
        json.dumps({"provider": provider.stats.to_dict(), "errors": errors}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
