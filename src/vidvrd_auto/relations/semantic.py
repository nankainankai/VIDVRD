"""Window-level pair relation classification with visual evidence."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np

from vidvrd_auto.providers import DashScopeProvider
from vidvrd_auto.agents import EvidencePacket, validate_semantic_actions
from vidvrd_auto.core.ontology import predicate_components
from vidvrd_auto.core.schema import serialize_relation_artifact
from vidvrd_auto.prompts.templates import semantic_relation_prompt
from vidvrd_auto.relations.candidate_router import expand_route, route_predicates
from vidvrd_auto.relations.evidence_features import trajectory_evidence
from vidvrd_auto.relations.object_candidates import GEOMETRY_PREDICATES, normalize_category
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


def _event_sample(values: List[int], evidences: Iterable[Dict[str, Any]], count: int, burst_size: int) -> List[int]:
    events = list(dict.fromkeys(frame for evidence in evidences for frame in evidence.get("event_frames", [])))
    priority: List[int] = []
    radius = max(0, burst_size // 2)
    available = set(values)
    for event in events:
        priority.extend(frame for offset in range(-radius, radius + 1) if (frame := event + offset) in available)
    priority.extend(_sample(values, count))
    return sorted(list(dict.fromkeys(priority))[: min(count, len(values))])


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
            annotated = frame.copy()
            for box, label, color in (
                (left, f"A ID{subject_id} {classes.get(subject_id, 'unknown')}", (0, 0, 255)),
                (right, f"B ID{object_id} {classes.get(object_id, 'unknown')}", (255, 128, 0)),
            ):
                bx1, by1, bx2, by2 = [int(value) for value in box]
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 3)
                cv2.putText(annotated, label, (max(2, bx1), max(20, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(annotated, f"frame {frame_index}", (8, annotated.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            crop = annotated[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = crop.copy()
            cv2.putText(crop, f"pair crop f{frame_index}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(crop, f"pair crop f{frame_index}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tile_height = 240
            context_width = max(1, int(annotated.shape[1] * tile_height / annotated.shape[0]))
            crop_width = max(1, int(crop.shape[1] * tile_height / crop.shape[0]))
            tiles.append(
                cv2.hconcat(
                    [
                        cv2.resize(annotated, (context_width, tile_height)),
                        cv2.resize(crop, (crop_width, tile_height)),
                    ]
                )
            )
    finally:
        capture.release()
    if not tiles:
        raise RuntimeError("track pair has no usable visual evidence")
    columns = min(2, len(tiles))
    target_width = max(tile.shape[1] for tile in tiles)
    blank = np.zeros((tiles[0].shape[0], target_width, 3), dtype=np.uint8)
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


def _direction_routes(
    left: int,
    right: int,
    tracks: Dict[int, List[Dict[str, Any]]],
    visible: List[int],
    predicate_split: str,
    candidate_limit: int,
) -> tuple[Dict[tuple[int, int], Dict[str, Any]], Dict[str, Any]]:
    routes: Dict[tuple[int, int], Dict[str, Any]] = {}
    evidence: Dict[str, Any] = {}
    for subject, obj in ((left, right), (right, left)):
        item = trajectory_evidence(tracks, visible, subject, obj)
        routes[(subject, obj)] = route_predicates(
            item, split=predicate_split, limit=candidate_limit, exclude=GEOMETRY_PREDICATES
        )
        evidence[f"{subject}->{obj}"] = item
    return routes, evidence


def _candidate_directions(
    routes: Dict[tuple[int, int], Dict[str, Any]], classes: Dict[int, str]
) -> List[Dict[str, Any]]:
    return [
        {
            "subject_track_id": subject,
            "subject_category": classes.get(subject, "unknown"),
            "object_track_id": obj,
            "object_category": classes.get(obj, "unknown"),
            **{key: value for key, value in route.items() if key != "ranked_predicates"},
        }
        for (subject, obj), route in routes.items()
    ]


def _track_evidence(
    tracks: Dict[int, List[Dict[str, Any]]], frames: List[int], left: int, right: int
) -> Dict[str, Any]:
    sources: Counter[str] = Counter()
    observed_pairs = 0
    for frame in frames:
        items = _by_id(tracks.get(frame, []))
        left_source = str(items[left].get("box_source", "observed"))
        right_source = str(items[right].get("box_source", "observed"))
        sources[f"{left_source}/{right_source}"] += 1
        if left_source == right_source == "observed":
            observed_pairs += 1
    return {
        "joint_visible_frames": len(frames),
        "joint_observed_frames": observed_pairs,
        "box_source_pairs": dict(sorted(sources.items())),
    }


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
    evidence_packets: List[Dict[str, Any]] = []
    agent_audit: List[Dict[str, Any]] = []
    max_windows = int(config.get("max_windows", 0) or 0)
    max_pairs = int(config.get("max_pairs_per_window", 0) or 0)
    min_pair_frames = max(1, int(config.get("min_pair_frames", 2) or 2))
    predicate_split = str(config.get("predicate_split", "all") or "all")
    allow_more_frames = bool(config.get("allow_request_more_frames", True))
    max_additional_frames = max(0, int(config.get("max_additional_frames", 4) or 0))
    candidate_limit = max(1, int(config.get("candidate_limit", 14) or 14))
    expanded_candidate_limit = max(
        candidate_limit, int(config.get("expanded_candidate_limit", 24) or 24)
    )
    event_burst_size = max(1, int(config.get("event_burst_size", 5) or 5))
    video_meta = windows_obj.get("video", {}) if isinstance(windows_obj, dict) else {}
    fps = float(video_meta.get("fps", 0.0) or 0.0)
    if max_windows > 0:
        windows = windows[:max_windows]
    coverage: List[Dict[str, Any]] = []

    for window_index, window in enumerate(windows, 1):
        if not isinstance(window, dict):
            continue
        start = int(window.get("start_frame", 0))
        end = int(window.get("end_frame", start))
        track_ids = sorted({int(value) for value in window.get("track_ids", [])})
        pairs: List[tuple[int, int, List[int]]] = []
        for position, left in enumerate(track_ids):
            for right in track_ids[position + 1 :]:
                visible = [
                    frame
                    for frame in range(start, end + 1)
                    if _box(_by_id(tracks.get(frame, [])).get(left)) is not None
                    and _box(_by_id(tracks.get(frame, [])).get(right)) is not None
                ]
                if len(visible) >= min_pair_frames:
                    pairs.append((left, right, visible))
        pairs.sort(key=lambda item: len(item[2]), reverse=True)
        selected_pairs = pairs if max_pairs <= 0 else pairs[:max_pairs]
        deferred_pairs = pairs[len(selected_pairs) :]
        coverage.append(
            {
                "window_id": int(window.get("window_id", window_index)),
                "candidate_pair_count": len(pairs),
                "processed_pair_count": len(selected_pairs),
                "deferred_pairs": [[left, right] for left, right, _ in deferred_pairs],
            }
        )

        for left, right, visible in selected_pairs:
            routes, directional_evidence = _direction_routes(
                left,
                right,
                tracks,
                visible,
                predicate_split,
                candidate_limit,
            )
            frames = _event_sample(
                visible,
                directional_evidence.values(),
                int(config.get("max_frames_per_window", 8)),
                event_burst_size,
            )
            packet = EvidencePacket(
                packet_id=f"{video_id}:w{int(window.get('window_id', window_index))}:A{left}:B{right}",
                video_id=video_id,
                window_id=int(window.get("window_id", window_index)),
                start_frame=start,
                end_frame=end,
                fps=fps,
                displayed_frames=frames,
                available_frames=visible,
                subject_track_id=left,
                subject_category=classes.get(left, "unknown"),
                object_track_id=right,
                object_category=classes.get(right, "unknown"),
                candidate_directions=_candidate_directions(routes, classes),
                track_evidence=_track_evidence(tracks, visible, left, right),
                trajectory_evidence=directional_evidence,
                candidate_policy="hierarchical_predicate_v1",
                evidence_mode="event_burst_dual_view",
                max_additional_frames=max_additional_frames,
            )
            evidence_packets.append(packet.to_dict())
            image = _pair_storyboard(video_path, frames, tracks, left, right, classes)
            image_path = storyboards_dir / f"window_{window_index:04d}_A{left}_B{right}.jpg"
            _save_jpeg(image_path, image)
            if dry_run or not routes:
                agent_audit.append({"packet_id": packet.packet_id, "state": "dry_run" if dry_run else "no_candidates"})
                continue
            result = provider.call(prompt=semantic_relation_prompt(packet), image_paths=[image_path])
            if not result.ok:
                errors.append({"window": window_index, "pair": [left, right], "error": result.error})
                agent_audit.append({"packet_id": packet.packet_id, "state": "provider_failed", "response": result.to_dict()})
                continue
            try:
                actions = _parse_json(result.text).get("actions", [])
            except Exception as exc:
                errors.append({"window": window_index, "pair": [left, right], "error": f"invalid JSON: {exc}"})
                agent_audit.append({"packet_id": packet.packet_id, "state": "invalid_json", "response": result.to_dict()})
                continue
            validation = validate_semantic_actions(
                actions if isinstance(actions, list) else [],
                packet,
                request_budget_available=allow_more_frames and max_additional_frames > 0,
            )
            call_audit: Dict[str, Any] = {
                "packet_id": packet.packet_id,
                "state": "validated",
                "initial_response": result.to_dict(),
                "initial_validation": validation,
                "supplemental_call_count": 0,
            }
            accepted = list(validation["accepted_relations"])
            requested_frames = list(validation["requested_frames"])
            requested_expansions = list(validation["requested_expansions"])
            if requested_frames or requested_expansions:
                supplemental_routes = {direction: dict(route) for direction, route in routes.items()}
                for request in requested_expansions:
                    direction = (
                        int(request["subject_track_id"]),
                        int(request["object_track_id"]),
                    )
                    supplemental_routes[direction] = expand_route(
                        supplemental_routes[direction],
                        list(request["candidate_families"]),
                        limit=expanded_candidate_limit,
                        split=predicate_split,
                    )
                supplemental_frames = sorted(set(frames + requested_frames))
                supplemental_packet = replace(
                    packet,
                    packet_id=f"{packet.packet_id}:supplemental",
                    displayed_frames=supplemental_frames,
                    candidate_directions=_candidate_directions(supplemental_routes, classes),
                    max_additional_frames=0,
                )
                evidence_packets.append(supplemental_packet.to_dict())
                call_audit["supplemental_packet_id"] = supplemental_packet.packet_id
                supplemental_path = storyboards_dir / f"window_{window_index:04d}_A{left}_B{right}_supplemental.jpg"
                _save_jpeg(
                    supplemental_path,
                    _pair_storyboard(video_path, supplemental_frames, tracks, left, right, classes),
                )
                supplemental_result = provider.call(
                    prompt=semantic_relation_prompt(supplemental_packet, supplemental=True),
                    image_paths=[supplemental_path],
                )
                call_audit["supplemental_call_count"] = 1
                call_audit["supplemental_response"] = supplemental_result.to_dict()
                if supplemental_result.ok:
                    try:
                        supplemental_actions = _parse_json(supplemental_result.text).get("actions", [])
                        supplemental_validation = validate_semantic_actions(
                            supplemental_actions if isinstance(supplemental_actions, list) else [],
                            supplemental_packet,
                            request_budget_available=False,
                        )
                    except Exception as exc:
                        supplemental_validation = {
                            "accepted_relations": [],
                            "actions": [],
                            "requested_frames": [],
                            "requested_expansions": [],
                            "rejected_actions": [{"reason": f"invalid_supplemental_json: {exc}"}],
                        }
                    call_audit["supplemental_validation"] = supplemental_validation
                    accepted.extend(supplemental_validation["accepted_relations"])
            agent_audit.append(call_audit)
            for action_index, relation in enumerate(accepted):
                subject = int(relation["subject_track_id"])
                obj = int(relation["object_track_id"])
                predicate = str(relation["predicate"])
                agent_score = float(relation["agent_score"])
                output.append(
                    {
                        "subject_track_id": subject,
                        "predicate": predicate,
                        "object_track_id": obj,
                        "start_frame": int(relation["start_frame"]),
                        "end_frame": int(relation["end_frame"]),
                        "evidence_frames": list(relation["evidence_frames"]),
                        "agent_score": agent_score,
                        "ranking_score": agent_score,
                        "score_kind": "agent_ranking",
                        "source": "window_semantic_vl",
                        "predicate_components": predicate_components(predicate),
                        "segment_id": int(window.get("window_id", window_index)),
                        "subject_category": classes.get(subject, "unknown"),
                        "object_category": classes.get(obj, "unknown"),
                        "evidence": str(relation["reason"]),
                        "agent_execution": {
                            "packet_id": packet.packet_id,
                            "action": "accept_relation",
                            "action_index": action_index,
                            "candidate_policy": packet.candidate_policy,
                            "evidence_mode": packet.evidence_mode,
                            "supplemental_call_count": call_audit["supplemental_call_count"],
                        },
                    }
                )

    write_json(out_path, {video_id: [serialize_relation_artifact(item) for item in output]})
    write_json(out_path.parent / "evidence_packets.json", {"video_id": video_id, "packets": evidence_packets})
    (out_path.parent / "run.log").write_text(
        json.dumps(
            {"provider": provider.stats.to_dict(), "errors": errors, "pair_coverage": coverage, "agent_audit": agent_audit},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
