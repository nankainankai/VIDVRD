from __future__ import annotations

"""Render predicted tracks and active relations onto an MP4 video."""

import argparse
from pathlib import Path
from typing import Any, Dict, Sequence

import cv2

from vidvrd_auto.core.schema import relation_span
from vidvrd_auto.utils.io import read_json


_COLORS = ((70, 210, 255), (255, 160, 70), (90, 220, 100), (220, 100, 220), (80, 120, 255))


def _select_video_id(relations: Dict[str, Any], trajectories: Dict[str, Any], requested: str, source: Path) -> str:
    if requested:
        return requested
    candidates = set(relations) | set(trajectories)
    if source.stem in candidates:
        return source.stem
    if len(candidates) == 1:
        return next(iter(candidates))
    raise ValueError("--video-id is required when prediction files contain multiple or non-matching video IDs")


def render_prediction_video(
    *, video: Path, relations_path: Path, trajectories_path: Path, output: Path, video_id: str = ""
) -> Dict[str, Any]:
    relation_obj = read_json(relations_path)
    trajectory_obj = read_json(trajectories_path)
    if not isinstance(relation_obj, dict) or not isinstance(trajectory_obj, dict):
        raise ValueError("relation and trajectory artifacts must be JSON objects keyed by video ID")
    selected = _select_video_id(relation_obj, trajectory_obj, video_id, video)
    relations = [item for item in relation_obj.get(selected, []) if isinstance(item, dict)]
    tracks = [item for item in trajectory_obj.get(selected, []) if isinstance(item, dict)]
    track_map = {int(item["track_id"]): item for item in tracks}
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"cannot create output video: {output}")
    frame_index = frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            visible: Dict[int, Sequence[float]] = {}
            for track_id, track in track_map.items():
                box = track.get("trajectory", {}).get(str(frame_index))
                if not isinstance(box, list) or len(box) != 4:
                    continue
                visible[track_id] = box
                x1, y1, x2, y2 = (int(round(float(value))) for value in box)
                color = _COLORS[track_id % len(_COLORS)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"#{track_id} {track.get('category', 'unknown')}"
                cv2.putText(frame, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            active_relations = []
            for relation in relations:
                span = relation_span(relation)
                if not (span.start_frame <= frame_index < span.end_frame):
                    continue
                subject_id = int(relation.get("subject_track_id", relation.get("subject_id", -1)))
                object_id = int(relation.get("object_track_id", relation.get("object_id", -1)))
                if subject_id not in visible or object_id not in visible:
                    continue
                active_relations.append((
                    float(relation.get("ranking_score", relation.get("agent_score", relation.get("rule_support", 0.0))) or 0.0),
                    subject_id,
                    object_id,
                    str(relation.get("predicate", "?")),
                ))
            active_relations.sort(key=lambda item: (-item[0], item[1], item[3], item[2]))
            shown = active_relations[:6]
            for _, subject_id, object_id, _ in shown:
                subject = visible[subject_id]
                obj = visible[object_id]
                left = (int((subject[0] + subject[2]) / 2), int((subject[1] + subject[3]) / 2))
                right = (int((obj[0] + obj[2]) / 2), int((obj[1] + obj[3]) / 2))
                cv2.arrowedLine(frame, left, right, (30, 230, 230), 2, tipLength=0.12)
            labels = [f"#{subject_id} {predicate} #{object_id}" for _, subject_id, object_id, predicate in shown]
            if len(active_relations) > len(shown):
                labels.append(f"+ {len(active_relations) - len(shown)} more active relations")
            if labels:
                panel_height = min(height, 12 + 25 * len(labels))
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
                for index, label in enumerate(labels):
                    cv2.putText(frame, label, (12, 24 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            writer.write(frame)
            frame_index += 1
            frame_count += 1
    finally:
        capture.release()
        writer.release()
    return {"video_id": selected, "frames": frame_count, "tracks": len(tracks), "relations": len(relations), "output": str(output)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Overlay VidVRD tracks and relations on a video")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--relations", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-id", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = render_prediction_video(
        video=args.video.expanduser().resolve(),
        relations_path=args.relations.expanduser().resolve(),
        trajectories_path=args.trajectories.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
        video_id=args.video_id,
    )
    print(f"rendered frames={result['frames']} tracks={result['tracks']} relations={result['relations']}")
    print(f"output={result['output']}")


if __name__ == "__main__":
    main()
