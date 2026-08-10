from __future__ import annotations

"""In-process Rex-Omni video detection stage."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import cv2

from vidvrd_auto.detection.rex import RexDetector
from vidvrd_auto.detection.temporal_fusion import annotate_batch_detections


DetectorFactory = Callable[..., RexDetector]


def _compact(objects: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for obj in objects:
        box = obj.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            raw_score = obj.get("score", obj.get("confidence"))
            score = None if raw_score is None else float(raw_score)
            item: Dict[str, Any] = {
                "bbox": [float(value) for value in box],
                "class": int(obj.get("class", -1)),
                "class_name": str(obj.get("class_name", "unknown")),
                "score": score,
                "confidence": score,
                "score_kind": str(obj.get("score_kind", "legacy_confidence" if score is not None else "unavailable")),
            }
        except (TypeError, ValueError):
            continue
        for key in ("source", "batch_id", "batch_frame_indices", "raw_class_name"):
            if key in obj:
                item[key] = obj[key]
        output.append(item)
    return output


def _draw(frame: Any, objects: Sequence[Dict[str, Any]]) -> Any:
    image = frame.copy()
    for obj in objects:
        box = obj.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(float(value)) for value in box]
        score = obj.get("score", obj.get("confidence"))
        label = str(obj.get("class_name", "unknown"))
        if score is not None:
            label += f" {float(score):.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(image, label, (x1 + 4, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return image


def _make_detector(config: Dict[str, Any], factory: DetectorFactory) -> RexDetector:
    return factory(
        model_path=str(config.get("rex_model_path", "")),
        backend=str(config.get("rex_backend", "transformers")),
        categories=config.get("rex_categories", "person"),
        detection_interval=1,
        min_box_area=float(config.get("rex_min_box_area", 500.0)),
        max_detections_per_frame=int(config.get("rex_max_detections_per_frame", 60)),
        max_tokens=int(config.get("rex_max_tokens", 512)),
        max_pixels=int(config.get("rex_max_pixels", 640 * 640)),
        category_aliases=dict(config.get("category_aliases", {})),
        temperature=float(config.get("temperature", 0.0)),
        top_p=float(config.get("top_p", 0.05)),
        top_k=int(config.get("top_k", 1)),
        repetition_penalty=float(config.get("repetition_penalty", 1.05)),
    )


def _thumbnail(frame: Any) -> Any:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)


def _change_score(previous: Any, current: Any) -> float:
    return float(cv2.absdiff(previous, current).mean() / 255.0)


def detect_video(
    *,
    video_path: Path,
    out_dir: Path,
    config: Dict[str, Any],
    log_path: Path,
    detector_factory: DetectorFactory = RexDetector,
) -> None:
    """Decode a video and write the established frame-wise detection artifacts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    detections_path = out_dir / "detections.jsonl"
    meta_path = out_dir / "meta.json"
    box_video_path = out_dir / "preview.mp4"
    batch_size = max(1, int(config.get("batch_size", 5)))
    interval = max(1, int(config.get("detection_interval", 5)))
    min_interval = min(interval, max(1, int(config.get("min_detection_interval", 2))))
    scene_threshold = max(0.0, float(config.get("scene_change_threshold", 0.2)))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = float(total_frames / float(max(1, fps)))

    writer = None
    if bool(config.get("save_box_video", False)):
        writer = cv2.VideoWriter(
            str(box_video_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1, fps), (max(1, width), max(1, height))
        )

    detector = _make_detector(config, detector_factory)
    try:
        detector.load_model()
    except Exception:
        cap.release()
        if writer is not None:
            writer.release()
        raise
    errors: List[str] = []
    frame_count = 0
    anchor_count = 0

    def write_row(handle: Any, frame_index: int, frame: Any, objects: List[Dict[str, Any]], batch: Dict[str, Any] | None = None) -> None:
        row: Dict[str, Any] = {
            "frame": int(frame_index),
            "timestamp": float(frame_index / max(1, fps)),
            "objects": objects,
        }
        if batch is not None:
            row["detection_batch"] = batch
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if writer is not None:
            image = _draw(frame, objects)
            if image.shape[1] != width or image.shape[0] != height:
                image = cv2.resize(image, (max(1, width), max(1, height)))
            writer.write(image)

    try:
        with detections_path.open("w", encoding="utf-8") as handle:
            pending: List[Tuple[int, Any, bool, str, float]] = []
            pending_anchors: List[Tuple[int, Any]] = []
            batch_id = 0
            last_anchor = -interval
            last_anchor_thumb = None
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                index = frame_count
                frame_count += 1
                thumb = _thumbnail(frame)
                gap = index - last_anchor
                score = _change_score(last_anchor_thumb, thumb) if last_anchor_thumb is not None else 0.0
                scheduled = gap >= interval
                changed = last_anchor_thumb is not None and gap >= min_interval and scene_threshold > 0 and score >= scene_threshold
                anchor = index == 0 or scheduled or changed
                reason = "first" if index == 0 else ("scene_change" if changed and not scheduled else "interval")
                pending.append((index, frame, anchor, reason if anchor else "sparse_schedule", score))
                if anchor:
                    pending_anchors.append((index, frame))
                    anchor_count += 1
                    last_anchor = index
                    last_anchor_thumb = thumb
                if len(pending_anchors) >= batch_size:
                    batch_id = _flush_records(
                        handle, pending, pending_anchors, batch_id, detector, write_row, errors, interval
                    )
                    pending = []
                    pending_anchors = []

            if pending:
                _flush_records(handle, pending, pending_anchors, batch_id, detector, write_row, errors, interval)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    meta = {
        "video": {
            "path": str(video_path),
            "fps": int(fps),
            "total_frames": int(total_frames),
            "duration": float(duration),
        },
        "detector_stats": detector.get_stats(),
        "sampling": {
            "mode": str(config.get("sampling_mode", "adaptive_sparse")),
            "detection_interval": interval,
            "min_detection_interval": min_interval,
            "scene_change_threshold": scene_threshold,
            "anchor_frames": anchor_count,
            "decoded_frames": frame_count,
        },
        "detections_jsonl": detections_path.name,
        "box_vis_video": box_video_path.name if bool(config.get("save_box_video", False)) else "",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path.write_text(
        "mode=in_process_rexomni\n"
        f"video={video_path}\nframes={frame_count}\nanchors={anchor_count}\n"
        f"detection_interval={interval}\nbatch_size={batch_size}\nerrors={len(errors)}\n"
        + "\n".join(errors[:20]),
        encoding="utf-8",
    )


def _flush_records(
    handle: Any,
    records: List[Tuple[int, Any, bool, str, float]],
    anchors: List[Tuple[int, Any]],
    batch_id: int,
    detector: RexDetector,
    write_row: Callable[[Any, int, Any, List[Dict[str, Any]], Dict[str, Any] | None], None],
    errors: List[str],
    interval: int,
) -> int:
    indices = [index for index, _ in anchors]
    frames = [frame for _, frame in anchors]
    error = ""
    detected_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    if anchors:
        try:
            detected = detector.detect_batch(frames)
            detected = annotate_batch_detections(
                detected, frame_indices=indices, batch_id=batch_id, source="rexomni"
            )
        except Exception as exc:
            detected = [[] for _ in frames]
            error = str(exc)
            errors.append(f"batch {batch_id} {indices}: {error}")
        detected_by_frame = {index: _compact(objects) for index, objects in zip(indices, detected)}
    for index, frame, anchor, reason, score in records:
        batch = {
            "batch_id": int(batch_id) if anchor else None,
            "frame_indices": indices if anchor else [],
            "source": "rexomni" if anchor else "sparse_schedule",
            "status": "observed" if anchor else "skipped",
            "reason": reason,
            "scene_change_score": round(score, 6),
            "detection_interval": interval,
            "error": error if anchor else "",
        }
        write_row(handle, index, frame, detected_by_frame.get(index, []), batch)
    return batch_id + 1
