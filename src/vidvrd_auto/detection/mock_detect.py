from __future__ import annotations

"""无模型检测：用于 dry-run / smoke test，不依赖 Rex-Omni 或 DINO-X。"""

import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("mock detection requires opencv-python (pip install opencv-python)") from e


def _mock_objects(frame_idx: int, width: int, height: int) -> List[Dict[str, Any]]:
    """两人物框：左侧略动，右侧静止，便于规则关系产出 left/right。"""
    w, h = float(max(64, width)), float(max(64, height))
    shift = float((frame_idx % 10) * 2)
    left = [0.12 * w + shift, 0.25 * h, 0.42 * w + shift, 0.85 * h]
    right = [0.55 * w, 0.28 * h, 0.88 * w, 0.82 * h]
    return [
        {"bbox": left, "class": 0, "class_name": "person", "confidence": 0.92},
        {"bbox": right, "class": 0, "class_name": "person", "confidence": 0.90},
    ]


def run_mock_detect(*, video_path: Path, out_dir: Path, config: Dict[str, Any], log_path: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    detections_path = out_dir / "detections_full.jsonl"
    meta_path = out_dir / "video_meta.json"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 10
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 320
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 240

    if total_frames <= 0:
        total_frames = max(1, int(config.get("mock_min_frames", 30)))

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with detections_path.open("w", encoding="utf-8") as det_fp, log_path.open("w", encoding="utf-8") as log:
        log.write(f"mock_detect video={video_path} fps={fps} frames={total_frames}\n")
        frame_idx = 0
        while frame_idx < total_frames:
            ret, _ = cap.read()
            if not ret and frame_idx > 0:
                break
            ts = float(frame_idx / max(1, fps))
            objects = _mock_objects(frame_idx, width, height)
            row = {"frame": int(frame_idx), "timestamp": float(ts), "objects": objects}
            det_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            frame_idx += 1
            if not ret:
                break

    cap.release()
    written_frames = frame_idx
    duration = float(written_frames / max(1, fps))

    meta = {
        "video": {
            "path": str(video_path.resolve()),
            "fps": int(fps),
            "total_frames": int(written_frames),
            "duration": float(duration),
        },
        "detector_stats": {"backend": "mock", "frames_written": int(written_frames)},
        "detections_jsonl": detections_path.name,
        "box_vis_video": "",
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"OK mock_detect frames={written_frames} -> {detections_path}\n")
