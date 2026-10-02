from __future__ import annotations

"""Offline COCO/Rex detection-fusion tracking pilot on one dev video.

This is an experiment, not part of the production pipeline. Rex remains the
source for every other class; a COCO pass supplies person/motorcycle boxes.
"""

import argparse
import json
from pathlib import Path

import cv2
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

from vidvrd_auto.config.loader import load_config
from vidvrd_auto.evaluation.diagnostic.track_aligned import align_trajectories, clear_mot_on_gold_frames
from vidvrd_auto.nodes.export import tracks_to_trajectories
from vidvrd_auto.tracking.video import track_video
from vidvrd_auto.utils.io import iter_jsonl, read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--rex-detections", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()
    rows = list(iter_jsonl(args.rex_detections))
    anchors = [row for row in rows if row.get("detection_batch", {}).get("status") == "observed"]
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")
    used = 0
    try:
        with torch.inference_mode():
            for anchor_index, row in enumerate(anchors):
                index = int(row["frame"])
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError(f"cannot decode frame {index}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(device)
                result = model([frame])[0]
                coco = []
                for box, label, score in zip(result["boxes"], result["labels"], result["scores"]):
                    category = str(weights.meta["categories"][int(label)])
                    if category not in {"person", "motorcycle"} or float(score) < args.threshold:
                        continue
                    coco.append({
                        "bbox": [float(value) for value in box],
                        "class_name": category,
                        "class": 0 if category == "person" else 1,
                        "score": float(score),
                        "score_kind": "coco_probability",
                        "source": "torchvision_coco_pilot",
                    })
                replace = {item["class_name"] for item in coco}
                row["objects"] = [item for item in row.get("objects", []) if item.get("class_name") not in replace] + coco
                used += len(coco)
                if len(anchors) <= 10 or anchor_index % 10 == 0:
                    print(f"frame={index} coco={len(coco)}", flush=True)
    finally:
        capture.release()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    detections_path = args.out_dir / "detections.jsonl"
    with detections_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tracking = dict(load_config(Path("configs/benchmark_dev.json"))["tracking"])
    track_dir = args.out_dir / "track"
    track_video(video_path=args.video, detections_path=detections_path, out_dir=track_dir, config=tracking)
    trajectory_path = args.out_dir / "trajectories.json"
    tracks_to_trajectories(track_dir / "tracks.jsonl", args.video.stem, trajectory_path)
    gold = read_json(Path("gold/vidvrd_50_trajectories.json"))[args.video.stem]
    prediction = read_json(trajectory_path)[args.video.stem]
    _, matches = align_trajectories(gold, prediction, 0.3)
    mot = clear_mot_on_gold_frames(gold, prediction)
    report = {
        "scope": "one development video; cached Rex plus COCO person/motorcycle; tracking-only",
        "coco_boxes": used,
        "matched_tracks": len(matches),
        "gold_tracks": len(gold),
        "predicted_tracks": len(prediction),
        "mota": mot["mota"],
        "motp": mot["motp"],
        "matches": matches,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
