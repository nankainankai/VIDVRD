from __future__ import annotations

"""Read-only COCO detector probe on selected development video frames."""

import argparse
import json
from pathlib import Path

import cv2
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")
    frames = []
    try:
        for index in args.frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"cannot decode frame {index}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0))
    finally:
        capture.release()
    rows = []
    with torch.inference_mode():
        for index, frame in zip(args.frames, frames):
            output = model([frame.to(device)])[0]
            objects = [
                {
                    "category": weights.meta["categories"][int(label)],
                    "score": round(float(score), 4),
                    "bbox": [round(float(value), 2) for value in box],
                }
                for box, label, score in zip(output["boxes"], output["labels"], output["scores"])
                if float(score) >= args.threshold
            ]
            rows.append({"frame": index, "objects": objects})
            print(f"frame={index} " + ", ".join(f"{item['category']}:{item['score']}" for item in objects), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"video": str(args.video), "rows": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
