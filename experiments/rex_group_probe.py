from __future__ import annotations

"""Small, reproducible category-group probe; no Gold labels are sent to Rex."""

import argparse
import json
from pathlib import Path

import cv2

from vidvrd_auto.config.loader import load_config
from vidvrd_auto.detection.rex import RexDetector
from vidvrd_auto.utils.io import read_json


GROUPS = {
    "carnivore": ["bear", "dog", "domestic_cat", "fox", "lion", "tiger"],
    "small_mammal": ["hamster", "rabbit", "red_panda", "squirrel"],
    "transport_person": ["airplane", "bicycle", "bus", "car", "motorcycle", "person", "skateboard", "train", "watercraft"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-categories", nargs="*", default=[])
    parser.add_argument("--append-categories", nargs="*", default=[])
    args = parser.parse_args()
    config = load_config()["detector"]
    vocabulary = read_json(args.vocabulary)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {args.video}")
    frames = []
    try:
        for index in args.frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"cannot decode frame {index}")
            frames.append(frame)
    finally:
        capture.release()
    detector = RexDetector(
        model_path=config["rex_model_path"], backend=config["rex_backend"],
        categories=vocabulary["categories"], min_box_area=config["rex_min_box_area"],
        max_detections_per_frame=config["rex_max_detections_per_frame"],
        max_tokens=config["rex_max_tokens"], max_pixels=config["rex_max_pixels"],
        temperature=config["temperature"], top_p=config["top_p"], top_k=config["top_k"],
        category_aliases=vocabulary["label_map"],
    )
    rows = []
    detector.load_model()
    try:
        groups = {"full": vocabulary["categories"], **GROUPS}
        if args.extra_categories:
            groups = {"custom": args.extra_categories}
        elif args.append_categories:
            groups = {"full_plus": list(vocabulary["categories"]) + list(args.append_categories)}
        for name, categories in groups.items():
            detector.categories = categories
            detections = detector.detect_batch(frames)
            rows.append({
                "group": name,
                "categories": categories,
                "frames": [{"index": index, "objects": objects} for index, objects in zip(args.frames, detections)],
            })
            print(f"group={name} " + ", ".join(
                f"f{index}:{[item['class_name'] for item in objects]}"
                for index, objects in zip(args.frames, detections)
            ), flush=True)
    finally:
        detector.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"video": str(args.video), "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
