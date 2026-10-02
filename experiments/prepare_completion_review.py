"""Select reproducible review clips using video metadata, without Gold spans."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2

from vidvrd_auto.utils.hashing import sha256_file
from vidvrd_auto.utils.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True, help="Frozen absolute input-path list")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--clip-frames", type=int, default=60)
    parser.add_argument("--seed", default="completion-eval-v1")
    parser.add_argument("--previously-inspected-id", action="append", default=[])
    args = parser.parse_args()
    if args.clip_frames < 1:
        parser.error("clip frames must be positive")
    paths = [Path(line.strip()).resolve() for line in args.videos.read_text(encoding="utf-8-sig").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not paths or len({p.stem for p in paths}) != len(paths):
        raise ValueError("input list must contain unique, nonempty video IDs")
    clips = []
    for path in paths:
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"cannot open video: {path}")
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        if count < 1 or fps <= 0:
            raise ValueError(f"invalid video metadata: {path}")
        length = min(args.clip_frames, count)
        digest = hashlib.sha256(f"{args.seed}:{path.stem}".encode()).hexdigest()
        start = int(digest, 16) % (count - length + 1)
        clips.append({
            "video_id": path.stem,
            "clip_id": f"{path.stem}:{start}-{start + length}",
            "source": str(path),
            "start_frame": start,
            "end_frame": start + length,
            "span_convention": "half_open",
            "fps": fps,
            "duration_sec": length / fps,
            "annotation_complete": False,
            "annotation_status": "pending_independent_complete_annotation",
            "previously_inspected_video": path.stem in args.previously_inspected_id,
        })
    write_json(args.out_dir / "clips.json", clips)
    write_json(args.out_dir / "protocol.json", {
        "protocol": "completion_presence_review_v1",
        "source_list": str(args.videos.resolve()),
        "source_list_sha256": sha256_file(args.videos),
        "seed": args.seed,
        "clip_frames": args.clip_frames,
        "selection": "SHA256(seed:video_id) modulo available fixed-length clip starts",
        "selection_reads_gold_annotations": False,
        "video_count": len(clips),
        "annotation_complete": False,
        "current_metrics_available": False,
        "required_annotation": "all object instances, per-frame boxes and all directed ontology relations within each selected clip",
        "intended_metrics": ["instance_pair_presence_precision_recall_f1", "human_acceptance_rate", "total_human_edit_time", "manual_baseline_time", "runtime_and_cost", "completion_rate"],
    })
    print(f"prepared {len(clips)} fixed clips; annotations pending; {args.out_dir}")


if __name__ == "__main__":
    main()
