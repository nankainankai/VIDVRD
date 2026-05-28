#!/usr/bin/env python3
"""在已有 run 结果上生成「检测框 + 关系」可视化视频。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vidvrd_auto.utils.io import read_json
from vidvrd_auto.utils.relation_viz import render_relation_video


def _resolve_video_path(video_dir: Path) -> Path:
    source_json = video_dir / "inputs" / "source.json"
    if source_json.exists():
        meta = read_json(source_json)
        p = Path(str(meta.get("video_path", "") or ""))
        if p.exists():
            return p
    windows = video_dir / "step2_track" / "windows.json"
    if windows.exists():
        obj = read_json(windows)
        if isinstance(obj, dict):
            vp = str(obj.get("video_path", "") or obj.get("video", {}).get("path", "") if isinstance(obj.get("video"), dict) else "")
            if vp:
                p = Path(vp)
                if p.exists():
                    return p
    raise FileNotFoundError(f"cannot resolve source video under {video_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render relation overlay video from an existing VIDVRD run")
    ap.add_argument("--run_dir", type=str, required=True, help="Run directory, e.g. runs/test1_kf2")
    ap.add_argument("--video_id", type=str, default="", help="Video id folder name; auto-detect if omitted")
    ap.add_argument("--output", type=str, default="", help="Output mp4 path; default export/relation_box_vis.mp4")
    ap.add_argument("--min_confidence", type=float, default=0.3)
    ap.add_argument("--max_confidence_spatial", type=float, default=0.95, help="高于此置信度的位置关系不显示")
    ap.add_argument("--spatial_max_center_distance_ratio", type=float, default=0.35, help="位置关系连线超过画面对角线该比例则隐藏，0=关闭")
    ap.add_argument("--max_relations_per_frame", type=int, default=8)
    ap.add_argument("--top_k_per_pair", type=int, default=1, help="过滤后每对 subject-object 最多显示几条")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    videos_root = run_dir / "videos"
    if not videos_root.exists():
        raise SystemExit(f"ERROR: missing videos dir: {videos_root}")

    if args.video_id.strip():
        video_ids = [args.video_id.strip()]
    else:
        video_ids = sorted(p.name for p in videos_root.iterdir() if p.is_dir())
    if not video_ids:
        raise SystemExit(f"ERROR: no video folders under {videos_root}")

    results = []
    for vid in video_ids:
        video_dir = videos_root / vid
        export_dir = video_dir / "export"
        relations_json = export_dir / "relations_pred.json"
        tracks_jsonl = video_dir / "step2_track" / "tracks_full.jsonl"
        if not relations_json.exists():
            relations_json = run_dir / "pred" / "relations_pred.json"
        out_path = Path(args.output).expanduser().resolve() if args.output.strip() else export_dir / "relation_box_vis.mp4"
        video_path = _resolve_video_path(video_dir)
        meta = render_relation_video(
            video_path=video_path,
            tracks_jsonl=tracks_jsonl,
            relations_json=relations_json,
            video_id=vid,
            out_path=out_path,
            config={
                "min_confidence": args.min_confidence,
                "max_confidence_spatial": args.max_confidence_spatial,
                "spatial_max_center_distance_ratio": args.spatial_max_center_distance_ratio,
                "max_relations_per_frame": args.max_relations_per_frame,
                "top_k_per_pair": args.top_k_per_pair,
            },
        )
        results.append(meta)
        print(f"OK {vid}: {meta['output_video']} ({meta['relation_count']} relations)")

    summary_path = run_dir / "reports" / "relation_viz.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
