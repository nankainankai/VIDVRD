from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _here() -> Path:
    return Path(__file__).resolve().parent


def _py() -> str:
    return sys.executable


def _run(cmd: List[str]) -> None:
    print("=" * 70)
    print("RUN:", " ".join(cmd))
    print("=" * 70)
    subprocess.check_call(cmd)


def _resolve_api_key(cli_api_key: str) -> str:
    api_key = (cli_api_key or "").strip()
    if api_key:
        return api_key
    # DashScope key is usually in env
    return (os.getenv("DASHSCOPE_API_KEY", "") or "").strip()


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="One-click pipeline runner (Step1~Step5)")

    ap.add_argument("--video", type=str, default="", help="视频路径；不传则 Step1 会弹窗选择")
    ap.add_argument("--output_dir", type=str, default="", help="输出目录；不传则使用各 step 的默认值(config.py)")

    # Step1 (detector)
    ap.add_argument("--backend", type=str, default="rexomni", choices=["dinox", "rexomni"], help="检测后端")
    ap.add_argument("--rex_model_path", type=str, default="", help="Rex-Omni 本地模型目录（或 repo id）")
    ap.add_argument("--rex_categories", type=str, default="person", help="Rex-Omni categories，逗号分隔")
    ap.add_argument("--keyframe_interval", type=int, default=25, help="每 N 帧检测一次（中间插值补全）")
    ap.add_argument("--interp_iou_thresh", type=float, default=0.1, help="插值匹配 IoU 阈值")
    ap.add_argument("--no_save_box_video", action="store_true", help="不导出 Step1 可视化视频（更快）")

    # API key for LLM steps (Step2 QC / Step3 / Step5)
    ap.add_argument("--api_key", type=str, default="", help="DashScope API Key（不传则读环境变量 DASHSCOPE_API_KEY）")

    return ap


def main() -> None:
    args = _build_parser().parse_args()

    base = _here()
    step1 = base / "step1_full_video_box_detection_dinox.py"
    step2 = base / "step2_full_video_tracking_ocsort_qc_pairviz.py"
    step3 = base / "step3_window_relation_classification.py"
    step4 = base / "step4_video_relation_event_aggregation.py"
    step5 = base / "step5_video_relation_natural_language_qwen.py"

    # basic sanity
    for p in [step1, step2, step3, step4, step5]:
        if not p.exists():
            raise SystemExit(f"ERROR: missing script: {p}")

    api_key = _resolve_api_key(str(args.api_key))

    out_dir: Optional[str] = str(args.output_dir or "").strip() or None
    video: Optional[str] = str(args.video or "").strip() or None

    # ---- Step1 ----
    cmd1: List[str] = [_py(), str(step1)]
    if video:
        cmd1 += ["--video", video]
    if out_dir:
        cmd1 += ["--output_dir", out_dir]

    cmd1 += ["--backend", str(args.backend)]

    if str(args.backend).strip().lower() == "rexomni":
        if str(args.rex_model_path).strip():
            cmd1 += ["--rex_model_path", str(args.rex_model_path).strip()]
        if str(args.rex_categories).strip():
            cmd1 += ["--rex_categories", str(args.rex_categories).strip()]

    # speed knobs
    if int(args.keyframe_interval) > 1:
        cmd1 += ["--keyframe_interval", str(int(args.keyframe_interval))]
        cmd1 += ["--interp_iou_thresh", str(float(args.interp_iou_thresh))]
    if bool(args.no_save_box_video):
        cmd1 += ["--no_save_box_video"]

    _run(cmd1)

    # ---- Step2 ----
    cmd2: List[str] = [_py(), str(step2)]
    if video:
        cmd2 += ["--video", video]
    if out_dir:
        cmd2 += ["--output_dir", out_dir]
    if api_key:
        cmd2 += ["--api_key", api_key]

    _run(cmd2)

    # ---- Step3~5 (optional, requires api_key) ----
    if not api_key:
        print("=" * 70)
        print("SKIP: Step3~Step5 (missing DashScope API key)")
        print("- Set env DASHSCOPE_API_KEY, or pass --api_key to run_all.py")
        print("=" * 70)
        return

    cmd3: List[str] = [_py(), str(step3), "--api_key", api_key]
    if out_dir:
        # Keep Step3 inputs/outputs under the same output_dir.
        cmd3 += ["--output_dir", out_dir]
    _run(cmd3)

    cmd4: List[str] = [_py(), str(step4)]
    if out_dir:
        cmd4 += ["--relations_final", str((Path(out_dir) / "relations_final.json").resolve())]
    _run(cmd4)

    cmd5: List[str] = [_py(), str(step5), "--api_key", api_key]
    if out_dir:
        cmd5 += ["--video_relations_json", str((Path(out_dir) / "video_relations.json").resolve())]
    _run(cmd5)


if __name__ == "__main__":
    main()
