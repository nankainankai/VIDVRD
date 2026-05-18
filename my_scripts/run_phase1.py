from __future__ import annotations

import argparse
import os
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _here() -> Path:
    return Path(__file__).resolve().parent


def _repo_root_from_here() -> Path:
    """Find workspace-ish root so defaults like gold/ pred/ work."""

    here = _here()
    for p in [here] + list(here.parents):
        if (p / "tools").exists() and (p / "gold").exists() and (p / "pred").exists():
            return p
    # fallback: assume VIDVRD/VIDVRD is the project root
    return here.parent


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
    return (os.getenv("DASHSCOPE_API_KEY", "") or "").strip()


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="One-click Phase-1 semi-auto labeling (Step2 -> candidates -> presence eval)")

    ap.add_argument("--video", type=str, default="", help="视频路径；不传则 Step2 会自动推断/弹窗")
    ap.add_argument("--output_dir", type=str, default="", help="输出目录；不传则使用 config.py 的 OUTPUT_DIR")
    ap.add_argument("--api_key", type=str, default="", help="DashScope API key（不传则读环境变量 DASHSCOPE_API_KEY）")

    ap.add_argument("--windows_json", type=str, default="", help="已有 windows.json（传了就不会再跑 Step2）")
    ap.add_argument("--tracks_jsonl", type=str, default="", help="已有 tracks_full.jsonl（传了就不会再跑 Step2）")
    ap.add_argument("--skip_step2", action="store_true", help="强制跳过 Step2（需要你自己提供 windows/tracks）")
    ap.add_argument("--force_step2", action="store_true", help="强制重跑 Step2（覆盖已有 windows/tracks）")

    ap.add_argument("--pred_json", type=str, default="", help="输出 Pred JSON（默认 pred/relations_pred.json）")
    ap.add_argument("--save_storyboards_dir", type=str, default="", help="保存 storyboard 证据图目录（可选）")
    ap.add_argument("--resume", action="store_true", help="断点续跑")
    ap.add_argument("--reset_progress", action="store_true", help="重置半自动断点文件（忽略并覆盖 .progress.json）")
    ap.add_argument("--dry_run", action="store_true", help="只生成 storyboard，不调用模型")
    ap.add_argument("--relations", type=str, default="", help="谓词列表(逗号分隔)，不传则用默认空间关系集合")
    ap.add_argument("--group_size", type=int, default=3, help="每次问模型的谓词数量")
    ap.add_argument("--max_windows", type=int, default=0, help="最多处理多少个 window（0=全部）")
    ap.add_argument("--max_frames_per_window", type=int, default=8, help="每个窗口抽帧上限")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--backoff_sec", type=float, default=1.5)
    ap.add_argument("--sleep_sec", type=float, default=0.0)

    ap.add_argument("--gold_json", type=str, default="", help="Gold JSON（默认 gold/relations_gold.json；存在则自动评测）")
    ap.add_argument("--report", type=str, default="", help="Presence 评测报告 markdown 输出路径")
    ap.add_argument("--skip_eval", action="store_true", help="跳过 presence 评测")

    return ap


def _windows_video_path_missing(windows_json_path: Path) -> bool:
    """Return True if windows.json exists but recorded video.path is missing on disk."""

    try:
        with windows_json_path.open("r", encoding="utf-8-sig") as f:
            obj = json.load(f)
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    video = obj.get("video", {})
    if not isinstance(video, dict):
        return False
    vp = str(video.get("path", "") or "").strip()
    if not vp:
        return False
    return not Path(vp).expanduser().exists()


def main() -> None:
    args = _build_parser().parse_args()

    base = _here()
    repo_root = _repo_root_from_here()

    step2 = base / "step2_full_video_tracking_ocsort_qc_pairviz.py"
    semi = base / "semi_auto_label_relations.py"
    eval_script = repo_root / "tools" / "evaluate_presence.py"

    if not step2.exists():
        raise SystemExit(f"ERROR: missing Step2 script: {step2}")
    if not semi.exists():
        raise SystemExit(f"ERROR: missing semi-auto script: {semi}")
    if not eval_script.exists():
        print(f"WARN: presence evaluator not found: {eval_script} (will skip eval)")

    api_key = _resolve_api_key(str(args.api_key))

    out_dir_arg = str(args.output_dir or "").strip() or None
    video_arg = str(args.video or "").strip() or None

    # Resolve Step2 outputs
    windows_json = str(args.windows_json or "").strip() or None
    tracks_jsonl = str(args.tracks_jsonl or "").strip() or None

    # Try infer from output_dir if not explicitly provided
    if out_dir_arg and (not windows_json):
        cand = Path(out_dir_arg).expanduser().resolve() / "windows.json"
        if cand.exists():
            windows_json = str(cand)
    if out_dir_arg and (not tracks_jsonl):
        cand = Path(out_dir_arg).expanduser().resolve() / "tracks_full.jsonl"
        if cand.exists():
            tracks_jsonl = str(cand)

    if bool(args.force_step2):
        need_step2 = not bool(args.skip_step2)
    else:
        need_step2 = (not args.skip_step2) and (not windows_json or not tracks_jsonl)

    # If windows.json is present but points to a missing video file, re-run Step2.
    if (not bool(args.skip_step2)) and (not bool(args.force_step2)) and windows_json:
        wpath = Path(windows_json).expanduser().resolve()
        if wpath.exists() and _windows_video_path_missing(wpath):
            print("=" * 70)
            print("WARN: windows.json references missing video.path; re-running Step2 to regenerate")
            print(f"windows_json={wpath}")
            print("TIP: You can also pass --force_step2 explicitly.")
            print("=" * 70)
            need_step2 = True

    if need_step2:
        cmd2: List[str] = [_py(), str(step2)]
        if video_arg:
            cmd2 += ["--video", video_arg]
        if out_dir_arg:
            cmd2 += ["--output_dir", out_dir_arg]
        if api_key:
            # Step2 uses api_key only for optional QC
            cmd2 += ["--api_key", api_key]
        _run(cmd2)

        # After Step2, infer paths
        if not out_dir_arg:
            # Step2 default output_dir comes from config.py; best effort: look for C:\video_output2
            # User can always pass --output_dir to be explicit.
            default_out = Path("C:/video_output2")
            if default_out.exists():
                out_dir_arg = str(default_out)

        if not out_dir_arg:
            raise SystemExit("ERROR: output_dir unknown. Please pass --output_dir.")

        out_dir = Path(out_dir_arg).expanduser().resolve()
        windows_path = out_dir / "windows.json"
        tracks_path = out_dir / "tracks_full.jsonl"
        if not windows_path.exists() or not tracks_path.exists():
            raise SystemExit(f"ERROR: Step2 outputs not found under {out_dir}.")

        windows_json = str(windows_path)
        tracks_jsonl = str(tracks_path)

    if not windows_json or not tracks_jsonl:
        raise SystemExit("ERROR: missing windows_json/tracks_jsonl. Provide them or run without --skip_step2.")

    # Pred output default
    pred_json = str(args.pred_json or "").strip()
    if not pred_json:
        pred_json = str((repo_root / "pred" / "relations_pred.json").resolve())

    # storyboards default: output_dir/storyboards if output_dir known
    save_storyboards_dir = str(args.save_storyboards_dir or "").strip()
    if (not save_storyboards_dir) and out_dir_arg:
        save_storyboards_dir = str((Path(out_dir_arg).expanduser().resolve() / "storyboards").resolve())

    # ---- semi auto candidates ----
    cmd_semi: List[str] = [_py(), str(semi), "--windows_json", windows_json, "--tracks_jsonl", tracks_jsonl, "--output_json", pred_json]

    if api_key:
        cmd_semi += ["--api_key", api_key]

    if bool(args.resume):
        cmd_semi += ["--resume"]
    if bool(args.reset_progress):
        cmd_semi += ["--reset_progress"]
    if bool(args.dry_run):
        cmd_semi += ["--dry_run"]

    if save_storyboards_dir:
        cmd_semi += ["--save_storyboards_dir", save_storyboards_dir]

    if str(args.relations or "").strip():
        cmd_semi += ["--relations", str(args.relations).strip()]

    cmd_semi += ["--group_size", str(int(args.group_size))]
    cmd_semi += ["--max_windows", str(int(args.max_windows))]
    cmd_semi += ["--max_frames_per_window", str(int(args.max_frames_per_window))]
    cmd_semi += ["--retries", str(int(args.retries))]
    cmd_semi += ["--backoff_sec", str(float(args.backoff_sec))]
    cmd_semi += ["--sleep_sec", str(float(args.sleep_sec))]

    _run(cmd_semi)

    # ---- presence eval (optional) ----
    if bool(args.skip_eval) or (not eval_script.exists()):
        return

    gold_json = str(args.gold_json or "").strip()
    if not gold_json:
        gold_json = str((repo_root / "gold" / "relations_gold.json").resolve())

    gold_path = Path(gold_json).expanduser().resolve()
    pred_path = Path(pred_json).expanduser().resolve()

    if not gold_path.exists():
        print("=" * 70)
        print("SKIP: presence eval (gold not found)")
        print(f"- expected: {gold_path}")
        print("=" * 70)
        return

    # If gold exists but is empty, skip to avoid confusing all-zeros report.
    try:
        with gold_path.open("r", encoding="utf-8-sig") as f:
            gold_obj = json.load(f)
        if isinstance(gold_obj, dict) and len(gold_obj) == 0:
            print("=" * 70)
            print("SKIP: presence eval (gold is empty {})")
            print(f"- gold: {gold_path}")
            print("TIP: Fill gold/relations_gold.json or pass a non-empty --gold_json.")
            print("=" * 70)
            return
    except Exception:
        # If gold can't be parsed, let evaluator report it.
        pass
    if not pred_path.exists():
        print("=" * 70)
        print("SKIP: presence eval (pred not found)")
        print(f"- expected: {pred_path}")
        print("=" * 70)
        return

    report = str(args.report or "").strip()
    if not report:
        if out_dir_arg:
            report = str((Path(out_dir_arg).expanduser().resolve() / "presence_report.md").resolve())
        else:
            report = str((repo_root / "presence_report.md").resolve())

    cmd_eval: List[str] = [_py(), str(eval_script), "--gold", str(gold_path), "--pred", str(pred_path), "--report", report]
    _run(cmd_eval)


if __name__ == "__main__":
    main()
