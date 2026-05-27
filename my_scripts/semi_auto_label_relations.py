"""半自动关系候选生成 — 兼容 CLI 包装器。

实现已迁入 `src/vidvrd_auto/relations/clip_relation.py`。
本脚本保留命令行参数，供 `my_scripts/run_phase1.py` 或手工调试直接调用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vidvrd_auto.relations.clip_relation import run_clip_relation  # noqa: E402
from vidvrd_auto.utils.io import read_json  # noqa: E402

try:
    import config  # type: ignore
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config  # type: ignore


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Semi-auto relation labeling (wrapper -> vidvrd_auto)")
    ap.add_argument("--windows_json", type=str, required=True)
    ap.add_argument("--tracks_jsonl", type=str, required=True)
    ap.add_argument("--output_json", type=str, default="pred/relations_pred.json")
    ap.add_argument("--api_key", type=str, default="")
    ap.add_argument("--model_vl", type=str, default=str(getattr(config, "API_MODEL", "qwen-vl-max")))
    ap.add_argument("--group_size", type=int, default=3)
    ap.add_argument("--relations", type=str, default="")
    ap.add_argument("--max_windows", type=int, default=0)
    ap.add_argument("--max_frames_per_window", type=int, default=8)
    ap.add_argument("--vggsound_label", type=str, default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reset_progress", action="store_true")
    ap.add_argument("--save_storyboards_dir", type=str, default="")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--backoff_sec", type=float, default=1.5)
    ap.add_argument("--sleep_sec", type=float, default=0.0)
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    windows_path = Path(args.windows_json).expanduser().resolve()
    tracks_path = Path(args.tracks_jsonl).expanduser().resolve()
    out_path = Path(args.output_json).expanduser().resolve()

    windows_obj = read_json(windows_path)
    video_meta = windows_obj.get("video", {}) if isinstance(windows_obj, dict) else {}
    video_path = Path(str(video_meta.get("path", "") or "")).expanduser().resolve()
    video_id = video_path.stem if video_path.name else "unknown"

    storyboards_dir = (
        Path(args.save_storyboards_dir).expanduser().resolve()
        if str(args.save_storyboards_dir or "").strip()
        else out_path.parent / "storyboards"
    )
    log_path = out_path.parent / "semi_auto_cli.log"

    cfg = {
        "api_model": str(args.model_vl),
        "group_size": int(args.group_size),
        "relations": str(args.relations or ""),
        "max_windows": int(args.max_windows),
        "max_frames_per_window": int(args.max_frames_per_window),
        "vggsound_label": str(args.vggsound_label or ""),
        "retries": int(args.retries),
        "backoff_sec": float(args.backoff_sec),
        "sleep_sec": float(args.sleep_sec),
        "reset_progress": bool(args.reset_progress),
    }

    api_key = (args.api_key or "").strip() or str(getattr(config, "API_KEY", "") or "").strip()

    run_clip_relation(
        windows_json=windows_path,
        tracks_jsonl=tracks_path,
        out_json=out_path,
        storyboards_dir=storyboards_dir,
        config=cfg,
        api_key=api_key,
        resume=bool(args.resume),
        dry_run=bool(args.dry_run),
        video_id=video_id,
        log_path=log_path,
    )


if __name__ == "__main__":
    main()
