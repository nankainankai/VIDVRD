from __future__ import annotations

import argparse
from pathlib import Path

from vidvrd_auto.pipeline.constants import NODE_ORDER
from vidvrd_auto.pipeline.runner import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="VIDVRD OpenClaw-first full-auto labeling runner")
    ap.add_argument("--video", type=str, default="", help="Single video path or URL")
    ap.add_argument("--videos", type=str, default="", help="Video list file or comma-separated paths/URLs")
    ap.add_argument("--run_dir", type=str, required=True, help="Run output directory, e.g. runs/exp001")
    ap.add_argument("--config", type=str, default="configs/default.json", help="JSON config path")
    ap.add_argument("--api_key", type=str, default="", help="DashScope API key; fallback to DASHSCOPE_API_KEY")
    ap.add_argument("--resume", action="store_true", help="Skip succeeded nodes when input hash matches")
    ap.add_argument("--force", action="store_true", help="Re-run selected nodes even when cache is valid")
    ap.add_argument("--from_node", type=str, default="", choices=[""] + NODE_ORDER)
    ap.add_argument("--to_node", type=str, default="", choices=[""] + NODE_ORDER)
    ap.add_argument("--dry_run_relations", action="store_true", help="Only generate storyboards in relation_llm")
    ap.add_argument("--skip_eval", action="store_true", help="Skip presence evaluation")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve() if str(args.config or "").strip() else None
    run_pipeline(args=args, config_path=config_path)


if __name__ == "__main__":
    main()
