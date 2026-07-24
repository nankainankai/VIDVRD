from __future__ import annotations

import argparse
from pathlib import Path

from vidvrd_auto.pipeline.run import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Automatic video relation detection pipeline")
    ap.add_argument("--video", type=str, default="", help="Single video path or URL")
    ap.add_argument("--videos", type=str, default="", help="Video list file or comma-separated paths/URLs")
    ap.add_argument("--run-dir", dest="run_dir", type=str, required=True, help="Run output directory")
    ap.add_argument("--config", type=str, default="configs/config.json", help="JSON config path")
    ap.add_argument("--api-key", dest="api_key", type=str, default="", help="DashScope API key")
    ap.add_argument("--resume", action="store_true", help="Skip succeeded nodes when input hash matches")
    ap.add_argument("--force", action="store_true", help="Re-run selected nodes even when cache is valid")
    ap.add_argument("--dry-run", dest="dry_run_relations", action="store_true", help="Generate semantic storyboards without cloud calls")
    ap.add_argument("--skip-eval", dest="skip_eval", action="store_true", help="Skip VidVRD evaluation")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve() if str(args.config or "").strip() else None
    run_pipeline(args=args, config_path=config_path)


if __name__ == "__main__":
    main()
