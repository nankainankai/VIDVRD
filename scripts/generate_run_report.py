from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vidvrd_auto.pipeline.report import write_run_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="从 run_manifest.json 生成运行报告")
    ap.add_argument("--run_dir", type=str, required=True, help="runs/<run_id>")
    ap.add_argument("--out", type=str, default="", help="输出 md 路径，默认 runs/<id>/reports/run_report.md")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"ERROR: manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = Path(args.out).expanduser() if args.out else run_dir / "reports" / "run_report.md"
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    path = write_run_report(manifest, out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
