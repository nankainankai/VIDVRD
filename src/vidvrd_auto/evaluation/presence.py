from __future__ import annotations

from pathlib import Path

from vidvrd_auto.utils.paths import repo_root
from vidvrd_auto.utils.process import python_executable, run_cmd


def run_presence_eval(*, gold_json: Path, pred_json: Path, report_path: Path, log_path: Path) -> None:
    cmd = [
        python_executable(),
        str(repo_root() / "tools" / "evaluate_presence.py"),
        "--gold",
        str(gold_json),
        "--pred",
        str(pred_json),
        "--report",
        str(report_path),
    ]
    run_cmd(cmd, cwd=repo_root(), log_path=log_path)
