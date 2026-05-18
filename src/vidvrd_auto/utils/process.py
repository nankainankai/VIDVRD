from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


def python_executable() -> str:
    return sys.executable


def run_cmd(cmd: Sequence[str], *, cwd: Path, log_path: Path, env: Mapping[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("RUN: " + " ".join(str(x) for x in cmd) + "\n")
        log.write("CWD: " + str(cwd) + "\n")
        log.write("=" * 80 + "\n")
        log.flush()
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(env) if env is not None else None,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit_code={proc.returncode}; see {log_path}")
