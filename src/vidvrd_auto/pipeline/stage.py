from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

from vidvrd_auto.core import VideoPaths
from vidvrd_auto.pipeline.node import NodeJob, run_job
from vidvrd_auto.utils.hashing import stable_hash
from vidvrd_auto.utils.paths import safe_rel


class StageRunner:
    """Execute selected pipeline stages with uniform state handling."""

    def __init__(self, args: Namespace, paths: VideoPaths) -> None:
        self.args = args
        self.paths = paths

    def run(self, stage: str, inputs: Dict[str, Any], outputs: Sequence[Path], action: Callable[[], None]) -> None:
        run_job(
            job=NodeJob(
                stage,
                stable_hash({"stage": stage, **inputs}),
                outputs,
                {path.name: safe_rel(path) for path in outputs},
                action,
            ),
            args=self.args,
            video_dir=self.paths.video_dir,
        )
