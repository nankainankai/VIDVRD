from __future__ import annotations

"""Small, reusable node executor for the resumable pipeline."""

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from vidvrd_auto.pipeline.manifest import mark_failed, mark_running, mark_succeeded, should_skip


@dataclass(frozen=True)
class NodeJob:
    """Everything needed to execute one pipeline node."""

    name: str
    input_hash: str
    required: Sequence[Path]
    outputs: Mapping[str, str]
    action: Callable[[], None]


def run_job(*, job: NodeJob, args: Namespace, video_dir: Path) -> bool:
    """Run one node and return ``True`` when work was performed."""

    if should_skip(
        resume=bool(args.resume),
        force=bool(args.force),
        video_dir=video_dir,
        node=job.name,
        input_hash=job.input_hash,
        required_outputs=job.required,
    ):
        print(f"SKIP {video_dir.name}/{job.name} (resume cache hit)")
        return False

    print(f"RUN {video_dir.name}/{job.name}")
    mark_running(video_dir, job.name, job.input_hash)
    try:
        job.action()
    except Exception as exc:
        mark_failed(video_dir, job.name, job.input_hash, str(exc))
        raise

    for path in job.required:
        if path.exists():
            continue
        error = f"required output missing: {path}"
        mark_failed(video_dir, job.name, job.input_hash, error)
        raise RuntimeError(error)

    mark_succeeded(video_dir, job.name, job.input_hash, dict(job.outputs))
    return True

