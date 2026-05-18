from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Sequence

from vidvrd_auto.utils.io import read_json, write_json


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def status_path(video_dir: Path, node: str) -> Path:
    return video_dir / node / "status.json"


def load_status(video_dir: Path, node: str) -> Dict[str, Any]:
    p = status_path(video_dir, node)
    if not p.exists():
        return {}
    try:
        obj = read_json(p)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def write_status(video_dir: Path, node: str, status: Dict[str, Any]) -> None:
    write_json(status_path(video_dir, node), status)


def should_skip(
    *,
    resume: bool,
    force: bool,
    video_dir: Path,
    node: str,
    input_hash: str,
    required_outputs: Sequence[Path],
) -> bool:
    if force or not resume:
        return False
    st = load_status(video_dir, node)
    if st.get("state") != "succeeded" or st.get("input_hash") != input_hash:
        return False
    return all(p.exists() for p in required_outputs)


def mark_running(video_dir: Path, node: str, input_hash: str) -> None:
    write_status(video_dir, node, {"node": node, "state": "running", "input_hash": input_hash, "started_at": now_text()})


def mark_succeeded(video_dir: Path, node: str, input_hash: str, outputs: Dict[str, str]) -> None:
    write_status(
        video_dir,
        node,
        {"node": node, "state": "succeeded", "input_hash": input_hash, "finished_at": now_text(), "outputs": outputs},
    )


def mark_failed(video_dir: Path, node: str, input_hash: str, error: str) -> None:
    write_status(
        video_dir,
        node,
        {"node": node, "state": "failed", "input_hash": input_hash, "finished_at": now_text(), "error": error},
    )


def collect_node_statuses(video_dir: Path, nodes: Sequence[str]) -> Dict[str, Any]:
    return {node: load_status(video_dir, node) for node in nodes}
