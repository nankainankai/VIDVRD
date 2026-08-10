from __future__ import annotations

"""视频读入节点。

输入：本地视频路径、URL 或视频列表中的单条 source。
输出：统一落盘的视频文件和 `inputs/source.json`。
该节点不调用模型，是后续所有节点的可复现输入边界。
"""

import shutil
import urllib.request
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

from vidvrd_auto.pipeline.manifest import now_text
from vidvrd_auto.utils.hashing import sha256_file
from vidvrd_auto.utils.io import write_json
from vidvrd_auto.utils.paths import repo_root, safe_rel


def is_url(source: str) -> bool:
    return urlparse(str(source)).scheme.lower() in ("http", "https")


def source_name(source: str) -> str:
    if is_url(source):
        return Path(urlparse(source).path).stem or "video"
    return Path(source).stem or "video"


def video_id_for_source(source: str, used: set[str]) -> str:
    base = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in source_name(source)).strip("_") or "video"
    vid = base
    i = 2
    while vid in used:
        vid = f"{base}_{i}"
        i += 1
    used.add(vid)
    return vid


def planned_video_path(source: str, inputs_dir: Path) -> Path:
    if is_url(source):
        suffix = Path(urlparse(source).path).suffix or ".mp4"
        return inputs_dir / f"source_video{suffix}"
    p = Path(source).expanduser()
    return (repo_root() / p).resolve() if not p.is_absolute() else p.resolve()


def source_fingerprint(source: str, inputs_dir: Path) -> Dict[str, Any]:
    """Build the cache identity available before the ingest stage runs.

    Local sources include the complete file digest, so replacing a video at
    the same path invalidates ``--resume``.  A cached URL can only be
    fingerprinted from its local materialization without making an additional
    network request.
    """

    video_path = planned_video_path(source, inputs_dir)
    identity: Dict[str, Any] = {
        "source": str(source),
        "source_type": "url" if is_url(source) else "local",
        "video_path": str(video_path),
        "exists": video_path.exists(),
    }
    if video_path.exists() and video_path.is_file():
        identity["file_size"] = int(video_path.stat().st_size)
        identity["file_hash"] = sha256_file(video_path)
    return identity


def materialize_video(source: str, inputs_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    inputs_dir.mkdir(parents=True, exist_ok=True)
    video_path = planned_video_path(source, inputs_dir)
    meta: Dict[str, Any] = {
        "source": source,
        "source_type": "url" if is_url(source) else "local",
        "video_path": str(video_path),
        "video_path_rel": safe_rel(video_path),
        "materialized_at": now_text(),
    }
    if is_url(source):
        if not video_path.exists() or bool(config.get("overwrite_download", False)):
            timeout = float(config.get("download_timeout_sec", 120))
            with urllib.request.urlopen(source, timeout=timeout) as resp:
                video_path.parent.mkdir(parents=True, exist_ok=True)
                with video_path.open("wb") as f:
                    shutil.copyfileobj(resp, f)
        meta["downloaded"] = True
    else:
        if not video_path.exists():
            raise FileNotFoundError(f"video not found: {video_path}")
        meta["downloaded"] = False
    meta["exists"] = video_path.exists()
    meta["file_size"] = int(video_path.stat().st_size) if video_path.exists() else 0
    meta["file_hash"] = sha256_file(video_path) if video_path.exists() else ""
    write_json(inputs_dir / "source.json", meta)
    return meta
