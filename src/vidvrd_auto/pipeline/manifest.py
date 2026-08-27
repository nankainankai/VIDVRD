from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Sequence

from vidvrd_auto.utils.io import read_json, write_json
from vidvrd_auto.utils.hashing import sha256_file, stable_hash


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def git_revision(root: Path) -> str:
    """Read the current Git revision without invoking a subprocess."""

    git_dir = root / ".git"
    if git_dir.is_file():
        try:
            marker = git_dir.read_text(encoding="utf-8").strip()
            if marker.startswith("gitdir:"):
                target = Path(marker.split(":", 1)[1].strip())
                git_dir = target if target.is_absolute() else (root / target).resolve()
        except OSError:
            return "unknown"
    head = git_dir / "HEAD"
    try:
        value = head.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"
    if not value.startswith("ref:"):
        return value or "unknown"
    ref_name = value.split(":", 1)[1].strip()
    ref_path = git_dir / ref_name
    try:
        return ref_path.read_text(encoding="utf-8").strip() or "unknown"
    except OSError:
        pass
    try:
        for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and line.endswith(f" {ref_name}"):
                return line.split(" ", 1)[0]
    except OSError:
        pass
    return "unknown"


def source_tree_fingerprint(root: Path) -> str:
    """Hash project-owned source files so dirty worktrees stay reproducible."""

    records = []
    source_root = root / "src"
    if source_root.exists():
        for path in sorted(source_root.rglob("*.py")):
            if path.is_file():
                records.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    return stable_hash(records)


def build_run_provenance(*, root: Path, config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    """Describe the effective implementation, models and serialization rules."""

    project = config.get("project", {}) if isinstance(config.get("project"), dict) else {}
    detector = config.get("detector", {}) if isinstance(config.get("detector"), dict) else {}
    vocabulary = config.get("vocabulary", {}) if isinstance(config.get("vocabulary"), dict) else {}
    relations = config.get("relations", {}) if isinstance(config.get("relations"), dict) else {}
    verify = config.get("relation_verify", {}) if isinstance(config.get("relation_verify"), dict) else {}
    tracking = config.get("tracking", {}) if isinstance(config.get("tracking"), dict) else {}
    evaluation = config.get("evaluate", {}) if isinstance(config.get("evaluate"), dict) else {}
    return {
        "run_mode": str(project.get("run_mode", "main")),
        "schema_version": str(project.get("schema_version", "1.3")),
        "artifact_span_convention": str(project.get("artifact_span_convention", "inclusive")),
        "canonical_span_convention": str(project.get("canonical_span_convention", "half_open")),
        "prompt_version": str(project.get("prompt_version", "main-v7-batched-direction-schema")),
        "code_revision": git_revision(root),
        "code_fingerprint": source_tree_fingerprint(root),
        "effective_config_hash": stable_hash(config),
        "config_file": str(config_path),
        "config_file_hash": sha256_file(config_path) if config_path.exists() else "",
        "algorithms": {
            "detector": {
                "name": "rex_omni",
                "backend": str(detector.get("rex_backend", "transformers")),
                "model": str(detector.get("rex_model_path", "")),
                "sampling": str(detector.get("sampling_mode", "adaptive_sparse")),
            },
            "tracker": {
                "name": str(tracking.get("algorithm", "sparse_ocsort")),
                "upstream": "oc_sort",
                "time_unit": "detector_anchor"
                if str(tracking.get("algorithm")) == "sparse_ocsort"
                else "video_frame",
                "adapter": "project_io_adapter",
                "offline_stitching": False,
            },
            "vocabulary_agent": str(vocabulary.get("discovery_model", "")),
            "relation_agent": str(relations.get("api_model", "")),
            "agent_policy": {
                "name": "bounded_batched_agent_v3",
                "candidate_policy": "hierarchical_predicate_v1",
                "evidence_mode": "event_burst_dual_view",
                "candidate_limit": int(relations.get("candidate_limit", 14) or 14),
                "expanded_candidate_limit": int(relations.get("expanded_candidate_limit", 24) or 24),
                "batch_windows_per_call": int(relations.get("batch_windows_per_call", 6) or 6),
                "max_supplemental_calls": 1 if bool(relations.get("allow_request_more_frames", True)) else 0,
                "max_additional_frames": int(relations.get("max_additional_frames", 4) or 0),
                "allowed_external_mutations": [],
            },
            "review_agent": str(verify.get("strong_model", "")),
            "official_evaluator": {
                "name": "imagenet_vidvrd_official_2017_compatible_v1",
                "viou_threshold": float(evaluation.get("official_viou_threshold", 0.5) or 0.5),
            },
            "diagnostic_evaluator": "diagnostic_track_aligned_v2",
        },
    }


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
