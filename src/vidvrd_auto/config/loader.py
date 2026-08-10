from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from vidvrd_auto.core.config import AppConfig

from vidvrd_auto.utils.io import read_json
from vidvrd_auto.utils.paths import repo_root


RUN_MODES = frozenset({"reference_dense", "main"})


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: Path | None = None) -> Dict[str, Any]:
    default_path = repo_root() / "configs" / "base.json"
    cfg = read_json(default_path)
    if config_path is not None and config_path.exists() and config_path.resolve() != default_path.resolve():
        user_cfg = read_json(config_path)
        if not isinstance(user_cfg, dict):
            raise SystemExit(f"ERROR: config must be a JSON object: {config_path}")
        cfg = deep_merge(cfg, user_cfg)
    project = cfg.setdefault("project", {})
    if not isinstance(project, dict):
        raise SystemExit("ERROR: config project section must be a JSON object")
    mode = str(project.setdefault("run_mode", "main"))
    if mode not in RUN_MODES:
        raise SystemExit(f"ERROR: unsupported project.run_mode '{mode}'")
    project.setdefault("schema_version", "1.3")
    project.setdefault("artifact_span_convention", "inclusive")
    project.setdefault("canonical_span_convention", "half_open")
    project.setdefault("prompt_version", "main-v4-hierarchical-agent")
    return cfg


def load_app_config(config_path: Path | None = None) -> AppConfig:
    """Load the application configuration."""

    return AppConfig(load_config(config_path))
