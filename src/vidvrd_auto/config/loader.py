from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from vidvrd_auto.core.config import AppConfig

from vidvrd_auto.utils.io import read_json
from vidvrd_auto.utils.paths import repo_root


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: Path | None = None) -> Dict[str, Any]:
    default_path = repo_root() / "configs" / "config.json"
    cfg = read_json(default_path)
    if config_path is not None and config_path.exists() and config_path.resolve() != default_path.resolve():
        user_cfg = read_json(config_path)
        if not isinstance(user_cfg, dict):
            raise SystemExit(f"ERROR: config must be a JSON object: {config_path}")
        cfg = deep_merge(cfg, user_cfg)
    return cfg


def load_app_config(config_path: Path | None = None) -> AppConfig:
    """Load the application configuration."""

    return AppConfig(load_config(config_path))
