"""Explicit runtime dependencies passed between pipeline stages."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from .config import AppConfig, ConfigSection
from .paths import VideoPaths

_API_KEY_ENV = "DASHSCOPE_API_KEY"


def _default_dotenv_path() -> Path:
    return Path(__file__).resolve().parents[3] / ".env"


def _read_dotenv_value(path: Path, key: str) -> str:
    if not path.is_file():
        return ""
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() != key:
            continue
        return value.strip().strip("'").strip('"')
    return ""


@dataclass(frozen=True)
class Secrets:
    dashscope_api_key: str = ""

    @classmethod
    def from_env(
        cls,
        *,
        dashscope_api_key: str | None = None,
        dotenv_path: Path | None = None,
    ) -> Secrets:
        explicit = (dashscope_api_key or "").strip()
        if explicit:
            return cls(dashscope_api_key=explicit)
        from_environment = os.getenv(_API_KEY_ENV, "").strip()
        if from_environment:
            return cls(dashscope_api_key=from_environment)
        from_file = _read_dotenv_value(dotenv_path or _default_dotenv_path(), _API_KEY_ENV)
        return cls(dashscope_api_key=from_file)

    @property
    def has_dashscope(self) -> bool:
        return bool(self.dashscope_api_key)


@dataclass(frozen=True)
class RunContext:
    config: AppConfig
    secrets: Secrets
    paths: VideoPaths
    video_id: str = ""
    source: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.config, AppConfig):
            raise TypeError("config must be an AppConfig")
        if not isinstance(self.secrets, Secrets):
            raise TypeError("secrets must be Secrets")
        if not isinstance(self.paths, VideoPaths):
            raise TypeError("paths must be VideoPaths")
        video_id = str(self.video_id or self.paths.video_id).strip()
        if video_id != self.paths.video_id:
            raise ValueError("RunContext video_id must match paths.video_id")
        object.__setattr__(self, "video_id", video_id)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "data", dict(self.data))

    @property
    def api_key(self) -> str:
        """Compatibility alias for the current DashScope credential."""

        return self.secrets.dashscope_api_key

    def section(self, name: str) -> ConfigSection:
        return self.config.section(name)

    def with_video(
        self,
        video_id: str,
        *,
        source: str = "",
        run_dir: Path | str | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> RunContext:
        paths = VideoPaths.for_video(
            run_dir or self.paths.run_dir,
            video_id,
            repo_dir=self.paths.repo_dir,
        )
        return replace(
            self,
            paths=paths,
            video_id=video_id,
            source=source,
            data=dict(data or {}),
        )
