from __future__ import annotations

"""Shared types for visual-language providers."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Protocol, runtime_checkable


@dataclass
class VLResult:
    """Normalized result returned by every visual-language provider."""

    ok: bool
    text: str
    model: str
    dry_run: bool = False
    error: str = ""
    attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VLStats:
    """Small, provider-local usage summary for logging and run reports."""

    calls: int = 0
    succeeded: int = 0
    failed: int = 0
    dry_runs: int = 0
    attempts: int = 0
    retries: int = 0
    images: int = 0

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@runtime_checkable
class VLProvider(Protocol):
    """Interface implemented by visual-language service providers."""

    model: str
    stats: VLStats

    def call(
        self,
        *,
        prompt: str,
        image_paths: Iterable[Path] | None = None,
        dry_run: bool | None = None,
    ) -> VLResult:
        ...
