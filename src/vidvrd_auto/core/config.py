"""Typed wrappers for the project's JSON configuration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from typing import Any, TypeVar, overload


T = TypeVar("T")


def _wrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return ConfigSection(value)
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, ConfigSection):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return deepcopy(value)


class ConfigSection(Mapping[str, Any]):
    """Read-only mapping with optional attribute access for section values."""

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        if values is not None and not isinstance(values, Mapping):
            raise TypeError("configuration section must be a mapping")
        self._values = {str(key): _plain(value) for key, value in (values or {}).items()}

    def __getitem__(self, key: str) -> Any:
        return _wrap(self._values[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @overload
    def get(self, key: str) -> Any | None: ...

    @overload
    def get(self, key: str, default: T) -> Any | T: ...

    def get(self, key: str, default: T | None = None) -> Any | T | None:
        try:
            return self[key]
        except KeyError:
            return default

    def section(self, name: str) -> ConfigSection:
        value = self.get(name)
        if value is None:
            return ConfigSection()
        if not isinstance(value, ConfigSection):
            raise TypeError(f"configuration value '{name}' is not a section")
        return value

    def to_dict(self) -> dict[str, Any]:
        """Return a detached mutable dictionary."""

        return {key: _plain(value) for key, value in self._values.items()}


class AppConfig(ConfigSection):
    """Root application configuration.

    It behaves like a mapping while making common stage access concise, for
    example both ``config["detector"]["batch_size"]`` and
    ``config.detector.batch_size`` are supported.
    """

    @property
    def detector(self) -> ConfigSection:
        return self.section("detector")

    @property
    def tracking(self) -> ConfigSection:
        return self.section("tracking")

    @property
    def relations(self) -> ConfigSection:
        return self.section("relations")
