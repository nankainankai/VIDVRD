from __future__ import annotations

"""DashScope implementation of the visual-language provider."""

import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .base import VLResult, VLStats


def image_to_data_uri(path: Path) -> str:
    """Encode a local image as a data URI accepted by DashScope."""

    image_path = Path(path)
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    suffix = image_path.suffix.lower().lstrip(".") or "jpeg"
    if suffix == "jpg":
        suffix = "jpeg"
    return f"data:image/{suffix};base64,{data}"


def _response_text(response: Any) -> str:
    """Extract text from both known DashScope message content shapes."""

    content = response.output.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and "text" in item:
                return str(item.get("text", "")).strip()
    return str(content or "").strip()


class DashScopeProvider:
    """Qwen-VL client with retries, dry-run support, and local call stats."""

    def __init__(self, config: Dict[str, Any] | None = None, *, api_key: str = "") -> None:
        cfg = config or {}
        self.model = str(cfg.get("model", cfg.get("api_model", "qwen-vl-max")) or "qwen-vl-max")
        self.api_key_env = str(cfg.get("api_key_env", "DASHSCOPE_API_KEY") or "DASHSCOPE_API_KEY")
        self.api_key = (api_key or os.getenv(self.api_key_env, "") or "").strip()
        retries = cfg.get("retries", 2)
        backoff_sec = cfg.get("backoff_sec", 1.5)
        sleep_sec = cfg.get("sleep_sec", 0.0)
        self.retries = max(0, int(2 if retries is None else retries))
        self.backoff_sec = max(0.0, float(1.5 if backoff_sec is None else backoff_sec))
        self.sleep_sec = max(0.0, float(0.0 if sleep_sec is None else sleep_sec))
        self.dry_run = bool(cfg.get("dry_run", False))
        self.stats = VLStats()

    def call(
        self,
        *,
        prompt: str,
        image_paths: Iterable[Path] | None = None,
        dry_run: bool | None = None,
    ) -> VLResult:
        effective_dry_run = self.dry_run if dry_run is None else bool(dry_run)
        images = [Path(path) for path in (image_paths or [])]
        self.stats.calls += 1
        self.stats.images += len(images)

        if effective_dry_run:
            self.stats.succeeded += 1
            self.stats.dry_runs += 1
            return VLResult(
                ok=True,
                text='{"decision":"keep","reason":"dry_run"}',
                model=self.model,
                dry_run=True,
                attempts=0,
            )

        if not self.api_key:
            return self._failure("missing DASHSCOPE api key", attempts=0)

        try:
            import dashscope  # type: ignore
            from dashscope import MultiModalConversation  # type: ignore
        except Exception as exc:
            return self._failure(f"dashscope unavailable: {exc}", attempts=0)

        try:
            content: List[Dict[str, str]] = [
                {"image": image_to_data_uri(image_path)} for image_path in images
            ]
        except Exception as exc:
            return self._failure(f"image encoding failed: {exc}", attempts=0)
        content.append({"text": prompt})
        messages: List[Dict[str, Any]] = [{"role": "user", "content": content}]
        dashscope.api_key = self.api_key

        last_error = ""
        max_attempts = self.retries + 1
        for attempt in range(1, max_attempts + 1):
            self.stats.attempts += 1
            if attempt > 1:
                self.stats.retries += 1
            try:
                if self.sleep_sec > 0:
                    time.sleep(self.sleep_sec)
                response = MultiModalConversation.call(model=self.model, messages=messages)
                text = _response_text(response)
                self.stats.succeeded += 1
                return VLResult(ok=True, text=text, model=self.model, attempts=attempt)
            except Exception as exc:
                last_error = str(exc)
                if attempt < max_attempts and self.backoff_sec > 0:
                    time.sleep(self.backoff_sec * attempt)

        return self._failure(last_error, attempts=max_attempts)

    def _failure(self, error: str, *, attempts: int) -> VLResult:
        self.stats.failed += 1
        return VLResult(ok=False, text="", model=self.model, error=error, attempts=attempts)
