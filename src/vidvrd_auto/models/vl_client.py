from __future__ import annotations

"""统一视觉语言模型客户端。

该客户端集中处理模型名、API key、重试、dry-run 和错误格式。
节点调用它后可以获得结构化 `VLResult`，避免各节点重复写 DashScope 调用逻辑。
"""

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass
class VLResult:
    ok: bool
    text: str
    model: str
    dry_run: bool = False
    error: str = ""
    attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "text": self.text,
            "model": self.model,
            "dry_run": self.dry_run,
            "error": self.error,
            "attempts": self.attempts,
        }


def _image_to_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower().lstrip(".") or "jpeg"
    if suffix == "jpg":
        suffix = "jpeg"
    return f"data:image/{suffix};base64,{data}"


def _bgr_to_data_uri(image_bgr: Any, *, quality: int = 90) -> str:
    import cv2  # type: ignore

    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed for storyboard image")
    data = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def _extract_message_text(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        parts: List[str] = []
        for item in message_content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item.get("text") or ""))
                elif item.get("type") == "text":
                    parts.append(str(item.get("content") or item.get("text") or ""))
        return "".join(parts)
    return str(message_content or "")


class VLClient:
    def __init__(self, config: Dict[str, Any] | None = None, *, api_key: str = "") -> None:
        cfg = config or {}
        self.model = str(cfg.get("model", cfg.get("api_model", "qwen-vl-max")) or "qwen-vl-max")
        key_env = str(cfg.get("api_key_env", "DASHSCOPE_API_KEY") or "DASHSCOPE_API_KEY")
        self.api_key = (api_key or os.getenv(key_env, "") or "").strip()
        self.retries = int(cfg.get("retries", 2) or 2)
        self.backoff_sec = float(cfg.get("backoff_sec", 1.5) or 1.5)
        self.sleep_sec = float(cfg.get("sleep_sec", 0.0) or 0.0)
        self.dry_run = bool(cfg.get("dry_run", False))

    def call(self, *, prompt: str, image_paths: Iterable[Path] | None = None, dry_run: bool | None = None) -> VLResult:
        effective_dry_run = self.dry_run if dry_run is None else bool(dry_run)
        images = [Path(p) for p in (image_paths or [])]
        if effective_dry_run:
            return VLResult(ok=True, text='{"decision":"keep","reason":"dry_run"}', model=self.model, dry_run=True, attempts=0)
        if not self.api_key:
            return VLResult(ok=False, text="", model=self.model, error="missing DASHSCOPE api key", attempts=0)

        try:
            import dashscope  # type: ignore
            from dashscope import MultiModalConversation  # type: ignore
        except Exception as exc:
            return VLResult(ok=False, text="", model=self.model, error=f"dashscope unavailable: {exc}", attempts=0)

        dashscope.api_key = self.api_key
        messages: List[Dict[str, Any]] = [{"role": "user", "content": []}]
        for image_path in images:
            messages[0]["content"].append({"image": _image_to_data_uri(image_path)})
        messages[0]["content"].append({"text": prompt})

        last_error = ""
        for attempt in range(1, self.retries + 2):
            try:
                if self.sleep_sec > 0:
                    time.sleep(self.sleep_sec)
                resp = MultiModalConversation.call(model=self.model, messages=messages)
                text = _extract_message_text(resp.output.choices[0].message.content).strip()
                return VLResult(ok=True, text=text, model=self.model, attempts=attempt)
            except Exception as exc:
                last_error = str(exc)
                if attempt <= self.retries:
                    time.sleep(self.backoff_sec * attempt)
        return VLResult(ok=False, text="", model=self.model, error=last_error, attempts=self.retries + 1)

    def call_bgr(self, *, prompt: str, image_bgr: Any, dry_run: bool | None = None) -> VLResult:
        """对内存中的 BGR 拼图调用多模态模型（用于 relation_llm storyboard）。"""
        effective_dry_run = self.dry_run if dry_run is None else bool(dry_run)
        if effective_dry_run:
            return VLResult(ok=True, text='{"triples":[]}', model=self.model, dry_run=True, attempts=0)
        if not self.api_key:
            return VLResult(ok=False, text="", model=self.model, error="missing DASHSCOPE api key", attempts=0)

        try:
            import dashscope  # type: ignore
            from dashscope import MultiModalConversation  # type: ignore
        except Exception as exc:
            return VLResult(ok=False, text="", model=self.model, error=f"dashscope unavailable: {exc}", attempts=0)

        dashscope.api_key = self.api_key
        image_data = _bgr_to_data_uri(image_bgr)
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": [{"image": image_data}, {"text": prompt}]}
        ]

        last_error = ""
        for attempt in range(1, self.retries + 2):
            try:
                if self.sleep_sec > 0:
                    time.sleep(self.sleep_sec)
                resp = MultiModalConversation.call(model=self.model, messages=messages)
                text = _extract_message_text(resp.output.choices[0].message.content).strip()
                return VLResult(ok=True, text=text, model=self.model, attempts=attempt)
            except Exception as exc:
                last_error = str(exc)
                if attempt <= self.retries:
                    time.sleep(self.backoff_sec * attempt)
        return VLResult(ok=False, text="", model=self.model, error=last_error, attempts=self.retries + 1)
