from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_read_json(path: Path) -> Any:
    """Read JSON with UTF-8 BOM tolerance (utf-8-sig)."""

    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def safe_write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=int(indent))


def safe_write_text(path: Path, text: str) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Iterate dict records in JSONL file, skipping invalid lines.

    Uses utf-8-sig to tolerate BOM on Windows.
    """

    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def try_read_json(path: Path) -> Optional[Any]:
    try:
        return safe_read_json(path)
    except Exception:
        return None
