from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_hash(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sha256_file(path: Path, limit_bytes: int | None = None) -> str:
    """Return a SHA-256 digest for a file.

    The previous default stopped after 16 MiB, so two large videos with the
    same prefix could share a cache identity.  The default now hashes the full
    file.  ``limit_bytes`` remains available only for explicitly requested
    diagnostic sampling.
    """

    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            remaining = None if limit_bytes is None else max(0, int(limit_bytes))
            while remaining is None or remaining > 0:
                size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
                chunk = f.read(size)
                if not chunk:
                    break
                h.update(chunk)
                if remaining is not None:
                    remaining -= len(chunk)
    except Exception:
        return ""
    return h.hexdigest()
