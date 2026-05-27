from __future__ import annotations

"""图像写入工具。Windows 下 cv2.imwrite 对非 ASCII 路径会失败，统一走 imencode + write_bytes。"""

from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore
    np = None  # type: ignore


def imwrite(path: Path, image_bgr) -> None:
    if cv2 is None or np is None:
        raise ImportError("opencv-python and numpy are required for imwrite")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        ext = ".jpg"
    elif suffix == ".png":
        ext = ".png"
    else:
        ext = ".jpg"
        if not path.suffix:
            path = path.with_suffix(ext)
    ok, buf = cv2.imencode(ext, image_bgr)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed: {path}")
    path.write_bytes(buf.tobytes())
