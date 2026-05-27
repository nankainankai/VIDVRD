from __future__ import annotations

"""生成 smoke test 用的短视频 data/validation_dummy.mp4。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "validation_dummy.mp4"


def main() -> None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("ERROR: need opencv-python and numpy: pip install opencv-python numpy")
        sys.exit(1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    w, h, fps, n = 320, 240, 10, 40
    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (30, 30, 30)
        cv2.rectangle(frame, (40 + i % 8, 50), (120 + i % 8, 200), (80, 180, 255), -1)
        cv2.rectangle(frame, (180, 60), (280, 190), (255, 160, 80), -1)
        writer.write(frame)
    writer.release()
    print(f"Wrote {OUT} ({n} frames @ {fps} fps)")


if __name__ == "__main__":
    main()
