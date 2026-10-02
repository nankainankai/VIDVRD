from __future__ import annotations

"""Materialize immutable development/test input lists from the local 50-video pack."""

import argparse
from pathlib import Path


def _ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-videos", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    videos = args.dataset_videos.resolve()
    for split in ("dev", "test"):
        ids = _ids(root / "experiments" / f"{split}_videos.txt")
        if len(ids) != 10 or len(set(ids)) != len(ids):
            raise ValueError(f"{split} must contain exactly 10 unique video IDs")
        paths = [videos / f"{video_id}.mp4" for video_id in ids]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{split} missing videos: {missing}")
        output = root / "experiments" / f"{split}_video_paths.txt"
        output.write_text("\n".join(str(path) for path in paths) + "\n", encoding="utf-8")
        print(f"{split}: {len(paths)} verified videos -> {output}")


if __name__ == "__main__":
    main()
