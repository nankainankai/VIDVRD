"""Export an allowlisted, size-limited result snapshot without videos or secrets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/performance_validation_20261002"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/performance_validation_20261002"))
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected: set[Path] = set()
    for run in ("additional_test3", "combined_test5", "qwen38_axis_baseline", "qwen38_axis_reserve", "qwen38_compositional"):
        folder = source / run
        selected.update((folder / "reports").rglob("*.json"))
        selected.update((folder / "reports").rglob("*.md"))
        for name in ("run_manifest.json", "selection.json", "aggregation_manifest.json"):
            file = folder / name
            if file.exists():
                selected.add(file)
    selected.update((source / "qwen38_flash_smoke" / "reports" / "completion_eval").glob("*.json"))
    selected.add(source / "qwen38_flash_smoke" / "reports" / "runtime.json")
    selected.add(source / "optimization_comparison.json")
    selected.update((source / "combined_test5" / "pred").glob("*.json"))
    for video in (source / "additional_test3" / "videos").iterdir():
        for name in ("detect/detections.jsonl", "detect/meta.json", "track/tracks.jsonl", "track_qc/report.json"):
            selected.add(video / name)

    def portable(value: object) -> object:
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, str):
            for prefix in (str(workspace) + "\\", workspace.as_posix() + "/"):
                value = value.replace(prefix, "")
            # Records occasionally contain JSON-escaped or doubled Windows paths.
            value = value.replace(str(workspace).replace("\\", "\\\\") + "\\\\", "")
        return value

    records = []
    total_bytes = 0
    for file in sorted(selected):
        raw = file.read_bytes()
        content = raw.decode("utf-8-sig")
        if file.suffix == ".json":
            content = json.dumps(portable(json.loads(content)), ensure_ascii=False, separators=(",", ":")) + "\n"
        elif file.suffix == ".jsonl":
            content = "".join(json.dumps(portable(json.loads(line)), ensure_ascii=False, separators=(",", ":")) + "\n" for line in content.splitlines() if line.strip())
        else:
            content = str(portable(content))
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        if re.search(r"sk-[A-Za-z0-9._-]{16,}", content):
            raise ValueError(f"Possible credential in {file.name}; export stopped")
        payload = content.encode("utf-8")
        total_bytes += len(payload)
        if len(payload) > 1024 * 1024 or total_bytes > 5 * 1024 * 1024:
            raise ValueError("Result snapshot exceeds the 1 MiB/file or 5 MiB total budget")
        relative = file.relative_to(source)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        records.append({"path": relative.as_posix(), "bytes": len(payload), "source_sha256": hashlib.sha256(raw).hexdigest(), "snapshot_sha256": hashlib.sha256(payload).hexdigest()})
    manifest = {
        "schema_version": 1,
        "source_run": args.source.as_posix(),
        "transforms": ["remove workspace absolute path prefixes", "compact JSON whitespace", "normalize snapshot line endings to LF"],
        "excluded": ["API credentials", "videos", "images", "models", "API response caches", "storyboards", "full execution logs"],
        "file_count": len(records), "payload_bytes": total_bytes,
        "complete_official_test": False,
        "files": records,
    }
    (output / "snapshot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(records), "payload_bytes": total_bytes, "output": args.output.as_posix()}))


if __name__ == "__main__":
    main()
