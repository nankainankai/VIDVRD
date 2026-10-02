from __future__ import annotations

"""Probe small/occluded-object vocabulary discovery on an existing storyboard."""

import argparse
from pathlib import Path

from vidvrd_auto.providers import DashScopeProvider


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storyboard", type=Path, required=True)
    args = parser.parse_args()
    prompt = (
        "Inspect every frame carefully, including small, partly occluded, held, or prey objects. "
        "List all visually identifiable trackable object categories across the video. "
        "Do not omit an object because it appears briefly or is partly hidden. "
        "Use concise singular English nouns, not actions or attributes. "
        'Return JSON only: {"objects":["category"]}.'
    )
    provider = DashScopeProvider({"model": "kimi-k2.6", "retries": 1, "request_timeout_sec": 90})
    response = provider.call(prompt=prompt, image_paths=[args.storyboard])
    print(response.text if response.ok else response.error)


if __name__ == "__main__":
    main()
