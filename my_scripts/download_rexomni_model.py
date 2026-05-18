from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Rex-Omni model files to local directory (HuggingFace snapshot)")
    ap.add_argument("--repo_id", type=str, default="IDEA-Research/Rex-Omni-AWQ")
    ap.add_argument("--revision", type=str, default="main")
    ap.add_argument(
        "--local_dir",
        type=str,
        default="models/Rex-Omni-AWQ",
        help="Where to store downloaded snapshot (a folder path)",
    )
    ap.add_argument(
        "--endpoint",
        type=str,
        default="",
        help="Optional HF endpoint/mirror, e.g. https://hf-mirror.com (also supports env HF_ENDPOINT)",
    )
    ap.add_argument(
        "--connect_timeout",
        type=int,
        default=30,
        help="HTTP connect timeout seconds (sets env HF_HUB_CONNECT_TIMEOUT)",
    )
    ap.add_argument(
        "--read_timeout",
        type=int,
        default=60,
        help="HTTP read timeout seconds (sets env HF_HUB_READ_TIMEOUT)",
    )
    args = ap.parse_args()

    endpoint = (args.endpoint or os.getenv("HF_ENDPOINT", "")).strip()
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint

    # Make timeouts explicit; useful for slow / unstable networks.
    os.environ["HF_HUB_CONNECT_TIMEOUT"] = str(int(args.connect_timeout))
    os.environ["HF_HUB_READ_TIMEOUT"] = str(int(args.read_timeout))

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as e:
        raise SystemExit(
            "ERROR: missing huggingface_hub. It should come with transformers, but if not:\n"
            "  python -m pip install -U huggingface_hub\n"
            f"Original error: {e}"
        )

    # Interpret relative path against repo root to avoid surprises when running from
    # different working directories (e.g. repo root vs my_scripts/).
    local_dir_arg = Path(args.local_dir).expanduser()
    if local_dir_arg.is_absolute():
        local_dir = local_dir_arg.resolve()
    else:
        repo_root = Path(__file__).resolve().parents[1]
        local_dir = (repo_root / local_dir_arg).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Downloading model snapshot")
    print("repo_id=", args.repo_id)
    print("revision=", args.revision)
    print("local_dir=", str(local_dir))
    if endpoint:
        print("HF_ENDPOINT=", endpoint)
    print("HF_HUB_CONNECT_TIMEOUT=", os.environ.get("HF_HUB_CONNECT_TIMEOUT"))
    print("HF_HUB_READ_TIMEOUT=", os.environ.get("HF_HUB_READ_TIMEOUT"))
    print("=" * 70)

    path = snapshot_download(
        repo_id=str(args.repo_id),
        revision=str(args.revision),
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print("=" * 70)
    print("DONE. Snapshot downloaded to:")
    print(path)
    print("Now run Step1 with:")
    print(
        f"  python my_scripts/step1_full_video_box_detection_dinox.py --backend rexomni --rex_model_path \"{path}\" --rex_categories \"person\""
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
