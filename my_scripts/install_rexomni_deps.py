from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


MINIMAL_PACKAGES: List[str] = [
    "numpy==1.26.4",
    "Pillow==10.4.0",
    "transformers==4.51.3",
    "accelerate==1.10.1",
    "qwen_vl_utils==0.0.14",
]


TORCH_INDEX = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu124": "https://download.pytorch.org/whl/cu124",
}


def _run(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd)


def _repo_root_from_here() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "Rex-Omni-master" / "Rex-Omni-master").exists():
            return p
    return here.parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Install Rex-Omni dependencies for this project")
    ap.add_argument(
        "--torch",
        type=str,
        default="skip",
        choices=["skip", "cpu", "cu121", "cu124"],
        help="Install torch/torchvision/torchaudio from selected channel",
    )
    ap.add_argument(
        "--install_local_rex",
        action="store_true",
        help="Optional: pip install -e local Rex-Omni package (uses --no-deps to avoid flash-attn on Windows)",
    )
    ap.add_argument("--upgrade_pip", action="store_true", help="upgrade pip/setuptools/wheel first")
    args = ap.parse_args()

    py = sys.executable

    if args.upgrade_pip:
        _run([py, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])

    _run([py, "-m", "pip", "install", *MINIMAL_PACKAGES])

    if args.torch != "skip":
        index_url = TORCH_INDEX[args.torch]
        _run(
            [
                py,
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                index_url,
            ]
        )

    if args.install_local_rex:
        root = _repo_root_from_here()
        rex_pkg = root / "Rex-Omni-master" / "Rex-Omni-master"
        if not rex_pkg.exists():
            raise SystemExit(f"Rex-Omni local package not found: {rex_pkg}")
        # IMPORTANT (Windows): Rex-Omni requirements include flash-attn which is hard/unavailable on win.
        # We already installed minimal runtime deps above; install package itself without deps.
        _run([py, "-m", "pip", "install", "-e", str(rex_pkg), "--no-deps"])

    print("=" * 70)
    print("Rex-Omni dependency installation finished.")
    print("Now you can run Step1 with --backend rexomni.")
    print("Tip: On Windows, do NOT install flash-attn/vllm unless you know what you're doing.")
    print("=" * 70)


if __name__ == "__main__":
    main()
