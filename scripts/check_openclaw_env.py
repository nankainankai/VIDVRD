from __future__ import annotations

"""OpenClaw / Agent 运行前环境检查。"""

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def main() -> int:
    errors = 0
    print("VIDVRD OpenClaw 环境检查")
    print("=" * 60)

    if sys.version_info < (3, 10):
        _fail(f"Python >= 3.10 required, got {sys.version.split()[0]}")
        errors += 1
    else:
        _ok(f"Python {sys.version.split()[0]}")

    try:
        import vidvrd_auto  # noqa: F401

        _ok("vidvrd_auto 可导入")
    except ImportError:
        _warn("vidvrd_auto 未安装，请运行: python -m pip install -e .")
        _warn("或使用: python scripts/run_vidvrd_auto.py")

    for mod, pkg in [("cv2", "opencv-python"), ("numpy", "numpy")]:
        if importlib.util.find_spec(mod):
            _ok(f"{pkg} 已安装")
        else:
            _warn(f"缺少 {pkg}（mock/dry-run 检测追踪需要）")

    cfg_default = ROOT / "configs" / "default.json"
    cfg_dry = ROOT / "configs" / "dry_run.json"
    if cfg_default.exists() and cfg_dry.exists():
        _ok("配置文件 configs/default.json, configs/dry_run.json")
    else:
        _fail("缺少 configs/default.json 或 configs/dry_run.json")
        errors += 1

    dummy = ROOT / "data" / "validation_dummy.mp4"
    if dummy.exists():
        _ok(f"测试视频 {dummy.relative_to(ROOT)}")
    else:
        _warn("无 data/validation_dummy.mp4，运行: python scripts/make_validation_dummy.py")

    dash = (os.getenv("DASHSCOPE_API_KEY", "") or "").strip()
    if dash:
        _ok("DASHSCOPE_API_KEY 已设置（正式 VL 节点可用）")
    else:
        _warn("未设置 DASHSCOPE_API_KEY（dry-run 可跳过；正式 relation_llm 需要）")

    dinox = (os.getenv("DINOX_API_TOKEN", "") or "").strip()
    if dinox:
        _ok("DINOX_API_TOKEN 已设置")
    else:
        _warn("未设置 DINOX_API_TOKEN（detector.backend=dinox 时需要）")

    if cfg_dry.exists():
        cfg = json.loads(cfg_dry.read_text(encoding="utf-8"))
        det = (cfg.get("detector") or {}).get("backend", "")
        trk = (cfg.get("tracking") or {}).get("backend", "")
        if str(det).lower() == "mock" and str(trk).lower() == "mock":
            _ok("dry_run.json 使用 mock 检测/追踪（无需 GPU/API）")
        else:
            _warn(f"dry_run detector={det} tracking={trk}，建议均为 mock")

    gold_rel = ROOT / "gold" / "relations_gold.json"
    if gold_rel.exists():
        _ok("Gold 样例 gold/relations_gold.json（可跑 Presence 评测）")
    else:
        _warn("缺少 gold/relations_gold.json，评测将 skipped")

    vggsound = ROOT / "data" / "vggsound" / "vggsound.csv"
    if vggsound.exists():
        _ok("VGGSound CSV 已就绪")
    else:
        _warn("无 data/vggsound/vggsound.csv，audio_prior 仅用 fallback")

    try:
        default_cfg = json.loads(cfg_default.read_text(encoding="utf-8"))
        rex_path = Path(str((default_cfg.get("detector") or {}).get("rex_model_path", "")))
        if not rex_path.is_absolute():
            rex_path = (ROOT / rex_path).resolve()
        if rex_path.exists():
            try:
                label = str(rex_path.relative_to(ROOT))
            except ValueError:
                label = str(rex_path)
            weights = rex_path / "model.safetensors"
            if weights.is_file():
                _ok(f"Rex-Omni 模型目录存在: {label}")
            else:
                _warn(f"Rex-Omni 目录存在但缺少 model.safetensors: {label}")
        else:
            _warn(f"Rex-Omni 未找到: {rex_path}（可改用 dinox 或 mock）")
    except Exception:
        pass

    print("=" * 60)
    if errors:
        print(f"检查未通过: {errors} 项失败")
        return 1
    print("检查完成。建议先跑:")
    print(
        "  python scripts/run_vidvrd_auto.py --video data/validation_dummy.mp4 "
        "--run_dir runs/smoke_openclaw --config configs/dry_run.json --resume --dry_run_relations --skip_eval"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
