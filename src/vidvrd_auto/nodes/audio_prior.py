from __future__ import annotations

"""音频先验节点。

输入：video_id、原始视频来源、VGGSound CSV 配置。
输出：`audio_prior.json`，供片段关系 Prompt 注入音频场景信息。
该节点不直接调用大模型，可缓存、可 dry-run。
"""

import csv
from pathlib import Path
from typing import Any, Dict

from vidvrd_auto.utils.io import write_json
from vidvrd_auto.utils.paths import repo_root


def _resolve(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (repo_root() / path).resolve()
    return path


def build_audio_prior(*, video_id: str, source: str, out_json: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    enabled = bool(config.get("enabled", True))
    csv_path_text = str(config.get("csv_path", "") or "").strip()
    fallback_label = str(config.get("fallback_label", "") or "").strip()
    source_name = Path(source).stem.lower()
    keys = {video_id.lower(), source_name}

    result: Dict[str, Any] = {
        "enabled": enabled,
        "video_id": video_id,
        "label": "",
        "confidence": 0.0,
        "source": "none",
        "matched_key": "",
    }
    if not enabled:
        result["source"] = "disabled"
        write_json(out_json, result)
        return result

    if csv_path_text:
        csv_path = _resolve(csv_path_text)
        if csv_path.exists():
            key_col = str(config.get("video_id_column", "video_id") or "video_id")
            label_col = str(config.get("label_column", "label") or "label")
            conf_col = str(config.get("confidence_column", "") or "").strip()
            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    row_key = str(row.get(key_col, "") or "").strip().lower()
                    if row_key not in keys:
                        continue
                    result["label"] = str(row.get(label_col, "") or "").strip()
                    result["confidence"] = float(row.get(conf_col, 1.0) or 1.0) if conf_col else 1.0
                    result["source"] = "vggsound_csv"
                    result["matched_key"] = row_key
                    break
        else:
            result["warning"] = f"csv not found: {csv_path}"

    if not result["label"] and fallback_label:
        result["label"] = fallback_label
        result["confidence"] = float(config.get("fallback_confidence", 0.5) or 0.5)
        result["source"] = "fallback"

    write_json(out_json, result)
    return result
