"""Step5: 基于 video_relations.json 调用 Qwen 生成自然语言描述。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import dashscope
    from dashscope import Generation

    HAS_DASHSCOPE = True
except Exception:
    dashscope = None
    Generation = None
    HAS_DASHSCOPE = False

import config

try:
    from utils_io import safe_read_json as _safe_read_json_utils
    from utils_io import safe_write_text as _safe_write_text_utils

    HAS_UTILS_IO = True
except Exception:
    HAS_UTILS_IO = False


def _extract_resp_error(resp: Any) -> str:
    status = getattr(resp, "status_code", "unknown")
    code = getattr(resp, "code", "")
    message = getattr(resp, "message", "")
    request_id = getattr(resp, "request_id", "")
    parts = [f"HTTP {status}"]
    if code:
        parts.append(f"code={code}")
    if message:
        parts.append(f"message={message}")
    if request_id:
        parts.append(f"request_id={request_id}")
    return " | ".join(parts)


def _supports_http_generation_error(msg: str) -> bool:
    m = (msg or "").lower()
    return (
        "does not support http call" in m
        or "url error" in m
        or "invalidparameter" in m and "http" in m
    )


def _read_json(path: Path) -> Any:
    if HAS_UTILS_IO:
        return _safe_read_json_utils(path)
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _write_text(path: Path, text: str) -> None:
    if HAS_UTILS_IO:
        _safe_write_text_utils(path, text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def _build_prompt(video: Dict[str, Any], events: List[Dict[str, Any]], max_events: int) -> str:
    lines: List[str] = []
    for e in (events or [])[: int(max_events)]:
        sid = e.get("subject_id")
        pred = e.get("predicate")
        oid = e.get("object_id")
        obj_label = e.get("object_label")
        score = e.get("score_max")

        span_txts: List[str] = []
        for sp in (e.get("spans", []) or [])[:4]:
            if not isinstance(sp, dict):
                continue
            try:
                s = float(sp.get("start_time", 0.0) or 0.0)
                t = float(sp.get("end_time", 0.0) or 0.0)
                span_txts.append(f"{s:.2f}-{t:.2f}s")
            except Exception:
                continue

        label = f" label={obj_label}" if obj_label else ""
        lines.append(f"- ID {sid} {pred} ID {oid}{label} | score={score} | spans={', '.join(span_txts)}")

    return f"""你是视频关系总结助手。请根据输入事件生成一段自然语言描述。

输出要求：
1) 输出一段 8-12 句中文描述，按时间顺序概述视频中的主要关系变化；
2) 尽量使用“主体-关系-客体”的表达；
3) 对置信度低或冲突关系，描述要保守；
4) 不要输出 JSON，不要输出项目符号。

视频信息：
- path: {video.get('path', '')}
- fps: {video.get('fps', '')}
- total_frames: {video.get('total_frames', '')}
- duration: {video.get('duration', '')}

关系事件：
{chr(10).join(lines)}
"""


def _call_qwen(api_key: str, model: str, prompt: str) -> str:
    if not HAS_DASHSCOPE or dashscope is None or Generation is None:
        raise RuntimeError("dashscope Generation not available")
    if not api_key.strip():
        raise RuntimeError("API key is empty")

    dashscope.api_key = api_key
    resp = Generation.call(model=str(model), prompt=prompt)
    if getattr(resp, "status_code", None) != 200:
        raise RuntimeError(f"API error: {_extract_resp_error(resp)}")

    if hasattr(resp, "output") and isinstance(resp.output, dict) and "text" in resp.output:
        return str(resp.output.get("text", "")).strip()
    if hasattr(resp, "output") and hasattr(resp.output, "text"):
        return str(resp.output.text).strip()
    return str(getattr(resp, "output", "")).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Step5: generate natural language summary from video_relations.json")
    ap.add_argument("--video_relations_json", type=str, default="", help="Step4输出 video_relations.json；不传则从OUTPUT_DIR推断")
    ap.add_argument("--output", type=str, default="")
    ap.add_argument("--api_key", type=str, default="")
    ap.add_argument("--model", type=str, default=str(getattr(config, "STEP5_QWEN_MODEL", "qwen-omni-turbo-realtime-latest")))
    ap.add_argument("--fallback_model", type=str, default=str(getattr(config, "STEP5_QWEN_FALLBACK_MODEL", "qwen-max")))
    ap.add_argument("--max_events", type=int, default=int(getattr(config, "STEP5_MAX_EVENTS", 60)))
    args = ap.parse_args()

    out_dir = Path(str(getattr(config, "OUTPUT_DIR", "C:/video_output"))).expanduser().resolve()
    vr_path = (
        Path(args.video_relations_json).expanduser().resolve()
        if str(args.video_relations_json or "").strip()
        else (out_dir / "video_relations.json").resolve()
    )
    if not vr_path.exists():
        print(f"ERROR: video_relations.json not found: {vr_path}")
        print("TIP: 传入 --video_relations_json，或先确认 config.OUTPUT_DIR 下存在 video_relations.json")
        return

    obj = _read_json(vr_path)
    if not isinstance(obj, dict):
        print("ERROR: invalid video_relations.json")
        return

    video = obj.get("video", {}) if isinstance(obj.get("video", {}), dict) else {}
    events = obj.get("events", []) if isinstance(obj.get("events", []), list) else []

    api_key = (args.api_key or "").strip()
    if not api_key:
        api_key = str(getattr(config, "STEP5_QWEN_API_KEY", "") or "").strip()
    if not api_key:
        api_key = str(getattr(config, "API_KEY", "") or "").strip()

    if not api_key:
        print("ERROR: missing API key. Set config.STEP5_QWEN_API_KEY or pass --api_key")
        return

    prompt = _build_prompt(video=video, events=events, max_events=int(args.max_events))
    used_model = str(args.model)
    try:
        text = _call_qwen(api_key=api_key, model=used_model, prompt=prompt)
    except Exception as e:
        first_err = str(e)
        fallback_candidates = [
            str(args.fallback_model or "").strip(),
            str(getattr(config, "API_MODEL", "") or "").strip(),
            "qwen-max",
        ]
        fallback_candidates = [m for m in fallback_candidates if m and m != used_model]

        if _supports_http_generation_error(first_err) and fallback_candidates:
            recovered = False
            for fb_model in fallback_candidates:
                try:
                    text = _call_qwen(api_key=api_key, model=fb_model, prompt=prompt)
                    used_model = fb_model
                    recovered = True
                    print(
                        "WARN: primary model call failed; auto-fallback succeeded. "
                        f"primary={args.model}, fallback={fb_model}, reason={first_err}"
                    )
                    break
                except Exception:
                    continue
            if not recovered:
                print(f"ERROR: step5 qwen call failed: {first_err}")
                return
        else:
            print(f"ERROR: step5 qwen call failed: {first_err}")
            return

    out_path = Path(args.output).expanduser().resolve() if args.output else (vr_path.parent / "video_relations_description.txt")
    _write_text(out_path, text)
    print(f"DONE: {out_path} (model={used_model})")


if __name__ == "__main__":
    main()
