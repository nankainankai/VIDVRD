"""Step4: 将窗口级最终关系聚合为视频级关系事件。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import config

try:
    from utils_io import safe_read_json as _safe_read_json_utils
    from utils_io import safe_write_json as _safe_write_json_utils

    HAS_UTILS_IO = True
except Exception:
    HAS_UTILS_IO = False


def _read_json(path: Path) -> Any:
    if HAS_UTILS_IO:
        return _safe_read_json_utils(path)
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    if HAS_UTILS_IO:
        _safe_write_json_utils(path, obj, indent=2)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _merge_spans(spans: List[Tuple[float, float, int]], gap: float) -> List[Dict[str, Any]]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    merged: List[Dict[str, Any]] = []

    cur_s, cur_e, cur_ids = spans[0][0], spans[0][1], [int(spans[0][2])]
    for s, e, sid in spans[1:]:
        if s <= cur_e + float(gap):
            cur_e = max(cur_e, e)
            cur_ids.append(int(sid))
        else:
            merged.append({"start_time": cur_s, "end_time": cur_e, "segment_ids": cur_ids})
            cur_s, cur_e, cur_ids = s, e, [int(sid)]

    merged.append({"start_time": cur_s, "end_time": cur_e, "segment_ids": cur_ids})
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description="Step4: aggregate relations_final into video-level relation events")
    ap.add_argument("--relations_final", type=str, default="", help="Step3 输出 relations_final.json；不传则从OUTPUT_DIR推断")
    ap.add_argument("--merge_gap", type=float, default=0.0, help="时间段合并容忍间隔(秒)")
    ap.add_argument("--output", type=str, default="", help="输出 video_relations.json 路径")
    args = ap.parse_args()

    out_dir = Path(str(getattr(config, "OUTPUT_DIR", "C:/video_output"))).expanduser().resolve()
    relations_final_path = (
        Path(args.relations_final).expanduser().resolve()
        if str(args.relations_final or "").strip()
        else (out_dir / "relations_final.json").resolve()
    )
    if not relations_final_path.exists():
        print(f"ERROR: relations_final not found: {relations_final_path}")
        print("TIP: 传入 --relations_final，或先确认 config.OUTPUT_DIR 下存在 relations_final.json")
        return

    obj = _read_json(relations_final_path)
    if not isinstance(obj, dict):
        print("ERROR: invalid relations_final.json")
        return

    windows_path_raw = str(obj.get("windows_json", "") or "").strip()
    if not windows_path_raw:
        print("ERROR: windows_json missing in relations_final")
        return

    windows_path = Path(windows_path_raw).expanduser().resolve()
    if not windows_path.exists():
        print(f"ERROR: windows_json not found: {windows_path}")
        return

    windows_obj = _read_json(windows_path)
    if not isinstance(windows_obj, dict):
        print("ERROR: invalid windows.json")
        return

    windows = windows_obj.get("windows", []) if isinstance(windows_obj.get("windows", []), list) else []
    window_map: Dict[int, Dict[str, Any]] = {}
    for w in windows:
        if not isinstance(w, dict):
            continue
        wid = int(w.get("window_id", 0) or 0)
        if wid > 0:
            window_map[wid] = w

    items = obj.get("items", []) if isinstance(obj.get("items", []), list) else []

    # key: (subject_id, predicate, object_id, object_label)
    bucket: Dict[Tuple[int, str, int, str], Dict[str, Any]] = {}

    for it in items:
        if not isinstance(it, dict):
            continue
        seg_id = int(it.get("segment_id", 0) or 0)
        triples = it.get("triples", [])
        if not isinstance(triples, list):
            continue

        w = window_map.get(seg_id, {})
        st = float(w.get("start_time", 0.0) or 0.0)
        et = float(w.get("end_time", 0.0) or 0.0)

        for t in triples:
            if not isinstance(t, dict):
                continue
            try:
                sid = int(t.get("subject_id"))
            except Exception:
                continue

            pred = str(t.get("predicate", "") or "").strip()
            if not pred:
                continue

            oid_raw = t.get("object_id", None)
            if oid_raw is None:
                oid = -1
            else:
                try:
                    oid = int(oid_raw)
                except Exception:
                    oid = -1

            obj_label = str(t.get("object_label", "") or "").strip()
            key = (sid, pred, oid, obj_label)

            if key not in bucket:
                bucket[key] = {
                    "subject_id": sid,
                    "predicate": pred,
                    "object_id": None if oid < 0 else oid,
                    "object_label": obj_label or None,
                    "occurrences": [],
                }

            bucket[key]["occurrences"].append(
                {
                    "segment_id": seg_id,
                    "start_time": st,
                    "end_time": et,
                    "confidence": float(t.get("confidence", 0.0) or 0.0),
                    "evidence": str(t.get("evidence", "") or "")[:300],
                    "sources": t.get("sources", []),
                }
            )

    events: List[Dict[str, Any]] = []
    for v in bucket.values():
        occ = v.get("occurrences", [])
        spans = [(float(o["start_time"]), float(o["end_time"]), int(o["segment_id"])) for o in occ]
        merged = _merge_spans(spans, gap=float(args.merge_gap))
        confs = [float(o.get("confidence", 0.0) or 0.0) for o in occ]

        events.append(
            {
                "subject_id": v["subject_id"],
                "predicate": v["predicate"],
                "object_id": v.get("object_id"),
                "object_label": v.get("object_label"),
                "score_max": max(confs) if confs else 0.0,
                "score_avg": (sum(confs) / max(1, len(confs))) if confs else 0.0,
                "spans": merged,
                "count": len(occ),
                "evidence_samples": [str(o.get("evidence", "")) for o in occ[:3]],
            }
        )

    events.sort(key=lambda x: (-(float(x.get("score_max", 0.0))), -(int(x.get("count", 0)))))

    out_path = Path(args.output).expanduser().resolve() if args.output else (relations_final_path.parent / "video_relations.json")
    _write_json(
        out_path,
        {
            "video": windows_obj.get("video", {}),
            "source": str(relations_final_path).replace("\\", "/"),
            "event_count": len(events),
            "events": events,
        },
    )

    print(f"DONE: {out_path}")


if __name__ == "__main__":
    main()
