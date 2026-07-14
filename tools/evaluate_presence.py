from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PRED_ALIASES: Dict[str, str] = {
    # left/right
    "左": "left",
    "左边": "left",
    "在左": "left",
    "在左侧": "left",
    "左侧": "left",
    "右": "right",
    "右边": "right",
    "在右": "right",
    "在右侧": "right",
    "右侧": "right",
    # above/below
    "上": "above",
    "上面": "above",
    "在上": "above",
    "在上方": "above",
    "上方": "above",
    "下": "below",
    "下面": "below",
    "在下": "below",
    "在下方": "below",
    "下方": "below",
    # front/behind
    "前": "front",
    "前面": "front",
    "在前": "front",
    "在前方": "front",
    "前方": "front",
    "后": "behind",
    "后面": "behind",
    "在后": "behind",
    "在后方": "behind",
    "后方": "behind",
    "骑": "ride",
    "骑乘": "ride",
    "滑滑板": "ride",
    "坐在": "sit_on",
    "拿": "hold",
    "拿着": "hold",
    "携带": "carry",
    "穿戴": "wear",
    "拥抱": "hug",
    "追": "chase",
    "追赶": "chase",
    "踢": "kick",
    "推": "push",
    "对话": "talk_to",
    "交谈": "talk_to",
    "注视": "look_at",
    "同行": "walk_with",
    "玩耍": "play_with",
    "对唱": "sing_with",
    "合唱": "sing_with",
    "sing with": "sing_with",
    "talk to": "talk_to",
}


def _canonical_predicate(p: str) -> str:
    s = str(p or "").strip()
    if not s:
        return ""
    low = s.lower()
    return PRED_ALIASES.get(s, PRED_ALIASES.get(low, low))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _to_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _iter_relation_items(obj: Any) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Yield (video_id, relation_item_dict)."""

    if not isinstance(obj, dict):
        return

    for vid, items in obj.items():
        video_id = str(vid)
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict):
                    yield video_id, it
        elif isinstance(items, dict):
            # allow {video_id: {"relations": [...]}}
            rels = items.get("relations")
            if isinstance(rels, list):
                for it in rels:
                    if isinstance(it, dict):
                        yield video_id, it


def _extract_presence_key(item: Dict[str, Any]) -> Optional[Tuple[int, str, int]]:
    # Support both semi_auto format and Step3-like keys.
    sid = _to_int(item.get("subject_track_id", item.get("subject_id")))
    oid = _to_int(item.get("object_track_id", item.get("object_id")))
    pred = _canonical_predicate(str(item.get("predicate", item.get("relationship_type", "")) or ""))

    if sid is None or oid is None or not pred:
        return None
    return int(sid), pred, int(oid)


def _presence_set(obj: Any) -> Dict[str, Dict[Tuple[int, str, int], List[Dict[str, Any]]]]:
    """Return mapping: video_id -> key -> list[raw_items]."""

    out: Dict[str, Dict[Tuple[int, str, int], List[Dict[str, Any]]]] = {}
    for video_id, it in _iter_relation_items(obj):
        key = _extract_presence_key(it)
        if key is None:
            continue
        out.setdefault(video_id, {}).setdefault(key, []).append(it)
    return out


def _f1(p: float, r: float) -> float:
    if p + r <= 0:
        return 0.0
    return 2.0 * p * r / (p + r)


def _format_key(k: Tuple[int, str, int]) -> str:
    return f"({k[0]}, {k[1]}, {k[2]})"


def _pick_example(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {}
    # Prefer one with evidence/segment_id if present.
    best = items[0]
    best_score = 0
    for it in items:
        score = 0
        if it.get("evidence"):
            score += 2
        if it.get("segment_id") is not None:
            score += 1
        if score > best_score:
            best = it
            best_score = score
    return best


def _markdown_report(
    *,
    per_video_rows: List[Dict[str, Any]],
    per_predicate_rows: List[Dict[str, Any]],
    overall: Dict[str, Any],
    fp_examples: List[Dict[str, Any]],
    fn_examples: List[Dict[str, Any]],
    gold_path: Path,
    pred_path: Path,
) -> str:
    lines: List[str] = []
    lines.append("# Presence Evaluation Report")
    lines.append("")
    lines.append(f"- gold: {str(gold_path).replace('\\\\', '/')}")
    lines.append(f"- pred: {str(pred_path).replace('\\\\', '/')}")
    lines.append("")

    lines.append("## Overall")
    lines.append("")
    lines.append(
        f"- TP={overall['tp']}  FP={overall['fp']}  FN={overall['fn']}  "
        f"Precision={overall['precision']:.4f}  Recall={overall['recall']:.4f}  F1={overall['f1']:.4f}"
    )
    lines.append("")

    lines.append("## Per Video")
    lines.append("")
    lines.append("| video_id | TP | FP | FN | Precision | Recall | F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in per_video_rows:
        lines.append(
            f"| {r['video_id']} | {r['tp']} | {r['fp']} | {r['fn']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |"
        )
    lines.append("")

    lines.append("## Per Predicate")
    lines.append("")
    lines.append("| predicate | TP | FP | FN | Precision | Recall | F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in per_predicate_rows:
        lines.append(
            f"| {r['predicate']} | {r['tp']} | {r['fp']} | {r['fn']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |"
        )
    lines.append("")

    def dump_examples(title: str, examples: List[Dict[str, Any]]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not examples:
            lines.append("(none)")
            lines.append("")
            return
        for ex in examples:
            lines.append(f"- video_id={ex.get('video_id','')} key={ex.get('key','')}")
            if ex.get("segment_id") is not None:
                lines.append(f"  - segment_id: {ex.get('segment_id')}")
            if ex.get("start_frame") is not None or ex.get("end_frame") is not None:
                lines.append(f"  - span: {ex.get('start_frame')}~{ex.get('end_frame')}")
            ev = str(ex.get("evidence", "") or "").strip()
            if ev:
                ev = ev.replace("\n", " ")
                lines.append(f"  - evidence: {ev[:240]}")
        lines.append("")

    dump_examples("False Positives (Pred only)", fp_examples)
    dump_examples("False Negatives (Gold only)", fn_examples)

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Presence-only evaluation for VIDVRD Phase-1 outputs")
    ap.add_argument("--gold", type=str, required=True, help="gold/relations_gold.json")
    ap.add_argument("--pred", type=str, required=True, help="pred/relations_pred.json")
    ap.add_argument("--report", type=str, default="", help="Output markdown path (optional)")
    ap.add_argument("--max_examples", type=int, default=50, help="Max FP/FN examples to include")
    args = ap.parse_args()

    gold_path = Path(args.gold).expanduser().resolve()
    pred_path = Path(args.pred).expanduser().resolve()

    if not gold_path.exists():
        raise SystemExit(f"ERROR: gold not found: {gold_path}")
    if not pred_path.exists():
        raise SystemExit(f"ERROR: pred not found: {pred_path}")

    gold_obj = _read_json(gold_path)
    pred_obj = _read_json(pred_path)

    gold_map = _presence_set(gold_obj)
    pred_map = _presence_set(pred_obj)

    video_ids = sorted(set(gold_map.keys()) | set(pred_map.keys()))

    per_video_rows: List[Dict[str, Any]] = []

    overall_tp = 0
    overall_fp = 0
    overall_fn = 0

    fp_examples: List[Dict[str, Any]] = []
    fn_examples: List[Dict[str, Any]] = []
    pred_tp: Counter[str] = Counter()
    pred_fp: Counter[str] = Counter()
    pred_fn: Counter[str] = Counter()

    for vid in video_ids:
        g = set(gold_map.get(vid, {}).keys())
        p = set(pred_map.get(vid, {}).keys())

        tp = len(g & p)
        fp = len(p - g)
        fn = len(g - p)

        precision = float(tp / max(1, tp + fp))
        recall = float(tp / max(1, tp + fn))
        f1 = _f1(precision, recall)

        per_video_rows.append(
            {
                "video_id": vid,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

        for k in g & p:
            pred_tp[k[1]] += 1
        for k in p - g:
            pred_fp[k[1]] += 1
        for k in g - p:
            pred_fn[k[1]] += 1

        # Collect examples (best-effort)
        for k in sorted(list(p - g)):
            if len(fp_examples) >= int(args.max_examples):
                break
            ex = _pick_example(pred_map.get(vid, {}).get(k, []))
            fp_examples.append(
                {
                    "video_id": vid,
                    "key": _format_key(k),
                    "segment_id": ex.get("segment_id"),
                    "start_frame": ex.get("start_frame"),
                    "end_frame": ex.get("end_frame"),
                    "evidence": ex.get("evidence"),
                }
            )
        for k in sorted(list(g - p)):
            if len(fn_examples) >= int(args.max_examples):
                break
            ex = _pick_example(gold_map.get(vid, {}).get(k, []))
            fn_examples.append(
                {
                    "video_id": vid,
                    "key": _format_key(k),
                    "segment_id": ex.get("segment_id"),
                    "start_frame": ex.get("start_frame"),
                    "end_frame": ex.get("end_frame"),
                    "evidence": ex.get("evidence"),
                }
            )

    overall_precision = float(overall_tp / max(1, overall_tp + overall_fp))
    overall_recall = float(overall_tp / max(1, overall_tp + overall_fn))
    overall_f1 = _f1(overall_precision, overall_recall)

    overall = {
        "tp": overall_tp,
        "fp": overall_fp,
        "fn": overall_fn,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
    }

    per_predicate_rows: List[Dict[str, Any]] = []
    for pred in sorted(set(pred_tp.keys()) | set(pred_fp.keys()) | set(pred_fn.keys())):
        tp = int(pred_tp[pred])
        fp = int(pred_fp[pred])
        fn = int(pred_fn[pred])
        precision = float(tp / max(1, tp + fp))
        recall = float(tp / max(1, tp + fn))
        per_predicate_rows.append(
            {
                "predicate": pred,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
            }
        )

    md = _markdown_report(
        per_video_rows=per_video_rows,
        per_predicate_rows=per_predicate_rows,
        overall=overall,
        fp_examples=fp_examples,
        fn_examples=fn_examples,
        gold_path=gold_path,
        pred_path=pred_path,
    )

    if str(args.report or "").strip():
        out_path = Path(args.report).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"DONE: wrote report: {out_path}")
    else:
        print(md)


if __name__ == "__main__":
    main()
