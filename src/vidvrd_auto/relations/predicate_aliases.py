from __future__ import annotations

"""谓词别名与音频先验谓词扩展。"""

from typing import Dict, List, Set

PREDICATE_ALIASES: Dict[str, str] = {
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
    "重叠": "overlap",
    "交叠": "overlap",
    "相交": "overlap",
    "近": "near",
    "靠近": "near",
    "附近": "near",
    "跟随": "follow",
    "跟着": "follow",
    "追随": "follow",
    "朝向": "toward",
    "朝": "toward",
    "面向": "toward",
}

DEFAULT_RELATION_PREDICATES: List[str] = [
    "left",
    "right",
    "front",
    "behind",
    "above",
    "below",
    "overlap",
    "near",
    "follow",
    "toward",
]


def canonical_predicate(p: str) -> str:
    s = str(p or "").strip()
    if not s:
        return ""
    low = s.lower()
    return PREDICATE_ALIASES.get(s, PREDICATE_ALIASES.get(low, low))


def audio_predicates_from_label(label: str) -> List[str]:
    s = (label or "").lower()
    if not s or s == "unknown":
        return []
    out: List[str] = []
    if "laugh" in s:
        out.append("laugh with")
    if "whisper" in s:
        out.append("whisper to")
    if "speech" in s or "speaking" in s:
        out.append("speech to")
    if "sing" in s and "bowl" not in s:
        out.append("sing with")
    if "growl" in s:
        out.append("growl at")
    if "bark" in s or "bow-wow" in s:
        out.append("bark at")
    if "chirp" in s or "tweet" in s:
        out.append("chirp at")
    seen: Set[str] = set()
    uniq: List[str] = []
    for p in out:
        c = canonical_predicate(p)
        if c and c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq
