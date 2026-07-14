from __future__ import annotations

"""Object-aware predicate candidates.

This module does not decide whether a relation is true. It only narrows the
predicate search space for a subject/object category pair before VL verification.
"""

from typing import Dict, List, Tuple


PAIR_CANDIDATES: Dict[Tuple[str, str], List[str]] = {
    ("person", "skateboard"): ["ride", "on", "hold", "push", "near"],
    ("person", "bicycle"): ["ride", "on", "push", "near"],
    ("person", "horse"): ["ride", "on", "near", "follow"],
    ("person", "dog"): ["walk_with", "hold", "follow", "chase", "near", "play_with"],
    ("person", "cat"): ["hold", "carry", "near", "play_with"],
    ("person", "ball"): ["hold", "kick", "carry", "near"],
    ("person", "guitar"): ["hold", "play_with", "carry"],
    ("person", "microphone"): ["hold", "near"],
    ("person", "chair"): ["sit_on", "near", "push"],
    ("person", "table"): ["near", "on"],
    ("person", "car"): ["ride", "near", "push"],
    ("person", "surfboard"): ["ride", "on", "hold", "near"],
    ("person", "bag"): ["hold", "carry", "wear"],
    ("person", "cup"): ["hold", "near"],
    ("person", "phone"): ["hold", "look_at", "near"],
    ("person", "person"): [
        "near",
        "follow",
        "chase",
        "hug",
        "push",
        "kick",
        "talk_to",
        "look_at",
        "walk_with",
        "sing_with",
        "play_with",
    ],
}

SYMMETRIC_PREDICATES = {"near", "hug", "talk_to", "walk_with", "sing_with", "play_with"}
GEOMETRY_PREDICATES = {"left", "right", "above", "below", "front", "behind", "near", "overlap", "contact"}

_CATEGORY_ALIASES = {
    "man": "person",
    "woman": "person",
    "boy": "person",
    "girl": "person",
    "people": "person",
    "skate board": "skateboard",
    "bike": "bicycle",
    "cycle": "bicycle",
    "mic": "microphone",
    "cell phone": "phone",
    "mobile phone": "phone",
}


def normalize_category(category: str) -> str:
    raw = str(category or "").strip().lower().replace("_", " ")
    if not raw:
        return "unknown"
    return _CATEGORY_ALIASES.get(raw, raw.replace(" ", "_"))


def get_candidate_predicates(subject_class: str, object_class: str, audio_label: str = "") -> List[str]:
    """Return candidate predicates worth verifying for a directed category pair."""

    s = normalize_category(subject_class)
    o = normalize_category(object_class)
    candidates = list(PAIR_CANDIDATES.get((s, o), []))

    if not candidates:
        reverse = PAIR_CANDIDATES.get((o, s), [])
        candidates = [p for p in reverse if p in SYMMETRIC_PREDICATES]

    if not candidates:
        candidates = ["near", "overlap"]

    audio = str(audio_label or "").lower()
    if s == "person" and o == "person":
        if "sing" in audio and "sing_with" not in candidates:
            candidates.insert(0, "sing_with")
        if ("speech" in audio or "talk" in audio or "speak" in audio) and "talk_to" not in candidates:
            candidates.insert(0, "talk_to")

    out: List[str] = []
    seen: set[str] = set()
    for pred in candidates:
        key = str(pred).strip().lower()
        if key and key not in seen:
            out.append(key)
            seen.add(key)
    return out
