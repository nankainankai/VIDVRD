from __future__ import annotations

"""Prompts for review steps that receive real visual evidence."""

import json
from typing import Any, Dict, List


def relation_verify_prompt(relations: List[Dict[str, Any]], issues: List[Dict[str, Any]]) -> str:
    return (
        "You are the final reviewer for video relations. The attached pair storyboards are the visual evidence. "
        "Review only the listed risky relations, and reference each one by its stable relation_id. "
        "Do not invent relations. Return JSON only: "
        '{"actions":[{"action":"keep|delete|change_predicate|adjust_span","relation_id":"r000001",'
        '"new_predicate":"optional","start_frame":0,"end_frame":29,"reason":"visual reason"}]}. '
        "For change_predicate, use a predicate already represented in the candidates. For adjust_span, require clear temporal evidence.\n"
        f"Relations: {json.dumps(relations, ensure_ascii=False)}\n"
        f"Review triggers: {json.dumps(issues, ensure_ascii=False)}"
    )
