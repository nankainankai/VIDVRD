from __future__ import annotations

"""Parse a JSON object from multimodal model text, including fenced replies."""

import json
from typing import Any


def parse_json_object(text: str, *, required_key: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and required_key in value:
            return value
    # Some models finish a complete action list but omit the final outer brace.
    # Repair only one or two trailing structural closers; never repair strings
    # or incomplete values.
    for index, char in enumerate(text):
        if char != "{":
            continue
        candidate = text[index:].rstrip("` \r\n\t")
        stack: list[str] = []
        in_string = False
        escaped = False
        valid = True
        for token in candidate:
            if in_string:
                if escaped:
                    escaped = False
                elif token == "\\":
                    escaped = True
                elif token == '"':
                    in_string = False
            elif token == '"':
                in_string = True
            elif token in "{[":
                stack.append("}" if token == "{" else "]")
            elif token in "}]":
                if not stack or stack.pop() != token:
                    valid = False
                    break
        if not valid or in_string or not 1 <= len(stack) <= 2:
            continue
        try:
            value = json.loads(candidate + "".join(reversed(stack)))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and required_key in value:
            return value
    return {}
