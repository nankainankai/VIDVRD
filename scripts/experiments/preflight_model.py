"""Text/image API preflight with the existing project credential."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from vidvrd_auto.core.context import Secrets
from vidvrd_auto.providers import DashScopeProvider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    client = DashScopeProvider({"model": args.model, "retries": 0, "request_timeout_sec": 60},
                              api_key=Secrets.from_env().dashscope_api_key)
    records = []
    for name, prompt, images in [
        ("text", 'Return only JSON: {"ok":true}', []),
        ("image", 'Describe the visible objects and activity. Return only JSON with objects and activity fields.', [args.image]),
    ]:
        result = client.call(prompt=prompt, image_paths=images)
        records.append({"name": name, **asdict(result)})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"results": records, "stats": client.stats.to_dict()},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)
        if not result.ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
