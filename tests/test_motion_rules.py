from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vidvrd_auto.relations.ops import generate_rule_relations  # noqa: E402


class MotionRuleTests(unittest.TestCase):
    def test_toward_from_approaching_tracks(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            tracks = root / "tracks.jsonl"
            windows = root / "windows.json"
            out = root / "relations.json"

            lines = []
            for f in range(10):
                shift = float(f * 8)
                lines.append(
                    json.dumps(
                        {
                            "frame": f,
                            "tracks": [
                                {"track_id": 1, "bbox": [10 + shift, 10, 50 + shift, 90]},
                                {"track_id": 2, "bbox": [200, 10, 240, 90]},
                            ],
                        },
                        ensure_ascii=False,
                    )
                )
            tracks.write_text("\n".join(lines) + "\n", encoding="utf-8")
            windows.write_text(
                json.dumps(
                    {
                        "windows": [
                            {
                                "window_id": 1,
                                "start_frame": 0,
                                "end_frame": 9,
                                "track_ids": [1, 2],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = generate_rule_relations(
                windows_json=windows,
                tracks_jsonl=tracks,
                out_json=out,
                video_id="demo",
                config={"min_vote_ratio": 0.5, "motion_distance_eps_ratio": 0.01},
            )
            preds = {r["predicate"] for r in result.get("demo", [])}
            self.assertIn("toward", preds)


if __name__ == "__main__":
    unittest.main()
