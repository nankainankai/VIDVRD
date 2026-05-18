from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vidvrd_auto.relations.merge import merge_relations


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


class RelationTests(unittest.TestCase):
    def test_merge_relations_adds_coupling(self) -> None:
        with TemporaryDirectory() as td:
            tmp_path = Path(td)
            src = tmp_path / "rule.json"
            out = tmp_path / "merged.json"
            _write(
                src,
                {
                    "v1": [
                        {
                            "subject_track_id": 1,
                            "object_track_id": 2,
                            "predicate": "left",
                            "start_frame": 0,
                            "end_frame": 10,
                            "confidence": 0.8,
                            "source": "rule_geometry",
                        }
                    ]
                },
            )

            merged = merge_relations(video_id="v1", relation_jsons=[src], out_json=out, apply_coupling=True)
            items = merged["v1"]
            keys = {(x["subject_track_id"], x["predicate"], x["object_track_id"]) for x in items}
            self.assertIn((1, "left", 2), keys)
            self.assertIn((2, "right", 1), keys)


if __name__ == "__main__":
    unittest.main()
