from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vidvrd_auto.core.ontology import object_names, predicate_components, predicate_names
from vidvrd_auto.evaluation.gold import build_gold


class OntologyTests(unittest.TestCase):
    def test_complete_official_ontology_and_composite_predicate(self) -> None:
        self.assertEqual(len(object_names()), 35)
        self.assertEqual(len(predicate_names()), 132)
        self.assertTrue(set(predicate_names(split="base")))
        self.assertTrue(set(predicate_names(split="novel")))
        self.assertEqual(
            predicate_components("walk_behind"),
            {"action": "walk", "spatial": "behind", "comparative": ""},
        )

    def test_gold_converter_uses_exclusive_official_end_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            annotation_dir = root / "anno" / "test"
            annotation_dir.mkdir(parents=True)
            annotation = {
                "video_id": "clip",
                "subject/objects": [{"tid": 0, "category": "person"}, {"tid": 1, "category": "dog"}],
                "trajectories": [
                    [{"tid": 0, "bbox": {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}}, {"tid": 1, "bbox": {"xmin": 20, "ymin": 0, "xmax": 30, "ymax": 10}}],
                    [{"tid": 0, "bbox": {"xmin": 1, "ymin": 0, "xmax": 11, "ymax": 10}}, {"tid": 1, "bbox": {"xmin": 21, "ymin": 0, "xmax": 31, "ymax": 10}}],
                ],
                "relation_instances": [{"subject_tid": 0, "predicate": "next_to", "object_tid": 1, "begin_fid": 0, "end_fid": 2}],
            }
            (annotation_dir / "clip.json").write_text(json.dumps(annotation), encoding="utf-8")
            relations, trajectories, manifest = root / "relations.json", root / "trajectories.json", root / "manifest.json"
            result = build_gold(annotations_dir=root / "anno", relations_path=relations, trajectories_path=trajectories, manifest_path=manifest)
            item = json.loads(relations.read_text(encoding="utf-8"))["clip"][0]
            self.assertEqual((item["start_frame"], item["end_frame"]), (0, 1))
            self.assertEqual(result["trajectory_count"], 2)


if __name__ == "__main__":
    unittest.main()
