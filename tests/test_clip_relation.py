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

from vidvrd_auto.relations.clip_relation import run_clip_relation  # noqa: E402
from vidvrd_auto.relations.storyboard import save_storyboard_image  # noqa: E402

try:
    import cv2
    import numpy as np

    HAS_CV = True
except ImportError:
    HAS_CV = False


class ClipRelationTests(unittest.TestCase):
    @unittest.skipUnless(HAS_CV, "opencv required")
    def test_dry_run_writes_storyboard_and_empty_relations(self) -> None:
        dummy = ROOT / "data" / "validation_dummy.mp4"
        if not dummy.exists():
            self.skipTest("validation_dummy.mp4 missing")

        with TemporaryDirectory() as td:
            root = Path(td)
            windows = root / "windows.json"
            tracks = root / "tracks.jsonl"
            out = root / "relations_llm.json"
            sb_dir = root / "storyboards"
            log = root / "run.log"

            windows.write_text(
                json.dumps(
                    {
                        "video": {"path": str(dummy.resolve()), "fps": 10, "total_frames": 40},
                        "windows": [
                            {
                                "window_id": 1,
                                "start_frame": 0,
                                "end_frame": 19,
                                "track_ids": [1, 2],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            lines = []
            for f in range(20):
                lines.append(
                    json.dumps(
                        {
                            "frame": f,
                            "tracks": [
                                {"track_id": 1, "bbox": [10, 10, 50, 90]},
                                {"track_id": 2, "bbox": [200, 10, 240, 90]},
                            ],
                        }
                    )
                )
            tracks.write_text("\n".join(lines) + "\n", encoding="utf-8")

            run_clip_relation(
                windows_json=windows,
                tracks_jsonl=tracks,
                out_json=out,
                storyboards_dir=sb_dir,
                config={"max_windows": 1, "max_frames_per_window": 4},
                api_key="",
                resume=False,
                dry_run=True,
                video_id="validation_dummy",
                log_path=log,
            )

            self.assertTrue(out.exists())
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("validation_dummy", data)
            self.assertIsInstance(data["validation_dummy"], list)
            storyboards = list(sb_dir.glob("seg_*.jpg"))
            self.assertGreaterEqual(len(storyboards), 1)
            for sb in storyboards:
                self.assertGreater(sb.stat().st_size, 0)

    @unittest.skipUnless(HAS_CV, "opencv required")
    def test_storyboard_save_unicode_dir(self) -> None:
        with TemporaryDirectory(prefix="vidvrd关系_") as td:
            path = Path(td) / "seg_0001.jpg"
            img = np.zeros((24, 24, 3), dtype=np.uint8)
            save_storyboard_image(path, img)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
