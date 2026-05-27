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

from vidvrd_auto.nodes.global_relation import run_global_relation  # noqa: E402
from vidvrd_auto.nodes.screen import screen_keyframes  # noqa: E402
from vidvrd_auto.nodes.track_qc import run_track_qc  # noqa: E402

try:
    import cv2  # noqa: F401

    HAS_CV = True
except ImportError:
    HAS_CV = False


class VLNodeImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dummy = ROOT / "data" / "validation_dummy.mp4"
        if not cls.dummy.exists():
            import subprocess

            subprocess.check_call([sys.executable, str(ROOT / "scripts" / "make_validation_dummy.py")], cwd=str(ROOT))

    @unittest.skipUnless(HAS_CV, "opencv required")
    def test_screen_vl_uses_video_frames(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            det = root / "detections.jsonl"
            meta = root / "video_meta.json"
            out = root / "screen.json"
            det.write_text(
                json.dumps({"frame": 0, "objects": [{"bbox": [10, 10, 50, 90], "confidence": 0.9}, {"bbox": [200, 10, 240, 90], "confidence": 0.9}]})
                + "\n"
                + json.dumps({"frame": 10, "objects": [{"bbox": [12, 10, 52, 90], "confidence": 0.9}, {"bbox": [200, 10, 240, 90], "confidence": 0.9}]})
                + "\n",
                encoding="utf-8",
            )
            meta.write_text(
                json.dumps({"video": {"path": str(self.dummy.resolve()), "fps": 10, "total_frames": 40}}, ensure_ascii=False),
                encoding="utf-8",
            )
            # detections in step1 dir layout: meta sibling
            det_dir = root / "step1"
            det_dir.mkdir()
            det_path = det_dir / "detections_full.jsonl"
            det_path.write_text(det.read_text(encoding="utf-8"), encoding="utf-8")
            (det_dir / "video_meta.json").write_text(meta.read_text(encoding="utf-8"), encoding="utf-8")

            result = screen_keyframes(
                detections_jsonl=det_path,
                out_json=out,
                config={"sample_frames": 2, "max_frame_index": 20, "min_objects": 2, "vl_enabled": True, "vl_dry_run": True},
            )
            self.assertTrue(result["vl_screen"].get("used_images"))

    @unittest.skipUnless(HAS_CV, "opencv required")
    def test_track_qc_vl_uses_risk_frames(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            tracks = root / "tracks.jsonl"
            windows = root / "windows.json"
            out = root / "track_qc.json"
            lines = []
            for f in range(5):
                bbox = [10 + f * 30, 10, 50 + f * 30, 90]
                lines.append(json.dumps({"frame": f, "tracks": [{"track_id": 1, "bbox": bbox, "class_name": "person"}]}))
            tracks.write_text("\n".join(lines) + "\n", encoding="utf-8")
            windows.write_text(
                json.dumps(
                    {"video": {"path": str(self.dummy.resolve()), "fps": 10, "total_frames": 40}, "windows": [{"window_id": 1, "start_frame": 0, "end_frame": 4, "track_ids": [1]}]},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            result = run_track_qc(
                tracks_jsonl=tracks,
                windows_json=windows,
                out_json=out,
                config={"min_track_frames": 10, "max_center_jump_ratio": 0.01, "vl_enabled": True, "vl_dry_run": True},
            )
            self.assertGreater(result["large_jump_count"], 0)
            self.assertTrue(result["vl_review"].get("used_images"))

    @unittest.skipUnless(HAS_CV, "opencv required")
    def test_global_relation_vl_uses_timeline_sample(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            rel = root / "merged.json"
            windows = root / "windows.json"
            out = root / "global.json"
            rel.write_text(json.dumps({"demo": [{"subject_track_id": 1, "object_track_id": 2, "predicate": "left", "start_frame": 0, "end_frame": 10, "confidence": 0.9, "source": "rule"}]}), encoding="utf-8")
            windows.write_text(
                json.dumps({"video": {"path": str(self.dummy.resolve()), "fps": 10, "total_frames": 40}, "windows": []}, ensure_ascii=False),
                encoding="utf-8",
            )
            result = run_global_relation(
                video_id="demo",
                relations_json=rel,
                out_json=out,
                config={"vl_enabled": True, "vl_dry_run": True, "vl_sample_frames": 4},
                windows_json=windows,
            )
            review = result.get("_global_review", {})
            self.assertTrue(review.get("used_images"))


if __name__ == "__main__":
    unittest.main()
