from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


class OfficialSourceTests(unittest.TestCase):
    def test_vendored_files_match_pinned_upstream_hashes(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "vidvrd_auto" / "tracking" / "third_party" / "oc_sort"
        expected = {
            "ocsort.py": "c900be251d1ce01483880ad5b144195072153cb0810d2f2d0c5db256628982db",
            "association.py": "71c5e97b6f93472b98f80cb754294daa7606d78fe40aa8a838a938e51507d504",
            "kalmanfilter.py": "96c859ec913640e3e6ebb1e5cdc2c6f0b94bdb140a6d54c3a8586868e5dc374a",
        }
        for name, digest in expected.items():
            normalized = root.joinpath(name).read_text(encoding="utf-8").replace("\r\n", "\n").encode()
            self.assertEqual(hashlib.sha256(normalized).hexdigest(), digest, name)


if __name__ == "__main__":
    unittest.main()
