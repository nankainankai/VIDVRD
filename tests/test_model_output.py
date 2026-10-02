from __future__ import annotations

import unittest

from vidvrd_auto.utils.model_output import parse_json_object


class ModelOutputTests(unittest.TestCase):
    def test_parses_explanation_before_fenced_json(self) -> None:
        response = 'I saw a person.\n```json\n{"actions":[{"action":"reject_relation"}]}\n```'
        self.assertEqual(parse_json_object(response, required_key="actions")["actions"][0]["action"], "reject_relation")

    def test_ignores_unrelated_json_objects(self) -> None:
        response = '{"example":true}\n{"packet_results":[]}'
        self.assertEqual(parse_json_object(response, required_key="packet_results"), {"packet_results": []})

    def test_missing_required_key_returns_empty_object(self) -> None:
        self.assertEqual(parse_json_object('no result {"foo": 1}', required_key="actions"), {})

    def test_repairs_only_trailing_structural_closer(self) -> None:
        truncated = '{"packet_results":[{"packet_id":"p1","actions":[]}]'
        self.assertEqual(
            parse_json_object(truncated, required_key="packet_results"),
            {"packet_results": [{"packet_id": "p1", "actions": []}]},
        )
        self.assertEqual(parse_json_object('{"actions":[{"reason":"unfinished', required_key="actions"), {})


if __name__ == "__main__":
    unittest.main()
