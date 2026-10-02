from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from vidvrd_auto.providers import DashScopeProvider, VLProvider, VLResult, image_to_data_uri


def _response(text: str) -> SimpleNamespace:
    message = SimpleNamespace(content=[{"text": text}])
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(output=SimpleNamespace(choices=[choice]))


class VLProviderTests(unittest.TestCase):
    def test_provider_dry_run_is_counted(self) -> None:
        client = DashScopeProvider({"dry_run": True, "model": "mock-vl"})

        result = client.call(prompt="review", image_paths=[Path("not-read.jpg")])

        self.assertIsInstance(client, VLProvider)
        self.assertIsInstance(result, VLResult)
        self.assertTrue(result.ok)
        self.assertTrue(result.dry_run)
        self.assertEqual(result.model, "mock-vl")
        self.assertEqual(client.stats.to_dict()["dry_runs"], 1)
        self.assertEqual(client.stats.images, 1)

    def test_image_encoding_uses_normalized_mime_type(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "frame.JPG"
            image.write_bytes(b"image-data")

            uri = image_to_data_uri(image)

        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))

    def test_dashscope_call_encodes_images_and_returns_text(self) -> None:
        fake = ModuleType("dashscope")
        fake.api_key = ""
        conversation = unittest.mock.Mock()
        conversation.call.return_value = _response("accepted")
        fake.MultiModalConversation = conversation

        with TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "frame.png"
            image.write_bytes(b"png-data")
            with patch.dict(sys.modules, {"dashscope": fake}):
                client = DashScopeProvider({"model": "qwen-test", "retries": 0}, api_key="secret")
                result = client.call(prompt="inspect", image_paths=[image])

        self.assertTrue(result.ok)
        self.assertEqual(result.text, "accepted")
        self.assertEqual(result.attempts, 1)
        self.assertEqual(fake.api_key, "secret")
        content = conversation.call.call_args.kwargs["messages"][0]["content"]
        self.assertTrue(content[0]["image"].startswith("data:image/png;base64,"))
        self.assertEqual(content[-1], {"text": "inspect"})
        self.assertEqual(conversation.call.call_args.kwargs["request_timeout"], 180)
        self.assertEqual(client.stats.succeeded, 1)
        self.assertEqual(client.stats.attempts, 1)

    def test_retry_and_failure_stats(self) -> None:
        fake = ModuleType("dashscope")
        fake.api_key = ""
        conversation = unittest.mock.Mock(side_effect=[RuntimeError("busy"), _response("ok")])
        fake.MultiModalConversation = SimpleNamespace(call=conversation)

        with patch.dict(sys.modules, {"dashscope": fake}), patch("vidvrd_auto.providers.dashscope.time.sleep"):
            client = DashScopeProvider({"retries": 1, "backoff_sec": 0.01}, api_key="secret")
            result = client.call(prompt="retry")

        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(client.stats.attempts, 2)
        self.assertEqual(client.stats.retries, 1)
        self.assertEqual(client.stats.failed, 0)

    def test_api_error_exposes_status_without_losing_provider_failure(self) -> None:
        fake = ModuleType("dashscope")
        fake.api_key = ""
        response = SimpleNamespace(
            status_code=403, code="AllocationQuota.FreeTierOnly",
            message="quota exhausted", output=None,
        )
        fake.MultiModalConversation = SimpleNamespace(call=unittest.mock.Mock(return_value=response))
        with patch.dict(sys.modules, {"dashscope": fake}):
            client = DashScopeProvider({"retries": 0}, api_key="secret")
            result = client.call(prompt="inspect")
        self.assertFalse(result.ok)
        self.assertIn("status=403", result.error)
        self.assertIn("AllocationQuota.FreeTierOnly", result.error)

    def test_content_inspection_failure_is_not_retried(self) -> None:
        fake = ModuleType("dashscope")
        fake.api_key = ""
        response = SimpleNamespace(
            status_code=400, code="DataInspectionFailed",
            message="Input image data may contain inappropriate content", output=None,
        )
        conversation = unittest.mock.Mock(return_value=response)
        fake.MultiModalConversation = SimpleNamespace(call=conversation)
        with patch.dict(sys.modules, {"dashscope": fake}):
            client = DashScopeProvider({"retries": 3}, api_key="secret")
            result = client.call(prompt="inspect")
        self.assertFalse(result.ok)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(conversation.call_count, 1)

    def test_missing_key_returns_structured_failure(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            client = DashScopeProvider({"api_key_env": "DASHSCOPE_API_KEY"})
            result = client.call(prompt="inspect")

        self.assertFalse(result.ok)
        self.assertIn("missing", result.error)
        self.assertEqual(result.attempts, 0)
        self.assertEqual(client.stats.failed, 1)

    def test_usage_accounting_preserves_unknown_usage(self) -> None:
        fake = ModuleType("dashscope")
        response = _response("ok")
        response.usage = {"input_tokens": 120, "output_tokens": 15}
        conversation = unittest.mock.Mock(side_effect=[response, _response("ok")])
        fake.MultiModalConversation = SimpleNamespace(call=conversation)
        with patch.dict(sys.modules, {"dashscope": fake}):
            client = DashScopeProvider({"retries": 0}, api_key="secret")
            measured = client.call(prompt="inspect")
            unknown = client.call(prompt="inspect")
        self.assertEqual(measured.input_tokens, 120)
        self.assertIsNone(unknown.input_tokens)
        self.assertEqual(client.stats.usage_reported_calls, 1)
        self.assertEqual(client.stats.output_tokens, 15)

    def test_explicit_non_thinking_mode_reaches_api(self) -> None:
        fake = ModuleType("dashscope")
        conversation = unittest.mock.Mock(return_value=_response("ok"))
        fake.MultiModalConversation = SimpleNamespace(call=conversation)
        with patch.dict(sys.modules, {"dashscope": fake}):
            client = DashScopeProvider({"model": "qwen3.6-plus", "enable_thinking": False}, api_key="secret")
            result = client.call(prompt="inspect")
        self.assertTrue(result.ok)
        self.assertIs(conversation.call_args.kwargs["enable_thinking"], False)
        with self.assertRaises(ValueError):
            DashScopeProvider({"enable_thinking": "false"})

    def test_quota_failure_stops_without_retries(self) -> None:
        fake = ModuleType("dashscope")
        conversation = unittest.mock.Mock(return_value=SimpleNamespace(
            status_code=403, code="AllocationQuota.FreeTierOnly", message="quota exhausted", output=None))
        fake.MultiModalConversation = SimpleNamespace(call=conversation)
        with patch.dict(sys.modules, {"dashscope": fake}):
            result = DashScopeProvider({"retries": 3}, api_key="secret").call(prompt="inspect")
        self.assertFalse(result.ok)
        self.assertEqual(conversation.call_count, 1)


if __name__ == "__main__":
    unittest.main()
