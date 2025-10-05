"""Tests for Tumblr Image Collector core utilities."""

import json
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import config_wizard
import pytest

import tumblr_image_collector
from tumblr_image_collector import TumblrImageCollector


@pytest.fixture()
def minimal_config(tmp_path):
    """Create a minimal configuration file pointing to a temporary directory."""

    output_dir = tmp_path / "images"
    config_payload = {
        "output_folder_name": str(output_dir),
        "cache": {"enabled": False},
        "enable_deep_model": False,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    return config_path


def _stub_collector_dependencies(stack: ExitStack) -> None:
    """Patch heavy side-effecting methods for safe instantiation during tests."""

    stack.enter_context(patch.object(TumblrImageCollector, "_setup_logging", return_value=None))
    stack.enter_context(patch.object(TumblrImageCollector, "_setup_proxy", return_value=None))
    stack.enter_context(patch.object(TumblrImageCollector, "_setup_output_directory", return_value=None))
    stack.enter_context(patch.object(TumblrImageCollector, "_load_downloaded_files", return_value=None))
    stack.enter_context(patch.object(TumblrImageCollector, "_initialize_client", return_value=MagicMock()))
    stack.enter_context(patch.object(TumblrImageCollector, "_create_requests_session", return_value=MagicMock()))
    stack.enter_context(patch.object(TumblrImageCollector, "_build_requests_proxies", return_value={}))
    stack.enter_context(patch.object(TumblrImageCollector, "_create_rate_limiter", return_value={}))
    stack.enter_context(patch.object(TumblrImageCollector, "_setup_credentials", return_value=None))
    stack.enter_context(patch.object(TumblrImageCollector, "_load_cache_index", return_value={}))
    stack.enter_context(patch("tumblr_image_collector.ImageClassifier", autospec=True))


def test_load_dotenv_called_when_available(minimal_config):
    with ExitStack() as stack:
        mock_dotenv = stack.enter_context(patch("tumblr_image_collector.load_dotenv"))
        stack.enter_context(patch("tumblr_image_collector._DOTENV_AVAILABLE", True))
        _stub_collector_dependencies(stack)

        TumblrImageCollector(config_file=str(minimal_config))

        mock_dotenv.assert_called_once()


def test_load_dotenv_skipped_when_unavailable(minimal_config):
    with ExitStack() as stack:
        mock_dotenv = stack.enter_context(patch("tumblr_image_collector.load_dotenv"))
        stack.enter_context(patch("tumblr_image_collector._DOTENV_AVAILABLE", False))
        _stub_collector_dependencies(stack)

        TumblrImageCollector(config_file=str(minimal_config))

        mock_dotenv.assert_not_called()


def test_config_wizard_main_outputs_sanitized_config(monkeypatch, capsys):
    raw_config = {"consumer_key": "secret", "output_folder_name": "images"}
    sanitized_config = {"consumer_key": "***REDACTED***", "output_folder_name": "images"}

    class DummyWizard:  # pragma: no cover - simple stub
        """Minimal wizard returning deterministic configuration for tests."""

        def run(self):
            return raw_config

        @staticmethod
        def _sanitize_config_for_display(config):
            assert config == raw_config
            return sanitized_config

    monkeypatch.setattr(config_wizard, "ConfigWizard", DummyWizard)

    config_wizard.main()

    captured = capsys.readouterr()
    assert "Configuration details" in captured.out
    assert "secret" not in captured.out

    json_start = captured.out.index("{")
    rendered = captured.out[json_start:]
    parsed = json.loads(rendered)
    assert parsed == sanitized_config
