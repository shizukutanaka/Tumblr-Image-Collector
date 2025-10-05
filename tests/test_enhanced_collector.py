import os
from unittest.mock import Mock, patch

import pytest

from tumblr_image_collector_enhanced import EnhancedTumblrImageCollector


class DummyCredentialsManager:
    """テスト用の簡易認証情報マネージャー"""

    def __init__(self):
        self.storage = {}

    def store_credential(self, key, value, encrypt=True):
        self.storage[key] = value
        return True

    def retrieve_credential(self, key, decrypt=True):
        return self.storage.get(key)


class DummyMetricsCollector:
    def __init__(self):
        self.recorded = []

    def record_metric(self, name, value, tags=None):
        self.recorded.append((name, value, tags))


class DummyErrorTracker:
    def __init__(self):
        self.errors = []

    def track_error(self, error, context=None):
        self.errors.append((error, context))


@pytest.mark.asyncio
@patch("security_manager.SecurityManager")
@patch("tumblr_image_collector_enhanced.pytumblr.TumblrRestClient")
async def test_initialize_api_uses_environment_credentials(mock_client_class, mock_security_manager):
    """環境変数で指定した資格情報を優先しセキュアストレージへ保存することを検証"""

    mock_security_manager.return_value.audit_log.return_value = None
    mock_client = Mock()
    mock_client.info.return_value = {"user": {"name": "env-user"}}
    mock_client_class.return_value = mock_client

    credentials_manager = DummyCredentialsManager()
    metrics_collector = DummyMetricsCollector()
    error_tracker = DummyErrorTracker()

    collector = EnhancedTumblrImageCollector.__new__(EnhancedTumblrImageCollector)
    collector.credentials_manager = credentials_manager
    collector.metrics_collector = metrics_collector
    collector.error_tracker = error_tracker
    collector.stats = {"failed_downloads": 0, "total_downloads": 0, "api_rate_limited": False}
    collector.tumblr_client = None
    collector.api_initialized = False

    env_values = {
        "TUMBLR_CONSUMER_KEY": "env_consumer_key",
        "TUMBLR_CONSUMER_SECRET": "env_consumer_secret",
        "TUMBLR_OAUTH_TOKEN": "env_token",
        "TUMBLR_OAUTH_TOKEN_SECRET": "env_token_secret",
    }

    with patch.dict(os.environ, env_values, clear=True):
        result = await collector.initialize_api()

    assert result is True
    assert collector.api_initialized is True
    assert credentials_manager.storage["tumblr_consumer_key"] == "env_consumer_key"
    assert credentials_manager.storage["tumblr_consumer_secret"] == "env_consumer_secret"
    assert credentials_manager.storage["tumblr_oauth_token"] == "env_token"
    assert credentials_manager.storage["tumblr_oauth_token_secret"] == "env_token_secret"
    assert ("api.initialized", 1, None) in metrics_collector.recorded


@pytest.mark.asyncio
@patch("security_manager.SecurityManager")
@patch("tumblr_image_collector_enhanced.pytumblr.TumblrRestClient")
async def test_initialize_api_requires_secure_credentials(mock_client_class, mock_security_manager):
    """セキュアストレージに資格情報が無い場合は初期化が失敗することを検証"""

    mock_security_manager.return_value.audit_log.return_value = None
    mock_client_class.return_value = Mock()

    credentials_manager = DummyCredentialsManager()
    metrics_collector = DummyMetricsCollector()
    error_tracker = DummyErrorTracker()

    collector = EnhancedTumblrImageCollector.__new__(EnhancedTumblrImageCollector)
    collector.credentials_manager = credentials_manager
    collector.metrics_collector = metrics_collector
    collector.error_tracker = error_tracker
    collector.stats = {"failed_downloads": 0, "total_downloads": 0, "api_rate_limited": False}
    collector.tumblr_client = None
    collector.api_initialized = False

    with patch.dict(os.environ, {}, clear=True):
        result = await collector.initialize_api()

    assert result is False
    assert collector.api_initialized is False
    mock_client_class.assert_not_called()
