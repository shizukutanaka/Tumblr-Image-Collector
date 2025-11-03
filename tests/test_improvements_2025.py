"""
Comprehensive tests for 2024-2025 improvements

Tests cover:
- Advanced HTTP client (async/sync)
- Structured logging
- Configuration validation
- Multi-tier caching
- Image deduplication
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from datetime import datetime
import time

# Import modules to test
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from advanced_http_client import AsyncHTTPClient, SyncHTTPClient, ClientConfig
from advanced_logging import (
    initialize_logging,
    get_logger,
    AdvancedLoggingConfig,
    SensitiveDataFilter,
)
from pydantic_config import (
    TumblrCollectorConfig,
    TumblrConfig,
    FilterMode,
    validate_config_file,
)
from advanced_cache import AdvancedCache, initialize_cache, get_cache
from advanced_deduplication import ImageDeduplicator


class TestAdvancedHttpClient:
    """Tests for advanced HTTP client."""

    @pytest.mark.asyncio
    async def test_async_client_initialization(self):
        """Test AsyncHTTPClient initialization."""
        config = ClientConfig(max_connections=50)
        client = AsyncHTTPClient(config)

        assert client.config.max_connections == 50
        assert client._circuit_breaker_state["failures"] == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_open(self):
        """Test circuit breaker opens after failures."""
        config = ClientConfig()
        client = AsyncHTTPClient(config)

        # Simulate failures
        for _ in range(5):
            client._record_failure()

        # Circuit breaker should be open
        assert client._check_circuit_breaker()

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        config = ClientConfig()
        client = AsyncHTTPClient(config)

        # Simulate failures
        for _ in range(5):
            client._record_failure()

        # Circuit breaker is open
        assert client._check_circuit_breaker()

        # Manually advance failure time beyond recovery timeout
        client._circuit_breaker_state["last_failure_time"] = time.time() - 61

        # Circuit breaker should be reset
        assert not client._check_circuit_breaker()

    @pytest.mark.asyncio
    async def test_rate_limiter_integration(self):
        """Test rate limiter is integrated."""
        config = ClientConfig(rate_limit_calls=100, rate_limit_period_seconds=60)
        client = AsyncHTTPClient(config)

        assert client._rate_limiter is not None

    def test_sync_client_initialization(self):
        """Test SyncHTTPClient initialization."""
        config = ClientConfig(max_connections=50)
        client = SyncHTTPClient(config)

        assert client.config.max_connections == 50
        assert client._circuit_breaker_state["failures"] == 0

        client.close()

    def test_sync_client_context_manager(self):
        """Test SyncHTTPClient as context manager."""
        with SyncHTTPClient() as client:
            assert client._client is not None

        # Client should be closed after context
        assert client._client is None


class TestAdvancedLogging:
    """Tests for advanced logging system."""

    def test_sensitive_data_filter_redaction(self):
        """Test sensitive data filter redacts credentials."""
        filter_obj = SensitiveDataFilter()

        # Test string redaction
        text = "API_KEY=secret123456"
        redacted = filter_obj._redact_string(text)
        assert "secret123456" not in redacted or "*" in redacted

        # Test dict redaction
        data = {"password": "secret", "username": "user", "name": "John"}
        redacted = filter_obj._redact_dict(data)
        assert redacted["password"] == "*" * 10
        assert redacted["username"] == "user"

    def test_logging_configuration(self):
        """Test logging configuration setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AdvancedLoggingConfig(
                log_dir=tmpdir,
                log_level="INFO",
                json_output=False,
            )

            assert Path(tmpdir).exists()
            logger = config.get_logger("test")
            assert logger is not None

    def test_logging_initialization(self):
        """Test global logging initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from advanced_logging import _logging_config, initialize_logging

            logging_config = initialize_logging(log_dir=tmpdir, log_level="DEBUG")

            assert logging_config is not None
            logger = get_logger("test")
            assert logger is not None

    def test_performance_logger(self):
        """Test performance logger."""
        from advanced_logging import get_performance_logger

        perf_logger = get_performance_logger()
        assert perf_logger is not None

        # Should not raise exceptions
        perf_logger.log_operation_time("test_op", 1.5)
        perf_logger.log_cache_hit("test_key", "memory")
        perf_logger.log_cache_miss("test_key", "memory")


class TestPydanticConfig:
    """Tests for Pydantic configuration."""

    def test_tumblr_config_validation(self):
        """Test Tumblr configuration validation."""
        config = TumblrConfig(
            consumer_key="key123",
            consumer_secret="secret123",
            token="token123",
            token_secret="token_secret123",
        )

        assert config.consumer_key == "key123"
        assert config.rate_limit_calls == 1000

    def test_tumblr_config_empty_credentials(self):
        """Test validation of empty credentials."""
        with pytest.raises(ValueError, match="Credentials cannot be empty"):
            TumblrConfig(
                consumer_key="",
                consumer_secret="secret",
                token="token",
                token_secret="token_secret",
            )

    def test_filter_config_date_validation(self):
        """Test date format validation."""
        from pydantic_config import FilterConfig

        # Valid dates
        config = FilterConfig(
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        assert config.start_date == "2024-01-01"

        # Invalid date format
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            FilterConfig(start_date="01/01/2024")

    def test_filter_config_date_range(self):
        """Test date range validation."""
        from pydantic_config import FilterConfig

        # start_date after end_date should fail
        with pytest.raises(ValueError, match="start_date must be before end_date"):
            FilterConfig(
                start_date="2024-12-31",
                end_date="2024-01-01",
            )

    def test_main_config_to_file(self):
        """Test saving configuration to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TumblrCollectorConfig(
                tumblr=TumblrConfig(
                    consumer_key="key",
                    consumer_secret="secret",
                    token="token",
                    token_secret="token_secret",
                ),
                log_level="INFO",
            )

            config_file = Path(tmpdir) / "config.json"
            config.to_file(str(config_file))

            assert config_file.exists()

    def test_main_config_from_file(self):
        """Test loading configuration from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_data = {
                "tumblr": {
                    "consumer_key": "key",
                    "consumer_secret": "secret",
                    "token": "token",
                    "token_secret": "token_secret",
                },
                "log_level": "DEBUG",
            }

            config_file = Path(tmpdir) / "config.json"
            with open(config_file, "w") as f:
                json.dump(config_data, f)

            config = TumblrCollectorConfig.from_file(str(config_file))
            assert config.log_level == "DEBUG"

    def test_config_summary(self):
        """Test configuration summary generation."""
        config = TumblrCollectorConfig(
            tumblr=TumblrConfig(
                consumer_key="key",
                consumer_secret="secret",
                token="token",
                token_secret="token_secret",
            ),
        )

        summary = config.get_summary()
        assert "Tumblr Image Collector" in summary
        assert "INFO" in summary  # Default log level


class TestAdvancedCache:
    """Tests for advanced caching system."""

    def test_cache_initialization(self):
        """Test cache initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AdvancedCache(cache_dir=tmpdir, ttl_hours=1)

            assert cache.cache_dir.exists()
            assert cache.ttl_seconds == 3600

    def test_cache_set_and_get(self):
        """Test setting and getting cache values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AdvancedCache(cache_dir=tmpdir)

            cache.set("test_key", "test_value", namespace="test")
            value = cache.get("test_key", namespace="test")

            assert value == "test_value"

    def test_cache_miss(self):
        """Test cache miss returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AdvancedCache(cache_dir=tmpdir)

            value = cache.get("nonexistent_key")
            assert value is None

    def test_cache_delete(self):
        """Test deleting cache values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AdvancedCache(cache_dir=tmpdir)

            cache.set("test_key", "test_value")
            cache.delete("test_key")

            value = cache.get("test_key")
            assert value is None

    def test_cache_clear_namespace(self):
        """Test clearing specific namespace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AdvancedCache(cache_dir=tmpdir)

            cache.set("key1", "value1", namespace="ns1")
            cache.set("key2", "value2", namespace="ns2")

            cache.clear(namespace="ns1")

            assert cache.get("key1", namespace="ns1") is None
            assert cache.get("key2", namespace="ns2") == "value2"

    def test_cache_clear_all(self):
        """Test clearing all cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AdvancedCache(cache_dir=tmpdir)

            cache.set("key1", "value1")
            cache.set("key2", "value2")

            cache.clear()

            assert cache.get("key1") is None
            assert cache.get("key2") is None

    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AdvancedCache(cache_dir=tmpdir)

            cache.set("key1", "value1")
            cache.set("key2", "value2")

            cache.get("key1")  # Hit
            cache.get("key1")  # Hit
            cache.get("missing")  # Miss

            stats = cache.get_stats()

            assert stats["hits"] >= 2
            assert stats["misses"] >= 1
            assert stats["hit_rate_percent"] > 0

    def test_cache_decorator(self):
        """Test cache decorator for function results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AdvancedCache(cache_dir=tmpdir)

            call_count = 0

            @cache.cache_function(namespace="functions")
            def expensive_function(n):
                nonlocal call_count
                call_count += 1
                return n * 2

            # First call - executes function
            result1 = expensive_function(5)
            assert result1 == 10
            assert call_count == 1

            # Second call - returns cached
            result2 = expensive_function(5)
            assert result2 == 10
            assert call_count == 1  # Not incremented

    def test_cache_expiration(self):
        """Test cache expiration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AdvancedCache(cache_dir=tmpdir, ttl_hours=0.0001)  # ~0.36 seconds

            cache.set("key1", "value1")

            # Immediately get - should hit
            assert cache.get("key1") == "value1"

            # After expiration - should miss
            time.sleep(1)
            assert cache.get("key1") is None


class TestImageDeduplication:
    """Tests for image deduplication."""

    def test_deduplicator_initialization(self):
        """Test deduplicator initialization."""
        deduplicator = ImageDeduplicator(method="perceptual")

        assert deduplicator.method == "perceptual"
        assert deduplicator.perceptual_threshold == 0.9

    def test_file_hash_calculation(self):
        """Test file hash calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            test_file = Path(tmpdir) / "test.bin"
            test_file.write_bytes(b"test content")

            deduplicator = ImageDeduplicator()
            file_hash = deduplicator._calculate_file_hash(test_file)

            assert len(file_hash) == 64  # SHA256 hex string
            assert all(c in "0123456789abcdef" for c in file_hash)

    def test_file_hash_consistency(self):
        """Test that file hash is consistent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.bin"
            test_file.write_bytes(b"same content")

            deduplicator = ImageDeduplicator()

            hash1 = deduplicator._calculate_file_hash(test_file)
            hash2 = deduplicator._calculate_file_hash(test_file)

            assert hash1 == hash2

    def test_duplicate_report_generation(self):
        """Test duplicate report generation."""
        from advanced_deduplication import DuplicateMatch

        deduplicator = ImageDeduplicator()

        match = DuplicateMatch(
            original=Path("original.jpg"),
            duplicate=Path("duplicate.jpg"),
            method="exact",
            similarity=1.0,
            size_original=1000,
            size_duplicate=1000,
        )

        report = deduplicator.get_duplicate_report([match])

        assert report["total_duplicates"] == 1
        assert report["total_wasted_bytes"] == 1000
        assert report["by_method"]["exact"] == 1


class TestIntegration:
    """Integration tests combining multiple modules."""

    @pytest.mark.asyncio
    async def test_http_client_with_logging(self):
        """Test HTTP client with structured logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            initialize_logging(log_dir=tmpdir)
            logger = get_logger("test")

            config = ClientConfig()
            client = AsyncHTTPClient(config)

            # Should not raise exceptions
            assert client._circuit_breaker_state["failures"] == 0

            await client.close()

    def test_cache_with_config(self):
        """Test cache with configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TumblrCollectorConfig(
                tumblr=TumblrConfig(
                    consumer_key="key",
                    consumer_secret="secret",
                    token="token",
                    token_secret="token_secret",
                ),
                cache={"cache_dir": tmpdir},
            )

            cache = AdvancedCache(cache_dir=str(config.cache.cache_dir))
            cache.set("config_key", "config_value")

            assert cache.get("config_key") == "config_value"

    def test_full_workflow(self):
        """Test full workflow with all modules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize logging
            initialize_logging(log_dir=f"{tmpdir}/logs")
            logger = get_logger("workflow")

            # Create configuration
            config = TumblrCollectorConfig(
                tumblr=TumblrConfig(
                    consumer_key="key",
                    consumer_secret="secret",
                    token="token",
                    token_secret="token_secret",
                ),
                cache={"cache_dir": f"{tmpdir}/cache"},
            )

            # Initialize cache
            cache = initialize_cache(cache_dir=str(config.cache.cache_dir))

            # Log workflow start
            logger.info("workflow_started", config_log_level=config.log_level)

            # Cache some data
            cache.set("workflow_data", {"status": "processing"})

            # Get cached data
            data = cache.get("workflow_data")
            assert data is not None

            # Get cache stats
            stats = cache.get_stats()
            assert stats["hits"] >= 0


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
