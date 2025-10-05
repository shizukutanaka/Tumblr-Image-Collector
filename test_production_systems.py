#!/usr/bin/env python3
"""
Production Systems Integration Tests
Comprehensive testing for production-grade components
"""

import pytest
import time
import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import production modules
from production_url_manager import ProductionURLManager, URLStatus
from production_security import (
    InputSanitizer, RateLimiter, DDoSProtection, SecurityAuditor
)
from production_error_handler import (
    CircuitBreaker, RetryStrategy, ErrorRecoveryManager,
    ErrorCategory, ErrorSeverity, GracefulDegradation
)
from production_monitoring import (
    MetricsCollector, SystemMonitor, HealthChecker, HealthStatus,
    PerformanceMonitor, MonitoringDashboard
)


class TestProductionURLManager:
    """Test URL management system"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.manager = ProductionURLManager(db_path=self.temp_db.name)

    def teardown_method(self):
        """Cleanup test environment"""
        self.manager.close()
        Path(self.temp_db.name).unlink(missing_ok=True)

    def test_url_security_validation(self):
        """Test URL security validation"""
        # Valid Tumblr URL
        is_safe, reason = self.manager.validate_url_security("https://example.tumblr.com")
        assert is_safe, f"Valid URL rejected: {reason}"

        # SSRF attack (private IP)
        is_safe, reason = self.manager.validate_url_security("http://192.168.1.1/test")
        assert not is_safe, "Private IP not blocked"
        assert "Private IP" in reason

        # Path traversal
        is_safe, reason = self.manager.validate_url_security("https://example.tumblr.com/../../../etc/passwd")
        assert not is_safe, "Path traversal not detected"

        # XSS attempt
        is_safe, reason = self.manager.validate_url_security("https://example.tumblr.com/<script>alert('xss')</script>")
        assert not is_safe, "XSS pattern not detected"

        # Non-Tumblr domain
        is_safe, reason = self.manager.validate_url_security("https://malicious-site.com")
        assert not is_safe, "Non-Tumblr domain not blocked"

    def test_url_classification(self):
        """Test URL type classification"""
        test_cases = [
            ("https://myblog.tumblr.com", "blog"),
            ("https://myblog.tumblr.com/post/123456789", "post"),
            ("https://64.media.tumblr.com/abc123/tumblr_xyz789_1280.jpg", "image"),
            ("https://myblog.tumblr.com/tagged/photography", "tag"),
            ("https://myblog.tumblr.com/archive", "archive"),
        ]

        for url, expected_type in test_cases:
            url_type = self.manager.classify_url_type(url)
            assert url_type == expected_type, f"URL {url} misclassified as {url_type}, expected {expected_type}"

    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        assert self.manager._check_circuit_breaker(), "Circuit should be closed initially"

        # Trigger failures
        for _ in range(self.manager._circuit_breaker['failure_threshold']):
            self.manager._record_failure()

        assert not self.manager._check_circuit_breaker(), "Circuit should be open after threshold"

    def test_rate_limiting(self):
        """Test rate limiting enforcement"""
        initial_count = self.manager._rate_limiter['request_count']

        # Make requests
        for _ in range(5):
            self.manager._enforce_rate_limit()

        assert self.manager._rate_limiter['request_count'] > initial_count, "Rate limiter not counting requests"

    def test_cleanup_invalid_urls(self):
        """Test invalid URL cleanup"""
        urls = [
            "https://valid-blog.tumblr.com",
            "https://192.168.1.1/ssrf",  # Invalid: private IP
            "https://malicious.com",  # Invalid: non-Tumblr
            "<script>xss</script>",  # Invalid: XSS
        ]

        valid_urls = self.manager.cleanup_invalid_urls(urls)

        assert len(valid_urls) == 1, f"Expected 1 valid URL, got {len(valid_urls)}"
        assert valid_urls[0] == "https://valid-blog.tumblr.com"


class TestInputSanitizer:
    """Test input sanitization"""

    def test_string_sanitization(self):
        """Test string input sanitization"""
        sanitizer = InputSanitizer()

        # Valid blog name
        is_valid, sanitized, error = sanitizer.sanitize_string("my-blog", "blog_name")
        assert is_valid, f"Valid blog name rejected: {error}"

        # XSS attempt
        is_valid, sanitized, error = sanitizer.sanitize_string("<script>alert('xss')</script>", "tag")
        assert not is_valid, "XSS pattern not detected"

        # SQL injection
        is_valid, sanitized, error = sanitizer.sanitize_string("'; DROP TABLE users--", "general")
        assert not is_valid, "SQL injection not detected"

        # Too long
        is_valid, sanitized, error = sanitizer.sanitize_string("a" * 10000, "blog_name")
        assert not is_valid, "Length limit not enforced"

    def test_path_sanitization(self):
        """Test path sanitization"""
        sanitizer = InputSanitizer()
        base_dir = Path("/safe/dir")

        # Valid path
        is_valid, path, error = sanitizer.sanitize_path("subdir/file.txt", base_dir)
        assert is_valid, f"Valid path rejected: {error}"

        # Path traversal
        is_valid, path, error = sanitizer.sanitize_path("../../etc/passwd", base_dir)
        assert not is_valid, "Path traversal not detected"

    def test_filename_sanitization(self):
        """Test filename sanitization"""
        sanitizer = InputSanitizer()

        # Valid filename
        is_valid, filename, error = sanitizer.sanitize_filename("image.jpg")
        assert is_valid, f"Valid filename rejected: {error}"

        # Dangerous extension
        is_valid, filename, error = sanitizer.sanitize_filename("malware.exe")
        assert not is_valid, "Dangerous extension not blocked"

        # Path in filename
        is_valid, filename, error = sanitizer.sanitize_filename("/etc/passwd")
        assert is_valid, "Path components should be stripped"
        assert filename == "passwd"


class TestRateLimiter:
    """Test rate limiting"""

    def test_token_bucket(self):
        """Test token bucket algorithm"""
        limiter = RateLimiter()

        # Should allow initial requests
        for i in range(10):
            is_allowed, msg = limiter.token_bucket("test_key", capacity=100, refill_rate=10.0, tokens_required=1)
            assert is_allowed, f"Request {i} blocked unexpectedly"

        # Exhaust tokens
        is_allowed, msg = limiter.token_bucket("test_key2", capacity=5, refill_rate=1.0, tokens_required=10)
        assert not is_allowed, "Rate limit not enforced"

    def test_sliding_window(self):
        """Test sliding window algorithm"""
        limiter = RateLimiter()

        # Should allow up to max_requests
        for i in range(10):
            is_allowed, msg = limiter.sliding_window("test_user", max_requests=10, window_seconds=60)
            assert is_allowed, f"Request {i} blocked unexpectedly"

        # Should block after limit
        is_allowed, msg = limiter.sliding_window("test_user", max_requests=10, window_seconds=60)
        assert not is_allowed, "Rate limit not enforced"

    def test_ip_blocking(self):
        """Test IP blocking"""
        limiter = RateLimiter()

        # Block IP
        limiter.block_ip("192.168.1.100", duration_seconds=5, reason="Testing")

        # Check if blocked
        is_blocked, reason = limiter.is_ip_blocked("192.168.1.100")
        assert is_blocked, "IP not blocked"
        assert reason == "Testing"

        # Wait for expiry
        time.sleep(6)

        # Should be unblocked
        is_blocked, reason = limiter.is_ip_blocked("192.168.1.100")
        assert not is_blocked, "IP still blocked after expiry"


class TestCircuitBreaker:
    """Test circuit breaker"""

    def test_circuit_states(self):
        """Test circuit breaker state transitions"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=2)

        assert cb.state == "closed", "Initial state should be closed"

        # Cause failures
        failing_func = Mock(side_effect=Exception("Test failure"))

        for _ in range(3):
            with pytest.raises(Exception):
                cb.call(failing_func)

        assert cb.state == "open", "Circuit should be open after failures"

        # Wait for recovery timeout
        time.sleep(2.5)

        assert cb.state == "half_open", "Circuit should be half-open after timeout"

        # Successful call should close circuit
        success_func = Mock(return_value="success")
        result = cb.call(success_func)

        assert result == "success"
        assert cb.state == "closed", "Circuit should be closed after success"

    def test_circuit_statistics(self):
        """Test circuit breaker statistics"""
        cb = CircuitBreaker(failure_threshold=5)

        # Make some calls
        success_func = Mock(return_value="ok")
        for _ in range(10):
            cb.call(success_func)

        stats = cb.get_statistics()

        assert stats['stats']['total_calls'] == 10
        assert stats['stats']['successful_calls'] == 10
        assert stats['stats']['failed_calls'] == 0


class TestRetryStrategy:
    """Test retry strategies"""

    def test_exponential_backoff(self):
        """Test exponential backoff retry"""
        attempt_count = 0

        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Network error")
            return "Success"

        success, result, error = RetryStrategy.exponential_backoff(
            flaky_function,
            max_retries=5,
            base_delay=0.1,
            jitter=False
        )

        assert success, "Retry should succeed"
        assert result == "Success"
        assert attempt_count == 3, f"Expected 3 attempts, got {attempt_count}"

    def test_retry_exhaustion(self):
        """Test retry exhaustion"""
        def always_fails():
            raise ValueError("Always fails")

        success, result, error = RetryStrategy.exponential_backoff(
            always_fails,
            max_retries=3,
            base_delay=0.1
        )

        assert not success, "Should fail after retries exhausted"
        assert isinstance(error, ValueError)


class TestErrorRecoveryManager:
    """Test error recovery manager"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.manager = ErrorRecoveryManager(db_path=self.temp_db.name)

    def teardown_method(self):
        """Cleanup test environment"""
        Path(self.temp_db.name).unlink(missing_ok=True)

    def test_error_recording(self):
        """Test error recording"""
        try:
            raise ValueError("Test error")
        except Exception as e:
            error_id = self.manager.record_error(
                e,
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.LOW,
                context={"test": "data"}
            )

        assert error_id is not None, "Error ID not returned"

        # Check statistics
        stats = self.manager.get_error_statistics(time_window_hours=1)
        assert stats['total_errors'] == 1

    def test_error_recovery(self):
        """Test error recovery tracking"""
        try:
            raise ConnectionError("Network issue")
        except Exception as e:
            error_id = self.manager.record_error(e)

        # Mark as recovered
        self.manager.mark_recovered(error_id, recovery_method="retry_with_backoff")

        # Check recovery rate
        stats = self.manager.get_error_statistics(time_window_hours=1)
        assert stats['recovery_rate'] == 100.0


class TestMonitoringSystem:
    """Test monitoring and metrics"""

    def test_metrics_collection(self):
        """Test metrics collection"""
        collector = MetricsCollector()

        # Record metrics
        collector.counter("requests", 1)
        collector.gauge("queue_size", 42)
        collector.timer("operation_duration", 123.45)

        # Get statistics
        stats = collector.get_metric_stats("operation_duration", time_window_seconds=60)

        assert stats['count'] > 0
        assert stats['latest'] == 123.45

    def test_system_monitor(self):
        """Test system resource monitoring"""
        monitor = SystemMonitor()

        cpu = monitor.get_cpu_usage()
        memory = monitor.get_memory_usage()
        disk = monitor.get_disk_usage()

        assert 'system_percent' in cpu
        assert 'process_rss_mb' in memory
        assert 'total_gb' in disk

    def test_health_checker(self):
        """Test health checking"""
        from production_monitoring import HealthCheckResult

        checker = HealthChecker()

        # Register custom check
        def check_database():
            return HealthCheckResult(
                component="database",
                status=HealthStatus.HEALTHY,
                timestamp=time.time(),
                response_time_ms=0,
                message="OK",
                metadata={}
            )

        checker.register_check("database", check_database)

        # Run checks
        results = checker.run_all_checks()

        assert "database" in results
        assert results["database"].status == HealthStatus.HEALTHY

    def test_performance_monitor(self):
        """Test performance monitoring"""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        monitor = PerformanceMonitor(db_path=temp_db.name)

        # Record operations
        monitor.record_operation("test_op", 100.0, success=True)
        monitor.record_operation("test_op", 200.0, success=True)
        monitor.record_operation("test_op", 150.0, success=False, error_message="Error")

        # Get statistics
        stats = monitor.get_operation_stats("test_op", time_window_hours=1)

        assert stats['total_count'] == 3
        assert stats['success_count'] == 2
        assert stats['avg_duration_ms'] > 0

        # Cleanup
        Path(temp_db.name).unlink(missing_ok=True)


class TestGracefulDegradation:
    """Test graceful degradation strategies"""

    def test_fallback(self):
        """Test fallback mechanism"""
        def primary():
            raise ConnectionError("Primary failed")

        def fallback():
            return "Fallback result"

        used_primary, result = GracefulDegradation.fallback(primary, fallback)

        assert not used_primary, "Should use fallback"
        assert result == "Fallback result"

    def test_cached_fallback(self):
        """Test cached fallback"""
        cache = {"key": "cached_value"}

        def failing_function():
            raise ValueError("Function failed")

        is_fresh, result = GracefulDegradation.cached_fallback(
            failing_function,
            cache,
            "key"
        )

        assert not is_fresh, "Should use cache"
        assert result == "cached_value"

    def test_timeout_with_default(self):
        """Test timeout with default value"""
        def slow_function():
            time.sleep(10)
            return "Never returned"

        completed, result = GracefulDegradation.timeout_with_default(
            slow_function,
            timeout_seconds=0.5,
            default_value="default"
        )

        assert not completed, "Should timeout"
        assert result == "default"


class TestIntegration:
    """Integration tests for production systems"""

    def test_end_to_end_workflow(self):
        """Test complete workflow with all production components"""
        # Initialize components
        with tempfile.TemporaryDirectory() as temp_dir:
            url_manager = ProductionURLManager(db_path=f"{temp_dir}/urls.db")
            error_manager = ErrorRecoveryManager(db_path=f"{temp_dir}/errors.db")
            dashboard = MonitoringDashboard()

            # Simulate workflow
            test_url = "https://test-blog.tumblr.com"

            # 1. Validate URL
            is_valid, message, record = url_manager.process_url(test_url)
            dashboard.metrics.counter("url.validation", 1)

            # 2. Handle errors
            if not is_valid:
                try:
                    raise ValueError(f"Invalid URL: {message}")
                except Exception as e:
                    error_id = error_manager.record_error(
                        e,
                        category=ErrorCategory.VALIDATION,
                        severity=ErrorSeverity.LOW
                    )
                    dashboard.metrics.counter("errors.validation", 1)

            # 3. Get summary
            summary = dashboard.get_dashboard_summary()

            assert 'health' in summary
            assert 'system' in summary
            assert 'metrics' in summary

            # Cleanup
            url_manager.close()


def test_security_hardening():
    """Test overall security hardening"""
    sanitizer = InputSanitizer()
    limiter = RateLimiter()

    # Test multiple attack vectors
    attack_vectors = [
        "<script>alert('xss')</script>",
        "'; DROP TABLE users--",
        "../../etc/passwd",
        "javascript:alert('xss')",
        "\x00malicious",
    ]

    for attack in attack_vectors:
        is_valid, _, _ = sanitizer.sanitize_string(attack, "general")
        assert not is_valid, f"Attack vector not blocked: {attack}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
