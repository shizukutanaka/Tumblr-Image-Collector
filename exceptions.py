#!/usr/bin/env python3
"""
Custom Exception Classes for Tumblr Image Collector

Provides a hierarchy of exceptions for better error handling and debugging.
"""

from typing import Optional, Dict, Any


class TumblrCollectorError(Exception):
    """Base exception for all Tumblr Collector errors.

    Attributes:
        message: Error message
        details: Additional error details
        original_exception: Original exception if wrapped
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.original_exception = original_exception

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message='{self.message}', details={self.details})"


# Network-related Exceptions
class NetworkError(TumblrCollectorError):
    """Base class for network-related errors."""
    pass


class ConnectionError(NetworkError):
    """Raised when connection to Tumblr API or image servers fails."""
    pass


class TimeoutError(NetworkError):
    """Raised when network request times out."""

    def __init__(self, message: str, timeout_seconds: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.timeout_seconds = timeout_seconds


class RateLimitError(NetworkError):
    """Raised when API rate limit is exceeded."""

    def __init__(
        self,
        message: str = "API rate limit exceeded",
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class DownloadError(NetworkError):
    """Raised when image download fails."""

    def __init__(self, message: str, url: Optional[str] = None, status_code: Optional[int] = None, **kwargs):
        details = kwargs.get('details', {})
        if url:
            details['url'] = url
        if status_code:
            details['status_code'] = status_code
        kwargs['details'] = details
        super().__init__(message, **kwargs)
        self.url = url
        self.status_code = status_code


# Validation Exceptions
class ValidationError(TumblrCollectorError):
    """Base class for validation errors."""
    pass


class URLValidationError(ValidationError):
    """Raised when URL validation fails."""

    def __init__(self, message: str, url: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if url:
            details['url'] = url
        kwargs['details'] = details
        super().__init__(message, **kwargs)
        self.url = url


class BlogValidationError(ValidationError):
    """Raised when blog name validation fails."""

    def __init__(self, message: str, blog_name: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if blog_name:
            details['blog_name'] = blog_name
        kwargs['details'] = details
        super().__init__(message, **kwargs)
        self.blog_name = blog_name


class CredentialsValidationError(ValidationError):
    """Raised when API credentials validation fails."""
    pass


class ConfigValidationError(ValidationError):
    """Raised when configuration validation fails."""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if config_key:
            details['config_key'] = config_key
        kwargs['details'] = details
        super().__init__(message, **kwargs)
        self.config_key = config_key


# Configuration Exceptions
class ConfigurationError(TumblrCollectorError):
    """Base class for configuration errors."""
    pass


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""

    def __init__(self, message: str, missing_keys: Optional[list] = None, **kwargs):
        details = kwargs.get('details', {})
        if missing_keys:
            details['missing_keys'] = missing_keys
        kwargs['details'] = details
        super().__init__(message, **kwargs)
        self.missing_keys = missing_keys or []


class InvalidConfigError(ConfigurationError):
    """Raised when configuration contains invalid values."""

    def __init__(self, message: str, invalid_key: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if invalid_key:
            details['invalid_key'] = invalid_key
        kwargs['details'] = details
        super().__init__(message, **kwargs)
        self.invalid_key = invalid_key


# Authentication Exceptions
class AuthenticationError(TumblrCollectorError):
    """Base class for authentication errors."""
    pass


class InvalidCredentialsError(AuthenticationError):
    """Raised when Tumblr API credentials are invalid."""
    pass


class TokenExpiredError(AuthenticationError):
    """Raised when authentication token has expired."""
    pass


# Image Processing Exceptions
class ImageProcessingError(TumblrCollectorError):
    """Base class for image processing errors."""
    pass


class ImageValidationError(ImageProcessingError):
    """Raised when image validation fails."""

    def __init__(
        self,
        message: str,
        image_path: Optional[str] = None,
        reason: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get('details', {})
        if image_path:
            details['image_path'] = image_path
        if reason:
            details['reason'] = reason
        kwargs['details'] = details
        super().__init__(message, **kwargs)
        self.image_path = image_path
        self.reason = reason


class ImageOptimizationError(ImageProcessingError):
    """Raised when image optimization fails."""
    pass


class ImageClassificationError(ImageProcessingError):
    """Raised when image classification fails."""
    pass


# File System Exceptions
class FileSystemError(TumblrCollectorError):
    """Base class for file system errors."""
    pass


class FileNotFoundError(FileSystemError):
    """Raised when required file is not found."""

    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if file_path:
            details['file_path'] = file_path
        kwargs['details'] = details
        super().__init__(message, **kwargs)
        self.file_path = file_path


class FileWriteError(FileSystemError):
    """Raised when file write operation fails."""

    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if file_path:
            details['file_path'] = file_path
        kwargs['details'] = details
        super().__init__(message, **kwargs)
        self.file_path = file_path


class DiskSpaceError(FileSystemError):
    """Raised when insufficient disk space is available."""

    def __init__(self, message: str, required_bytes: Optional[int] = None, available_bytes: Optional[int] = None, **kwargs):
        details = kwargs.get('details', {})
        if required_bytes:
            details['required_bytes'] = required_bytes
        if available_bytes:
            details['available_bytes'] = available_bytes
        kwargs['details'] = details
        super().__init__(message, **kwargs)
        self.required_bytes = required_bytes
        self.available_bytes = available_bytes


# Cache Exceptions
class CacheError(TumblrCollectorError):
    """Base class for cache-related errors."""
    pass


class CacheReadError(CacheError):
    """Raised when cache read operation fails."""
    pass


class CacheWriteError(CacheError):
    """Raised when cache write operation fails."""
    pass


class CacheCorruptionError(CacheError):
    """Raised when cache data is corrupted."""
    pass


# Security Exceptions
class SecurityError(TumblrCollectorError):
    """Base class for security-related errors."""
    pass


class SSRFError(SecurityError):
    """Raised when SSRF (Server-Side Request Forgery) attack is detected."""

    def __init__(self, message: str, blocked_url: Optional[str] = None, **kwargs):
        details = kwargs.get('details', {})
        if blocked_url:
            details['blocked_url'] = blocked_url
        kwargs['details'] = details
        super().__init__(message, **kwargs)
        self.blocked_url = blocked_url


class EncryptionError(SecurityError):
    """Raised when encryption/decryption operation fails."""
    pass


class IntegrityError(SecurityError):
    """Raised when data integrity check fails."""
    pass


# Resource Exceptions
class ResourceError(TumblrCollectorError):
    """Base class for resource-related errors."""
    pass


class MemoryError(ResourceError):
    """Raised when insufficient memory is available."""

    def __init__(self, message: str, required_mb: Optional[int] = None, available_mb: Optional[int] = None, **kwargs):
        details = kwargs.get('details', {})
        if required_mb:
            details['required_mb'] = required_mb
        if available_mb:
            details['available_mb'] = available_mb
        kwargs['details'] = details
        super().__init__(message, **kwargs)
        self.required_mb = required_mb
        self.available_mb = available_mb


class WorkerPoolError(ResourceError):
    """Raised when worker pool operations fail."""
    pass


# Circuit Breaker Exceptions
class CircuitBreakerError(TumblrCollectorError):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str = "Circuit breaker is open", retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


# Monitoring Exceptions
class MonitoringError(TumblrCollectorError):
    """Base class for monitoring-related errors."""
    pass


class HealthCheckError(MonitoringError):
    """Raised when health check fails."""
    pass


class MetricsError(MonitoringError):
    """Raised when metrics collection fails."""
    pass


# Convenience functions
def wrap_exception(original: Exception, new_class: type, message: str = None) -> TumblrCollectorError:
    """Wrap an exception in a TumblrCollectorError.

    Args:
        original: Original exception
        new_class: New exception class to wrap with
        message: Optional custom message

    Returns:
        Wrapped exception instance
    """
    msg = message or str(original)
    return new_class(msg, original_exception=original)
