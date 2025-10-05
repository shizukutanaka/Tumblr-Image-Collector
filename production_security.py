#!/usr/bin/env python3
"""
Production Security Module
Comprehensive security hardening for nation-state level deployments
"""

import re
import hashlib
import hmac
import secrets
import logging
import time
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from functools import wraps
import ipaddress

logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    """Security event record"""
    timestamp: float
    event_type: str  # rate_limit, input_validation, auth_failure, etc.
    severity: str  # info, warning, critical
    source_ip: Optional[str]
    user_id: Optional[str]
    description: str
    metadata: Dict[str, Any]


class InputSanitizer:
    """
    Input validation and sanitization for all user inputs
    Prevents: XSS, SQL Injection, Path Traversal, Command Injection
    """

    # Maximum input lengths
    MAX_LENGTHS = {
        'url': 2048,
        'path': 512,
        'filename': 255,
        'tag': 100,
        'blog_name': 63,
        'search_query': 500,
        'config_value': 1024
    }

    # Allowed characters by context
    ALLOWED_PATTERNS = {
        'blog_name': re.compile(r'^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$', re.IGNORECASE),
        'tag': re.compile(r'^[a-zA-Z0-9_\-\s]{1,100}$'),
        'filename': re.compile(r'^[a-zA-Z0-9_\-\.]{1,255}$'),
        'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$'),
        'safe_path': re.compile(r'^[a-zA-Z0-9_\-/\.]+$')
    }

    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        re.compile(r'<script', re.IGNORECASE),
        re.compile(r'javascript:', re.IGNORECASE),
        re.compile(r'data:', re.IGNORECASE),
        re.compile(r'vbscript:', re.IGNORECASE),
        re.compile(r'on\w+\s*=', re.IGNORECASE),  # Event handlers
        re.compile(r'\.\./'),  # Path traversal
        re.compile(r';\s*\w+', re.MULTILINE),  # Command injection
        re.compile(r'[\'";].*(--)'),  # SQL comment injection
        re.compile(r'union\s+select', re.IGNORECASE),
        re.compile(r'exec(\s|\+)+(s|x)p\w+', re.IGNORECASE),
    ]

    @staticmethod
    def sanitize_string(value: str, context: str = 'general', max_length: Optional[int] = None) -> Tuple[bool, str, str]:
        """
        Sanitize string input
        Returns: (is_valid, sanitized_value, error_message)
        """
        if not isinstance(value, str):
            return False, "", "Input must be a string"

        # Strip whitespace
        value = value.strip()

        # Check length
        max_len = max_length or InputSanitizer.MAX_LENGTHS.get(context, 1024)
        if len(value) > max_len:
            return False, "", f"Input exceeds maximum length ({max_len})"

        if len(value) == 0:
            return False, "", "Input cannot be empty"

        # Check for dangerous patterns
        for pattern in InputSanitizer.DANGEROUS_PATTERNS:
            if pattern.search(value):
                logger.warning(f"Dangerous pattern detected in input: {pattern.pattern}")
                return False, "", "Input contains potentially malicious content"

        # Context-specific validation
        if context in InputSanitizer.ALLOWED_PATTERNS:
            if not InputSanitizer.ALLOWED_PATTERNS[context].match(value):
                return False, "", f"Input does not match required format for {context}"

        return True, value, "OK"

    @staticmethod
    def sanitize_path(path: str, base_dir: Optional[Path] = None) -> Tuple[bool, Path, str]:
        """
        Sanitize file path to prevent path traversal
        Returns: (is_valid, sanitized_path, error_message)
        """
        if not isinstance(path, str):
            return False, Path(), "Path must be a string"

        try:
            path_obj = Path(path)

            # Block absolute paths if base_dir is specified
            if base_dir and path_obj.is_absolute():
                return False, Path(), "Absolute paths not allowed"

            # Resolve path and check for traversal
            if base_dir:
                full_path = (base_dir / path_obj).resolve()
                if not str(full_path).startswith(str(base_dir.resolve())):
                    logger.warning(f"Path traversal attempt detected: {path}")
                    return False, Path(), "Path traversal attempt detected"
            else:
                full_path = path_obj.resolve()

            # Check for dangerous characters
            path_str = str(full_path)
            if '..' in path_str or '~' in path_str:
                return False, Path(), "Path contains dangerous characters"

            # Length check
            if len(path_str) > InputSanitizer.MAX_LENGTHS['path']:
                return False, Path(), "Path too long"

            return True, full_path, "OK"

        except Exception as e:
            return False, Path(), f"Path validation error: {e}"

    @staticmethod
    def sanitize_filename(filename: str) -> Tuple[bool, str, str]:
        """
        Sanitize filename
        Returns: (is_valid, sanitized_filename, error_message)
        """
        if not isinstance(filename, str):
            return False, "", "Filename must be a string"

        # Remove path components
        filename = Path(filename).name

        # Length check
        if len(filename) > InputSanitizer.MAX_LENGTHS['filename']:
            return False, "", "Filename too long"

        # Pattern check
        if not InputSanitizer.ALLOWED_PATTERNS['filename'].match(filename):
            return False, "", "Filename contains invalid characters"

        # Block dangerous extensions
        dangerous_extensions = {'.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js'}
        if Path(filename).suffix.lower() in dangerous_extensions:
            return False, "", "Dangerous file extension"

        return True, filename, "OK"


class RateLimiter:
    """
    Advanced rate limiting with multiple strategies:
    - Token bucket algorithm
    - Sliding window
    - Per-IP and per-user limits
    """

    def __init__(self):
        self._locks = defaultdict(threading.Lock)
        self._buckets = {}  # key -> (tokens, last_update)
        self._windows = defaultdict(deque)  # key -> deque of timestamps
        self._blocked_ips = {}  # ip -> (block_until, reason)
        self._stats = {
            'requests_allowed': 0,
            'requests_blocked': 0,
            'ips_blocked': 0
        }

    def token_bucket(
        self,
        key: str,
        capacity: int = 100,
        refill_rate: float = 10.0,  # tokens per second
        tokens_required: int = 1
    ) -> Tuple[bool, str]:
        """
        Token bucket rate limiting
        Returns: (is_allowed, message)
        """
        with self._locks[key]:
            current_time = time.time()

            # Initialize bucket if needed
            if key not in self._buckets:
                self._buckets[key] = (capacity, current_time)

            tokens, last_update = self._buckets[key]

            # Refill tokens
            time_elapsed = current_time - last_update
            tokens = min(capacity, tokens + (time_elapsed * refill_rate))

            # Check if enough tokens
            if tokens >= tokens_required:
                tokens -= tokens_required
                self._buckets[key] = (tokens, current_time)
                self._stats['requests_allowed'] += 1
                return True, "OK"
            else:
                self._buckets[key] = (tokens, current_time)
                self._stats['requests_blocked'] += 1
                wait_time = (tokens_required - tokens) / refill_rate
                return False, f"Rate limit exceeded. Retry after {wait_time:.1f}s"

    def sliding_window(
        self,
        key: str,
        max_requests: int = 60,
        window_seconds: int = 60
    ) -> Tuple[bool, str]:
        """
        Sliding window rate limiting
        Returns: (is_allowed, message)
        """
        with self._locks[key]:
            current_time = time.time()
            window = self._windows[key]

            # Remove old timestamps
            cutoff_time = current_time - window_seconds
            while window and window[0] < cutoff_time:
                window.popleft()

            # Check limit
            if len(window) < max_requests:
                window.append(current_time)
                self._stats['requests_allowed'] += 1
                return True, "OK"
            else:
                self._stats['requests_blocked'] += 1
                oldest_request = window[0]
                retry_after = oldest_request + window_seconds - current_time
                return False, f"Rate limit exceeded. Retry after {retry_after:.1f}s"

    def block_ip(self, ip: str, duration_seconds: int = 3600, reason: str = "Rate limit exceeded"):
        """Block an IP address temporarily"""
        with self._locks[ip]:
            block_until = time.time() + duration_seconds
            self._blocked_ips[ip] = (block_until, reason)
            self._stats['ips_blocked'] += 1
            logger.warning(f"Blocked IP {ip} for {duration_seconds}s: {reason}")

    def is_ip_blocked(self, ip: str) -> Tuple[bool, str]:
        """
        Check if IP is blocked
        Returns: (is_blocked, reason)
        """
        if ip not in self._blocked_ips:
            return False, ""

        block_until, reason = self._blocked_ips[ip]

        if time.time() < block_until:
            return True, reason
        else:
            # Unblock expired IP
            del self._blocked_ips[ip]
            return False, ""

    def cleanup_old_entries(self, max_age_seconds: int = 3600):
        """Clean up old entries to prevent memory leak"""
        current_time = time.time()

        # Clean up buckets
        keys_to_remove = [
            key for key, (_, last_update) in self._buckets.items()
            if current_time - last_update > max_age_seconds
        ]
        for key in keys_to_remove:
            del self._buckets[key]

        # Clean up windows
        cutoff_time = current_time - max_age_seconds
        for key in list(self._windows.keys()):
            window = self._windows[key]
            while window and window[0] < cutoff_time:
                window.popleft()
            if not window:
                del self._windows[key]

        # Clean up blocked IPs
        self._blocked_ips = {
            ip: (block_until, reason)
            for ip, (block_until, reason) in self._blocked_ips.items()
            if current_time < block_until
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        return {
            'active_buckets': len(self._buckets),
            'active_windows': len(self._windows),
            'blocked_ips': len(self._blocked_ips),
            'stats': dict(self._stats)
        }


class DDoSProtection:
    """
    DDoS protection mechanisms
    - Connection limiting
    - Request pattern analysis
    - Automatic IP blocking
    """

    def __init__(self, max_connections_per_ip: int = 10):
        self.max_connections_per_ip = max_connections_per_ip
        self._connections = defaultdict(int)
        self._request_patterns = defaultdict(list)  # ip -> [(timestamp, endpoint)]
        self._lock = threading.Lock()
        self._rate_limiter = RateLimiter()

    def check_connection_limit(self, ip: str) -> Tuple[bool, str]:
        """
        Check if IP has exceeded connection limit
        Returns: (is_allowed, message)
        """
        # Check if IP is blocked
        is_blocked, reason = self._rate_limiter.is_ip_blocked(ip)
        if is_blocked:
            return False, f"IP blocked: {reason}"

        with self._lock:
            current_connections = self._connections[ip]

            if current_connections >= self.max_connections_per_ip:
                logger.warning(f"Connection limit exceeded for IP {ip}: {current_connections}")
                self._rate_limiter.block_ip(ip, duration_seconds=300, reason="Too many connections")
                return False, "Connection limit exceeded"

            self._connections[ip] += 1
            return True, "OK"

    def release_connection(self, ip: str):
        """Release a connection slot"""
        with self._lock:
            if self._connections[ip] > 0:
                self._connections[ip] -= 1
                if self._connections[ip] == 0:
                    del self._connections[ip]

    def analyze_request_pattern(self, ip: str, endpoint: str) -> Tuple[bool, str]:
        """
        Analyze request patterns to detect suspicious behavior
        Returns: (is_allowed, message)
        """
        current_time = time.time()

        with self._lock:
            patterns = self._request_patterns[ip]

            # Remove old entries (older than 60 seconds)
            patterns[:] = [(ts, ep) for ts, ep in patterns if current_time - ts < 60]

            # Add current request
            patterns.append((current_time, endpoint))

            # Check for suspicious patterns
            if len(patterns) > 100:
                logger.warning(f"Suspicious request rate from IP {ip}: {len(patterns)} requests/min")
                self._rate_limiter.block_ip(ip, duration_seconds=600, reason="Suspicious request pattern")
                return False, "Suspicious request pattern detected"

            # Check for rapid-fire same endpoint requests
            same_endpoint_count = sum(1 for _, ep in patterns[-20:] if ep == endpoint)
            if same_endpoint_count > 15:
                logger.warning(f"Rapid-fire requests to {endpoint} from IP {ip}")
                self._rate_limiter.block_ip(ip, duration_seconds=300, reason="Rapid-fire requests")
                return False, "Too many requests to same endpoint"

        return True, "OK"

    def get_statistics(self) -> Dict[str, Any]:
        """Get DDoS protection statistics"""
        with self._lock:
            return {
                'active_connections': dict(self._connections),
                'total_ips_tracked': len(self._request_patterns),
                'rate_limiter': self._rate_limiter.get_statistics()
            }


class SecurityAuditor:
    """
    Security event logging and auditing
    """

    def __init__(self, log_file: str = "security_audit.log"):
        self.log_file = Path(log_file)
        self._events = deque(maxlen=10000)  # Keep last 10k events in memory
        self._lock = threading.Lock()
        self._setup_logging()

    def _setup_logging(self):
        """Setup security audit logging"""
        self.logger = logging.getLogger('security_audit')
        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log security event"""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            source_ip=source_ip,
            user_id=user_id,
            description=description,
            metadata=metadata or {}
        )

        with self._lock:
            self._events.append(event)

        # Log to file
        log_message = f"[{event_type}] {description}"
        if source_ip:
            log_message += f" | IP: {source_ip}"
        if user_id:
            log_message += f" | User: {user_id}"
        if metadata:
            log_message += f" | Metadata: {json.dumps(metadata)}"

        if severity == 'critical':
            self.logger.critical(log_message)
        elif severity == 'warning':
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

    def get_recent_events(self, count: int = 100, event_type: Optional[str] = None) -> List[SecurityEvent]:
        """Get recent security events"""
        with self._lock:
            events = list(self._events)

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[-count:]

    def get_event_summary(self, time_window_seconds: int = 3600) -> Dict[str, Any]:
        """Get summary of security events within time window"""
        cutoff_time = time.time() - time_window_seconds

        with self._lock:
            recent_events = [e for e in self._events if e.timestamp > cutoff_time]

        summary = {
            'total_events': len(recent_events),
            'by_type': defaultdict(int),
            'by_severity': defaultdict(int),
            'unique_ips': set()
        }

        for event in recent_events:
            summary['by_type'][event.event_type] += 1
            summary['by_severity'][event.severity] += 1
            if event.source_ip:
                summary['unique_ips'].add(event.source_ip)

        summary['unique_ips'] = len(summary['unique_ips'])
        summary['by_type'] = dict(summary['by_type'])
        summary['by_severity'] = dict(summary['by_severity'])

        return summary


# Decorators for security enforcement

def require_sanitized_input(param_name: str, context: str = 'general'):
    """Decorator to enforce input sanitization"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            value = kwargs.get(param_name)
            if value is None and len(args) > 0:
                # Try to get from positional args
                import inspect
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                if param_name in params:
                    idx = params.index(param_name)
                    if idx < len(args):
                        value = args[idx]

            if value is not None:
                is_valid, sanitized, error = InputSanitizer.sanitize_string(value, context)
                if not is_valid:
                    raise ValueError(f"Input validation failed for {param_name}: {error}")

                # Replace value
                if param_name in kwargs:
                    kwargs[param_name] = sanitized

            return func(*args, **kwargs)
        return wrapper
    return decorator


def rate_limit(key_func: Callable, max_requests: int = 60, window_seconds: int = 60):
    """Decorator for rate limiting"""
    limiter = RateLimiter()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = key_func(*args, **kwargs)
            is_allowed, message = limiter.sliding_window(key, max_requests, window_seconds)

            if not is_allowed:
                raise PermissionError(f"Rate limit exceeded: {message}")

            return func(*args, **kwargs)
        return wrapper
    return decorator


# Global instances
_input_sanitizer = InputSanitizer()
_rate_limiter = RateLimiter()
_ddos_protection = DDoSProtection()
_security_auditor = SecurityAuditor()


def get_input_sanitizer() -> InputSanitizer:
    """Get global input sanitizer"""
    return _input_sanitizer


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter"""
    return _rate_limiter


def get_ddos_protection() -> DDoSProtection:
    """Get global DDoS protection"""
    return _ddos_protection


def get_security_auditor() -> SecurityAuditor:
    """Get global security auditor"""
    return _security_auditor


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test input sanitization
    sanitizer = InputSanitizer()

    test_inputs = [
        ("normal-blog", "blog_name"),
        ("../etc/passwd", "blog_name"),
        ("<script>alert('xss')</script>", "tag"),
        ("safe_filename.jpg", "filename"),
        ("../../etc/passwd", "safe_path")
    ]

    for value, context in test_inputs:
        is_valid, sanitized, error = sanitizer.sanitize_string(value, context)
        print(f"{value} ({context}): Valid={is_valid}, Error={error}")

    # Test rate limiting
    rate_limiter = RateLimiter()

    for i in range(65):
        is_allowed, msg = rate_limiter.sliding_window("test_user", max_requests=60, window_seconds=60)
        if not is_allowed:
            print(f"Request {i+1}: {msg}")

    # Test DDoS protection
    ddos = DDoSProtection(max_connections_per_ip=5)

    for i in range(7):
        is_allowed, msg = ddos.check_connection_limit("192.168.1.100")
        print(f"Connection {i+1}: Allowed={is_allowed}, Message={msg}")

    print("\nStatistics:", json.dumps(ddos.get_statistics(), indent=2))
