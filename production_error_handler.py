#!/usr/bin/env python3
"""
Production Error Handling and Recovery System
Comprehensive error handling, retry logic, graceful degradation
"""

import logging
import time
import traceback
import sys
import json
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable, Type
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from functools import wraps
from collections import defaultdict, deque
import sqlite3

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"  # Recoverable, non-critical
    MEDIUM = "medium"  # Degraded functionality
    HIGH = "high"  # Service disruption
    CRITICAL = "critical"  # System failure


class ErrorCategory(Enum):
    """Error categories"""
    NETWORK = "network"
    VALIDATION = "validation"
    PERMISSION = "permission"
    RESOURCE = "resource"
    DATA = "data"
    CONFIGURATION = "configuration"
    EXTERNAL_API = "external_api"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


@dataclass
class ErrorRecord:
    """Comprehensive error record"""
    timestamp: float
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    exception_type: str
    error_message: str
    stack_trace: str
    context: Dict[str, Any]
    retry_count: int
    recovered: bool
    recovery_method: Optional[str]


class CircuitBreaker:
    """
    Circuit breaker pattern implementation
    Prevents cascading failures
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self._failure_count = 0
        self._last_failure_time = 0
        self._state = "closed"  # closed, open, half_open
        self._lock = threading.Lock()

        self._stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'circuit_opens': 0
        }

    @property
    def state(self) -> str:
        """Get current circuit state"""
        with self._lock:
            # Transition from open to half-open if timeout expired
            if self._state == "open":
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = "half_open"
                    logger.info("Circuit breaker: transitioning to half-open state")

            return self._state

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._lock:
            self._stats['total_calls'] += 1

        # Check circuit state
        if self.state == "open":
            raise Exception("Circuit breaker is OPEN - service unavailable")

        try:
            result = func(*args, **kwargs)

            # Success - close circuit if half-open
            with self._lock:
                self._stats['successful_calls'] += 1
                if self._state == "half_open":
                    self._state = "closed"
                    self._failure_count = 0
                    logger.info("Circuit breaker: closed after successful call")

            return result

        except self.expected_exception as e:
            with self._lock:
                self._stats['failed_calls'] += 1
                self._failure_count += 1
                self._last_failure_time = time.time()

                # Open circuit if threshold exceeded
                if self._failure_count >= self.failure_threshold:
                    self._state = "open"
                    self._stats['circuit_opens'] += 1
                    logger.error(f"Circuit breaker: OPEN after {self._failure_count} failures")

            raise

    def reset(self):
        """Manually reset circuit breaker"""
        with self._lock:
            self._state = "closed"
            self._failure_count = 0
            logger.info("Circuit breaker: manually reset")

    def get_statistics(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            'state': self.state,
            'failure_count': self._failure_count,
            'stats': dict(self._stats)
        }


class RetryStrategy:
    """
    Advanced retry strategies with exponential backoff and jitter
    """

    @staticmethod
    def exponential_backoff(
        func: Callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ) -> Tuple[bool, Any, Optional[Exception]]:
        """
        Execute function with exponential backoff retry
        Returns: (success, result, last_exception)
        """
        import random

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                result = func()
                return True, result, None

            except retryable_exceptions as e:
                last_exception = e

                if attempt == max_retries:
                    logger.error(f"All retry attempts exhausted for {func.__name__}: {e}")
                    return False, None, e

                # Calculate delay
                delay = min(base_delay * (backoff_factor ** attempt), max_delay)

                # Add jitter to prevent thundering herd
                if jitter:
                    delay = delay * (0.5 + random.random())

                logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__} after {delay:.2f}s: {e}")
                time.sleep(delay)

            except Exception as e:
                # Non-retryable exception
                logger.error(f"Non-retryable exception in {func.__name__}: {e}")
                return False, None, e

        return False, None, last_exception

    @staticmethod
    def retry_decorator(
        max_retries: int = 3,
        base_delay: float = 1.0,
        backoff_factor: float = 2.0,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        """Decorator for retry logic"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                success, result, exception = RetryStrategy.exponential_backoff(
                    lambda: func(*args, **kwargs),
                    max_retries=max_retries,
                    base_delay=base_delay,
                    backoff_factor=backoff_factor,
                    retryable_exceptions=retryable_exceptions
                )

                if not success:
                    raise exception

                return result

            return wrapper
        return decorator


class ErrorRecoveryManager:
    """
    Centralized error recovery management
    """

    def __init__(self, db_path: str = "error_records.db"):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._initialize_database()

        # Error statistics
        self._stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'critical_errors': 0,
            'by_category': defaultdict(int),
            'by_severity': defaultdict(int)
        }

        # Circuit breakers for different services
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

    def _initialize_database(self):
        """Initialize error tracking database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS errors (
                    error_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    exception_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    stack_trace TEXT NOT NULL,
                    context TEXT,
                    retry_count INTEGER DEFAULT 0,
                    recovered INTEGER DEFAULT 0,
                    recovery_method TEXT
                )
            """)

            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON errors(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_severity ON errors(severity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON errors(category)")

            conn.commit()

    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service"""
        if service_name not in self._circuit_breakers:
            self._circuit_breakers[service_name] = CircuitBreaker()

        return self._circuit_breakers[service_name]

    def record_error(
        self,
        exception: Exception,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        retry_count: int = 0
    ) -> str:
        """Record error occurrence"""
        import uuid

        error_id = str(uuid.uuid4())
        timestamp = time.time()

        # Extract error details
        exception_type = type(exception).__name__
        error_message = str(exception)
        stack_trace = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))

        # Create error record
        record = ErrorRecord(
            timestamp=timestamp,
            error_id=error_id,
            category=category,
            severity=severity,
            exception_type=exception_type,
            error_message=error_message,
            stack_trace=stack_trace,
            context=context or {},
            retry_count=retry_count,
            recovered=False,
            recovery_method=None
        )

        # Store in database
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO errors (
                        error_id, timestamp, category, severity, exception_type,
                        error_message, stack_trace, context, retry_count, recovered
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    error_id, timestamp, category.value, severity.value,
                    exception_type, error_message, stack_trace,
                    json.dumps(context or {}), retry_count, 0
                ))
                conn.commit()

            # Update statistics
            self._stats['total_errors'] += 1
            self._stats['by_category'][category.value] += 1
            self._stats['by_severity'][severity.value] += 1

            if severity == ErrorSeverity.CRITICAL:
                self._stats['critical_errors'] += 1

        # Log error
        log_message = f"Error recorded [{error_id}]: {exception_type} - {error_message}"
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        else:
            logger.warning(log_message)

        return error_id

    def mark_recovered(self, error_id: str, recovery_method: str):
        """Mark error as recovered"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE errors
                    SET recovered = 1, recovery_method = ?
                    WHERE error_id = ?
                """, (recovery_method, error_id))
                conn.commit()

            self._stats['recovered_errors'] += 1

        logger.info(f"Error {error_id} recovered using: {recovery_method}")

    def get_error_statistics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for time window"""
        cutoff_time = time.time() - (time_window_hours * 3600)

        with sqlite3.connect(self.db_path) as conn:
            # Total errors
            total = conn.execute(
                "SELECT COUNT(*) FROM errors WHERE timestamp > ?",
                (cutoff_time,)
            ).fetchone()[0]

            # By category
            by_category = dict(conn.execute("""
                SELECT category, COUNT(*)
                FROM errors
                WHERE timestamp > ?
                GROUP BY category
            """, (cutoff_time,)).fetchall())

            # By severity
            by_severity = dict(conn.execute("""
                SELECT severity, COUNT(*)
                FROM errors
                WHERE timestamp > ?
                GROUP BY severity
            """, (cutoff_time,)).fetchall())

            # Recovery rate
            recovered = conn.execute(
                "SELECT COUNT(*) FROM errors WHERE timestamp > ? AND recovered = 1",
                (cutoff_time,)
            ).fetchone()[0]

            # Most common errors
            common_errors = conn.execute("""
                SELECT exception_type, COUNT(*) as count
                FROM errors
                WHERE timestamp > ?
                GROUP BY exception_type
                ORDER BY count DESC
                LIMIT 10
            """, (cutoff_time,)).fetchall()

        return {
            'time_window_hours': time_window_hours,
            'total_errors': total,
            'recovered_errors': recovered,
            'recovery_rate': (recovered / total * 100) if total > 0 else 0,
            'by_category': by_category,
            'by_severity': by_severity,
            'common_errors': dict(common_errors),
            'circuit_breakers': {
                name: cb.get_statistics()
                for name, cb in self._circuit_breakers.items()
            }
        }

    def get_recent_errors(self, count: int = 50, severity: Optional[ErrorSeverity] = None) -> List[Dict[str, Any]]:
        """Get recent error records"""
        with sqlite3.connect(self.db_path) as conn:
            if severity:
                query = """
                    SELECT * FROM errors
                    WHERE severity = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                rows = conn.execute(query, (severity.value, count)).fetchall()
            else:
                query = "SELECT * FROM errors ORDER BY timestamp DESC LIMIT ?"
                rows = conn.execute(query, (count,)).fetchall()

        errors = []
        for row in rows:
            errors.append({
                'error_id': row[0],
                'timestamp': datetime.fromtimestamp(row[1]).isoformat(),
                'category': row[2],
                'severity': row[3],
                'exception_type': row[4],
                'error_message': row[5],
                'retry_count': row[8],
                'recovered': bool(row[9]),
                'recovery_method': row[10]
            })

        return errors

    def cleanup_old_errors(self, max_age_days: int = 30):
        """Clean up old error records"""
        cutoff_time = time.time() - (max_age_days * 86400)

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute(
                    "DELETE FROM errors WHERE timestamp < ?",
                    (cutoff_time,)
                )
                removed_count = result.rowcount
                conn.commit()

        logger.info(f"Cleaned up {removed_count} old error records")
        return removed_count


class GracefulDegradation:
    """
    Graceful degradation strategies for service failures
    """

    @staticmethod
    def fallback(primary_func: Callable, fallback_func: Callable, *args, **kwargs) -> Tuple[bool, Any]:
        """
        Try primary function, fallback to secondary on failure
        Returns: (used_primary, result)
        """
        try:
            result = primary_func(*args, **kwargs)
            return True, result
        except Exception as e:
            logger.warning(f"Primary function failed, using fallback: {e}")
            try:
                result = fallback_func(*args, **kwargs)
                return False, result
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise

    @staticmethod
    def cached_fallback(
        func: Callable,
        cache: Dict[str, Any],
        cache_key: str,
        *args,
        **kwargs
    ) -> Tuple[bool, Any]:
        """
        Try function, use cached result on failure
        Returns: (is_fresh, result)
        """
        try:
            result = func(*args, **kwargs)
            cache[cache_key] = result
            return True, result
        except Exception as e:
            logger.warning(f"Function failed, using cached result: {e}")
            if cache_key in cache:
                return False, cache[cache_key]
            else:
                raise Exception("No cached fallback available")

    @staticmethod
    def timeout_with_default(
        func: Callable,
        timeout_seconds: float,
        default_value: Any,
        *args,
        **kwargs
    ) -> Tuple[bool, Any]:
        """
        Execute function with timeout, return default on timeout
        Returns: (completed, result)
        """
        from concurrent.futures import ThreadPoolExecutor, TimeoutError

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                result = future.result(timeout=timeout_seconds)
                return True, result
            except TimeoutError:
                logger.warning(f"Function timed out after {timeout_seconds}s, using default value")
                return False, default_value


# Global error recovery manager
_error_manager = None


def get_error_manager() -> ErrorRecoveryManager:
    """Get global error recovery manager"""
    global _error_manager
    if _error_manager is None:
        _error_manager = ErrorRecoveryManager()
    return _error_manager


# Decorators for error handling

def handle_errors(
    category: ErrorCategory = ErrorCategory.INTERNAL,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    retry: bool = False,
    max_retries: int = 3
):
    """Decorator for comprehensive error handling"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_error_manager()

            if retry:
                # With retry
                success, result, exception = RetryStrategy.exponential_backoff(
                    lambda: func(*args, **kwargs),
                    max_retries=max_retries
                )

                if not success:
                    error_id = manager.record_error(exception, category, severity)
                    raise exception

                return result
            else:
                # Without retry
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_id = manager.record_error(e, category, severity)
                    raise

        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test circuit breaker
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5)

    def failing_function():
        raise ValueError("Simulated failure")

    for i in range(10):
        try:
            cb.call(failing_function)
        except Exception as e:
            print(f"Call {i+1}: {e}")

        time.sleep(1)

    print("\nCircuit Breaker Stats:", json.dumps(cb.get_statistics(), indent=2))

    # Test retry strategy
    attempt_count = 0

    def flaky_function():
        global attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError("Network error")
        return "Success!"

    success, result, error = RetryStrategy.exponential_backoff(
        flaky_function,
        max_retries=5,
        base_delay=0.5
    )

    print(f"\nRetry result: Success={success}, Result={result}, Attempts={attempt_count}")

    # Test error manager
    manager = ErrorRecoveryManager()

    try:
        raise ValueError("Test error")
    except Exception as e:
        error_id = manager.record_error(e, ErrorCategory.VALIDATION, ErrorSeverity.LOW)
        print(f"\nRecorded error: {error_id}")

    print("\nError Statistics:", json.dumps(manager.get_error_statistics(), indent=2))
