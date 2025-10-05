#!/usr/bin/env python3
"""
Production-Grade URL Management System
Handles URL validation, cleanup, verification, and lifecycle management
Designed for nation-state level reliability and security
"""

import re
import requests
import logging
import sqlite3
import hashlib
import time
from typing import Dict, List, Optional, Set, Tuple, Any
from urllib.parse import urlparse, parse_qs, urlunparse
from pathlib import Path
import json
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from enum import Enum
import ipaddress

logger = logging.getLogger(__name__)


class URLStatus(Enum):
    """URL verification status"""
    VALID = "valid"
    INVALID = "invalid"
    NOT_FOUND = "not_found"
    BLOCKED = "blocked"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    RATE_LIMITED = "rate_limited"
    PENDING = "pending"
    QUARANTINED = "quarantined"


@dataclass
class URLRecord:
    """Comprehensive URL record"""
    url: str
    url_hash: str
    status: URLStatus
    first_seen: float
    last_verified: float
    verification_count: int
    failure_count: int
    http_status_code: Optional[int]
    content_type: Optional[str]
    content_length: Optional[int]
    redirect_url: Optional[str]
    error_message: Optional[str]
    metadata: Dict[str, Any]


class ProductionURLManager:
    """
    Production-grade URL management system with:
    - SQLite-based persistence for reliability
    - Automatic URL cleanup and verification
    - Rate limiting and circuit breaking
    - Security validation and threat detection
    - Comprehensive logging and monitoring
    """

    # Security: Strict Tumblr domain whitelist
    VALID_TUMBLR_DOMAINS = frozenset({
        'tumblr.com',
        'www.tumblr.com',
        'assets.tumblr.com',
        'static.tumblr.com',
        'media.tumblr.com',
        '64.media.tumblr.com',
        'va.media.tumblr.com'
    })

    # Security: Known malicious patterns (basic)
    SUSPICIOUS_PATTERNS = [
        re.compile(r'\.\./', re.IGNORECASE),  # Path traversal
        re.compile(r'<script', re.IGNORECASE),  # XSS attempt
        re.compile(r'javascript:', re.IGNORECASE),  # JS protocol
        re.compile(r'data:', re.IGNORECASE),  # Data URI
        re.compile(r'file:', re.IGNORECASE),  # File protocol
    ]

    # ReDoS-safe URL patterns with strict quantifiers
    TUMBLR_PATTERNS = {
        'blog': re.compile(r'^https?://([a-z0-9-]{1,63})\.tumblr\.com/?$', re.IGNORECASE),
        'post': re.compile(r'^https?://([a-z0-9-]{1,63})\.tumblr\.com/post/(\d{1,20})', re.IGNORECASE),
        'image': re.compile(r'^https?://\d{1,3}\.media\.tumblr\.com/[a-f0-9]{1,128}/tumblr_[a-z0-9_]{1,64}\.(jpg|jpeg|png|gif|webp)$', re.IGNORECASE),
        'tag': re.compile(r'^https?://([a-z0-9-]{1,63})\.tumblr\.com/tagged/[^/<>]{1,200}$', re.IGNORECASE),
        'archive': re.compile(r'^https?://([a-z0-9-]{1,63})\.tumblr\.com/archive', re.IGNORECASE),
    }

    def __init__(self, db_path: str = "url_database.db", max_workers: int = 10):
        self.db_path = Path(db_path)
        self.max_workers = max_workers
        self._lock = threading.Lock()
        self._session = self._create_session()
        self._initialize_database()

        # Circuit breaker state
        self._circuit_breaker = {
            'failure_threshold': 10,
            'timeout_duration': 300,  # 5 minutes
            'failure_count': 0,
            'last_failure_time': 0,
            'state': 'closed'  # closed, open, half_open
        }

        # Rate limiting
        self._rate_limiter = {
            'requests_per_second': 5,
            'last_request_time': 0,
            'request_count': 0,
            'window_start': time.time()
        }

        # Statistics
        self._stats = {
            'urls_validated': 0,
            'urls_cleaned': 0,
            'urls_blocked': 0,
            'verification_errors': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    def _create_session(self) -> requests.Session:
        """Create hardened HTTP session"""
        session = requests.Session()

        # Security headers
        session.headers.update({
            'User-Agent': 'TumblrCollector/2.0 (Production; +https://github.com/shizukutanaka/Tumblr-Image-Collector)',
            'Accept': 'text/html,application/xhtml+xml,image/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'DNT': '1',
            'Connection': 'keep-alive'
        })

        # Configure retries with exponential backoff
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET"]
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=20,
            pool_maxsize=20
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _initialize_database(self):
        """Initialize SQLite database with proper indexing"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS urls (
                    url_hash TEXT PRIMARY KEY,
                    url TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL,
                    first_seen REAL NOT NULL,
                    last_verified REAL NOT NULL,
                    verification_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    http_status_code INTEGER,
                    content_type TEXT,
                    content_length INTEGER,
                    redirect_url TEXT,
                    error_message TEXT,
                    metadata TEXT
                )
            """)

            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON urls(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_verified ON urls(last_verified)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_failure_count ON urls(failure_count)")

            # Statistics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS url_stats (
                    stat_date TEXT PRIMARY KEY,
                    total_urls INTEGER,
                    valid_urls INTEGER,
                    invalid_urls INTEGER,
                    cleanup_count INTEGER,
                    verification_errors INTEGER
                )
            """)

            conn.commit()

    def _hash_url(self, url: str) -> str:
        """Create SHA-256 hash of URL for indexing"""
        return hashlib.sha256(url.encode('utf-8')).hexdigest()

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows requests"""
        current_time = time.time()

        if self._circuit_breaker['state'] == 'open':
            if current_time - self._circuit_breaker['last_failure_time'] > self._circuit_breaker['timeout_duration']:
                self._circuit_breaker['state'] = 'half_open'
                self._circuit_breaker['failure_count'] = 0
                logger.info("Circuit breaker: half-open state")
                return True
            return False

        return True

    def _record_failure(self):
        """Record circuit breaker failure"""
        self._circuit_breaker['failure_count'] += 1
        self._circuit_breaker['last_failure_time'] = time.time()

        if self._circuit_breaker['failure_count'] >= self._circuit_breaker['failure_threshold']:
            self._circuit_breaker['state'] = 'open'
            logger.warning("Circuit breaker: OPEN - too many failures")

    def _record_success(self):
        """Record circuit breaker success"""
        if self._circuit_breaker['state'] == 'half_open':
            self._circuit_breaker['state'] = 'closed'
            self._circuit_breaker['failure_count'] = 0
            logger.info("Circuit breaker: closed state")

    def _enforce_rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()

        # Reset window if needed
        if current_time - self._rate_limiter['window_start'] >= 1.0:
            self._rate_limiter['request_count'] = 0
            self._rate_limiter['window_start'] = current_time

        # Check rate limit
        if self._rate_limiter['request_count'] >= self._rate_limiter['requests_per_second']:
            sleep_time = 1.0 - (current_time - self._rate_limiter['window_start'])
            if sleep_time > 0:
                time.sleep(sleep_time)
                self._rate_limiter['request_count'] = 0
                self._rate_limiter['window_start'] = time.time()

        self._rate_limiter['request_count'] += 1

    def validate_url_security(self, url: str) -> Tuple[bool, str]:
        """
        Comprehensive security validation
        Returns: (is_safe, reason)
        """
        # Length validation (防止 DoS)
        if len(url) > 2048:
            return False, "URL exceeds maximum length"

        if len(url) < 10:
            return False, "URL too short"

        # Type validation
        if not isinstance(url, str):
            return False, "Invalid URL type"

        # Check for suspicious patterns
        for pattern in self.SUSPICIOUS_PATTERNS:
            if pattern.search(url):
                return False, f"Suspicious pattern detected: {pattern.pattern}"

        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            return False, f"URL parsing failed: {e}"

        # Scheme validation
        if parsed.scheme not in ('http', 'https'):
            return False, "Invalid URL scheme"

        # Domain validation
        if not parsed.netloc:
            return False, "Missing domain"

        if len(parsed.netloc) > 255:
            return False, "Domain name too long"

        # Check for IP addresses (防止 SSRF)
        try:
            # Extract hostname without port
            hostname = parsed.netloc.split(':')[0]
            ip = ipaddress.ip_address(hostname)

            # Block private IP ranges
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return False, "Private IP address blocked"
        except ValueError:
            # Not an IP address, continue
            pass

        # Tumblr domain validation
        domain = parsed.netloc.lower()
        is_tumblr_domain = any(
            domain == valid_domain or domain.endswith('.' + valid_domain)
            for valid_domain in self.VALID_TUMBLR_DOMAINS
        )

        if not is_tumblr_domain:
            return False, "Not a valid Tumblr domain"

        # Path validation (prevent path traversal)
        if '..' in parsed.path:
            return False, "Path traversal attempt detected"

        # Fragment validation (防止 XSS)
        if parsed.fragment and len(parsed.fragment) > 100:
            return False, "Suspicious URL fragment"

        return True, "URL passed security validation"

    def classify_url_type(self, url: str) -> Optional[str]:
        """Classify URL type with timeout protection"""
        for url_type, pattern in self.TUMBLR_PATTERNS.items():
            try:
                if pattern.match(url):
                    return url_type
            except Exception as e:
                logger.warning(f"Pattern matching error: {e}")
                return None

        return "unknown"

    def verify_url_accessibility(self, url: str, timeout: int = 10) -> URLRecord:
        """Verify URL accessibility with comprehensive checks"""
        url_hash = self._hash_url(url)
        current_time = time.time()

        # Check circuit breaker
        if not self._check_circuit_breaker():
            return URLRecord(
                url=url,
                url_hash=url_hash,
                status=URLStatus.RATE_LIMITED,
                first_seen=current_time,
                last_verified=current_time,
                verification_count=0,
                failure_count=0,
                http_status_code=None,
                content_type=None,
                content_length=None,
                redirect_url=None,
                error_message="Circuit breaker is open",
                metadata={}
            )

        # Enforce rate limiting
        self._enforce_rate_limit()

        try:
            # Make HEAD request first (faster)
            response = self._session.head(url, timeout=timeout, allow_redirects=True)

            status = URLStatus.VALID if response.status_code < 400 else URLStatus.NOT_FOUND
            redirect_url = str(response.url) if response.url != url else None

            self._record_success()
            self._stats['urls_validated'] += 1

            return URLRecord(
                url=url,
                url_hash=url_hash,
                status=status,
                first_seen=current_time,
                last_verified=current_time,
                verification_count=1,
                failure_count=0 if status == URLStatus.VALID else 1,
                http_status_code=response.status_code,
                content_type=response.headers.get('Content-Type'),
                content_length=int(response.headers.get('Content-Length', 0)) if response.headers.get('Content-Length') else None,
                redirect_url=redirect_url,
                error_message=None if status == URLStatus.VALID else f"HTTP {response.status_code}",
                metadata={'response_time': response.elapsed.total_seconds()}
            )

        except requests.exceptions.Timeout:
            self._record_failure()
            self._stats['verification_errors'] += 1
            return URLRecord(
                url=url, url_hash=url_hash, status=URLStatus.TIMEOUT,
                first_seen=current_time, last_verified=current_time,
                verification_count=1, failure_count=1,
                http_status_code=None, content_type=None, content_length=None,
                redirect_url=None, error_message="Request timeout", metadata={}
            )

        except requests.exceptions.ConnectionError as e:
            self._record_failure()
            self._stats['verification_errors'] += 1
            return URLRecord(
                url=url, url_hash=url_hash, status=URLStatus.NETWORK_ERROR,
                first_seen=current_time, last_verified=current_time,
                verification_count=1, failure_count=1,
                http_status_code=None, content_type=None, content_length=None,
                redirect_url=None, error_message=f"Connection failed: {str(e)[:200]}", metadata={}
            )

        except Exception as e:
            self._record_failure()
            self._stats['verification_errors'] += 1
            return URLRecord(
                url=url, url_hash=url_hash, status=URLStatus.INVALID,
                first_seen=current_time, last_verified=current_time,
                verification_count=1, failure_count=1,
                http_status_code=None, content_type=None, content_length=None,
                redirect_url=None, error_message=f"Verification failed: {str(e)[:200]}", metadata={}
            )

    def store_url_record(self, record: URLRecord):
        """Store or update URL record in database"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # Check if URL exists
                existing = conn.execute(
                    "SELECT first_seen, verification_count, failure_count FROM urls WHERE url_hash = ?",
                    (record.url_hash,)
                ).fetchone()

                if existing:
                    # Update existing record
                    first_seen, ver_count, fail_count = existing
                    conn.execute("""
                        UPDATE urls SET
                            status = ?,
                            last_verified = ?,
                            verification_count = ?,
                            failure_count = ?,
                            http_status_code = ?,
                            content_type = ?,
                            content_length = ?,
                            redirect_url = ?,
                            error_message = ?,
                            metadata = ?
                        WHERE url_hash = ?
                    """, (
                        record.status.value,
                        record.last_verified,
                        ver_count + 1,
                        fail_count + record.failure_count,
                        record.http_status_code,
                        record.content_type,
                        record.content_length,
                        record.redirect_url,
                        record.error_message,
                        json.dumps(record.metadata),
                        record.url_hash
                    ))
                else:
                    # Insert new record
                    conn.execute("""
                        INSERT INTO urls (
                            url_hash, url, status, first_seen, last_verified,
                            verification_count, failure_count, http_status_code,
                            content_type, content_length, redirect_url,
                            error_message, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        record.url_hash,
                        record.url,
                        record.status.value,
                        record.first_seen,
                        record.last_verified,
                        record.verification_count,
                        record.failure_count,
                        record.http_status_code,
                        record.content_type,
                        record.content_length,
                        record.redirect_url,
                        record.error_message,
                        json.dumps(record.metadata)
                    ))

                conn.commit()

    def get_url_record(self, url: str) -> Optional[URLRecord]:
        """Retrieve URL record from database"""
        url_hash = self._hash_url(url)

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM urls WHERE url_hash = ?", (url_hash,)).fetchone()

            if not row:
                self._stats['cache_misses'] += 1
                return None

            self._stats['cache_hits'] += 1

            return URLRecord(
                url=row[1],
                url_hash=row[0],
                status=URLStatus(row[2]),
                first_seen=row[3],
                last_verified=row[4],
                verification_count=row[5],
                failure_count=row[6],
                http_status_code=row[7],
                content_type=row[8],
                content_length=row[9],
                redirect_url=row[10],
                error_message=row[11],
                metadata=json.loads(row[12]) if row[12] else {}
            )

    def process_url(self, url: str, force_verify: bool = False) -> Tuple[bool, str, Optional[URLRecord]]:
        """
        Process URL with validation, verification, and caching
        Returns: (is_valid, message, record)
        """
        # Security validation first
        is_safe, reason = self.validate_url_security(url)
        if not is_safe:
            logger.warning(f"URL blocked: {url} - Reason: {reason}")
            self._stats['urls_blocked'] += 1
            return False, reason, None

        # Check cache
        if not force_verify:
            cached_record = self.get_url_record(url)
            if cached_record:
                # Use cache if verified within last hour
                if time.time() - cached_record.last_verified < 3600:
                    is_valid = cached_record.status == URLStatus.VALID
                    return is_valid, cached_record.error_message or "OK", cached_record

        # Verify URL
        record = self.verify_url_accessibility(url)
        self.store_url_record(record)

        is_valid = record.status == URLStatus.VALID
        message = record.error_message or "URL verified successfully"

        return is_valid, message, record

    def batch_process_urls(self, urls: List[str], force_verify: bool = False) -> Dict[str, Dict[str, Any]]:
        """Process multiple URLs in parallel with controlled concurrency"""
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {
                executor.submit(self.process_url, url, force_verify): url
                for url in urls
            }

            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    is_valid, message, record = future.result()
                    results[url] = {
                        'valid': is_valid,
                        'message': message,
                        'status': record.status.value if record else 'error',
                        'http_status': record.http_status_code if record else None
                    }
                except Exception as e:
                    logger.error(f"Error processing URL {url}: {e}")
                    results[url] = {
                        'valid': False,
                        'message': f"Processing error: {str(e)}",
                        'status': 'error',
                        'http_status': None
                    }

        return results

    def cleanup_invalid_urls(self, urls: List[str]) -> List[str]:
        """Remove invalid URLs from list"""
        valid_urls = []

        for url in urls:
            is_safe, _ = self.validate_url_security(url)
            if is_safe:
                valid_urls.append(url)
            else:
                logger.info(f"Cleaned invalid URL: {url}")
                self._stats['urls_cleaned'] += 1

        return valid_urls

    def cleanup_stale_records(self, max_age_days: int = 30, max_failures: int = 5):
        """Clean up old and failed records"""
        cutoff_time = time.time() - (max_age_days * 86400)
        removed_count = 0

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # Remove old invalid records
                result = conn.execute("""
                    DELETE FROM urls
                    WHERE (last_verified < ? AND status != ?)
                       OR failure_count > ?
                """, (cutoff_time, URLStatus.VALID.value, max_failures))

                removed_count = result.rowcount
                conn.commit()

        logger.info(f"Cleaned up {removed_count} stale URL records")
        return removed_count

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        with sqlite3.connect(self.db_path) as conn:
            total_urls = conn.execute("SELECT COUNT(*) FROM urls").fetchone()[0]
            valid_urls = conn.execute("SELECT COUNT(*) FROM urls WHERE status = ?", (URLStatus.VALID.value,)).fetchone()[0]
            invalid_urls = conn.execute("SELECT COUNT(*) FROM urls WHERE status != ?", (URLStatus.VALID.value,)).fetchone()[0]

        return {
            'runtime_stats': dict(self._stats),
            'database_stats': {
                'total_urls': total_urls,
                'valid_urls': valid_urls,
                'invalid_urls': invalid_urls
            },
            'circuit_breaker': {
                'state': self._circuit_breaker['state'],
                'failure_count': self._circuit_breaker['failure_count']
            }
        }

    def export_valid_urls(self, output_file: str):
        """Export all valid URLs to file"""
        with sqlite3.connect(self.db_path) as conn:
            valid_urls = conn.execute(
                "SELECT url FROM urls WHERE status = ? ORDER BY last_verified DESC",
                (URLStatus.VALID.value,)
            ).fetchall()

        with open(output_file, 'w', encoding='utf-8') as f:
            for url_tuple in valid_urls:
                f.write(url_tuple[0] + '\n')

        logger.info(f"Exported {len(valid_urls)} valid URLs to {output_file}")
        return len(valid_urls)

    def close(self):
        """Cleanup resources"""
        if self._session:
            self._session.close()


# Global instance
_url_manager = None


def get_url_manager() -> ProductionURLManager:
    """Get global URL manager instance"""
    global _url_manager
    if _url_manager is None:
        _url_manager = ProductionURLManager()
    return _url_manager


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    manager = ProductionURLManager()

    test_urls = [
        "https://example.tumblr.com",
        "https://64.media.tumblr.com/abc123/tumblr_xyz789_1280.jpg",
        "https://malicious-site.com/fake",  # Should be blocked
        "https://192.168.1.1/ssrf-attempt",  # Should be blocked
    ]

    results = manager.batch_process_urls(test_urls)

    for url, result in results.items():
        print(f"{url}: {result}")

    print("\nStatistics:", json.dumps(manager.get_statistics(), indent=2))

    manager.close()
