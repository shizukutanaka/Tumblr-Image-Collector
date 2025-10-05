#!/usr/bin/env python3
"""
URL Validation and Verification Module
Ensures only valid, accessible Tumblr URLs are processed
"""

import re
import requests
import logging
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, parse_qs
import time
from pathlib import Path
import json
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TumblrURLValidator:
    """Advanced URL validation specifically for Tumblr content"""

    # Valid Tumblr domains
    VALID_TUMBLR_DOMAINS = {
        'tumblr.com',
        'www.tumblr.com',
        'assets.tumblr.com',
        'static.tumblr.com',
        'media.tumblr.com',
        '64.media.tumblr.com',
        'va.media.tumblr.com'
    }

    # URL patterns for different Tumblr content types
    # ReDoS対策: 量指定子の上限を設定し、複雑なネストを回避
    TUMBLR_PATTERNS = {
        'blog': re.compile(r'^https?://([a-z0-9-]{1,63})\.tumblr\.com/?$'),
        'post': re.compile(r'^https?://([a-z0-9-]{1,63})\.tumblr\.com/post/(\d{1,20})'),
        'image': re.compile(r'^https?://\d{1,3}\.media\.tumblr\.com/[a-f0-9]{1,128}/tumblr_[a-z0-9]{1,64}_\d{1,10}\.(jpg|jpeg|png|gif|webp)$'),
        'tag': re.compile(r'^https?://([a-z0-9-]{1,63})\.tumblr\.com/tagged/([^/]{1,200})'),
        'archive': re.compile(r'^https?://([a-z0-9-]{1,63})\.tumblr\.com/archive'),
    }

    def __init__(self, cache_file: str = "url_cache.json"):
        self.cache_file = Path(cache_file)
        self.url_cache: Dict[str, Dict] = self._load_cache()
        self.session = self._create_session()
        self._lock = threading.Lock()

    def _create_session(self) -> requests.Session:
        """Create optimized requests session"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Configure retries and timeouts
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _load_cache(self) -> Dict[str, Dict]:
        """Load URL validation cache"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)

                # Clean expired entries (older than 7 days)
                current_time = datetime.now().timestamp()
                cleaned_cache = {}
                for url, data in cache.items():
                    if current_time - data.get('timestamp', 0) < 7 * 24 * 3600:
                        cleaned_cache[url] = data

                return cleaned_cache
            except Exception as e:
                logger.warning(f"Failed to load URL cache: {e}")

        return {}

    def _save_cache(self):
        """Save URL validation cache"""
        try:
            with self._lock:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.url_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save URL cache: {e}")

    def validate_url_format(self, url: str) -> Tuple[bool, str, str]:
        """Validate URL format and determine type"""
        if not url or not isinstance(url, str):
            return False, "invalid", "Empty or invalid URL"

        # URL長の検証（攻撃対策）
        if len(url) > 2048:
            return False, "invalid", "URL too long"

        # Basic URL validation
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False, "invalid", "Malformed URL"

            # スキームの検証
            if parsed.scheme not in ('http', 'https'):
                return False, "invalid", "Invalid URL scheme"

            # ドメイン長の検証
            if len(parsed.netloc) > 255:
                return False, "invalid", "Domain name too long"

        except Exception:
            return False, "invalid", "URL parsing failed"

        # Check if it's a Tumblr URL
        domain = parsed.netloc.lower()
        is_tumblr_domain = any(
            domain == valid_domain or domain.endswith('.' + valid_domain)
            for valid_domain in self.VALID_TUMBLR_DOMAINS
        )

        if not is_tumblr_domain:
            return False, "non_tumblr", "Not a Tumblr URL"

        # Determine URL type with timeout protection
        for url_type, pattern in self.TUMBLR_PATTERNS.items():
            try:
                if pattern.match(url):
                    return True, url_type, f"Valid {url_type} URL"
            except Exception:
                # ReDoS攻撃から保護
                return False, "invalid", "Pattern matching failed"

        return True, "unknown", "Valid Tumblr URL of unknown type"

    def check_url_accessibility(self, url: str, timeout: int = 10) -> Tuple[bool, int, str]:
        """Check if URL is accessible and returns appropriate response"""
        try:
            # Check cache first
            cache_key = f"access_{url}"
            if cache_key in self.url_cache:
                cached = self.url_cache[cache_key]
                # Use cache if less than 1 hour old
                if time.time() - cached['timestamp'] < 3600:
                    return cached['accessible'], cached['status_code'], cached['message']

            # Make request
            response = self.session.head(url, timeout=timeout, allow_redirects=True)
            accessible = response.status_code < 400
            message = f"HTTP {response.status_code}"

            # Cache result
            with self._lock:
                self.url_cache[cache_key] = {
                    'accessible': accessible,
                    'status_code': response.status_code,
                    'message': message,
                    'timestamp': time.time()
                }

            return accessible, response.status_code, message

        except requests.exceptions.Timeout:
            return False, 0, "Request timeout"
        except requests.exceptions.ConnectionError:
            return False, 0, "Connection failed"
        except Exception as e:
            return False, 0, f"Request failed: {str(e)}"

    def extract_blog_name(self, url: str) -> Optional[str]:
        """Extract blog name from Tumblr URL"""
        try:
            parsed = urlparse(url)
            if parsed.netloc.endswith('.tumblr.com'):
                blog_name = parsed.netloc.replace('.tumblr.com', '')
                return blog_name if blog_name else None
        except Exception:
            pass
        return None

    def validate_blog_exists(self, blog_name: str) -> Tuple[bool, str]:
        """Validate that a Tumblr blog exists and is accessible"""
        if not blog_name or not isinstance(blog_name, str):
            return False, "Invalid blog name"

        # 長さ検証
        if len(blog_name) > 63:  # DNS標準の最大ラベル長
            return False, "Blog name too long"

        # Sanitize blog name - より厳格な検証
        sanitized = blog_name.lower().strip()
        # 英数字とハイフンのみ許可（ハイフンは先頭・末尾に不可）
        if not re.match(r'^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$', sanitized):
            return False, "Invalid characters in blog name"

        blog_url = f"https://{sanitized}.tumblr.com"

        # Check cache first
        cache_key = f"blog_{sanitized}"
        if cache_key in self.url_cache:
            cached = self.url_cache[cache_key]
            # Use cache if less than 24 hours old
            if time.time() - cached['timestamp'] < 24 * 3600:
                return cached['exists'], cached['message']

        try:
            # Check if blog exists by making a request
            response = self.session.get(blog_url, timeout=10, allow_redirects=True)

            # Check for indicators that blog exists
            exists = (
                response.status_code == 200 and
                'tumblr' in response.text.lower() and
                'not found' not in response.text.lower()
            )

            message = "Blog exists" if exists else "Blog not found or private"

            # Cache result
            with self._lock:
                self.url_cache[cache_key] = {
                    'exists': exists,
                    'message': message,
                    'timestamp': time.time()
                }

            return exists, message

        except Exception as e:
            return False, f"Validation failed: {str(e)}"

    def get_image_urls_from_post(self, post_url: str) -> List[str]:
        """Extract image URLs from a Tumblr post"""
        try:
            # コンテンツサイズ制限
            response = self.session.get(post_url, timeout=15, stream=True)
            if response.status_code != 200:
                return []

            # レスポンスサイズを制限（DoS対策）
            max_content_size = 10 * 1024 * 1024  # 10MB
            content = b''
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > max_content_size:
                    logger.warning(f"Content too large for {post_url}")
                    return []

            text_content = content.decode('utf-8', errors='ignore')

            # Look for image URLs in the HTML (ReDoS対策版)
            import re
            image_pattern = re.compile(
                r'https?://\d{1,3}\.media\.tumblr\.com/[a-f0-9]{1,128}/tumblr_[a-z0-9]{1,64}_\d{1,10}\.(jpg|jpeg|png|gif|webp)',
                re.IGNORECASE
            )

            # findall の結果を検証
            matches = image_pattern.findall(text_content)
            if not matches:
                return []

            # タプルの場合は完全なマッチを取得
            if matches and isinstance(matches[0], tuple):
                # findall がグループを返す場合、完全マッチを再取得
                images = list(set([match[0] if isinstance(match, tuple) else match for match in image_pattern.finditer(text_content)]))
                return [m.group(0) for m in image_pattern.finditer(text_content)][:50]  # 最大50件に制限
            else:
                return list(set(matches))[:50]  # 最大50件に制限

        except Exception as e:
            logger.warning(f"Failed to extract images from post {post_url}: {e}")
            return []

    def batch_validate_urls(self, urls: List[str], max_workers: int = 5) -> Dict[str, Dict]:
        """Validate multiple URLs in parallel"""
        import concurrent.futures

        results = {}

        def validate_single_url(url: str) -> Tuple[str, Dict]:
            # Format validation
            is_valid, url_type, format_msg = self.validate_url_format(url)
            if not is_valid:
                return url, {
                    'valid': False,
                    'type': url_type,
                    'format_error': format_msg,
                    'accessible': False,
                    'blog_exists': False
                }

            # Accessibility check
            accessible, status_code, access_msg = self.check_url_accessibility(url)

            # Blog existence check (for blog URLs)
            blog_exists = True
            blog_msg = "N/A"
            if url_type == 'blog':
                blog_name = self.extract_blog_name(url)
                if blog_name:
                    blog_exists, blog_msg = self.validate_blog_exists(blog_name)

            return url, {
                'valid': True,
                'type': url_type,
                'accessible': accessible,
                'status_code': status_code,
                'access_message': access_msg,
                'blog_exists': blog_exists,
                'blog_message': blog_msg
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(validate_single_url, url): url for url in urls}

            for future in concurrent.futures.as_completed(future_to_url):
                try:
                    url, result = future.result()
                    results[url] = result
                except Exception as e:
                    url = future_to_url[future]
                    results[url] = {
                        'valid': False,
                        'error': str(e)
                    }

        # Save cache after batch operation
        self._save_cache()

        return results

    def cleanup_invalid_urls(self, urls: List[str]) -> List[str]:
        """Remove invalid URLs from a list"""
        valid_urls = []

        for url in urls:
            is_valid, url_type, _ = self.validate_url_format(url)
            if is_valid:
                valid_urls.append(url)
            else:
                logger.info(f"Removed invalid URL: {url}")

        return valid_urls

    def get_validation_stats(self) -> Dict[str, int]:
        """Get statistics about URL validation cache"""
        stats = {
            'total_cached': len(self.url_cache),
            'blogs_checked': 0,
            'urls_checked': 0,
            'cache_age_hours': 0
        }

        if self.url_cache:
            current_time = time.time()
            timestamps = []

            for key, data in self.url_cache.items():
                timestamps.append(data['timestamp'])
                if key.startswith('blog_'):
                    stats['blogs_checked'] += 1
                elif key.startswith('access_'):
                    stats['urls_checked'] += 1

            if timestamps:
                oldest_timestamp = min(timestamps)
                stats['cache_age_hours'] = int((current_time - oldest_timestamp) / 3600)

        return stats

    def cleanup_cache(self, max_age_days: int = 7):
        """Clean up old cache entries"""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600

        with self._lock:
            keys_to_remove = []
            for key, data in self.url_cache.items():
                if current_time - data['timestamp'] > max_age_seconds:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.url_cache[key]

        self._save_cache()
        logger.info(f"Cleaned up {len(keys_to_remove)} old cache entries")


# Global URL validator instance
_url_validator = None

def get_url_validator() -> TumblrURLValidator:
    """Get global URL validator instance"""
    global _url_validator
    if _url_validator is None:
        _url_validator = TumblrURLValidator()
    return _url_validator