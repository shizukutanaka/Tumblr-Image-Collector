import os
import random
import shutil
import tempfile
import time
import logging
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional, Dict, Any, Iterable

import requests
from PIL import Image
import imagehash

# Assuming these modules will be siblings or in an accessible path
from exceptions import DiskSpaceError, DownloadError
from resource_manager import DiskSpaceManager

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 8192

class Downloader:
    """Manages the downloading and storing of media (images and videos)."""

    def __init__(self, session, output_folder, downloaded_hashes, config, download_stats, cache_manager):
        self.session = session
        self.output_folder = Path(output_folder)
        self.downloaded_hashes = downloaded_hashes
        self.config = config
        self._download_stats = download_stats
        self.cache_manager = cache_manager

        network_cfg = self.config.get('network', {})
        self.download_timeout = int(network_cfg.get('download_timeout_seconds', 30))
        self.max_retries = int(network_cfg.get('max_retries', 3))
        self.backoff_factor = float(network_cfg.get('backoff_factor', 0.5))
        self.max_backoff_seconds = int(network_cfg.get('max_backoff_seconds', 60))

        security_cfg = self.config.get('security', {})
        self.allowed_domains = self._load_allowed_domains(security_cfg.get('allowed_domains'))

        self.IMAGE_FILTERS = self.config.get('filters', {})
        self.slow_response_threshold = int(self.config.get('slow_response_threshold', 45))

        self.proxies = self._build_requests_proxies(self.config.get('proxy'))

    def _update_download_stats(self, outcome: str) -> None:
        """Updates download statistics based on the outcome."""
        if outcome == 'success':
            self._download_stats['successful_downloads'] += 1
        elif outcome == 'failure':
            self._download_stats['failed_downloads'] += 1
        elif outcome == 'duplicate':
            self._download_stats['skipped_duplicates'] += 1
        self._download_stats['total_images_processed'] += 1

    def _load_allowed_domains(self, values: Optional[Iterable[str]]) -> set:
        default_domains = {
            "tumblr.com",
            "media.tumblr.com",
            "data.tumblr.com",
        }
        if not values:
            return default_domains
        if isinstance(values, str):
            candidates = [part.strip() for part in values.split(',') if part.strip()]
        else:
            candidates = [str(item).strip() for item in values if str(item).strip()]
        normalized = {candidate.lower() for candidate in candidates if candidate}
        return normalized or default_domains

    def _build_requests_proxies(self, proxy_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
        if not proxy_config or not proxy_config.get('type'):
            return None

        proxy_type = proxy_config.get('type')
        if proxy_type not in {'http', 'https', 'socks4', 'socks5'}:
            return None

        host = proxy_config.get('host')
        port = proxy_config.get('port')
        if not host or not port:
            return None

        auth = ''
        username = proxy_config.get('username')
        password = proxy_config.get('password')
        if username and password:
            auth = f"{username}:{password}@"
        elif username:
            auth = f"{username}@"

        proxy_url = f"{proxy_type}://{auth}{host}:{port}"
        return {
            'http': proxy_url,
            'https': proxy_url
        }

    def download_image(self, image_url, post_data=None, max_retries=None):
        """Downloads an image with advanced retry mechanisms and caching."""
        if max_retries is None:
            max_retries = self.max_retries

        if not self._validate_image_domain(image_url):
            logger.warning(f"Blocked image from disallowed domain: {image_url}")
            self._update_download_stats('failure')
            return False

        self._download_stats['total_attempts'] += 1

        cached_file = self.cache_manager.check_cache(image_url)
        if cached_file:
            try:
                filename = self._generate_filename_from_path(cached_file)
                filepath = self.output_folder / filename
                shutil.copy2(cached_file, filepath)
                logger.info(f"Copied from cache: {filename}")
                return True
            except Exception as e:
                logger.error(f"Error copying from cache: {e}")

        def exponential_backoff(retry_count):
            base_delay = max(0.5, self.backoff_factor)
            max_delay = max(1, self.max_backoff_seconds)
            jitter = random.uniform(0, 0.5)
            delay = min(max_delay, base_delay * (2 ** retry_count) + jitter)
            logger.info(f"Retry {retry_count + 1}: waiting {delay:.2f} seconds")
            time.sleep(delay)

        last_exception = None
        for retry_count in range(max_retries):
            try:
                result = self._download_and_store_image(image_url, post_data)
                if result:
                    return True
                return False  # Duplicate or invalid, no retry
            except requests.RequestException as e:
                last_exception = e
                logger.warning(f"Download attempt {retry_count + 1} failed: {e}")
                if self._is_network_error(e):
                    if retry_count < max_retries - 1:
                        exponential_backoff(retry_count)
                    else:
                        logger.error(f"Failed to download {image_url} after {max_retries} retries.")
                        self._update_download_stats('failure')
                        return False
                else:
                    logger.error(f"Non-recoverable error: {e}")
                    self._update_download_stats('failure')
                    return False
            except IOError as e:
                logger.error(f"File processing error: {e}")
                self._update_download_stats('failure')
                return False

        if last_exception:
            self._log_download_failure(image_url, post_data, last_exception)

        return False

    def _is_network_error(self, exception):
        network_error_types = (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ProxyError,
            requests.exceptions.SSLError
        )
        return isinstance(exception, network_error_types)

    def _validate_image_domain(self, image_url: str) -> bool:
        try:
            hostname = urlparse(image_url).hostname or ""
            return self._is_allowed_domain(hostname.lower())
        except Exception:
            return False

    def _is_allowed_domain(self, hostname: str) -> bool:
        if not hostname:
            return False
        return any(hostname == domain or hostname.endswith(f".{domain}") for domain in self.allowed_domains)

    def _log_download_failure(self, image_url, post_data, exception):
        failure_log_path = self.output_folder / 'download_failures.log'
        with open(failure_log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(
                f"Time: {datetime.datetime.now().isoformat()}\nURL: {image_url}\nError Type: {type(exception).__name__}\nError: {str(exception)}\n---\n"
            )

    def _generate_filename_from_path(self, cached_path: Path) -> str:
        cached_path = Path(cached_path)
        extension = cached_path.suffix.lower() or '.jpg'
        safe_name = cached_path.stem
        return f"cached_{safe_name}{extension}"

    def _download_and_store_image(self, image_url, post_data=None, allow_duplicate_skip=True):
        """Downloads, validates, and stores a single image."""
        try:
            disk_manager = DiskSpaceManager(min_free_space_mb=100)
            disk_manager.ensure_space_for_file(5, str(self.output_folder))
        except DiskSpaceError as e:
            logger.error(f"Insufficient disk space for download: {e}")
            raise

        request_started = time.monotonic()
        with self.session.get(
            image_url, stream=True, proxies=self.proxies, timeout=self.download_timeout
        ) as response:
            response.raise_for_status()

            # Security checks
            if not self._is_response_safe(response, image_url):
                return False

            # Download to a temporary file
            temp_path = self._download_to_temp_file(response, image_url)
            if not temp_path:
                return False

        try:
            with Image.open(temp_path) as img:
                # Validation and duplicate check
                if not self._is_image_valid(img) or (allow_duplicate_skip and self._is_image_duplicate(img)):
                    os.unlink(temp_path)
                    return False

                # Save file and metadata
                metadata = self._extract_image_metadata(img, post_data)
                filename = self._generate_output_filename(temp_path, metadata, image_url, post_data)
                filepath = self.output_folder / filename

                if filepath.exists() and allow_duplicate_skip:
                    os.unlink(temp_path)
                    self._update_download_stats('duplicate')
                    logger.info(f"Skipping duplicate file: {filename}")
                    return False

                shutil.move(temp_path, filepath)
                self.cache_manager.save_to_cache(filepath, image_url)

            self._update_download_stats('success')
            logger.info(f"Successfully downloaded: {filename}")

            # Persist metadata
            self._save_metadata(filepath, metadata)

            self.downloaded_files.add(filename)
            if metadata.get('phash'):
                self.downloaded_hashes.add(metadata['phash'])

            elapsed = time.monotonic() - request_started
            if elapsed > self.slow_response_threshold:
                logger.info(f"Slow download detected: {elapsed:.1f}s for {image_url}")
            return filepath

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _is_response_safe(self, response, image_url):
        final_url = response.url or image_url
        parsed_final = urlparse(final_url)
        final_host = (parsed_final.hostname or "").lower()

        if parsed_final.scheme.lower() != 'https':
            logger.warning(f"Non-HTTPS scheme detected, aborting download: {final_url}")
            self._update_download_stats('failure')
            return False

        if not self._is_allowed_domain(final_host):
            logger.warning(f"Blocked response from disallowed domain: {final_host}. Original URL: {image_url}")
            self._update_download_stats('failure')
            return False

        content_type = (response.headers.get('Content-Type') or '').lower()
        if content_type and not content_type.startswith('image/'):
            logger.warning(f"Non-image Content-Type detected, aborting: {content_type} for {image_url}")
            self._update_download_stats('failure')
            return False

        return True

    def _download_to_temp_file(self, response, image_url):
        max_size_mb = self.IMAGE_FILTERS.get('max_file_size_mb', 10)
        max_download_bytes = max_size_mb * 1024 * 1024

        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=self._get_file_extension(image_url))
            downloaded_bytes = 0
            for chunk in response.iter_content(chunk_size=DEFAULT_CHUNK_SIZE):
                downloaded_bytes += len(chunk)
                if downloaded_bytes > max_download_bytes:
                    logger.warning(f"Download exceeded max size, aborting: {image_url}")
                    temp_file.close()
                    os.unlink(temp_file.name)
                    self._update_download_stats('failure')
                    return None
                temp_file.write(chunk)
            temp_file.close()
            return temp_file.name
        except Exception as e:
            logger.error(f"Failed to download to temp file: {e}")
            if 'temp_file' in locals() and temp_file:
                temp_file.close()
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
            return None

    def _get_file_extension(self, url):
        path = urlparse(url).path
        ext = os.path.splitext(path)[1]
        return ext if ext else '.jpg'

    def _is_image_valid(self, image):
        try:
            width, height = image.size
            if width < self.IMAGE_FILTERS.get('min_width', 100) or height < self.IMAGE_FILTERS.get('min_height', 100):
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating image: {e}")
            return False

    def _is_image_duplicate(self, image):
        try:
            current_hash = str(imagehash.phash(image))
            if current_hash in self.downloaded_hashes:
                self._update_download_stats('duplicate')
                logger.info(f"Skipping duplicate image with hash: {current_hash}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to compute image hash: {e}")
            return False

    def _extract_image_metadata(self, image, post_data):
        phash = str(imagehash.phash(image))
        metadata = {
            'width': image.width,
            'height': image.height,
            'format': image.format,
            'phash': phash,
            'tags': post_data.get('tags', []) if post_data else []
        }
        return metadata

    def _generate_output_filename(self, source_path, metadata, image_url, post_data):
        path_obj = Path(source_path)
        extension = path_obj.suffix.lower() or '.jpg'
        blog_name = (post_data.get('blog_name') or '') if post_data else ''
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        size_token = f"{metadata.get('width')}x{metadata.get('height')}"
        base_token = path_obj.stem
        raw_name = '_'.join(filter(None, [blog_name, timestamp, size_token, base_token]))
        sanitized = self._sanitize_filename(raw_name)
        return f"{sanitized}{extension}"

    def _sanitize_filename(self, filename):
        return "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()

    def _save_metadata(self, filepath, metadata):
        metadata_file = filepath.with_suffix('.json')
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to save metadata: {e}")