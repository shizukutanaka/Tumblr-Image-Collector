#!/usr/bin/env python3
"""
Lightweight Download Manager with Resume Support
Simple, efficient, and practical implementation
"""

import os
import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class DownloadManager:
    """Manages file downloads with resume capability"""

    def __init__(self, download_dir: str, state_file: str = "download_state.json"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.download_dir / state_file
        self.state = self._load_state()
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry logic"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        return session

    def _load_state(self) -> Dict:
        """Load download state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
        return {"downloads": {}, "completed": {}}

    def _save_state(self):
        """Save download state to file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _get_file_hash(self, url: str) -> str:
        """Generate unique hash for URL"""
        return hashlib.md5(url.encode()).hexdigest()

    def _get_file_path(self, url: str, filename: Optional[str] = None) -> Path:
        """Get file path for download"""
        if not filename:
            parsed = urlparse(url)
            filename = Path(parsed.path).name or f"{self._get_file_hash(url)}.bin"

        return self.download_dir / filename

    def is_downloaded(self, url: str) -> bool:
        """Check if file is already downloaded"""
        file_hash = self._get_file_hash(url)
        return file_hash in self.state.get("completed", {})

    def get_partial_size(self, file_path: Path) -> int:
        """Get size of partial download"""
        if file_path.exists():
            return file_path.stat().st_size
        return 0

    def download(
        self,
        url: str,
        filename: Optional[str] = None,
        timeout: int = 30,
        chunk_size: int = 8192,
        resume: bool = True
    ) -> Tuple[bool, str, Optional[Path]]:
        """Download file with resume support.

        Args:
            url: URL to download
            filename: Optional custom filename
            timeout: Request timeout in seconds
            chunk_size: Download chunk size in bytes
            resume: Enable resume capability

        Returns:
            Tuple of (success, message, file_path)
        """
        if not url or not isinstance(url, str):
            return False, "Invalid URL", None

        if len(url) > 2048:
            return False, "URL too long", None

        file_hash = self._get_file_hash(url)
        file_path = self._get_file_path(url, filename)

        # Check if already completed
        if self.is_downloaded(url):
            if file_path.exists():
                return True, "Already downloaded", file_path
            else:
                # File was deleted, remove from completed
                self.state["completed"].pop(file_hash, None)

        # Get existing file size for resume
        downloaded_size = self.get_partial_size(file_path) if resume else 0

        headers = {}
        if resume and downloaded_size > 0:
            headers['Range'] = f'bytes={downloaded_size}-'
            logger.info(f"Resuming download from byte {downloaded_size}")

        try:
            # Make request
            response = self.session.get(url, headers=headers, stream=True, timeout=timeout)

            # Check if resume is supported
            if resume and downloaded_size > 0:
                if response.status_code != 206:
                    logger.warning("Server doesn't support resume, starting fresh")
                    downloaded_size = 0
                    file_path.unlink(missing_ok=True)

            response.raise_for_status()

            # Get total size
            total_size = int(response.headers.get('content-length', 0))
            if resume and downloaded_size > 0:
                total_size += downloaded_size

            # Download file
            mode = 'ab' if resume and downloaded_size > 0 else 'wb'

            with open(file_path, mode) as f:
                start_time = time.time()
                bytes_written = 0

                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)

                        # Progress logging (every 1MB)
                        if bytes_written % (1024 * 1024) < chunk_size:
                            elapsed = time.time() - start_time
                            speed = bytes_written / elapsed if elapsed > 0 else 0
                            logger.debug(
                                f"Downloaded {bytes_written / 1024 / 1024:.2f}MB "
                                f"@ {speed / 1024 / 1024:.2f}MB/s"
                            )

            # Mark as completed
            self.state["completed"][file_hash] = {
                "url": url,
                "path": str(file_path),
                "size": file_path.stat().st_size,
                "timestamp": time.time()
            }
            self.state["downloads"].pop(file_hash, None)
            self._save_state()

            elapsed = time.time() - start_time
            logger.info(
                f"Download completed: {file_path.name} "
                f"({file_path.stat().st_size / 1024 / 1024:.2f}MB in {elapsed:.1f}s)"
            )

            return True, "Download successful", file_path

        except requests.exceptions.Timeout:
            return False, "Download timeout", None

        except requests.exceptions.ConnectionError as e:
            return False, f"Connection error: {str(e)[:100]}", None

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False, f"Download error: {str(e)[:100]}", None

    def batch_download(
        self,
        urls: list,
        max_concurrent: int = 3,
        skip_existing: bool = True
    ) -> Dict[str, Tuple[bool, str, Optional[Path]]]:
        """
        Download multiple files
        Returns: dict of url -> (success, message, path)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}

        # Filter already downloaded if requested
        if skip_existing:
            urls_to_download = [url for url in urls if not self.is_downloaded(url)]
            skipped = len(urls) - len(urls_to_download)
            if skipped > 0:
                logger.info(f"Skipping {skipped} already downloaded files")
        else:
            urls_to_download = urls

        # Download in parallel
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_url = {
                executor.submit(self.download, url): url
                for url in urls_to_download
            }

            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    results[url] = future.result()
                except Exception as e:
                    logger.error(f"Batch download error for {url}: {e}")
                    results[url] = (False, str(e), None)

        return results

    def cleanup_incomplete(self):
        """Remove incomplete downloads that are too old"""
        cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days

        cleaned = 0
        for file_hash, info in list(self.state.get("downloads", {}).items()):
            if info.get("timestamp", 0) < cutoff_time:
                file_path = Path(info.get("path", ""))
                if file_path.exists():
                    try:
                        file_path.unlink()
                        cleaned += 1
                    except Exception as e:
                        logger.warning(f"Failed to cleanup {file_path}: {e}")

                self.state["downloads"].pop(file_hash, None)

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} incomplete downloads")
            self._save_state()

        return cleaned

    def get_statistics(self) -> Dict:
        """Get download statistics"""
        return {
            "total_completed": len(self.state.get("completed", {})),
            "in_progress": len(self.state.get("downloads", {})),
            "total_size_mb": sum(
                info.get("size", 0)
                for info in self.state.get("completed", {}).values()
            ) / 1024 / 1024
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    manager = DownloadManager("./downloads")

    # Single download
    success, message, path = manager.download(
        "https://via.placeholder.com/1000",
        filename="test_image.png"
    )

    print(f"Download: {success} - {message}")
    if path:
        print(f"Saved to: {path}")

    # Statistics
    stats = manager.get_statistics()
    print(f"Statistics: {stats}")
