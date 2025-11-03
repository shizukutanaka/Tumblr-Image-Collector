"""
Cache Manager

画像キャッシュの管理、保存、検索を担当するモジュール
"""

import json
import time
import uuid
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CacheManager:
    """
    キャッシュ管理を担当するクラス
    """

    def __init__(self, cache_dir: Path, config: Dict[str, Any]):
        self.cache_dir = cache_dir
        self.config = config
        self.cache_enabled = config.get('cache', {}).get('enabled', True)
        self.cache_index: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = config.get('cache', {}).get('ttl_seconds', 24 * 60 * 60)
        self.max_entries = config.get('cache', {}).get('max_entries', 2048)

        # キャッシュ統計カウンター
        self.cache_hits = 0
        self.cache_misses = 0

        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache_index()

    def _load_cache_index(self) -> Dict[str, Dict[str, Any]]:
        """キャッシュインデックスを読み込む"""
        index_path = self.cache_dir / "index.json"
        if not index_path.exists():
            return {}

        try:
            with open(index_path, 'r', encoding='utf-8') as index_file:
                data = json.load(index_file)
            if isinstance(data, dict):
                self.cache_index = data
                return data
        except json.JSONDecodeError as decode_err:
            logger.warning(f"キャッシュインデックスの読み込みに失敗しました: {decode_err}")
        except OSError as os_err:
            logger.warning(f"キャッシュインデックスファイルにアクセスできません: {os_err}")
        return {}

    def _persist_cache_index(self) -> None:
        """キャッシュインデックスを保存"""
        if not self.cache_enabled:
            return

        index_path = self.cache_dir / "index.json"
        try:
            with open(index_path, 'w', encoding='utf-8') as index_file:
                json.dump(self.cache_index, index_file, ensure_ascii=False, indent=2)
        except OSError as os_err:
            logger.warning(f"キャッシュインデックスの保存に失敗しました: {os_err}")

    def _cache_key(self, image_url: str) -> str:
        """キャッシュキーを生成"""
        return image_url.strip().lower()

    def check_cache(self, image_url: str) -> Optional[Path]:
        """
        キャッシュ済みファイルのパスを返す

        Args:
            image_url: 画像URL

        Returns:
            キャッシュされたファイルのパス、またはNone
        """
        if not self.cache_enabled:
            self.cache_misses += 1
            return None

        key = self._cache_key(image_url)
        cached_entry = self.cache_index.get(key)
        if not cached_entry:
            self.cache_misses += 1
            return None

        cached_path = self.cache_dir / cached_entry.get('filename', '')
        if not cached_path.exists():
            self.cache_index.pop(key, None)
            self.cache_misses += 1
            return None

        # TTLチェック
        expires_at = cached_entry.get('expires_at')
        if expires_at and time.time() > expires_at:
            logger.debug(f"キャッシュの有効期限切れ: {image_url}")
            try:
                cached_path.unlink(missing_ok=True)
            except OSError:
                pass
            self.cache_index.pop(key, None)
            self.cache_misses += 1
            return None

        self.cache_hits += 1
        return cached_path

    def save_to_cache(self, file_path: Path, image_url: str) -> None:
        """
        ダウンロードしたファイルをキャッシュに保存

        Args:
            file_path: 保存するファイルのパス
            image_url: 元の画像URL
        """
        if not self.cache_enabled:
            return

        key = self._cache_key(image_url)
        cached_name = f"{uuid.uuid4().hex}{file_path.suffix.lower()}"
        cached_path = self.cache_dir / cached_name

        try:
            shutil.copy2(file_path, cached_path)
        except OSError as copy_err:
            logger.warning(f"キャッシュへの保存に失敗: {copy_err}")
            return

        self.cache_index[key] = {
            'filename': cached_name,
            'stored_at': time.time(),
            'expires_at': time.time() + self.ttl_seconds if self.ttl_seconds else None
        }

        self._prune_cache()
        self._persist_cache_index()

    def _prune_cache(self) -> None:
        """キャッシュの最大件数を超える場合に古いエントリを削除"""
        current_time = time.time()

        # 期限切れエントリを収集・削除
        expired_keys = []
        for key, info in self.cache_index.items():
            expires_at = info.get('expires_at')
            if expires_at and current_time > expires_at:
                expired_keys.append(key)

        for key in expired_keys:
            info = self.cache_index.get(key)
            if info:
                filename = info.get('filename')
                if filename:
                    file_path = self.cache_dir / filename
                    try:
                        file_path.unlink(missing_ok=True)
                    except OSError as e:
                        logger.debug(f"期限切れキャッシュファイルの削除失敗: {file_path} - {e}")
                self.cache_index.pop(key, None)

        # 最大件数チェック
        if len(self.cache_index) <= self.max_entries:
            return

        # LRUで古いエントリを削除
        sorted_items = sorted(
            self.cache_index.items(),
            key=lambda item: item[1].get('stored_at', 0)
        )

        removed_count = 0
        target_removal = len(self.cache_index) - self.max_entries

        for key, info in sorted_items:
            if removed_count >= target_removal:
                break

            filename = info.get('filename')
            if filename:
                file_path = self.cache_dir / filename
                try:
                    file_path.unlink(missing_ok=True)
                    removed_count += 1
                except OSError as e:
                    logger.debug(f"キャッシュファイルの削除失敗: {file_path} - {e}")

            self.cache_index.pop(key, None)

        if removed_count > 0:
            logger.info(f"キャッシュを整理: {removed_count}件のエントリを削除")

    def clear_cache(self) -> None:
        """キャッシュを完全にクリア"""
        if not self.cache_enabled:
            return

        try:
            for filename in self.cache_dir.glob("*"):
                if filename.is_file() and filename.name != "index.json":
                    filename.unlink()
            self.cache_index.clear()
            self._persist_cache_index()
            logger.info("キャッシュをクリアしました")
        except Exception as e:
            logger.error(f"キャッシュクリア中にエラー: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests) * 100 if total_requests > 0 else 0.0

        total_size = 0
        file_count = 0

        try:
            for filename in self.cache_dir.glob("*"):
                if filename.is_file() and filename.name != "index.json":
                    total_size += filename.stat().st_size
                    file_count += 1
        except Exception as e:
            logger.warning(f"キャッシュ統計取得中にエラー: {e}")

        return {
            'enabled': self.cache_enabled,
            'total_entries': len(self.cache_index),
            'total_files': file_count,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'max_entries': self.max_entries,
            'ttl_seconds': self.ttl_seconds,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_requests': total_requests,
            'hit_rate_percent': round(hit_rate, 2)
        }

    def cleanup_expired(self) -> int:
        """期限切れのキャッシュエントリをクリーンアップ"""
        if not self.cache_enabled:
            return 0

        current_time = time.time()
        expired_keys = []
        cleaned_count = 0

        for key, info in self.cache_index.items():
            expires_at = info.get('expires_at')
            if expires_at and current_time > expires_at:
                expired_keys.append(key)

        for key in expired_keys:
            info = self.cache_index.get(key)
            if info:
                filename = info.get('filename')
                if filename:
                    file_path = self.cache_dir / filename
                    try:
                        file_path.unlink(missing_ok=True)
                        cleaned_count += 1
                    except OSError as e:
                        logger.debug(f"期限切れキャッシュファイルの削除失敗: {file_path} - {e}")
                self.cache_index.pop(key, None)

        if cleaned_count > 0:
            self._persist_cache_index()
            logger.info(f"期限切れキャッシュを{cleaned_count}件クリーンアップしました")

        return cleaned_count
