#!/usr/bin/env python3
"""
Tumblr Image Collector - エンタープライズグレード版
高度なアーキテクチャとデザインパターンを適用した実装
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from urllib.parse import urlparse
import hashlib
import os

logger = logging.getLogger(__name__)

# ドメインモデル
@dataclass
class ImageMetadata:
    """画像メタデータ"""
    url: str
    filename: str
    size: int
    width: int
    height: int
    format: str
    hash_value: str
    tags: List[str] = field(default_factory=list)
    blog_name: str = ""
    downloaded_at: float = field(default_factory=time.time)

@dataclass
class DownloadResult:
    """ダウンロード結果"""
    success: bool
    metadata: Optional[ImageMetadata] = None
    error_message: str = ""
    retry_count: int = 0

@dataclass
class CollectionStats:
    """収集統計"""
    total_downloads: int = 0
    successful_downloads: int = 0
    failed_downloads: int = 0
    duplicates_skipped: int = 0
    bytes_downloaded: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0

# インターフェース定義
class ImageValidator(ABC):
    """画像検証インターフェース"""

    @abstractmethod
    def validate_image(self, metadata: ImageMetadata) -> bool:
        """画像を検証する"""
        pass

    @abstractmethod
    def validate_url(self, url: str) -> bool:
        """URLを検証する"""
        pass

class ImageProcessor(ABC):
    """画像処理インターフェース"""

    @abstractmethod
    def process_image(self, image_path: str) -> Optional[ImageMetadata]:
        """画像を処理する"""
        pass

    @abstractmethod
    def calculate_hash(self, image_path: str) -> str:
        """画像のハッシュを計算する"""
        pass

class ImageDownloader(ABC):
    """画像ダウンロードインターフェース"""

    @abstractmethod
    async def download_image(self, url: str) -> DownloadResult:
        """画像をダウンロードする"""
        pass

class MetadataExtractor(ABC):
    """メタデータ抽出インターフェース"""

    @abstractmethod
    def extract_metadata(self, image_path: str, url: str) -> ImageMetadata:
        """画像からメタデータを抽出する"""
        pass

# 実装クラス
class EnterpriseImageValidator(ImageValidator):
    """エンタープライズグレード画像検証器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_file_size = config.get('max_image_size', 10 * 1024 * 1024)
        self.allowed_formats = config.get('allowed_formats', ['jpg', 'jpeg', 'png', 'gif', 'webp'])
        self.min_resolution = config.get('min_resolution', (300, 300))

    def validate_image(self, metadata: ImageMetadata) -> bool:
        """画像を検証する"""
        # ファイルサイズチェック
        if metadata.size > self.max_file_size:
            logger.warning(f"Image too large: {metadata.size} > {self.max_file_size}")
            return False

        # 解像度チェック
        if metadata.width < self.min_resolution[0] or metadata.height < self.min_resolution[1]:
            logger.warning(f"Image resolution too low: {metadata.width}x{metadata.height}")
            return False

        # フォーマットチェック
        if metadata.format.lower() not in self.allowed_formats:
            logger.warning(f"Unsupported format: {metadata.format}")
            return False

        return True

    def validate_url(self, url: str) -> bool:
        """URLを検証する"""
        try:
            parsed = urlparse(url)
            return parsed.scheme in ['http', 'https'] and parsed.netloc
        except Exception as e:
            logger.error(f"URL validation error: {e}")
            return False

class AdvancedImageProcessor(ImageProcessor):
    """高度な画像処理器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hash_algorithm = config.get('hash_algorithm', 'sha256')

    def process_image(self, image_path: str) -> Optional[ImageMetadata]:
        """画像を処理する"""
        try:
            # 画像の基本情報を取得
            from PIL import Image
            with Image.open(image_path) as img:
                metadata = ImageMetadata(
                    url="",  # これは別途設定される
                    filename=Path(image_path).name,
                    size=Path(image_path).stat().st_size,
                    width=img.size[0],
                    height=img.size[1],
                    format=img.format or "UNKNOWN",
                    hash_value=self.calculate_hash(image_path)
                )
                return metadata
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return None

    def calculate_hash(self, image_path: str) -> str:
        """画像のハッシュを計算する"""
        try:
            # 複数のハッシュアルゴリズムをサポート
            if self.hash_algorithm == 'perceptual':
                # 知覚的ハッシュ（類似画像検出用）
                from imagehash import average_hash
                from PIL import Image
                with Image.open(image_path) as img:
                    hash_obj = average_hash(img)
                    return str(hash_obj)
            else:
                # 標準ハッシュ
                hash_obj = hashlib.new(self.hash_algorithm)
                with open(image_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hash_obj.update(chunk)
                return hash_obj.hexdigest()
        except Exception as e:
            logger.error(f"Hash calculation error: {e}")
            return ""

class SecureImageDownloader(ImageDownloader):
    """セキュア画像ダウンローダー"""

    def __init__(self, config: Dict[str, Any], validator: ImageValidator):
        self.config = config
        self.validator = validator
        self.session = None
        self._init_session()

    def _init_session(self):
        """セキュアなセッションを初期化"""
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        self.session = requests.Session()

        # リトライ戦略
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # セキュリティヘッダー
        self.session.headers.update({
            'User-Agent': 'TumblrImageCollector/Enterprise-1.0',
            'Accept': 'image/*',
        })

    async def download_image(self, url: str) -> DownloadResult:
        """画像をダウンロードする"""
        # URL検証
        if not self.validator.validate_url(url):
            return DownloadResult(
                success=False,
                error_message="Invalid URL"
            )

        try:
            # 非同期ダウンロード（requestsは同期だが、ThreadPoolExecutorで並列化）
            import requests

            response = self.session.get(url, timeout=30, stream=True)
            response.raise_for_status()

            # レスポンス検証
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                return DownloadResult(
                    success=False,
                    error_message=f"Invalid content type: {content_type}"
                )

            # 一時ファイルに保存
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                temp_path = tmp_file.name

            # 画像処理
            processor = AdvancedImageProcessor(self.config)
            metadata = processor.process_image(temp_path)

            if metadata:
                metadata.url = url
                # ファイルサイズ更新
                metadata.size = Path(temp_path).stat().st_size

                # 最終検証
                if self.validator.validate_image(metadata):
                    # 正式な場所に移動
                    final_path = self._get_final_path(metadata)
                    Path(temp_path).rename(final_path)

                    return DownloadResult(
                        success=True,
                        metadata=metadata
                    )
                else:
                    Path(temp_path).unlink()  # 削除
                    return DownloadResult(
                        success=False,
                        error_message="Image validation failed"
                    )
            else:
                Path(temp_path).unlink()
                return DownloadResult(
                    success=False,
                    error_message="Image processing failed"
                )

        except Exception as e:
            logger.error(f"Download error: {e}")
            return DownloadResult(
                success=False,
                error_message=str(e)
            )

    def _get_final_path(self, metadata: ImageMetadata) -> Path:
        """最終的な保存パスを取得"""
        output_dir = Path(self.config.get('output_dir', 'images'))
        output_dir.mkdir(exist_ok=True)

        # ハッシュベースのファイル名
        hash_suffix = metadata.hash_value[:16] if metadata.hash_value else "unknown"
        extension = metadata.format.lower() if metadata.format else 'jpg'

        filename = f"{hash_suffix}.{extension}"
        return output_dir / filename

class MetadataManager:
    """メタデータ管理"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metadata_file = Path(config.get('metadata_file', 'metadata.json'))
        self._metadata_cache: Dict[str, ImageMetadata] = {}
        self._load_metadata()

    def _load_metadata(self):
        """メタデータを読み込む"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        metadata = ImageMetadata(**item)
                        self._metadata_cache[metadata.hash_value] = metadata
            except Exception as e:
                logger.error(f"Metadata loading error: {e}")

    def add_metadata(self, metadata: ImageMetadata):
        """メタデータを追加"""
        self._metadata_cache[metadata.hash_value] = metadata
        self._save_metadata()

    def get_metadata(self, hash_value: str) -> Optional[ImageMetadata]:
        """メタデータを取得"""
        return self._metadata_cache.get(hash_value)

    def is_duplicate(self, hash_value: str) -> bool:
        """重複かどうかをチェック"""
        return hash_value in self._metadata_cache

    def _save_metadata(self):
        """メタデータを保存"""
        try:
            data = [vars(meta) for meta in self._metadata_cache.values()]
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Metadata saving error: {e}")

# メインコレクター（ファサードパターン）
class EnterpriseTumblrCollector:
    """エンタープライズグレードTumblrコレクター"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stats = CollectionStats()

        # コンポーネントの初期化
        self.validator = EnterpriseImageValidator(config)
        self.processor = AdvancedImageProcessor(config)
        self.downloader = SecureImageDownloader(config, self.validator)
        self.metadata_manager = MetadataManager(config)

        # 並列処理用のスレッドプール
        self.executor = ThreadPoolExecutor(
            max_workers=config.get('max_workers', 10),
            thread_name_prefix="TumblrCollector"
        )

    def collect_images(self, blog_names: List[str], limit_per_blog: int = 100) -> Dict[str, Any]:
        """画像を収集する"""
        logger.info(f"Starting collection from blogs: {blog_names}")

        futures = []
        for blog_name in blog_names:
            future = self.executor.submit(self._collect_from_blog, blog_name, limit_per_blog)
            futures.append((blog_name, future))

        results = {}
        for blog_name, future in futures:
            try:
                results[blog_name] = future.result()
            except Exception as e:
                logger.error(f"Error collecting from {blog_name}: {e}")
                results[blog_name] = {'error': str(e)}

        self.stats.end_time = time.time()
        return {
            'results': results,
            'stats': self._get_stats_dict(),
            'duration': self.stats.end_time - self.stats.start_time
        }

    def _collect_from_blog(self, blog_name: str, limit: int) -> Dict[str, Any]:
        """1つのブログから画像を収集"""
        blog_stats = CollectionStats()

        try:
            # Tumblr APIから投稿を取得（実際の実装ではpytumblrを使用）
            posts = self._get_blog_posts(blog_name, limit)

            for post in posts:
                if blog_stats.successful_downloads >= limit:
                    break

                result = self._process_post(post)
                if result.success:
                    blog_stats.successful_downloads += 1
                    blog_stats.bytes_downloaded += result.metadata.size if result.metadata else 0
                else:
                    blog_stats.failed_downloads += 1

        except Exception as e:
            logger.error(f"Blog collection error: {e}")
            return {'error': str(e)}

        return {
            'successful_downloads': blog_stats.successful_downloads,
            'failed_downloads': blog_stats.failed_downloads,
            'bytes_downloaded': blog_stats.bytes_downloaded
        }

    def _process_post(self, post: Dict[str, Any]) -> DownloadResult:
        """投稿を処理する"""
        # 実際の実装では投稿から画像URLを抽出
        # ここではサンプルとして実装
        image_urls = self._extract_image_urls(post)

        for url in image_urls:
            result = self._download_single_image(url)
            if result.success:
                return result

        return DownloadResult(success=False, error_message="No valid images found")

    def _download_single_image(self, url: str) -> DownloadResult:
        """単一画像をダウンロード"""
        try:
            # 重複チェック
            temp_processor = AdvancedImageProcessor(self.config)
            # URLから仮のハッシュを生成（実際にはHEADリクエストでサイズを取得）
            url_hash = hashlib.md5(url.encode()).hexdigest()

            if self.metadata_manager.is_duplicate(url_hash):
                return DownloadResult(
                    success=False,
                    error_message="Duplicate image"
                )

            # ダウンロード実行
            result = self.downloader.download_image(url)

            if result.success and result.metadata:
                # ハッシュを正しく設定
                result.metadata.hash_value = url_hash
                self.metadata_manager.add_metadata(result.metadata)
                self.stats.successful_downloads += 1
            else:
                self.stats.failed_downloads += 1

            return result

        except Exception as e:
            logger.error(f"Single image download error: {e}")
            return DownloadResult(success=False, error_message=str(e))

    def _extract_image_urls(self, post: Dict[str, Any]) -> List[str]:
        """投稿から画像URLを抽出"""
        # 実際の実装では投稿の構造に応じて抽出
        return []  # サンプル実装

    def _get_blog_posts(self, blog_name: str, limit: int) -> List[Dict[str, Any]]:
        """ブログ投稿を取得"""
        # 実際の実装ではpytumblr APIを使用
        return []  # サンプル実装

    def _get_stats_dict(self) -> Dict[str, Any]:
        """統計情報を辞書形式で取得"""
        return {
            'total_downloads': self.stats.successful_downloads,
            'failed_downloads': self.stats.failed_downloads,
            'bytes_downloaded': self.stats.bytes_downloaded,
            'duration': self.stats.end_time - self.stats.start_time,
            'images_per_second': self.stats.successful_downloads / (self.stats.end_time - self.stats.start_time) if self.stats.end_time > self.stats.start_time else 0
        }

    def cleanup(self):
        """クリーンアップ"""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("Cleanup completed")

# ファクトリーパターンによるコレクター作成
class TumblrCollectorFactory:
    """Tumblrコレクターファクトリ"""

    @staticmethod
    def create_collector(config: Dict[str, Any]) -> EnterpriseTumblrCollector:
        """適切なコレクターを作成"""
        deployment_mode = config.get('deployment_mode', 'standard')

        if deployment_mode == 'enterprise':
            return EnterpriseTumblrCollector(config)
        else:
            # 簡易版コレクター
            return SimpleTumblrCollector(config)

class SimpleTumblrCollector:
    """簡易版コレクター（後方互換性用）"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("Simple collector initialized")

    def collect_images(self, blog_names: List[str], limit_per_blog: int = 100) -> Dict[str, Any]:
        """簡易収集"""
        return {'message': 'Simple collector - feature not implemented'}

# 使用例
def main():
    """メイン関数"""
    config = {
        'max_workers': 10,
        'max_image_size': 10 * 1024 * 1024,
        'allowed_formats': ['jpg', 'jpeg', 'png', 'gif', 'webp'],
        'min_resolution': (300, 300),
        'output_dir': 'images',
        'deployment_mode': 'enterprise',
        'hash_algorithm': 'sha256'
    }

    collector = TumblrCollectorFactory.create_collector(config)

    try:
        results = collector.collect_images(['blog1', 'blog2'], limit_per_blog=50)
        print(f"Collection completed: {results}")
    finally:
        collector.cleanup()

if __name__ == "__main__":
    main()
