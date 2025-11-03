"""
Download Manager

画像ダウンロード、URL検証、並列処理を担当するモジュール
"""

import logging
import time
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from urllib.parse import urlparse
import requests
from PIL import Image
import io

logger = logging.getLogger(__name__)

class DownloadManager:
    """
    画像ダウンロードを管理するクラス
    """

    def __init__(self, config: Dict[str, Any], session: requests.Session):
        self.config = config
        self.session = session

        # ダウンロード設定
        network_cfg = config.get('network', {})
        self.download_timeout = network_cfg.get('download_timeout_seconds', 30)
        self.max_retries = network_cfg.get('max_retries', 3)
        self.backoff_factor = network_cfg.get('backoff_factor', 0.5)
        self.max_backoff_seconds = network_cfg.get('max_backoff_seconds', 10)

        # 並列処理設定
        self.max_workers = config.get('max_download_workers', 4)

        # 統計情報
        self.stats = {
            'total_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_bytes': 0
        }

    def download_image(self, image_url: str, output_path: Path,
                      progress_callback: Optional[Callable] = None) -> bool:
        """
        単一画像をダウンロード

        Args:
            image_url: 画像URL
            output_path: 保存先パス
            progress_callback: 進捗コールバック関数

        Returns:
            ダウンロード成功かどうか
        """
        self.stats['total_downloads'] += 1

        try:
            # URL検証
            if not self._validate_image_url(image_url):
                logger.warning(f"Invalid image URL: {image_url}")
                self.stats['failed_downloads'] += 1
                return False

            # ダウンロード実行
            response = self._download_with_retry(image_url, progress_callback)
            if not response:
                self.stats['failed_downloads'] += 1
                return False

            # 画像データの検証と保存
            if self._save_image_data(response.content, output_path):
                self.stats['successful_downloads'] += 1
                self.stats['total_bytes'] += len(response.content)
                return True
            else:
                self.stats['failed_downloads'] += 1
                return False

        except Exception as e:
            logger.error(f"Error downloading image {image_url}: {e}")
            self.stats['failed_downloads'] += 1
            return False

    def download_images_batch(self, image_urls: List[str], output_dir: Path,
                             progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        画像をバッチダウンロード

        Args:
            image_urls: 画像URLのリスト
            output_dir: 出力ディレクトリ
            progress_callback: 進捗コールバック関数

        Returns:
            ダウンロード結果の統計
        """
        results = {
            'total': len(image_urls),
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'results': []
        }

        # 出力ディレクトリ作成
        output_dir.mkdir(parents=True, exist_ok=True)

        # 並列ダウンロード
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # ダウンロードタスクを作成
            future_to_url = {}
            for i, url in enumerate(image_urls):
                filename = self._generate_filename(url, i)
                output_path = output_dir / filename
                future = executor.submit(self.download_image, url, output_path, progress_callback)
                future_to_url[future] = (url, output_path)

            # 結果収集
            for future in concurrent.futures.as_completed(future_to_url):
                url, output_path = future_to_url[future]
                try:
                    success = future.result()
                    if success:
                        results['successful'] += 1
                        results['results'].append({'url': url, 'path': str(output_path), 'status': 'success'})
                    else:
                        results['failed'] += 1
                        results['results'].append({'url': url, 'status': 'failed'})
                except Exception as e:
                    logger.error(f"Download task failed for {url}: {e}")
                    results['failed'] += 1
                    results['results'].append({'url': url, 'status': 'error', 'error': str(e)})

        logger.info(f"Batch download completed: {results['successful']}/{results['total']} successful")
        return results

    def _download_with_retry(self, url: str, progress_callback: Optional[Callable] = None) -> Optional[requests.Response]:
        """リトライ付きダウンロード"""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    url,
                    timeout=self.download_timeout,
                    stream=True
                )
                response.raise_for_status()

                # コンテンツタイプチェック
                content_type = response.headers.get('content-type', '').lower()
                if not content_type.startswith('image/'):
                    logger.warning(f"Non-image content type: {content_type} for {url}")
                    return None

                # コンテンツ長チェック（DoS対策）
                content_length = response.headers.get('content-length')
                if content_length:
                    content_length = int(content_length)
                    max_size = 10 * 1024 * 1024  # 10MB
                    if content_length > max_size:
                        logger.warning(f"Content too large: {content_length} bytes for {url}")
                        return None

                return response

            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    # 指数バックオフ
                    delay = min(self.backoff_factor * (2 ** attempt), self.max_backoff_seconds)
                    logger.debug(f"Download attempt {attempt + 1} failed for {url}, retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"Download failed after {self.max_retries} attempts for {url}: {e}")

        return None

    def _validate_image_url(self, url: str) -> bool:
        """画像URLの検証"""
        try:
            parsed = urlparse(url)

            # スキームチェック
            if parsed.scheme not in ['http', 'https']:
                return False

            # ホストチェック
            if not parsed.netloc:
                return False

            # Tumblrドメイン以外は拒否（セキュリティ）
            if 'tumblr.com' not in parsed.netloc:
                logger.debug(f"Non-Tumblr URL rejected: {url}")
                return False

            # URL長チェック（DoS対策）
            if len(url) > 2048:
                return False

            return True

        except Exception as e:
            logger.error(f"URL validation error for {url}: {e}")
            return False

    def _save_image_data(self, data: bytes, output_path: Path) -> bool:
        """画像データを検証して保存"""
        try:
            # PILで画像として読み込み検証
            image = Image.open(io.BytesIO(data))

            # 画像形式の検証
            if image.format not in ['JPEG', 'PNG', 'GIF', 'WEBP']:
                logger.warning(f"Unsupported image format: {image.format}")
                return False

            # 画像サイズチェック
            width, height = image.size
            if width < 100 or height < 100:
                logger.warning(f"Image too small: {width}x{height}")
                return False

            # 出力ディレクトリ作成
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存
            # 元の形式を維持しつつ、必要に応じて最適化
            if image.format == 'JPEG':
                image.save(output_path, 'JPEG', quality=85, optimize=True)
            else:
                image.save(output_path, image.format)

            logger.debug(f"Image saved: {output_path} ({width}x{height})")
            return True

        except Exception as e:
            logger.error(f"Error saving image data: {e}")
            return False

    def _generate_filename(self, url: str, index: int) -> str:
        """ダウンロードファイル名の生成"""
        try:
            # URLから元のファイル名を抽出
            url_path = urlparse(url).path
            if url_path and '.' in url_path:
                original_name = Path(url_path).name
                # 拡張子が画像関連なら使用
                ext = Path(original_name).suffix.lower()
                if ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                    return f"{index:04d}_{original_name}"

            # URLからファイル名を生成できない場合はインデックスベース
            return f"image_{index:04d}.jpg"

        except Exception as e:
            logger.error(f"Error generating filename for {url}: {e}")
            return f"image_{index:04d}.jpg"

    def get_download_stats(self) -> Dict[str, Any]:
        """ダウンロード統計を取得"""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """統計をリセット"""
        self.stats = {
            'total_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_bytes': 0
        }
