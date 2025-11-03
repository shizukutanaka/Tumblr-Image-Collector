"""
Tumblr Image Collector - Core Module

このモジュールは、Tumblr Image Collectorのコア機能を分割したものです。
巨大な単一ファイルだった実装を、責任分離に基づいて複数のモジュールに分割しています。

主なコンポーネント:
- collector: メインのコレクタークラスと基本機能
- api_manager: Tumblr APIとの通信管理
- download_manager: 画像ダウンロード処理
- processor: 画像処理と分類
- cache_manager: キャッシュ管理
"""

from .collector import TumblrImageCollector
from .api_manager import ApiManager
from .download_manager import DownloadManager
from .processor import ImageProcessor
from .cache_manager import CacheManager

__all__ = [
    'TumblrImageCollector',
    'ApiManager',
    'DownloadManager',
    'ImageProcessor',
    'CacheManager'
]
