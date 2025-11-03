import pytumblr
import requests
import os
import time
import random
from PIL import Image
from io import BytesIO
import datetime
import webbrowser
import json
import logging
import csv
from logging.handlers import RotatingFileHandler
import concurrent.futures
import shutil
from pathlib import Path
import argparse
from functools import partial
import socket
import threading
import queue
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import imagehash
import urllib3
import socks
from urllib.parse import urlparse
import traceback
import platform
import uuid
import sys
import tempfile
import atexit
import signal
from typing import Optional, Dict, Any, Iterable

from billing import LicenseManager, LicenseStatus, LicenseInfo, StripeBillingManager

try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None
    _DOTENV_AVAILABLE = False

# Import new utility modules
from ui import InteractiveCLI, ProgressDisplay
from config import ConfigWizard
from exceptions import (
    TumblrCollectorError, NetworkError, ValidationError, ConfigurationError,
    DownloadError, URLValidationError, DiskSpaceError,
    MemoryError as CustomMemoryError, ResourceError
)
from logging_utils import LoggingConfig, SensitiveDataFilter, create_context_logger
from resource_manager import (
    ResourceMonitor, ResourceLimits, ResourceGuard,
    get_file_handle_manager, get_cleanup_manager,
    resource_guarded_operation
)

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None
    _NUMPY_AVAILABLE = False

try:
    import skimage.feature
    _SKIMAGE_AVAILABLE = True
except ImportError:
    skimage = None
    _SKIMAGE_AVAILABLE = False

try:
    import browser_cookie3
    _BROWSER_COOKIE_AVAILABLE = True
except ImportError:
    browser_cookie3 = None
    _BROWSER_COOKIE_AVAILABLE = False

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from core import constants

# Import new utility modules
from ui import InteractiveCLI, ProgressDisplay
from config import ConfigWizard
from exceptions import (
    TumblrCollectorError, NetworkError, ValidationError, ConfigurationError,
    DownloadError, URLValidationError, DiskSpaceError,
    MemoryError as CustomMemoryError, ResourceError
)
from logging_utils import LoggingConfig, SensitiveDataFilter, create_context_logger
from resource_manager import (
    ResourceMonitor, ResourceLimits, ResourceGuard,
    get_file_handle_manager, get_cleanup_manager,
    resource_guarded_operation
)

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None
    _NUMPY_AVAILABLE = False

try:
    import skimage.feature
    _SKIMAGE_AVAILABLE = True
except ImportError:
    skimage = None
    _SKIMAGE_AVAILABLE = False

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from core import constants

# --- Global Logger ---
# Logger will be configured in main() based on args
logger = logging.getLogger(__name__)

class TumblrImageCollector:
    """
    Tumblrブログから画像を収集し、関連タグの画像もダウンロードするクラス。
    設定管理、API認証、並列ダウンロード、重複防止、レート制限対応を行う。
    高度な画像フィルタリングと重複排除を実装。
    パフォーマンスと安定性を最適化。
    """

    # ダウンロード統計
    _download_stats = {
        'total_attempts': 0,
        'successful_downloads': 0,
        'failed_downloads': 0,
        'skipped_duplicates': 0,

        # 基本ダウンロード統計
        'total_images_processed': 0,
        'total_images_downloaded': 0,
        'total_images_skipped': 0,

        # AI画像分類統計
        'ai_classification_stats': {
            'valid_images': 0,
            'invalid_images': 0,
            'high_resolution_images': 0,
            'low_resolution_images': 0,
            'potentially_nsfw_images': 0,
            'image_type_distribution': {},
            'metrics_summary': {}
        }
    }

    # Note: Using centralized exception hierarchy from exceptions.py
    # Keeping this for backward compatibility
    DownloadError = DownloadError

    # 画像ハッシュの類似度閾値
    IMAGE_HASH_THRESHOLD = 5

    # 画像フィルタリングオプション
    IMAGE_FILTERS = {
        'min_width': constants.MIN_IMAGE_DIMENSION,
        'min_height': constants.MIN_IMAGE_DIMENSION,
        'allowed_formats': ['jpg', 'jpeg', 'png', 'gif', 'webp'],
        'max_file_size_mb': 10,
        'aspect_ratio_range': (0.5, 2.0),  # 縦横比の制限
        'color_threshold': 0.1,  # カラー画像の判定閾値
        'blur_threshold': constants.BLUR_THRESHOLD,  # ぼかし度の閾値
        'nsfw_detection': True  # NSFWコンテンツの検出
    }

    def __init__(self, config_file=constants.DEFAULT_CONFIG_FILE, output_dir_override=None, workers_override=None, proxy_config=None):
        if _DOTENV_AVAILABLE:
            load_dotenv()

        self.config_file = Path(config_file).resolve()
        self.config = self._load_config()
        self.script_dir = Path(__file__).parent.resolve()
        self.billing_manager: Optional[StripeBillingManager] = None

        # ロギングシステムを初期化
        self._setup_logging()

        # Initialize cleanup manager for proper resource cleanup
        self.cleanup_manager = get_cleanup_manager()

        # プロキシ設定の初期化
        self.proxy_config = proxy_config or self.config.get('proxy', constants.DEFAULT_PROXY_CONFIG)
        self._setup_proxy()

        # Determine output folder: CLI override > config > default
        default_values = ConfigWizard.default_config_values()
        output_folder_name = output_dir_override or self.config.get("output_folder_name", default_values["output_folder_name"])
        # Ensure output_folder is absolute path
        if Path(output_folder_name).is_absolute():
             self.output_folder = Path(output_folder_name)
        else:
             self.output_folder = self.script_dir / output_folder_name

        # Determine max workers: Use dynamic resource calculation if not overridden
        if workers_override:
            self.max_workers = workers_override
        elif "max_download_workers" in self.config:
            self.max_workers = self.config["max_download_workers"]
        else:
            # Calculate optimal worker count based on system resources
            self.max_workers = ResourceMonitor.get_optimal_worker_count(max_workers=15)
            logger.info(f"Dynamically calculated worker count: {self.max_workers}")

        self.api_batch_sleep = self.config.get("api_batch_sleep_seconds", 2)
        self.api_wait_hours = self.config.get("api_wait_hours", 1)

        cache_cfg = self.config.get('cache', {})
        self.cache_enabled = bool(cache_cfg.get('enabled', True))
        default_cache_cfg = ConfigWizard.default_config_values().get('cache', {})
        self.cache_ttl_seconds = int(cache_cfg.get('ttl_seconds', default_cache_cfg.get('ttl_seconds', 24 * 60 * 60)))
        self.cache_max_entries = int(cache_cfg.get('max_entries', default_cache_cfg.get('max_entries', 2048)))

        self.downloaded_files = set()
        self.downloaded_hashes = set()
        self._setup_output_directory()
        self._cache_dir = self.output_folder / "cache"
        self._cache_index = {}
        if self.cache_enabled:
            try:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                self._cache_index = self._load_cache_index()
            except (OSError, IOError) as cache_err:
                logger.warning(f"キャッシュディレクトリの作成に失敗しました: {cache_err}")
                self.cache_enabled = False
            except Exception as cache_err:
                logger.error(f"予期しないキャッシュエラー: {cache_err}", exc_info=True)
                self.cache_enabled = False
        self._load_downloaded_files()

        self.consumer_key = None
        self.consumer_secret = None
        self.token = None
        self.token_secret = None
        self._setup_credentials() # Reads from self.config

        self.image_classifier = ImageClassifier(
            enable_deep_model=self.config.get('enable_deep_model', default_values['enable_deep_model'])
        )

        filters_cfg = self.config.get('filters', {})
        default_filters = default_values.get('filters', {})
        self.nsfw_threshold = float(filters_cfg.get('nsfw_threshold', default_filters.get('nsfw_threshold', 0.35)))

        security_cfg = self.config.get('security', {})
        self.allowed_domains = self._load_allowed_domains(security_cfg.get('allowed_domains'))
        self.max_download_limit = int(self.config.get('max_download_limit', 1000))
        self.slow_response_threshold = int(self.config.get('slow_response_threshold', 45))

        network_cfg = self.config.get('network', {})
        default_network = default_values['network']
        self.download_timeout = int(network_cfg.get('download_timeout_seconds', default_network['download_timeout_seconds']))
        self.max_retries = int(network_cfg.get('max_retries', default_network['max_retries']))
        self.backoff_factor = float(network_cfg.get('backoff_factor', default_network['backoff_factor']))
        self.max_backoff_seconds = int(network_cfg.get('max_backoff_seconds', default_network['max_backoff_seconds']))

        from download_manager import DownloadManager
        self.download_manager = DownloadManager(self.output_folder, self.config)

        self.client = self._initialize_client()
        self.executor = None # Will be managed by 'with' statement in run()

        # The session is now managed by DownloadManager
        self.session = self.download_manager.session
        self.proxies = self._build_requests_proxies(self.proxy_config)

        # CLIフィルタの初期値
        self._cli_tags = []
        self._cli_start_date = None
        self._cli_end_date = None
        self._include_likes = False

        # レート制限管理
        self._rate_limiter = self._create_rate_limiter()
        self._request_timestamps = []
        self._download_stats.setdefault('cache_hits', 0)
        self._download_stats.setdefault('cache_misses', 0)

        # シャットダウンハンドラーを登録
        import atexit
        atexit.register(self._cleanup_resources)

        # シグナルハンドラーを登録（graceful shutdown）
        self._setup_signal_handlers()

        # 起動時の設定検証
        self._validate_configuration()

        # Stripe課金連携の初期化
        self.billing_manager = self._initialize_billing_manager()

        # ライセンス情報の読み込み
        license_storage = self.script_dir / "licenses" / "license.json"
        self.license_manager = LicenseManager(license_storage)
        self._load_license_from_config()

        # プライベートブログ対応のためのクッキー管理
        self._browser_cookies = {}
        self._load_browser_cookies()

    def _load_browser_cookies(self):
        """ブラウザからTumblrのクッキーを読み込む"""
        if not _BROWSER_COOKIE_AVAILABLE:
            logger.warning("browser_cookie3が利用できないためクッキーインポートは無効です")
            return

        try:
            # Chromeのクッキーを読み込み
            cookies = browser_cookie3.chrome(domain_name='tumblr.com')
            for cookie in cookies:
                self._browser_cookies[cookie.name] = cookie.value
            logger.info(f"{len(self._browser_cookies)}件のTumblrクッキーを読み込みました")
        except Exception as e:
            logger.error(f"ブラウザクッキーの読み込みに失敗しました: {e}")

    def _initialize_client(self):
        """プライベートブログ対応のクライアントを初期化"""
        if self.token and self.token_secret:
            client = pytumblr.TumblrRestClient(
                self.consumer_key,
                self.consumer_secret,
                self.token,
                self.token_secret
            )
        else:
            client = pytumblr.TumblrRestClient(
                self.consumer_key,
                self.consumer_secret
            )

        # プライベートブログ対応のためのセッション設定
        if self._browser_cookies:
            # クッキーを使用してプライベートブログにアクセス
            client.session = requests.Session()
            for name, value in self._browser_cookies.items():
                client.session.cookies.set(name, value, domain='tumblr.com')

        return client

    def _simulate_private_api(self, blog_name, endpoint, params=None):
        """プライベートAPIをシミュレートしてアクセス"""
        url = f"https://api.tumblr.com/v2/blog/{blog_name}/{endpoint}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        if self._browser_cookies:
            session = requests.Session()
            for name, value in self._browser_cookies.items():
                session.cookies.set(name, value, domain='tumblr.com')
        else:
            session = requests.Session()

        try:
            response = session.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"プライベートAPIシミュレーションエラー: {e}")
            return None

    def _cleanup_resources(self):
        """リソースのクリーンアップとメモリリーク防止"""
        try:
            # スレッドプールのクリーンアップ
            if self.executor:
                self.executor.shutdown(wait=False)
                self.executor = None

            # HTTPセッションのクリーンアップ
            if self.session:
                self.session.close()
                self.session = None

            # キャッシュのクリーンアップ
            if hasattr(self, '_image_cache'):
                self._image_cache.clear()
                delattr(self, '_image_cache')

            # ダウンロード済みファイルのセットをクリア
            if hasattr(self, 'downloaded_files'):
                self.downloaded_files.clear()

            # 統計情報を保存
            self._save_statistics()
            logger.info("リソースをクリーンアップしました")

            # ガベージコレクションの実行
            import gc
            gc.collect()

        except Exception as e:
            logger.error(f"リソースクリーンアップ中にエラー: {e}")

    def _initialize_billing_manager(self) -> Optional[StripeBillingManager]:
        """Stripe課金マネージャーを初期化"""

        stripe_cfg = self.config.get('stripe') or {}
        if not stripe_cfg or not stripe_cfg.get('secret_key'):
            logger.debug("Stripe設定が見つからないため課金機能は無効化されています")
            return None

        try:
            manager = StripeBillingManager.from_config(self.config)
            logger.info("Stripe課金管理を初期化しました")
            return manager
        except Exception as exc:
            logger.error(f"Stripe課金管理の初期化に失敗しました: {exc}")
            return None

    def _load_license_from_config(self) -> None:
        """設定ファイルからライセンス情報を読み込む"""

        stripe_cfg = self.config.get('stripe') or {}
        license_cfg = stripe_cfg.get('license')

        payload: Optional[Dict[str, Any]] = None

        if isinstance(license_cfg, dict):
            payload = license_cfg
        elif isinstance(license_cfg, str) and license_cfg:
            license_path = Path(license_cfg)
            if not license_path.is_absolute():
                license_path = (self.script_dir / license_path).resolve()

            if license_path.exists():
                try:
                    payload = json.loads(license_path.read_text(encoding='utf-8'))
                except Exception as exc:
                    logger.error(f"ライセンスファイルの読み込みに失敗しました: {exc}")
            else:
                logger.warning(f"指定されたライセンスファイルが見つかりません: {license_path}")

        if not payload:
            logger.debug("適用可能なライセンス情報が設定から見つかりませんでした")
            return

        self._apply_license_payload(payload)

    def _apply_license_payload(self, payload: Dict[str, Any]) -> None:
        """ライセンス情報を保存"""

        try:
            status = LicenseStatus(payload.get('status', LicenseStatus.NONE))
        except ValueError:
            logger.warning("ライセンス状態の値が不正のため 'none' として扱います")
            status = LicenseStatus.NONE

        license_info = LicenseInfo(
            status=status,
            plan_key=payload.get('plan_key'),
            current_period_end=payload.get('current_period_end'),
            customer_email=payload.get('customer_email'),
            stripe_subscription_id=payload.get('stripe_subscription_id'),
            metadata=payload.get('metadata')
        )

        self.license_manager.set_license(license_info)
        logger.info("ライセンス情報を適用しました: %s", license_info.status.value)

    def _ensure_license_for_feature(self, feature_name: str) -> None:
        """指定機能にライセンスが必要か確認"""

        if not self.license_manager.is_active():
            raise ConfigurationError(
                f"機能 '{feature_name}' の利用には有効なライセンスが必要です。Stripe Checkoutでプランを購入してライセンスを適用してください。"
            )

    def list_billing_plans(self) -> Dict[str, Dict[str, Any]]:
        """利用可能な課金プランを一覧取得"""

        if not self.billing_manager:
            raise ConfigurationError("Stripe課金が設定されていません。configのstripeセクションを確認してください。")

        plans = self.billing_manager.list_products()
        return {plan['key']: plan for plan in plans}

    def create_checkout_session(self, plan_key: str, customer_email: Optional[str] = None) -> Dict[str, Any]:
        """Stripe Checkoutセッションを作成してURLを返す"""

        if not self.billing_manager:
            raise ConfigurationError("Stripe課金が初期化されていないためCheckoutを作成できません。")

        metadata = {
            'application': 'tumblr-image-collector',
            'plan_key': plan_key,
        }

        session = self.billing_manager.create_checkout_session(
            plan_key=plan_key,
            customer_email=customer_email,
            metadata=metadata
        )

        return {
            'id': session.id,
            'url': getattr(session, 'url', ''),
            'plan_key': plan_key
        }

    def apply_license_file(self, license_file: str) -> None:
        """ライセンスファイルを読み込んで適用する"""

        license_path = Path(license_file)
        if not license_path.is_absolute():
            license_path = (self.script_dir / license_path).resolve()

        if not license_path.exists():
            raise ConfigurationError(f"ライセンスファイルが存在しません: {license_path}")

        try:
            payload = json.loads(license_path.read_text(encoding='utf-8'))
        except Exception as exc:
            raise ConfigurationError(f"ライセンスファイルの読み込みに失敗しました: {exc}") from exc

        self._apply_license_payload(payload)

    def export_license(self) -> Dict[str, Any]:
        """現在のライセンス情報を辞書で返す"""

        info = self.license_manager.get_license()
        return {
            'status': info.status.value,
            'plan_key': info.plan_key,
            'current_period_end': info.current_period_end,
            'customer_email': info.customer_email,
            'stripe_subscription_id': info.stripe_subscription_id,
            'metadata': info.metadata or {}
        }

    def _create_rate_limiter(self):
        """
        レート制限機能を作成

        Returns:
            dict: レート制限設定
        """
        return {
            'requests_per_minute': self.config.get('rate_limit', {}).get('requests_per_minute', 30),
            'burst_limit': self.config.get('rate_limit', {}).get('burst_limit', 5),
            'window_seconds': 60
        }

    def _load_cache_index(self) -> Dict[str, Dict[str, Any]]:
        """キャッシュインデックスを読み込む"""
        index_path = self._cache_dir / "index.json"
        if not index_path.exists():
            return {}
        try:
            with open(index_path, 'r', encoding='utf-8') as index_file:
                data = json.load(index_file)
            if isinstance(data, dict):
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
        index_path = self._cache_dir / "index.json"
        try:
            with open(index_path, 'w', encoding='utf-8') as index_file:
                json.dump(self._cache_index, index_file, ensure_ascii=False, indent=2)
        except OSError as os_err:
            logger.warning(f"キャッシュインデックスの保存に失敗しました: {os_err}")

    def _cache_key(self, image_url: str) -> str:
        return image_url.strip().lower()

    def _check_cache(self, image_url: str) -> Optional[Path]:
        """キャッシュ済みファイルのパスを返す"""
        if not self.cache_enabled:
            return None
        key = self._cache_key(image_url)
        cached_entry = self._cache_index.get(key)
        if not cached_entry:
            self._download_stats['cache_misses'] += 1
            return None

        cached_path = self._cache_dir / cached_entry.get('filename', '')
        if not cached_path.exists():
            self._download_stats['cache_misses'] += 1
            self._cache_index.pop(key, None)
            return None

        expires_at = cached_entry.get('expires_at')
        if expires_at and time.time() > expires_at:
            logger.debug(f"キャッシュの有効期限切れ: {image_url}")
            try:
                cached_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._cache_index.pop(key, None)
            self._download_stats['cache_misses'] += 1
            return None

        self._download_stats['cache_hits'] += 1
        return cached_path

    def _prune_cache_index(self) -> None:
        """キャッシュの最大件数を超える場合に古いエントリを削除（TTL強制実施）"""
        current_time = time.time()
        expired_keys = []

        # 期限切れエントリを収集
        for key, info in self._cache_index.items():
            expires_at = info.get('expires_at')
            if expires_at and current_time > expires_at:
                expired_keys.append(key)

        # 期限切れエントリを削除
        for key in expired_keys:
            info = self._cache_index.get(key)
            if info:
                filename = info.get('filename')
                if filename:
                    file_path = self._cache_dir / filename
                    try:
                        file_path.unlink(missing_ok=True)
                    except OSError as e:
                        logger.debug(f"期限切れキャッシュファイルの削除失敗: {file_path} - {e}")
                self._cache_index.pop(key, None)

        # 最大件数チェック
        if len(self._cache_index) <= self.cache_max_entries:
            return

        # LRU: 最も古いエントリから削除
        sorted_items = sorted(
            self._cache_index.items(),
            key=lambda item: item[1].get('stored_at', 0)
        )

        removed_count = 0
        target_removal = len(self._cache_index) - self.cache_max_entries

        for key, info in sorted_items:
            if removed_count >= target_removal:
                break

            filename = info.get('filename')
            if filename:
                file_path = self._cache_dir / filename
                try:
                    file_path.unlink(missing_ok=True)
                    removed_count += 1
                except OSError as e:
                    logger.debug(f"キャッシュファイルの削除失敗: {file_path} - {e}")

            self._cache_index.pop(key, None)

        if removed_count > 0:
            logger.info(f"キャッシュを整理: {removed_count}件のエントリを削除")

    def _save_to_cache(self, file_path: Path, image_url: str) -> None:
        """ダウンロードしたファイルをキャッシュに保存"""
        if not self.cache_enabled:
            return
        key = self._cache_key(image_url)
        cached_name = f"{uuid.uuid4().hex}{file_path.suffix.lower()}"
        cached_path = self._cache_dir / cached_name
        try:
            shutil.copy2(file_path, cached_path)
        except OSError as copy_err:
            logger.warning(f"キャッシュへの保存に失敗: {copy_err}")
            return

        self._cache_index[key] = {
            'filename': cached_name,
            'stored_at': time.time(),
            'expires_at': time.time() + self.cache_ttl_seconds if self.cache_ttl_seconds else None
        }
        self._prune_cache_index()
        self._persist_cache_index()

    def _is_metadata_size_safe(self, metadata: Dict[str, Any]) -> bool:
        try:
            serialized = json.dumps(metadata, ensure_ascii=False)
        except (TypeError, OverflowError):
            return False
        return len(serialized.encode('utf-8')) <= MAX_METADATA_SIZE_BYTES

    def _check_rate_limit(self):
        """
        レート制限をチェックし、必要に応じて待機

        Returns:
            bool: リクエスト可能かどうか
        """
        current_time = time.time()
        rate_limiter = self._rate_limiter

        # 古いタイムスタンプを削除
        self._request_timestamps = [
            ts for ts in self._request_timestamps
            if current_time - ts < rate_limiter['window_seconds']
        ]

        # レート制限チェック
        if len(self._request_timestamps) >= rate_limiter['requests_per_minute']:
            # バースト制限チェック
            recent_requests = [ts for ts in self._request_timestamps
                             if current_time - ts < 1]  # 1秒以内のリクエスト

            if len(recent_requests) >= rate_limiter['burst_limit']:
                sleep_time = 1.0
                logger.warning(f"レート制限に達しました。{sleep_time}秒待機します。")
                time.sleep(sleep_time)
                return self._check_rate_limit()

        # リクエストタイムスタンプを記録
        self._request_timestamps.append(current_time)
        return True

    def _handle_rate_limit_with_retry(self, func, *args, max_retries=3, backoff_factor=1.5, **kwargs):
        """
        レート制限を考慮したリトライメカニズム付き関数実行

        Args:
            func: 実行する関数
            *args: 関数の引数
            max_retries: 最大リトライ回数
            backoff_factor: バックオフ係数
            **kwargs: 関数のキーワード引数

        Returns:
            関数の戻り値または例外
        """
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    # バックオフ計算
                    sleep_time = backoff_factor ** (attempt - 1)
                    logger.warning(f"リトライ {attempt}/{max_retries} - {sleep_time:.2f}秒待機")
                    time.sleep(sleep_time)
                return func(*args, **kwargs)
            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    if attempt == max_retries:
                        logger.error(f"レート制限によるリトライ上限に達しました: {e}")
                        raise
                    logger.warning(f"レート制限エラーでリトライします: {e}")
                    continue
                else:
                    # 他のエラーは即座にraise
                    raise
        return None

    def _execute_with_retry(self, func, *args, max_retries=3, backoff_factor=1.5, retryable_errors=None, **kwargs):
        """
        一般的なエラーハンドリングと再試行メカニズム付き関数実行

        Args:
            func: 実行する関数
            *args: 関数の引数
            max_retries: 最大リトライ回数
            backoff_factor: バックオフ係数
            retryable_errors: 再試行可能なエラータイプのリスト
            **kwargs: 関数のキーワード引数

        Returns:
            関数の戻り値または例外
        """
        if retryable_errors is None:
            retryable_errors = (NetworkError, ConnectionError, TimeoutError)

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except retryable_errors as e:
                if attempt == max_retries:
                    logger.error(f"再試行上限に達しました: {e}")
                    raise
                sleep_time = backoff_factor ** attempt
                logger.warning(f"エラーでリトライします (試行 {attempt + 1}/{max_retries + 1}): {e} - {sleep_time:.2f}秒待機")
                time.sleep(sleep_time)
            except Exception as e:
                # 再試行対象外のエラーは即座にraise
                logger.error(f"非再試行エラー: {e}")
                raise
        return None

    def _serialize_cli_filters(self):
        return {
            'tags': list(self._cli_tags) if self._cli_tags else [],
            'start_date': self._cli_start_date.isoformat() if self._cli_start_date else None,
            'end_date': self._cli_end_date.isoformat() if self._cli_end_date else None,
            'include_likes': bool(self._include_likes)
        }

    def _restore_cli_filters(self, cli_filters):
        if not cli_filters:
            return

        tags = cli_filters.get('tags') or []
        self._cli_tags = [str(tag).lower() for tag in tags]

        start_iso = cli_filters.get('start_date')
        end_iso = cli_filters.get('end_date')

        self._cli_start_date = datetime.datetime.fromisoformat(start_iso) if start_iso else None
        self._cli_end_date = datetime.datetime.fromisoformat(end_iso) if end_iso else None

        self._include_likes = bool(cli_filters.get('include_likes', False))

    def _save_statistics(self):
        """統計情報を保存"""
        stats_path_json = self.script_dir / 'download_statistics.json'
        stats_path_csv = self.script_dir / 'download_statistics.csv'

        try:
            # JSON保存
            with open(stats_path_json, 'w', encoding='utf-8') as f:
                json.dump(self._download_stats, f, ensure_ascii=False, indent=2)

            # CSV保存
            with open(stats_path_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                writer.writerow(['metric', 'value'])
                writer.writerow(['total_attempts', self._download_stats['total_attempts']])
                writer.writerow(['successful_downloads', self._download_stats['successful_downloads']])
                writer.writerow(['failed_downloads', self._download_stats['failed_downloads']])
                writer.writerow(['skipped_duplicates', self._download_stats['skipped_duplicates']])

                ai_stats = self._download_stats.get('ai_classification_stats', {})
                writer.writerow([])
                writer.writerow(['AI Classification Stats'])
                writer.writerow(['valid_images', ai_stats.get('valid_images', 0)])
                writer.writerow(['invalid_images', ai_stats.get('invalid_images', 0)])
                writer.writerow(['potentially_nsfw_images', ai_stats.get('potentially_nsfw_images', 0)])

                metrics_summary = ai_stats.get('metrics_summary', {})
                if metrics_summary:
                    writer.writerow([])
                    writer.writerow(['Metric', 'count', 'mean', 'min', 'max'])
                    for metric_name, metric_data in metrics_summary.items():
                        count = metric_data.get('count', 0)
                        mean_value = (metric_data['sum'] / count) if count else 0.0
                        writer.writerow([
                            metric_name,
                            count,
                            f"{mean_value:.4f}",
                            f"{metric_data['min']:.4f}",
                            f"{metric_data['max']:.4f}"
                        ])

            logger.info(f"統計情報を保存しました: {stats_path_json}, {stats_path_csv}")
        except (IOError, OSError) as e:
            logger.error(f"統計情報の保存に失敗 (ファイルシステムエラー): {e}")
        except (TypeError, ValueError) as e:
            logger.error(f"統計情報の保存に失敗 (データエラー): {e}")
        except Exception as e:
            logger.error(f"統計情報の保存に失敗 (予期しないエラー): {e}", exc_info=True)

    def _setup_logging(self):
        """高度なロギングシステムを設定する（セキュリティフィルタ付き）"""
        # ログディレクトリを作成
        log_dir = self.script_dir / 'logs'
        log_dir.mkdir(exist_ok=True)

        log_filename = log_dir / "tumblr_collector.log"

        # Use centralized logging configuration from logging_utils
        LoggingConfig.setup_logging(
            config=self.config,
            log_file=str(log_filename),
            enable_colors=True,
            enable_sanitization=True
        )

        # 未処理の例外をキャッチするハンドラーを追加
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return

            logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
            self._generate_crash_report(exc_type, exc_value, exc_traceback)

        sys.excepthook = handle_exception

    def _load_config(self):
        """設定ファイルを読み込む"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"{self.config_file} is corrupted. Using defaults.")
                return {}
            except Exception as e:
                logger.error(f"Error loading config file {self.config_file}: {e}")
                return {}
        logger.info(f"Config file {self.config_file} not found. Using defaults.")
        return {}

    def _save_config(self):
        """設定ファイルに書き込む"""
        try:
            # Ensure parent directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4, sort_keys=True)
            logger.debug(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving config file {self.config_file}: {e}")

    def _setup_output_directory(self):
        """保存先ディレクトリを作成する"""
        try:
            self.output_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory set to: {self.output_folder}")
        except Exception as e:
            logger.error(f"Failed to create output directory {self.output_folder}: {e}")
            # Consider raising an exception instead of exiting directly
            raise IOError(f"Cannot create output directory: {self.output_folder}") from e

    def _load_downloaded_files(self):
        """出力フォルダから既存のファイル情報を読み込み、キャッシュを初期化する"""
        if not self.output_folder.exists():
             logger.warning(f"Output folder {self.output_folder} does not exist yet.")
             return
        try:
            existing_files = {f.name for f in self.output_folder.iterdir() if f.is_file() and f.suffix.lower() not in {'.json', '.log'}}
            self.downloaded_files.update(existing_files)

            # 既存ファイルのハッシュをメタデータから読み込み、可能であればキャッシュする
            loaded_hashes = 0
            skipped_hashes = 0
            for filename in existing_files:
                metadata_path = (self.output_folder / filename).with_suffix('.json')
                if not metadata_path.exists():
                    skipped_hashes += 1
                    continue
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as meta_file:
                        metadata = json.load(meta_file)
                    image_hash = metadata.get('image_hash')
                    if image_hash:
                        self.downloaded_hashes.add(image_hash)
                        loaded_hashes += 1
                    else:
                        skipped_hashes += 1
                except Exception as e:
                    skipped_hashes += 1
                    logger.debug(f"Failed to load metadata hash for {filename}: {e}")

            logger.info(
                "Loaded %d existing files and %d cached hashes from %s (skipped %d).",
                len(existing_files),
                loaded_hashes,
                self.output_folder,
                skipped_hashes
            )
        except Exception as e:
            logger.error(f"Error reading existing files from {self.output_folder}: {e}")

    def _get_oauth_token(self):
        """OAuthアクセストークンを取得する"""
        if not self.consumer_key or not self.consumer_secret:
             logger.error("Consumer key and secret must be set before getting OAuth token.")
             return None, None

        oauth_client = pytumblr.TumblrRestClient(self.consumer_key, self.consumer_secret)
        try:
            url = oauth_client.get_authorize_url()

            # URL検証（セキュリティチェック）
            if not url or not isinstance(url, str):
                logger.error("Invalid OAuth URL received")
                return None, None

            # URLがTumblrドメインであることを確認
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if not parsed.netloc.endswith('tumblr.com'):
                logger.error(f"OAuth URL is not from Tumblr domain: {parsed.netloc}")
                return None, None

            logger.info("Please visit the following URL in your browser to get the OAuth verifier:")
            logger.info(url)

            # Try to open browser, but don't fail if it doesn't work
            try:
                webbrowser.open(url)
            except Exception as browser_error:
                 logger.warning(f"Could not automatically open browser: {browser_error}")

            # Verifierの入力と検証
            verifier = input("Enter the OAuth verifier here: ").strip()
            if not verifier:
                 logger.error("OAuth verifier is required.")
                 return None, None

            # Verifierの形式検証（英数字のみ、適切な長さ）
            if not verifier.isalnum() or len(verifier) < 6 or len(verifier) > 128:
                logger.error("Invalid OAuth verifier format")
                return None, None

            oauth_client.get_access_token(verifier)

            # トークンの検証
            if not oauth_client.token or not oauth_client.token_secret:
                logger.error("Failed to obtain valid OAuth tokens")
                return None, None

            logger.info("OAuth access token obtained successfully!")
            # セキュリティ: トークンの全体をログに出力しない
            logger.debug(f"OAuth Token (first 10 chars): {oauth_client.token[:10]}...")
            return oauth_client.token, oauth_client.token_secret

        except KeyboardInterrupt:
            logger.warning("OAuth flow cancelled by user")
            return None, None
        except Exception as e:
            logger.error(f"Error obtaining OAuth token: {e}")
            return None, None

    def _setup_proxy(self):
        """プロキシ設定を初期化する"""
        proxy_type = self.proxy_config.get('type')
        if not proxy_type:
            logger.info("No proxy configuration found. Using direct connection.")
            return

        try:
            if proxy_type == 'socks4':
                socks.set_default_proxy(socks.SOCKS4, 
                    self.proxy_config['host'], 
                    int(self.proxy_config['port']),
                    username=self.proxy_config.get('username'))
            elif proxy_type == 'socks5':
                socks.set_default_proxy(socks.SOCKS5, 
                    self.proxy_config['host'], 
                    int(self.proxy_config['port']),
                    username=self.proxy_config.get('username'),
                    password=self.proxy_config.get('password'))
            elif proxy_type in ['http', 'https']:
                # urllib3のプロキシ設定
                proxy_url = f"{proxy_type}://{self.proxy_config['host']}:{self.proxy_config['port']}"
                if self.proxy_config.get('username') and self.proxy_config.get('password'):
                    proxy_url = f"{proxy_type}://{self.proxy_config['username']}:{self.proxy_config['password']}@{self.proxy_config['host']}:{self.proxy_config['port']}"
                
                self.proxy = urllib3.ProxyManager(proxy_url)
                logger.info(f"Proxy configured: {proxy_type.upper()} at {self.proxy_config['host']}:{self.proxy_config['port']}")
            
            # デフォルトソケットをSOCKSに変更
            socket.socket = socks.socksocket
            logger.info(f"Proxy type {proxy_type} initialized successfully.")
        except Exception as e:
            logger.error(f"Proxy configuration failed: {e}")
            # プロキシ設定をリセット
            self.proxy_config = DEFAULT_PROXY_CONFIG

    def _validate_input(self, input_value, input_type, max_length=None, allowed_chars=None):
        """
        入力値を検証する

        Args:
            input_value (str): 検証する入力値
            input_type (str): 入力の種類 ('url', 'filename', 'path', 'text')
            max_length (int): 最大長
            allowed_chars (str): 許可する文字セット

        Returns:
            bool: 検証結果
        """
        if not isinstance(input_value, str):
            return False

        # 空文字列チェック
        if not input_value.strip():
            return False

        # 最大長チェック
        if max_length and len(input_value) > max_length:
            return False

        # 文字セットチェック
        if allowed_chars:
            if not all(c in allowed_chars for c in input_value):
                return False

        # 入力タイプ別の追加検証
        if input_type == 'url':
            try:
                from urllib.parse import urlparse
                parsed = urlparse(input_value)
                return bool(parsed.scheme and parsed.netloc)
            except Exception:
                return False

        elif input_type == 'filename':
            import re
            # ファイル名として危険な文字をチェック
            dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            return not any(char in input_value for char in dangerous_chars)

        elif input_type == 'path':
            # パストラバーサルとnullバイト攻撃をチェック
            if '..' in input_value:
                return False
            if '\0' in input_value:
                return False
            # Windowsパストラバーサル対策
            if input_value.startswith(('/', '\\')):
                return False
            # ドライブレター検証
            if len(input_value) > 1 and input_value[1] == ':':
                return False
            return True

        return True

    def _sanitize_filename(self, filename):
        """
        ファイル名をサニタイズする

        Args:
            filename (str): 元のファイル名

        Returns:
            str: サニタイズされたファイル名
        """
        import re

        if not isinstance(filename, str):
            return f"image_{int(time.time())}.jpg"

        # 制御文字とnullバイトを除去
        sanitized = ''.join(char for char in filename if ord(char) >= 32 and char != '\0')

        # 危険な文字を置換
        dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '_')

        # 連続するアンダースコアを1つに
        sanitized = re.sub(r'_+', '_', sanitized)

        # 先頭と末尾の空白・ピリオドを除去（Windowsファイルシステム対策）
        sanitized = sanitized.strip('. ')

        # 予約語チェック（Windows）
        reserved_names = {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
                         'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2',
                         'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'}
        name_part = sanitized.split('.')[0].upper()
        if name_part in reserved_names:
            sanitized = f"file_{sanitized}"

        # ファイル名が長すぎる場合は切り詰め（拡張子を保持）
        max_length = 200  # 安全なマージンを確保
        if len(sanitized) > max_length:
            name, ext = os.path.splitext(sanitized)
            # 拡張子を保持しつつファイル名を短縮
            available_length = max_length - len(ext)
            sanitized = name[:available_length] + ext

        # ファイル名が空になった場合はデフォルト名を使用
        if not sanitized.strip():
            sanitized = f"image_{int(time.time())}.jpg"

        return sanitized

    def _process_image_efficiently(self, image_path, max_dimension=constants.MAX_DIMENSION_RESIZE):
        """メモリ効率を重視した画像処理を行う"""
        if not _NUMPY_AVAILABLE:
            logger.debug("NumPyが利用できないため、簡易的な画像処理のみ実施します。")

        image_copy = None
        img_array = None

        try:
            # ファイルサイズチェック（メモリ保護）
            file_size = Path(image_path).stat().st_size
            max_file_size = 100 * 1024 * 1024  # 100MB制限
            if file_size > max_file_size:
                logger.warning(f"ファイルサイズが大きすぎます: {file_size / 1024 / 1024:.2f}MB")
                return {}

            with Image.open(image_path) as img:
                original_width, original_height = img.size

                # 画像サイズの検証（爆弾画像対策）
                max_pixels = 178956970  # PIL の DECOMPRESSION_BOMB_CHECK のデフォルト
                if original_width * original_height > max_pixels:
                    logger.warning(f"画像サイズが大きすぎます: {original_width}x{original_height}")
                    return {}

                image_copy = img.copy()
                if max(original_width, original_height) > max_dimension:
                    image_copy.thumbnail((max_dimension, max_dimension), Image.LANCZOS)

                features = {
                    'dimensions': {
                        'width': image_copy.size[0],
                        'height': image_copy.size[1]
                    },
                    'aspect_ratio': image_copy.size[0] / image_copy.size[1] if image_copy.size[1] else 0,
                    'color_mode': image_copy.mode,
                }

                if _NUMPY_AVAILABLE:
                    img_array = np.array(image_copy)
                    color_channels = img_array.shape[2] if len(img_array.shape) > 2 else 1
                    features.update({
                        'color_channels': color_channels,
                        'mean_color': np.mean(img_array, axis=(0, 1)).tolist() if color_channels > 1 else []
                    })
                else:
                    features.update({
                        'color_channels': len(image_copy.getbands()),
                        'mean_color': []
                    })

                return features

        except FileNotFoundError:
            logger.error(f"画像ファイルが見つかりません: {image_path}")
        except Exception as e:
            logger.error(f"メモリ効率の良い画像処理中にエラー: {e}")
        finally:
            # メモリ解放
            if image_copy is not None:
                del image_copy
            if img_array is not None:
                del img_array
            import gc
            gc.collect()

        return {}

    def _extract_image_metadata(self, image_path):
        """画像ファイルからメタデータを抽出する"""
        try:
            path_obj = Path(image_path)
            if not path_obj.exists():
                logger.warning(f"メタデータ抽出対象のファイルが存在しません: {image_path}")
                return None

            with Image.open(path_obj) as img:
                width, height = img.size
                file_size = path_obj.stat().st_size
                file_format = img.format or 'UNKNOWN'
                color_mode = img.mode

                try:
                    phash = str(imagehash.phash(img))
                except Exception as hash_error:
                    logger.warning(f"画像ハッシュ算出に失敗しました: {hash_error}")
                    phash = None

                quality_score = 0.0
                try:
                    quality_score = float(self._calculate_image_quality(img))
                except Exception as quality_error:
                    logger.warning(f"画質評価に失敗しました: {quality_error}")

                nsfw_score = None
                if _CV2_AVAILABLE and _NUMPY_AVAILABLE:
                    try:
                        nsfw_input = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        nsfw_score = float(self._estimate_nsfw_content(nsfw_input))
                    except Exception as nsfw_error:
                        logger.warning(f"NSFWスコア算出に失敗しました: {nsfw_error}")
                else:
                    nsfw_score = None

            classification_result = {}
            if getattr(self, "image_classifier", None) is not None:
                try:
                    classification_result = self.image_classifier.analyze_image(str(path_obj)) or {}
                except Exception as clf_error:
                    logger.warning(f"AI画像分類に失敗しました: {clf_error}")
                    classification_result = {}

            metadata = {
                'width': width,
                'height': height,
                'file_size': file_size,
                'format': file_format,
                'color_mode': color_mode,
                'phash': phash,
                'quality_score': quality_score,
                'ai_classification': {
                    'is_valid': classification_result.get('is_valid', False),
                    'is_high_resolution': classification_result.get('is_high_resolution', False),
                    'is_potentially_nsfw': classification_result.get('is_potentially_nsfw', False),
                    'top_predictions': classification_result.get('top_predictions', []),
                    'metrics': dict(classification_result.get('metrics', {}) or {})
                }
            }

            processed_features = self._process_image_efficiently(str(path_obj))
            if processed_features:
                metadata['processed_features'] = processed_features

            if nsfw_score is not None:
                metrics_section = metadata['ai_classification'].setdefault('metrics', {})
                metrics_section['nsfw_score'] = nsfw_score
                metadata['ai_classification']['is_potentially_nsfw'] = (
                    metadata['ai_classification']['is_potentially_nsfw']
                    or nsfw_score >= self.nsfw_threshold
                )

            return metadata
        except Exception as e:
            logger.error(f"メタデータ抽出エラー: {e}")
            return None

    def _generate_filename_from_path(self, cached_path: Path) -> str:
        """キャッシュまたはテンポラリファイルパスから保存用ファイル名を生成する"""
        cached_path = Path(cached_path)
        extension = cached_path.suffix.lower() or '.jpg'
        safe_name = cached_path.stem
        return f"cached_{safe_name}{extension}"

    def _generate_output_filename(self, source_path, metadata, image_url=None, post_data=None):
        """画像メタデータと投稿情報から保存ファイル名を生成する"""
        path_obj = Path(source_path)
        extension = path_obj.suffix.lower() or '.jpg'

        blog_name = ''
        if isinstance(post_data, dict):
            blog_name = post_data.get('blog_name') or post_data.get('blog', '')

        if not blog_name and image_url:
            try:
                blog_name = urlparse(image_url).netloc.split('.')[0]
            except Exception:
                blog_name = ''

        timestamp_token = ''
        if isinstance(post_data, dict) and post_data.get('timestamp'):
            try:
                timestamp_token = datetime.datetime.fromtimestamp(post_data['timestamp']).strftime('%Y%m%d_%H%M%S')
            except Exception:
                timestamp_token = ''

        if not timestamp_token:
            timestamp_token = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')

        width = metadata.get('width') if isinstance(metadata, dict) else None
        height = metadata.get('height') if isinstance(metadata, dict) else None
        size_token = f"{width}x{height}" if width and height else ''

        base_token = path_obj.stem
        components = [blog_name, timestamp_token, size_token, base_token]
        raw_name = '_'.join(filter(None, components)) or f"image_{timestamp_token}"

        sanitized = self._sanitize_filename(raw_name)
        if not sanitized.lower().endswith(extension):
            sanitized = f"{sanitized}{extension}"

        return sanitized

    def _advanced_image_analysis(self, image_path):
        """
        高度な画像分析を実行し、詳細なメタデータを生成
        
        Args:
            image_path (Path): 分析する画像のパス
        
        Returns:
            dict: 画像の詳細な分析結果
        """
        try:
            # OpenCVを使用した画像分析
            if not _CV2_AVAILABLE or not _NUMPY_AVAILABLE:
                return {}

            # 画像をロード
            img = cv2.imread(str(image_path))
            
            # 基本的な画像プロパティ
            height, width, channels = img.shape
            
            # カラーヒストグラム分析
            color_hist = {
                'red': cv2.calcHist([img], [2], None, [256], [0, 256]).flatten(),
                'green': cv2.calcHist([img], [1], None, [256], [0, 256]).flatten(),
                'blue': cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            }
            
            # エッジ検出
            edges = cv2.Canny(img, CANNY_LOWER_THRESHOLD, CANNY_UPPER_THRESHOLD)
            edge_density = np.sum(edges) / (height * width)
            
            # ぼかし度の検出
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # テクスチャ分析
            glcm = self._calculate_glcm(gray)
            texture_features = self._extract_glcm_features(glcm)
            
            # エッジ方向ヒストグラム
            edge_orientation_hist = self._calculate_edge_orientation_histogram(edges)
            
            # 色の均一性と変化
            color_uniformity = self._calculate_color_uniformity(img)
            
            # 画像の複雑さ指標
            image_entropy = self._calculate_image_entropy(gray)
            
            # 顔検出の簡易実装（オプション）
            face_detection_result = self._detect_faces(img)
            
            # 画像の構図分析
            composition_analysis = self._analyze_image_composition(img)
            
            # コントラスト分析
            contrast = np.std(gray)
            
            # 平均輝度
            mean_brightness = np.mean(gray)
            
            # NSFW検出プレースホルダー（将来の拡張用）
            nsfw_score = self._estimate_nsfw_content(img)

            return {
                'dimensions': {
                    'width': width,
                    'height': height,
                    'aspect_ratio': width / height
                },
                'color_analysis': {
                    'dominant_colors': self._extract_dominant_colors(img),
                    'color_histogram': color_hist
                },
                'quality_metrics': {
                    'blur_score': variance_of_laplacian,
                    'edge_density': edge_density,
                    'contrast': contrast,
                    'mean_brightness': mean_brightness
                },
                'nsfw_score': nsfw_score,
                'channels': channels
            }
        except Exception as e:
            logger.error(f"画像分析中にエラー: {e}")
            return {}


    def _estimate_nsfw_content(self, image):
        """
        画像のNSFW（Not Safe For Work）スコアを推定
        
        Args:
            image (numpy.ndarray): OpenCV形式の画像
        
        Returns:
            float: NSFWスコア（0.0〜1.0）
        """
        if not _CV2_AVAILABLE or not _NUMPY_AVAILABLE:
            logger.debug("NSFW推定に必要なOpenCV/Numpyが利用できないため、スコア0.0を返します。")
            return 0.0

        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            lower_skin = np.array([0, 40, 60], dtype=np.uint8)
            upper_skin = np.array([25, 255, 255], dtype=np.uint8)

            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            skin_ratio = np.sum(skin_mask > 0) / float(image.shape[0] * image.shape[1])

            saturation = hsv[:, :, 1]
            brightness = hsv[:, :, 2]
            high_saturation_ratio = np.mean(saturation > 120)
            high_brightness_ratio = np.mean(brightness > 150)
            brightness_mean = float(brightness.mean())
            brightness_ratio = brightness_mean / 255.0

            contour_result = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contour_result) == 2:
                contours, _ = contour_result
            elif len(contour_result) == 3:
                _, contours, _ = contour_result
            else:  # pragma: no cover
                contours = []

            largest_region = 0.0
            if contours:
                largest_region = max(cv2.contourArea(cnt) for cnt in contours) / float(image.shape[0] * image.shape[1])

            score = (
                0.5 * skin_ratio +
                0.2 * high_saturation_ratio +
                0.15 * largest_region +
                0.15 * high_brightness_ratio
            )

            if brightness_mean < 40:
                score *= max(0.2, brightness_ratio)
            return float(min(max(score, 0.0), 1.0))
        except Exception as e:
            logger.error(f"NSFW推定中にエラー: {e}")
            return 0.0

    def _generate_image_report(self, image_path):
        """
        画像の包括的な分析レポートを生成
        
        Args:
            image_path (Path): 分析する画像のパス
        
        Returns:
            dict: 画像分析レポート
        """
        try:
            # メタデータと高度な分析を統合
            basic_metadata = self._extract_image_metadata(str(image_path)) or {}

            advanced_analysis = self._advanced_image_analysis(image_path)
            
            # レポートを統合
            last_modified = datetime.datetime.fromtimestamp(image_path.stat().st_mtime).isoformat()
            comprehensive_report = {
                'basic_metadata': basic_metadata,
                'advanced_analysis': advanced_analysis,
                'file_info': {
                    'path': str(image_path),
                    'size_bytes': image_path.stat().st_size,
                    'last_modified': last_modified
                }
            }
            
            # JSONファイルに保存
            report_path = image_path.with_suffix('.report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"画像レポートを生成: {report_path}")
            
            return comprehensive_report
        except Exception as e:
            logger.error(f"画像レポート生成中にエラー: {e}")
            return {}

    def _calculate_glcm(self, gray_image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """
        グレイレベル共起行列（GLCM）を計算
        
        Args:
            gray_image (numpy.ndarray): グレースケール画像
            distances (list): 画素間の距離
            angles (list): 画素間の角度
        
        Returns:
            numpy.ndarray: GLCM
        """
        try:
            import skimage.feature
            glcm = skimage.feature.graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
            return glcm
        except Exception as e:
            logger.error(f"GLCM計算中にエラー: {e}")
            return None

    def _generate_crash_report(self, exc_type, exc_value, exc_traceback):
        """
        クラッシュレポートを生成して保存

        Args:
            exc_type: 例外の型
            exc_value: 例外の値
            exc_traceback: 例外のトレースバック
        """
        try:
            crash_id = str(uuid.uuid4())
            crash_time = datetime.datetime.now().isoformat()

            crash_report = {
                'crash_id': crash_id,
                'timestamp': crash_time,
                'platform': platform.platform(),
                'python_version': sys.version,
                'exception_type': str(exc_type.__name__) if exc_type else 'Unknown',
                'exception_message': str(exc_value) if exc_value else 'No message',
                'traceback': traceback.format_exception(exc_type, exc_value, exc_traceback) if exc_traceback else [],
                'download_stats': self._download_stats,
                'config': {k: v for k, v in self.config.items() if k not in ['consumer_key', 'consumer_secret', 'token', 'token_secret']}
            }

            report_path = CRASH_REPORT_DIR / f"crash_{crash_time.replace(':', '-')}_{crash_id[:8]}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(crash_report, f, ensure_ascii=False, indent=2, default=str)

            logger.error(f"クラッシュレポートを保存しました: {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"クラッシュレポートの生成に失敗: {e}")
            return None

    def _extract_glcm_features(self, glcm):
        """
        GLCMから特徴量を抽出

        Args:
            glcm (numpy.ndarray): グレイレベル共起行列

        Returns:
            dict: テクスチャ特徴量
        """
        try:
            import skimage.feature
            features = {
                'contrast': skimage.feature.graycoprops(glcm, 'contrast')[0, 0],
                'dissimilarity': skimage.feature.graycoprops(glcm, 'dissimilarity')[0, 0],
                'homogeneity': skimage.feature.graycoprops(glcm, 'homogeneity')[0, 0],
                'energy': skimage.feature.graycoprops(glcm, 'energy')[0, 0],
                'correlation': skimage.feature.graycoprops(glcm, 'correlation')[0, 0]
            }
            return features
        except Exception as e:
            logger.error(f"GLCM特徴量抽出中にエラー: {e}")
            return {}

    def _calculate_edge_orientation_histogram(self, edges, num_bins=8):
        """
        エッジの方向ヒストグラムを計算
        
        Args:
            edges (numpy.ndarray): エッジ画像
            num_bins (int): 方向ビンの数
        
        Returns:
            numpy.ndarray: エッジ方向ヒストグラム
        """
        try:
            if not _CV2_AVAILABLE or not _NUMPY_AVAILABLE:
                return {}
            
            # エッジの勾配方向を計算
            gx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
            
            # 勾配の大きさと角度を計算
            mag, angle = cv2.cartToPolar(gx, gy)
            
            # 角度をビンに分割
            hist, _ = np.histogram(angle, bins=num_bins, range=(0, np.pi))
            
            return hist
        except Exception as e:
            logger.error(f"エッジ方向ヒストグラム計算中にエラー: {e}")
            return None

    def _calculate_color_uniformity(self, image):
        """
        画像の色の均一性を計算
        
        Args:
            image (numpy.ndarray): カラー画像
        
        Returns:
            float: 色の均一性スコア
        """
        try:
            if not _CV2_AVAILABLE or not _NUMPY_AVAILABLE:
                return {}
            
            # HSV色空間に変換
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 色相と彩度のヒストグラム
            h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            
            # エントロピーを計算
            h_entropy = -np.sum(h_hist * np.log2(h_hist + 1e-7))
            s_entropy = -np.sum(s_hist * np.log2(s_hist + 1e-7))
            
            return (h_entropy + s_entropy) / 2
        except Exception as e:
            logger.error(f"色の均一性計算中にエラー: {e}")
            return 0.0



    def _detect_faces(self, image, min_neighbors=constants.MIN_NEIGHBORS_FACE, scale_factor=constants.SCALE_FACTOR_FACE, advanced_analysis=True, detection_method='cascade'):
        """
        画像内の顔を高度に検出
        
        Args:
            image (numpy.ndarray): カラー画像
            min_neighbors (int): 最小隣接矩形数
            scale_factor (float): スケールファクター
            advanced_analysis (bool): 詳細な顔分析を有効化
            detection_method (str): 顔検出手法を指定
        
        Returns:
            dict: 顔検出結果
        """
        try:
            if not _CV2_AVAILABLE or not _NUMPY_AVAILABLE:
                return {}
            
            # 画像の前処理
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray)
            equalized = cv2.equalizeHist(denoised)
            
            # 顔検出結果格納用の変数
            faces = []
            face_details = []
            
            # 顔検出手法の選択
            if detection_method == 'cascade':
                # カスケード分類器を読み込み
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                
                # 正面とプロファイル顔を検出
                faces_frontal = face_cascade.detectMultiScale(equalized, scaleFactor=scale_factor, minNeighbors=min_neighbors)
                faces_profile_left = profile_face_cascade.detectMultiScale(equalized, scaleFactor=scale_factor, minNeighbors=min_neighbors)
                faces_profile_right = profile_face_cascade.detectMultiScale(cv2.flip(equalized, 1), scaleFactor=scale_factor, minNeighbors=min_neighbors)
                
                faces = np.concatenate([faces_frontal, faces_profile_left, faces_profile_right])
            
            elif detection_method == 'dnn':
                # DNNベースの顔検出（OpenCV深層学習モデル）
                net = cv2.dnn.readNetFromCaffe(
                    'path/to/deploy.prototxt', 
                    'path/to/res10_300x300_ssd_iter_140000.caffemodel'
                )
                blob = cv2.dnn.blobFromImage(image, 1.0, (constants.FACE_MAX_SIZE, constants.FACE_MAX_SIZE), (104.0, 177.0, 123.0))
                net.setInput(blob)
                detections = net.forward()
                
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    
                    # 信頼度のしきい値を設定
                    if confidence > constants.CONFIDENCE_THRESHOLD:  # 50%以上の信頼度
                        # 顔の位置座標を計算
                        box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                        (startX, startY, endX, endY) = box.astype('int')
                        
                        # 顔の幅と高さを計算
                        width = endX - startX
                        height = endY - startY
                        
                        # 小さすぎる顔や大きすぎる顔を除外
                        if constants.FACE_MIN_SIZE <= width <= constants.FACE_MAX_SIZE and constants.FACE_MIN_SIZE <= height <= constants.FACE_MAX_SIZE:
                            faces.append([startX, startY, width, height])
                    
                    # 検出された顔の数が一定数を超えたら処理を中断
                    if len(faces) >= 10:  # 最大10個の顔を検出
                        break
            
            # 高度な顔分析
            if advanced_analysis:
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # 目の検出
                    eyes = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml').detectMultiScale(face_roi)
                    
                    # 顔の特徴量計算
                    face_brightness = np.mean(face_roi)
                    face_contrast = np.std(face_roi)
                    
                    # 顔の方向性
                    face_orientation = self._estimate_face_orientation(face_roi)
                    
                    # 顔の品質評価
                    face_quality = self._calculate_face_quality(face_roi, eyes)
                    
                    # 顔の特徴抽出
                    face_features = self._extract_face_features(face_roi)
                    
                    face_details.append({
                        'location': [x, y, w, h],
                        'brightness': float(face_brightness),
                        'contrast': float(face_contrast),
                        'eyes_count': len(eyes),
                        'quality_score': face_quality,
                        'features': face_features,
                        'orientation': face_orientation
                    })
            
            return {
                'detected_faces': len(faces),
                'face_locations': faces.tolist(),
                'face_details': face_details if advanced_analysis else []
            }
        except Exception as e:
            logger.error(f"顔検出中にエラー: {e}")
            return {'detected_faces': 0, 'face_locations': [], 'face_details': []}

    def _calculate_face_quality(self, face_roi, eyes):
        """
        顔の品質を評価するメソッド
        
        Args:
            face_roi (numpy.ndarray): 顔のローカルイメージ
            eyes (list): 検出された目のリスト
        
        Returns:
            float: 顔の品質スコア
        """
        try:
            if not _CV2_AVAILABLE or not _NUMPY_AVAILABLE:
                return {}
            
            # ノイズレベルの評価
            noise_level = cv2.meanStdDev(face_roi)[1][0][0]
            
            # シャープネスの評価
            laplacian = cv2.Laplacian(face_roi, cv2.CV_64F).var()
            
            # 目の存在を考慮
            eye_factor = 1.0 if len(eyes) > 0 else 0.5
            
            # 品質スコアの計算
            quality_score = (
                (1.0 / (1.0 + noise_level)) *  # ノイズの少なさ
                (laplacian / 100.0) *  # シャープな画像
                eye_factor  # 目の存在
            )
            
            return max(0.0, min(1.0, quality_score))
        except Exception as e:
            logger.error(f"顔品質評価中にエラー: {e}")
            return 0.5
    
    def _extract_face_features(self, face_roi):
        """
        顔の特徴を抽出するメソッド
        
        Args:
            face_roi (numpy.ndarray): 顔のローカルイメージ
        
        Returns:
            dict: 顔の特徴情報
        """
        try:
            if not _CV2_AVAILABLE or not _NUMPY_AVAILABLE:
                return {}
            
            # グレースケールに変換
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            
            # ヒストグラム均等化
            equalized_face = cv2.equalizeHist(gray_face)
            
            # 特徴量の計算
            features = {
                'mean_intensity': float(np.mean(equalized_face)),
                'std_intensity': float(np.std(equalized_face)),
                'entropy': self._calculate_image_entropy(equalized_face),
                'dominant_colors': self._extract_dominant_colors(face_roi)
            }
            
            return features
        except Exception as e:
            logger.error(f"顔特徴抽出中にエラー: {e}")
            return {}
    
    def _extract_dominant_colors(self, image, k=DEFAULT_COLOR_CLUSTERS):
        """
        画像から主要な色を抽出
        
        Args:
            image (numpy.ndarray): 入力画像
            k (int): 抽出する色の数
        
        Returns:
            list: 主要な色のリスト
        """
        try:
            if not _CV2_AVAILABLE or not _NUMPY_AVAILABLE:
                return {}
            from sklearn.cluster import MiniBatchKMeans
            
            # 画像のリサイズと前処理
            resized_image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
            pixels = resized_image.reshape(-1, 3)
            
            # メモリ効率的なミニバッチK-meansを使用
            kmeans = MiniBatchKMeans(
                n_clusters=k, 
                random_state=42, 
                batch_size=1000, 
                max_iter=10
            ).fit(pixels)
            
            # 主要な色を抽出
            dominant_colors = kmeans.cluster_centers_.astype(int).tolist()
            
            return dominant_colors
        except Exception as e:
            logger.error(f"主要色抽出中にエラー: {e}")
            return []
    
    def _parallel_face_detection(self, image, detection_methods=['cascade', 'dnn']):
        """
        並列処理による顔検出の高速化
        
        Args:
            image (numpy.ndarray): 入力画像
            detection_methods (list): 使用する顔検出手法
        
        Returns:
            dict: 顔検出結果
        """
        try:
            import concurrent.futures
            if not _NUMPY_AVAILABLE:
                return np.array([])
            
            def detect_faces_method(method):
                return self._detect_faces(
                    image, 
                    detection_method=method, 
                    advanced_analysis=False
                )
            
            # 並列処理で顔検出
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(detection_methods)) as executor:
                results = list(executor.map(detect_faces_method, detection_methods))
            
            # 結果の統合
            total_faces = 0
            combined_locations = []
            
            for result in results:
                total_faces += result['detected_faces']
                combined_locations.extend(result['face_locations'])
            
            # 重複する顔領域を除去
            unique_faces = self._remove_duplicate_faces(combined_locations)
            
            return {
                'detected_faces': len(unique_faces),
                'face_locations': unique_faces
            }
        except Exception as e:
            logger.error(f"並列顔検出中にエラー: {e}")
            return {'detected_faces': 0, 'face_locations': []}
    
    def _remove_duplicate_faces(self, faces, iou_threshold=constants.IOU_THRESHOLD, confidence_threshold=constants.CONFIDENCE_THRESHOLD):
        """
        重複する顔領域を除去し、高品質の顔を選択
        
        Args:
            faces (list): 顔の座標リスト
            iou_threshold (float): 重複判定の閾値
            confidence_threshold (float): 顔の信頼度閾値
        
        Returns:
            list: 重複を除去した高品質の顔の座標リスト
        """
        def calculate_iou(box1, box2):
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            
            # 交差領域の計算
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            # 交差領域の面積
            intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
            
            # 和集合の面積
            box1_area = w1 * h1
            box2_area = w2 * h2
            union_area = box1_area + box2_area - intersection_area
            
            # IoU (Intersection over Union)
            return intersection_area / union_area if union_area > 0 else 0
        
        def calculate_face_quality(face):
            """
            顔領域の品質を評価
            """
            x, y, w, h = face
            area = w * h
            aspect_ratio = w / h
            
            # 顔領域の大きさと形状に基づく品質スコア
            size_score = min(1.0, area / (constants.FACE_MAX_SIZE * constants.FACE_MAX_SIZE))
            aspect_score = 1.0 - abs(aspect_ratio - 1.0)  # 正方形に近いほど高いスコア
            
            return size_score * aspect_score
        
        # 品質スコアに基づく顔をソート
        sorted_faces = sorted(faces, key=calculate_face_quality, reverse=True)
        
        unique_faces = []
        for face in sorted_faces:
            is_duplicate = False
            for unique_face in unique_faces:
                if calculate_iou(face, unique_face) > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_faces.append(face)
                
                # 最大顔数に制限
                if len(unique_faces) >= 5:
                    break
        
        return unique_faces
    
    def _adaptive_face_detection(self, image, initial_methods=['cascade', 'dnn'], fallback_methods=['haar']):
        """
        適応的な顔検出手法を選択
        
        Args:
            image (numpy.ndarray): 入力画像
            initial_methods (list): 初期に試行する顔検出手法
            fallback_methods (list): フォールバック手法
        
        Returns:
            dict: 顔検出結果
        """
        try:
            # 初期手法で顔検出を試みる
            result = self._parallel_face_detection(image, detection_methods=initial_methods)
            
            # 顔が検出されなかった場合はフォールバック手法を使用
            if result['detected_faces'] == 0 and fallback_methods:
                result = self._parallel_face_detection(image, detection_methods=fallback_methods)
            
            return result
        except Exception as e:
            logger.error(f"適応的顔検出中にエラー: {e}")
            return {'detected_faces': 0, 'face_locations': []}
    
    def _face_detection_performance_monitor(self, detection_results, image_metadata=None):
        """
        顔検出性能をモニタリング
        
        Args:
            detection_results (dict): 顔検出結果
            image_metadata (dict, optional): 画像のメタデータ
        
        Returns:
            dict: 性能分析結果
        """
        try:
            import time
            
            performance_data = {
                'detected_faces': detection_results['detected_faces'],
                'detection_time': time.time(),
                'image_size': image_metadata.get('size', None) if image_metadata else None,
                'image_format': image_metadata.get('format', None) if image_metadata else None
            }
            
            # 性能ロギング
            if performance_data['detected_faces'] > 0:
                logger.info(f"Face Detection Performance: {performance_data}")
            
            return performance_data
        except Exception as e:
            logger.error(f"性能モニタリング中にエラー: {e}")
            return {}
    
    def extract_image_metadata(self, image_path, post_data=None):
        metadata = self._extract_image_metadata(image_path) or {}

        if isinstance(post_data, dict):
            metadata.setdefault('tags', post_data.get('tags', []))
        else:
            metadata.setdefault('tags', [])

        if metadata.get('tags'):
            try:
                classification = self._classify_by_tags(metadata['tags'])
                metadata['tag_classification'] = classification
            except Exception as e:
                logger.warning(f"タグ分類に失敗しました: {e}")

        return metadata
    
    def _translate_tags(self, tag):
        """
        タグの多言語翻訳と同義語マッピング
        
        Args:
            tag (str): 元のタグ
        
        Returns:
            list: 関連する多言語タグのリスト
        """
        # 多言語・同義語マッピング
        tag_translations = {
            # 人物関連タグ
            'portrait': ['肖像', 'ポートレート', 'プロフィール写真'],
            'selfie': ['自撮り', 'セルフポートレート', 'セルフショット'],
            'person': ['人', '人物', 'ヒト', 'individual'],
            'face': ['顔', '表情', 'フェイス', 'visage'],
            'people': ['人々', '集団', 'folks', 'crowd'],
            'human': ['人間', 'ホト', 'mankind', '人類'],
            'model': ['モデル', 'モデリング', 'mannequin', 'ポーザー'],
            
            # 詳細な人物カテゴリ
            'cosplay': ['コスプレ', 'costume play', '仮装', 'character dress'],
            'celebrity': ['セレブ', '有名人', 'スター', 'famous person'],
            'influencer': ['インフルエンサー', '影響力のある人', 'social media star'],
            'actor': ['俳優', '演技者', 'performer', 'artiste'],
            'actress': ['女優', '女性演技者', 'female performer'],
            'musician': ['音楽家', 'ミュージシャン', 'artist', '演奏家'],
            'artist': ['アーティスト', '芸術家', 'creator', '表現者'],
            'performer': ['パフォーマー', '実演家', 'entertainer', '舞台人'],
            'dancer': ['ダンサー', '踊り手', 'choreographer', '舞踊家'],
            
            # 感情・姿勢タグ
            'smile': ['笑顔', 'スマイル', '微笑', 'grin'],
            'laugh': ['笑う', '笑い', 'chuckle', 'giggle'],
            'pose': ['ポーズ', '姿勢', 'stance', 'posture'],
            'expression': ['表情', '感情表現', 'facial expression', 'look'],
            'emotion': ['感情', '気持ち', 'sentiment', 'feeling'],
            
            # NSFWタグ
            'nsfw': ['エロ', '不適切', 'センシティブ', 'mature content'],
            'adult': ['アダルト', '成人向け', 'mature', '18禁'],
            'sexy': ['セクシー', 'エロティック', 'provocative', '官能的'],
            'nude': ['ヌード', '裸', 'bare', 'undressed'],
            'provocative': ['挑発的', 'センシティブ', 'suggestive', 'risqué']
        }
        
        # 大文字小文字を区別しない検索
        tag_lower = tag.lower()
        
        # 完全一致と部分一致の両方を検索
        translations = []
        for key, values in tag_translations.items():
            if tag_lower == key.lower() or tag_lower in [v.lower() for v in values]:
                translations.append(key)
                translations.extend(values)
        
        return list(set(translations))
    
    def _classify_by_tags(self, tags):
        """
        タグに基づいて画像が人物関連かどうかを判定
        
        Args:
            tags (list): タグリスト
        
        Returns:
            dict: 画像分類結果の詳細情報
        """
        # 多言語タグの展開
        expanded_tags = []
        for tag in tags:
            expanded_tags.extend(self._translate_tags(tag))
        
        # 人物関連タグ（拡張版）
        person_tags = [
            # 一般的な人物タグ
            'portrait', 'selfie', 'person', 'face', 'people', 'human', 'model',
            
            # 詳細な人物カテゴリ
            'cosplay', 'celebrity', 'influencer', 'actor', 'actress', 
            'musician', 'artist', 'performer', 'dancer',
            
            # 感情や姿勢に関連するタグ
            'smile', 'laugh', 'pose', 'expression', 'emotion'
        ]
        
        # アダルト・センシティブコンテンツタグ
        nsfw_tags = [
            'nsfw', 'adult', 'sexy', 'nude', 'provocative'
        ]
        
        # タグを小文字化してマッチング
        normalized_tags = [tag.lower() for tag in expanded_tags]
        
        # 分類結果の詳細
        classification_result = {
            'is_person': False,
            'is_nsfw': False,
            'confidence': 0.0,
            'tags': []
        }
        
        # 人物関連タグの検出
        person_matches = [tag for tag in person_tags if tag.lower() in normalized_tags]
        if person_matches:
            classification_result['is_person'] = True
            classification_result['tags'] = person_matches
            classification_result['confidence'] = len(person_matches) / len(person_tags)
        
        # NSFWタグの検出
        nsfw_matches = [tag for tag in nsfw_tags if tag.lower() in normalized_tags]
        if nsfw_matches:
            classification_result['is_nsfw'] = True
        
        return classification_result
    
    def _generate_recommended_tags(self, image_path):
        """
        画像から推奨タグを自動生成
        
        Args:
            image_path (str): 画像ファイルパス
        
        Returns:
            list: 推奨タグのリスト
        """
        if not _CV2_AVAILABLE or not _NUMPY_AVAILABLE:
            return {}
        from PIL import Image
        
        recommended_tags = []
        
        try:
            # 画像読み込み
            image = cv2.imread(image_path)
            
            # 色分析
            dominant_colors = self._extract_dominant_colors(image)
            color_tags = [
                f'{color_name} tone' for color_name, _ in dominant_colors
            ]
            recommended_tags.extend(color_tags)
            
            # 画像の明るさと雰囲気
            brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) / 255.0
            if brightness < BRIGHTNESS_THRESHOLD_LOW:
                recommended_tags.append('dark')
            elif brightness > 0.7:
                recommended_tags.append('bright')
            
            # アスペクト比分析
            height, width = image.shape[:2]
            aspect_ratio = width / height
            if aspect_ratio < 0.75:
                recommended_tags.append('portrait')
            elif aspect_ratio > 1.33:
                recommended_tags.append('landscape')
            
            # 画像の質感タグ
            quality_score = self._calculate_image_quality(image)
            if quality_score > 0.8:
                recommended_tags.append('high-quality')
            elif quality_score < QUALITY_THRESHOLD_LOW:
                recommended_tags.append('low-quality')
            
            return list(set(recommended_tags))
        
        except Exception as e:
            logger.error(f'タグ生成中にエラー: {e}')
            return []
    
    def _calculate_image_quality(self, image, fast_mode=True, advanced_analysis=False):
        """
        画像品質の総合的な評価
        
        Args:
            image (numpy.ndarray): 入力画像
            fast_mode (bool): 高速モードを有効化する
            advanced_analysis (bool): 高度な画像分析を有効化
        
        Returns:
            dict or float: 画像品質スコアまたは詳細情報
        """
        if not _CV2_AVAILABLE or not _NUMPY_AVAILABLE:
            return {}
        
        try:
            # 画像サイズの縮小で計算間隔を前値する
            original_shape = image.shape
            if fast_mode and image.shape[0] > RESIZE_THRESHOLD:
                scale_factor = RESIZE_THRESHOLD / image.shape[0]
                image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            
            # グレースケール変換
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # シャープネス評価
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000, 1.0)
            
            # 明度均一性評価
            brightness_mean = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness_mean - 0.5) * 2
            
            # ノイズレベル評価
            noise_score = 1.0 - np.std(gray) / 255.0
            
            # 総合品質スコア
            quality_score = (sharpness_score + brightness_score + noise_score) / 3
            
            # 高度な分析を有効化した場合の詳細情報
            if advanced_analysis:
                # コントラスト評価
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                contrast_enhanced = clahe.apply(gray)
                contrast_score = np.std(contrast_enhanced) / 255.0
                
                # エッジ検出
                edges = cv2.Canny(gray, CANNY_LOWER_THRESHOLD, CANNY_UPPER_THRESHOLD)
                edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
                
                return {
                    'quality_score': quality_score,
                    'sharpness': sharpness_score,
                    'brightness': brightness_score,
                    'noise_level': noise_score,
                    'contrast': contrast_score,
                    'edge_density': edge_density,
                    'original_size': original_shape[:2],
                    'resized': fast_mode and original_shape[0] > RESIZE_THRESHOLD
                }
            
            return quality_score
        except Exception as e:
            logger.error(f"画像品質評価中にエラー: {e}")
            return 0.0
    
    def _robust_image_processing(self, image_path, processing_func, max_retries=DEFAULT_RETRY_ATTEMPTS, timeout=10):
        """
        復復力とタイムアウトを考慮した画像処理
        
        Args:
            image_path (str): 画像ファイルパス
            processing_func (callable): 処理関数
            max_retries (int): 最大再試行回数
            timeout (int): タイムアウト秒数
        
        Returns:
            Any: 処理結果
        """
        import signal
        import functools
        
        class TimeoutError(Exception):
            pass
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Processing timed out")
        
        @functools.wraps(processing_func)
        def timeout_wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            try:
                result = processing_func(*args, **kwargs)
                signal.alarm(0)  # タイムアウト解除
                return result
            except TimeoutError:
                logger.warning(f"Processing timed out for {image_path}")
                return None
            finally:
                signal.alarm(0)
        
        for attempt in range(max_retries):
            try:
                # メモリ効率的な読み込みと処理
                result = self._memory_efficient_image_processing(
                    image_path, 
                    lambda img: timeout_wrapper(img)
                )
                
                if result is not None:
                    return result
                
                logger.warning(f"Attempt {attempt + 1} failed for {image_path}")
            
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
        
        logger.error(f"Failed to process {image_path} after {max_retries} attempts")
        return None
    
    def _image_cache_manager(self, cache_dir=None, max_cache_size_mb=DEFAULT_CACHE_SIZE_MB, cleanup_threshold=CLEANUP_THRESHOLD):
        """
        画像キャッシュ管理システム
        
        Args:
            cache_dir (str, optional): キャッシュディレクトリ
            max_cache_size_mb (int): 最大キャッシュサイズ（MB）
            cleanup_threshold (float): クリーンアップ開始閾値
        
        Returns:
            dict: キャッシュ管理情報
        """
        import os
        import shutil
        
        # デフォルトキャッシュディレクトリ
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser('~'), '.tumblr_image_collector', 'cache')
        
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            # 現在のキャッシュサイズを評価
            total_size = sum(
                os.path.getsize(os.path.join(cache_dir, f)) 
                for f in os.listdir(cache_dir) 
                if os.path.isfile(os.path.join(cache_dir, f))
            ) / BYTES_TO_MB_DIVISOR  # MB単位
            
            # キャッシュクリーンアップ
            if total_size > max_cache_size_mb * cleanup_threshold:
                # 古い順にソートし、上限を超えるファイルを削除
                files = sorted(
                    [(f, os.path.getatime(os.path.join(cache_dir, f))) for f in os.listdir(cache_dir)],
                    key=lambda x: x[1]
                )
                
                while total_size > max_cache_size_mb * cleanup_threshold and files:
                    oldest_file, _ = files.pop(0)
                    file_path = os.path.join(cache_dir, oldest_file)
                    file_size = os.path.getsize(file_path) / BYTES_TO_MB_DIVISOR
                    os.remove(file_path)
                    total_size -= file_size
            
            return {
                'cache_dir': cache_dir,
                'total_size_mb': total_size,
                'max_size_mb': max_cache_size_mb,
                'cleanup_performed': total_size > max_cache_size_mb * cleanup_threshold
            }
        
        except Exception as e:
            logger.error(f"Cache management error: {e}")
            return None
    
    def _memory_efficient_image_processing(self, image_path, processing_func, chunk_size=MEMORY_CHUNK_SIZE):
        """
        メモリ効率の高い画像処理方法

        Args:
            image_path (str): 画像ファイルパス
            processing_func (callable): 処理関数
            chunk_size (int): メモリチャンクサイズ

        Returns:
            Any: 処理結果
        """
        try:
            with Image.open(image_path) as img:
                # 画像サイズが大きい場合はリサイズして処理
                max_dimension = 2048
                if max(img.size) > max_dimension:
                    ratio = max_dimension / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                # 処理関数を実行
                return processing_func(img)
        except Exception as e:
            logger.error(f"メモリ効率的画像処理中にエラー: {e}")
            return None

    def organize_images_with_metadata(self, source_folder, destination_folder=None):
        """
        タグ情報とメタデータに基づく画像管理
        
        Args:
            source_folder (str): 入力画像フォルダ
            destination_folder (str, optional): 出力先フォルダ
        
        Returns:
            dict: 分類結果
        """
        import os
        import json
        import shutil
        
        # 出力先フォルダの設定
        if destination_folder is None:
            destination_folder = os.path.join(source_folder, 'organized_images')
        
        # 出力先フォルダ作成
        os.makedirs(destination_folder, exist_ok=True)
        metadata_folder = os.path.join(destination_folder, 'metadata')
        os.makedirs(metadata_folder, exist_ok=True)
        
        # 結果記録用変数
        results = {
            'total_images': 0,
            'person_images': 0,
            'non_person_images': 0
        }
        
        # 画像ファイルの調査
        for filename in os.listdir(source_folder):
            file_path = os.path.join(source_folder, filename)
            
            # 画像ファイルのみ対象
            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                results['total_images'] += 1
                
                try:
                    # メタデータ抽出
                    metadata = self.extract_image_metadata(file_path)
                    
                    # メタデータ保存
                    metadata_path = os.path.join(metadata_folder, f"{os.path.splitext(filename)[0]}_metadata.json")
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                    
                    # 人物関連の画像をカウント
                    if metadata['is_person']:
                        results['person_images'] += 1
                    else:
                        results['non_person_images'] += 1
                    
                except Exception as e:
                    logger.error(f"{filename}の分類中にエラー: {e}")
        
        # 結果のロギング
        logger.info(f"Image Organization Results: {results}")
        return results

    def _estimate_face_orientation(self, face_image, orientation_method='gradient'):
        """
        顔の方向性を推定
        
        Args:
            face_image (numpy.ndarray): 顔領域のグレースケール画像
            orientation_method (str): 方向性推定手法
        
        Returns:
            dict: 顔の方向性情報
        """
        try:
            if not _CV2_AVAILABLE or not _NUMPY_AVAILABLE:
                return {}
            
            if orientation_method == 'gradient':
                # 勾配ベースの方向性推定
                gx = cv2.Sobel(face_image, cv2.CV_64F, 1, 0, ksize=SOBEL_KERNEL_SIZE)
                gy = cv2.Sobel(face_image, cv2.CV_64F, 0, 1, ksize=SOBEL_KERNEL_SIZE)
                
                # 勾配の大きさと角度
                mag, angle = cv2.cartToPolar(gx, gy)
                
                # 主要な勾配方向
                dominant_angle = np.mean(angle)
                dominant_magnitude = np.mean(mag)
                
                return {
                    'dominant_angle': float(dominant_angle),
                    'gradient_magnitude': float(dominant_magnitude)
                }
            
            return {}
        except Exception as e:
            logger.error(f"顔の方向性推定中にエラー: {e}")
            return {}

    def _analyze_image_composition(self, image):
        """
        画像の構図を分析
        
        Args:
            image (numpy.ndarray): カラー画像
        
        Returns:
            dict: 構図分析結果
        """
        try:
            if not _CV2_AVAILABLE or not _NUMPY_AVAILABLE:
                return {}
            
            height, width = image.shape[:2]
            
            # ルールオブサードの分析
            third_height = height // 3
            third_width = width // 3
            
            # 各セクションの平均色を計算
            sections = [
                image[0:third_height, 0:third_width],
                image[0:third_height, third_width:2*third_width],
                image[0:third_height, 2*third_width:width],
                image[third_height:2*third_height, 0:third_width],
                image[third_height:2*third_height, third_width:2*third_width],
                image[third_height:2*third_height, 2*third_width:width],
                image[2*third_height:height, 0:third_width],
                image[2*third_height:height, third_width:2*third_width],
                image[2*third_height:height, 2*third_width:width]
            ]
            
            # 各セクションの平均色を計算
            section_colors = [np.mean(section, axis=(0, 1)) for section in sections]
            
            return {
                'rule_of_thirds': {
                    'section_colors': section_colors,
                    'dominant_section': np.argmax([np.mean(color) for color in section_colors])
                },
                'aspect_ratio': width / height
            }
        except Exception as e:
            logger.error(f"構図分析中にエラー: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error extracting image metadata: {e}")
            return None

    def _analyze_color_distribution(self, image):
        """画像のカラー分布を分析"""
        try:
            if not _NUMPY_AVAILABLE:
                return np.array([])
            from PIL import ImageStat, ImageColor

            # RGBヒストグラムを計算
            img_array = np.array(image)
            r_hist, g_hist, b_hist = [np.histogram(img_array[:,:,i], bins=HISTOGRAM_BINS)[0] for i in range(3)]

            # 主色を検出
            dominant_color = self._get_dominant_color(img_array)

            return {
                'color_entropy': self._calculate_color_entropy(img_array),
                'dominant_color': dominant_color,
                'color_variance': np.var(img_array),
                'red_histogram': r_hist.tolist(),
                'green_histogram': g_hist.tolist(),
                'blue_histogram': b_hist.tolist()
            }
        except Exception as e:
            logger.error(f"Error analyzing color distribution: {e}")
            return {}

    def _get_dominant_color(self, img_array):
        """画像の主色を検出"""
        try:
            if not _NUMPY_AVAILABLE:
                return np.array([])

            # ピクセルをグルーピング
            pixels = img_array.reshape(-1, 3)
            unique, counts = np.unique(pixels, axis=0, return_counts=True)
            dominant_color = unique[np.argmax(counts)]

            return {
                'r': int(dominant_color[0]),
                'g': int(dominant_color[1]),
                'b': int(dominant_color[2])
            }
        except Exception as e:
            logger.error(f"Error finding dominant color: {e}")
            return None

    def _calculate_color_entropy(self, img_array):
        """カラーエントロピーを計算"""
        try:
            if not _NUMPY_AVAILABLE:
                return np.array([])

            # RGB値のヒストグラムを計算
            hist, _ = np.histogramdd(img_array.reshape(-1, 3), bins=(COLOR_HISTOGRAM_BINS, COLOR_HISTOGRAM_BINS, COLOR_HISTOGRAM_BINS))
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-10))

            return float(entropy)
        except Exception as e:
            logger.error(f"Error calculating color entropy: {e}")
            return None


    def _setup_credentials(self):
        """APIキーとOAuthトークンを設定ファイルから読み込むか、ユーザーに要求する"""
        env_consumer_key = os.environ.get("TUMBLR_CONSUMER_KEY")
        env_consumer_secret = os.environ.get("TUMBLR_CONSUMER_SECRET")

        if env_consumer_key and env_consumer_secret:
            self.consumer_key = env_consumer_key.strip()
            self.consumer_secret = env_consumer_secret.strip()
            logger.info("Using Tumblr consumer credentials from environment variables.")
        else:
            # Prioritize config file values when environment is not fully provided
            self.consumer_key = self.config.get("consumer_key")
            self.consumer_secret = self.config.get("consumer_secret")

            if not (self.consumer_key and self.consumer_secret):
                logger.info("Tumblr API keys not found. Please enter them:")
                # Use input only if keys are missing
                self.consumer_key = input("Enter your Tumblr Consumer Key: ").strip()
                self.consumer_secret = input("Enter your Tumblr Consumer Secret: ").strip()
                if not (self.consumer_key and self.consumer_secret):
                    logger.error("Consumer Key and Secret are required.")
                    raise ValueError("Missing API credentials")  # Raise exception
                self.config["consumer_key"] = self.consumer_key
                self.config["consumer_secret"] = self.consumer_secret
                self._save_config()  # Save immediately after getting them
                logger.info("API keys saved to config.")

        env_token = os.environ.get("TUMBLR_OAUTH_TOKEN")
        env_token_secret = os.environ.get("TUMBLR_OAUTH_TOKEN_SECRET")

        if env_token and env_token_secret:
            self.token = env_token.strip()
            self.token_secret = env_token_secret.strip()
            logger.info("Using Tumblr OAuth tokens from environment variables.")
        else:
            self.token = self.config.get("token")
            self.token_secret = self.config.get("token_secret")

            if not (self.token and self.token_secret):
                logger.info("OAuth tokens not found in config. Attempting to obtain them...")
                self.token, self.token_secret = self._get_oauth_token()
                if not (self.token and self.token_secret):
                    logger.error("Failed to get OAuth token.")
                    raise ValueError("Missing OAuth credentials")  # Raise exception
                self.config["token"] = self.token
                self.config["token_secret"] = self.token_secret
                self._save_config()  # Save immediately
                logger.info("OAuth tokens saved to config.")

    def _initialize_client(self):
        """Tumblr APIクライアントを初期化する"""
        if not all([self.consumer_key, self.consumer_secret, self.token, self.token_secret]):
            logger.error("Cannot initialize Tumblr client: Credentials missing.")
            raise ValueError("Cannot initialize client due to missing credentials")
        try:
            client = pytumblr.TumblrRestClient(
                self.consumer_key,
                self.consumer_secret,
                self.token,
                self.token_secret
            )
            # Optional: Test connection by fetching user info
            # user_info = client.info()
            # logger.debug(f"Connected to Tumblr API as user: {user_info.get('user', {}).get('name')}")
            logger.info("Tumblr client initialized successfully.")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Tumblr client: {e}")
            raise ConnectionError("Failed to initialize Tumblr client") from e

    def get_blog_posts(self, blog_name, limit=20, offset=0):
        """Tumblrブログの投稿を取得する（レート制限対応）"""
        if not self._check_rate_limit():
            return None

        try:
            if not self._validate_input(blog_name, 'text', max_length=100):
                logger.error(f"無効なブログ名: {blog_name}")
                return None

            normalized_limit = max(1, min(int(limit or 1), self.max_download_limit))
            posts_data = self.client.posts(blog_name, limit=normalized_limit, offset=offset)
            return posts_data.get('posts', [])
        except Exception as e:
            if "limit" in str(e).lower() or "429" in str(e) or "too many requests" in str(e).lower():
                logger.warning(f"Rate limit likely hit while fetching posts for '{blog_name}'.")
                return None
            logger.error(f"Error fetching posts for {blog_name}: {e}")
            return []

    def _record_classification_stats(self, classification_result):
        """AI分類結果をダウンロード統計に反映する"""
        if not classification_result:
            return

        stats = self._download_stats['ai_classification_stats']

        if classification_result.get('is_valid'):
            stats['valid_images'] += 1
        else:
            stats['invalid_images'] += 1

        if classification_result.get('is_high_resolution'):
            stats['high_resolution_images'] += 1
        else:
            stats['low_resolution_images'] += 1

        if classification_result.get('is_potentially_nsfw'):
            stats['potentially_nsfw_images'] += 1

        for prediction in classification_result.get('top_predictions', []):
            label = prediction.get('label')
            confidence = float(prediction.get('confidence', 0.0))
            if not label:
                continue
            distribution = stats['image_type_distribution'].setdefault(
                label,
                {'count': 0, 'total_confidence': 0.0}
            )
            distribution['count'] += 1
            distribution['total_confidence'] += confidence

        metrics_summary = stats.setdefault('metrics_summary', {})
        for metric_name, value in classification_result.get('metrics', {}).items():
            if not isinstance(value, (int, float)):
                continue
            metric_entry = metrics_summary.setdefault(
                metric_name,
                {'count': 0, 'sum': 0.0, 'min': value, 'max': value}
            )
            metric_entry['count'] += 1
            metric_entry['sum'] += value
            metric_entry['min'] = min(metric_entry['min'], value)
            metric_entry['max'] = max(metric_entry['max'], value)

    def _update_download_stats(self, outcome: str) -> None:
        """ダウンロード結果に応じて統計情報を更新"""
        if outcome == 'success':
            self._download_stats['total_images_processed'] += 1
            self._download_stats['successful_downloads'] += 1
        elif outcome == 'failure':
            self._download_stats['total_images_processed'] += 1
            self._download_stats['failed_downloads'] += 1
        elif outcome == 'duplicate':
            self._download_stats['total_images_processed'] += 1
            self._download_stats['skipped_duplicates'] += 1

    def _generate_ar_markers(self, image_path, marker_type='aruco'):
        """
        ARマーカーを生成して画像に埋め込み
        def exponential_backoff(retry_count):
            """エクスポネンシャルバックオフ戦略"""
            base_delay = max(0.5, self.backoff_factor)
            max_delay = max(1, self.max_backoff_seconds)
            jitter = random.uniform(0, 0.5)
            delay = min(max_delay, base_delay * (2 ** retry_count) + jitter)
            logger.info(f"再試行 {retry_count + 1}: {delay}秒待機")
            time.sleep(delay)

        def is_network_error(exception):
            """ネットワークエラーを判定
            
            Args:
                exception (Exception): 発生した例外
            
            Returns:
                bool: ネットワークエラーかどうか
            """
            network_error_types = (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ProxyError,
                requests.exceptions.SSLError
            )
            return isinstance(exception, network_error_types)

        last_exception = None
        for retry_count in range(max_retries):
            try:
                result = self._download_and_store_video(video_url, post_data)
                if result:
                    return True
                # result False implies duplicate or invalid video; do not retry.
                return False
            except requests.RequestException as e:
                last_exception = e
                logger.warning(f"動画ダウンロード試行 {retry_count + 1} 失敗: {e}")

                if is_network_error(e):
                    if retry_count < max_retries - 1:
                        exponential_backoff(retry_count)
                    else:
                        logger.error(f"{video_url}のダウンロードに{max_retries}回失敗")
                        self._update_download_stats('failure')
                        return False
                else:
                    logger.error(f"回復不能なエラー: {e}")
                    self._update_download_stats('failure')
                    return False
            except IOError as e:
                logger.error(f"動画ファイル処理エラー: {e}")
                self._update_download_stats('failure')
                return False

        if last_exception:
            self._log_download_failure(video_url, post_data, last_exception)

        return False


    def _download_and_store_video(self, video_url, post_data=None, allow_duplicate_skip=True):
        """動画をダウンロードして保存する共通処理"""
        # Check disk space before download
        try:
            from resource_manager import DiskSpaceManager
            disk_manager = DiskSpaceManager(min_free_space_mb=100)
            # Estimate 50MB per video (much larger than images)
            disk_manager.ensure_space_for_file(50, str(self.output_folder))
        except DiskSpaceError as e:
            logger.error(f"Insufficient disk space for video download: {e}")
            raise

        request_started = time.monotonic()
        with self.session.get(
            video_url,
            stream=True,
            proxies=self.proxies,
            timeout=self.download_timeout,
            headers={
                'User-Agent': self.session.headers.get('User-Agent', 'TumblrImageCollector/1.0'),
                'Accept': 'video/*,*/*;q=0.8'
            }
        ) as response:
            response.raise_for_status()

            final_url = response.url or video_url
            parsed_final = urlparse(final_url)
            final_host = (parsed_final.hostname or "").lower()

            if parsed_final.scheme.lower() != 'https':
                logger.warning(
                    "HTTPS以外のスキームを検出したため動画ダウンロードを中止しました: %s", final_url
                )
                self._update_download_stats('failure')
                return False

            if not self._is_allowed_domain(final_host):
                logger.warning(
                    f"許可されていないドメインからの応答をブロックしました: {final_host}. 元URL: {video_url}"
                )
                self._update_download_stats('failure')
                return False

            content_type = (response.headers.get('Content-Type') or '').lower()
            if content_type and not content_type.startswith('video/'):
                logger.warning(
                    "動画以外のContent-Typeを検出したためダウンロードを中止しました: %s (%s)",
                    video_url,
                    content_type
                )
                self._update_download_stats('failure')
                return False

            max_size_mb = self.IMAGE_FILTERS.get('max_file_size_mb', MAX_FILE_SIZE_MB) * 10  # Videos can be larger
            max_download_bytes = max_size_mb * 1024 * 1024
            content_length = response.headers.get('Content-Length')
            if content_length:
                try:
                    if int(content_length) > max_download_bytes:
                        logger.warning(
                            "動画コンテンツサイズが上限を超えたためダウンロードを中止しました: %s", video_url
                        )
                        self._update_download_stats('failure')
                        return False
                except ValueError:
                    logger.debug("Content-Lengthヘッダーを解析できませんでした: %s", content_length)

            temp_path = None
            temp_file = None
            downloaded_bytes = 0
            try:
                # 動画ファイルの拡張子を決定
                video_extension = self._get_video_extension(video_url, content_type)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=video_extension)
                for chunk in response.iter_content(chunk_size=DEFAULT_CHUNK_SIZE):
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > max_download_bytes:
                        logger.warning(
                            "動画ダウンロードサイズが上限を超えたため処理を中断しました: %s", video_url
                        )
                        temp_file.close()
                        os.unlink(temp_file.name)
                        self._update_download_stats('failure')
                        return False
                    temp_file.write(chunk)
                temp_path = temp_file.name
            finally:
                if temp_file is not None:
                    temp_file.close()

        try:
            # 動画ファイルの基本検証
            if not self._is_video_valid(temp_path):
                os.unlink(temp_path)
                logger.warning(f"無効な動画ファイル: {video_url}")
                self._update_download_stats('failure')
                return False

            # 動画ハッシュの計算と重複チェック
            video_hash = self._compute_video_hash(temp_path)
            if video_hash and video_hash in self.downloaded_hashes and allow_duplicate_skip:
                os.unlink(temp_path)
                self._update_download_stats('duplicate')
                logger.info("動画ハッシュ重複のためスキップ: %s", video_url)
                return False

            # 動画メタデータの抽出
            metadata = self._extract_video_metadata(temp_path) or {}

            # ファイル名の生成
            filename = self._generate_video_filename(
                temp_path,
                metadata,
                video_url=video_url,
                post_data=post_data
            )
            filepath = self.output_folder / filename

            if filepath.exists() and allow_duplicate_skip:
                os.unlink(temp_path)
                self._update_download_stats('duplicate')
                logger.info(f"重複動画をスキップ: {filename}")
                return False

            shutil.move(temp_path, filepath)
            if self.cache_enabled:
                self._save_to_cache(filepath, video_url)

            self._update_download_stats('success')
            logger.info(f"動画ダウンロード成功: {filename}")

            if 'video_hash' not in metadata and video_hash:
                metadata['video_hash'] = video_hash

            # メタデータファイルの保存
            metadata_file = filepath.with_suffix('.json')
            try:
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
            except (OSError, json.JSONDecodeError) as metadata_err:
                logger.error(f"動画メタデータの保存に失敗しました: {metadata_err}")
                if metadata_file.exists():
                    metadata_file.unlink(missing_ok=True)

            self.downloaded_files.add(filename)
            if video_hash:
                self.downloaded_hashes.add(video_hash)

            elapsed = time.monotonic() - request_started
            if elapsed > self.slow_response_threshold:
                logger.info(
                    f"動画ダウンロードに時間を要しました: {elapsed:.1f} 秒 ({video_url})"
                )
            return filepath
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def _get_video_extension(self, video_url: str, content_type: str = None) -> str:
        """動画ファイルの拡張子を決定する"""
        # Content-Typeから拡張子を決定
        if content_type:
            if 'mp4' in content_type:
                return '.mp4'
            elif 'webm' in content_type:
                return '.webm'
            elif 'avi' in content_type:
                return '.avi'
            elif 'mov' in content_type:
                return '.mov'
            elif 'mkv' in content_type:
                return '.mkv'

        # URLから拡張子を抽出
        url_path = urlparse(video_url).path
        if '.' in url_path:
            ext = '.' + url_path.split('.')[-1].lower()
            video_extensions = {'.mp4', '.webm', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
            if ext in video_extensions:
                return ext

        # デフォルト拡張子
        return '.mp4'

    def _is_video_valid(self, video_path: str) -> bool:
        """動画ファイルが有効かチェックする"""
        try:
            # ファイルサイズチェック
            file_size = Path(video_path).stat().st_size
            if file_size < 1024:  # 1KB未満は無効
                logger.debug(f"動画ファイルサイズが小さすぎます: {file_size} bytes")
                return False

            if file_size > 500 * 1024 * 1024:  # 500MBを超える場合は警告だが有効
                logger.warning(f"非常に大きな動画ファイル: {file_size / 1024 / 1024:.1f} MB")

            # OpenCVで動画を開けるかチェック
            if _CV2_AVAILABLE:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logger.debug("OpenCVで動画を開けません")
                    return False

                # フレーム数とサイズを取得
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                cap.release()

                if frame_count <= 0:
                    logger.debug("動画にフレームがありません")
                    return False

                if width <= 0 or height <= 0:
                    logger.debug("動画の解像度が無効です")
                    return False

            return True

        except Exception as e:
            logger.error(f"動画検証エラー: {e}")
            return False

    @staticmethod
    def _compute_video_hash(video_path: str) -> Optional[str]:
        """動画ファイルの知覚ハッシュを計算"""
        try:
            if not _CV2_AVAILABLE:
                # OpenCVが利用できない場合はファイルハッシュを使用
                import hashlib
                with open(video_path, 'rb') as f:
                    return hashlib.md5(f.read()).hexdigest()

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            # 動画の中央フレームを抽出
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                cap.release()
                return None

            # 中央付近のフレームを抽出
            target_frame = max(1, frame_count // 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                return None

            # グレースケールに変換
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ハッシュ計算
            hash_obj = imagehash.phash(Image.fromarray(gray))
            return str(hash_obj)

        except Exception as e:
            logger.debug(f"動画ハッシュ計算エラー: {e}")
            return None

    def _extract_video_metadata(self, video_path: str) -> Optional[Dict]:
        """動画ファイルからメタデータを抽出"""
        try:
            path_obj = Path(video_path)
            if not path_obj.exists():
                return None

            metadata = {
                'file_size': path_obj.stat().st_size,
                'format': 'video',
            }

            if _CV2_AVAILABLE:
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    metadata.update({
                        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                        'fps': float(cap.get(cv2.CAP_PROP_FPS)),
                        'duration': float(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
                        'codec': 'unknown',  # OpenCVでは直接取得できない
                    })
                    cap.release()

            return metadata

        except Exception as e:
            logger.error(f"動画メタデータ抽出エラー: {e}")
            return None

    def _generate_video_filename(self, video_path, metadata, video_url=None, post_data=None):
        """動画ファイルの保存ファイル名を生成"""
        path_obj = Path(video_path)
        extension = path_obj.suffix.lower() or '.mp4'

        blog_name = ''
        if isinstance(post_data, dict):
            blog_name = post_data.get('blog_name') or post_data.get('blog', '')

        if not blog_name and video_url:
            try:
                blog_name = urlparse(video_url).netloc.split('.')[0]
            except Exception:
                blog_name = ''

        timestamp_token = ''
        if isinstance(post_data, dict) and post_data.get('timestamp'):
            try:
                timestamp_token = datetime.datetime.fromtimestamp(post_data['timestamp']).strftime('%Y%m%d_%H%M%S')
            except Exception:
                pass

        if not timestamp_token:
            timestamp_token = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')

        # 動画の解像度情報を取得
        size_token = ''
        if isinstance(metadata, dict):
            width = metadata.get('width')
            height = metadata.get('height')
            if width and height:
                size_token = f"{width}x{height}"

        base_token = path_obj.stem
        components = [blog_name, timestamp_token, size_token, base_token]
        raw_name = '_'.join(filter(None, components)) or f"video_{timestamp_token}"

        sanitized = self._sanitize_filename(raw_name)
        if not sanitized.lower().endswith(extension):
            sanitized = f"{sanitized}{extension}"

        return sanitized

    def _validate_image_domain(self, image_url: str) -> bool:

    def _validate_image_domain(self, image_url: str) -> bool:
        """画像URLのホストが許可ドメインに含まれるか確認する"""
        try:
            hostname = urlparse(image_url).hostname or ""
            return self._is_allowed_domain(hostname.lower())
        except Exception:
            return False

    def _is_allowed_domain(self, hostname: str) -> bool:
        if not hostname:
            return False
        return any(hostname == domain or hostname.endswith(f".{domain}") for domain in self.allowed_domains)

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

    @staticmethod
    def _build_requests_proxies(proxy_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
        """Requests用のプロキシ辞書を生成"""
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
        if proxy_type.startswith('socks'):
            return {
                'http': proxy_url,
                'https': proxy_url
            }

        return {
            'http': proxy_url,
            'https': proxy_url
        }

    @staticmethod
    def _compute_image_hash(image: Image.Image) -> Optional[str]:
        """画像の知覚ハッシュを計算し、文字列表現として返す"""
        try:
            return str(imagehash.phash(image))
        except Exception as exc:
            logger.debug(f"画像ハッシュ計算に失敗: {exc}")
            return None

    def _setup_signal_handlers(self):
        """シグナルハンドラーを設定してグレースフルシャットダウンを実装"""
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            logger.info(f"シグナル {signal_name} を受信しました。グレースフルシャットダウンを開始します...")

            # 統計を保存
            self._save_statistics()

            # キャッシュインデックスを保存
            if self.cache_enabled:
                self._persist_cache_index()

            # リソースクリーンアップ
            self._cleanup_resources()

            logger.info("グレースフルシャットダウン完了")
            sys.exit(0)

        # SIGINT (Ctrl+C) と SIGTERM を処理
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Windowsでは SIGBREAK も処理
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)

    def _validate_configuration(self):
        """起動時に設定を検証し、問題があれば警告またはエラーを出力"""
        validation_errors = []
        validation_warnings = []

        # 必須設定の検証
        if not self.consumer_key or not self.consumer_secret:
            validation_errors.append("Tumblr API credentials (consumer_key/consumer_secret) are missing")

        if not self.token or not self.token_secret:
            validation_errors.append("OAuth tokens (token/token_secret) are missing")

        # 数値範囲の検証
        if self.max_workers <= 0 or self.max_workers > 50:
            validation_warnings.append(f"max_workers ({self.max_workers}) is out of recommended range (1-50)")

        if self.download_timeout <= 0 or self.download_timeout > 300:
            validation_warnings.append(f"download_timeout ({self.download_timeout}) is out of recommended range (1-300)")

        if self.max_retries < 0 or self.max_retries > 10:
            validation_warnings.append(f"max_retries ({self.max_retries}) is out of recommended range (0-10)")

        if self.nsfw_threshold < 0.0 or self.nsfw_threshold > 1.0:
            validation_warnings.append(f"nsfw_threshold ({self.nsfw_threshold}) must be between 0.0 and 1.0")

        # キャッシュ設定の検証
        if self.cache_enabled:
            if self.cache_ttl_seconds <= 0:
                validation_warnings.append(f"cache_ttl_seconds ({self.cache_ttl_seconds}) should be positive")

            if self.cache_max_entries <= 0:
                validation_warnings.append(f"cache_max_entries ({self.cache_max_entries}) should be positive")

        # 出力ディレクトリの検証
        if not self.output_folder.exists():
            try:
                self.output_folder.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                validation_errors.append(f"Cannot create output directory {self.output_folder}: {e}")

        # エラーがあれば例外を発生
        if validation_errors:
            error_message = "Configuration validation failed:\n" + "\n".join(f"  - {err}" for err in validation_errors)
            logger.error(error_message)
            raise ValueError(error_message)

        # 警告があればログ出力
        if validation_warnings:
            for warning in validation_warnings:
                logger.warning(f"Configuration warning: {warning}")

        logger.info("Configuration validation passed")

    def _log_download_failure(self, image_url, post_data, exception):
        """
        ダウンロード失敗の詳細をログに記録
        
        Args:
            image_url (str): ダウンロードに失敗したURL
            exception (Exception): 発生した例外
        """
        failure_log_path = self.output_folder / 'download_failures.log'
        with open(failure_log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(
                "時刻: {ts}\nURL: {url}\nエラータイプ: {etype}\nエラー詳細: {detail}\n---\n".format(
                    ts=datetime.datetime.now().isoformat(),
                    url=image_url,
                    etype=type(exception).__name__,
                    detail=str(exception)
                )
            )

        try:
            result = self._download_and_store_image(image_url, post_data=post_data, allow_duplicate_skip=False)
            if result:
                return result
        except Exception as retry_exc:
            logger.error(f"Failed during failure-handling download: {retry_exc}")
            self._download_stats['failed_downloads'] += 1
            return None

        self._download_stats['failed_downloads'] += 1
        return None

    def print_download_stats(self):
        """ダウンロード統計を表示"""
        stats = self._download_stats
        total_attempts = stats['total_attempts']
        success_rate = (stats['successful_downloads'] / total_attempts * 100) if total_attempts > 0 else 0

        logger.info("\n=== ダウンロード統計 ===")
        logger.info(f"総試行回数: {total_attempts}")
        logger.info(f"成功ダウンロード数: {stats['successful_downloads']}")
        logger.info(f"失敗ダウンロード数: {stats['failed_downloads']}")
        logger.info(f"重複スキップ数: {stats['skipped_duplicates']}")
        logger.info(f"キャッシュヒット数: {stats.get('cache_hits', 0)}")
        logger.info(f"キャッシュミス数: {stats.get('cache_misses', 0)}")
        logger.info(f"成功率: {success_rate:.2f}%")

        ai_stats = stats.get('ai_classification_stats', {})
        if ai_stats:
            logger.info("\n--- AI分類統計 ---")
            logger.info(f"有効な画像数: {ai_stats.get('valid_images', 0)}")
            logger.info(f"無効な画像数: {ai_stats.get('invalid_images', 0)}")
            logger.info(f"潜在的にNSFWな画像数: {ai_stats.get('potentially_nsfw_images', 0)}")

            metrics_summary = ai_stats.get('metrics_summary', {})
            if metrics_summary:
                logger.info("\nメトリクスサマリー:")
                for metric_name, metric_data in metrics_summary.items():
                    count = metric_data.get('count', 0)
                    average = (metric_data['sum'] / count) if count else 0.0
                    logger.info(
                        f"{metric_name}: 件数 = {count}, 平均 = {average:.4f}, "
                        f"最小 = {metric_data['min']:.4f}, 最大 = {metric_data['max']:.4f}"
                    )

    def _log_final_stats(self):
        """最終的なダウンロード統計をログに記録する"""
        logger.info("--- ダウンロード統計 ---")
        logger.info(f"総ダウンロード試行回数: {self._download_stats['total_attempts']}")
        logger.info(f"成功したダウンロード数: {self._download_stats['successful_downloads']}")
        logger.info(f"失敗したダウンロード数: {self._download_stats['failed_downloads']}")
        logger.info(f"キャッシュヒット数: {self._download_stats.get('cache_hits', 0)}")
        logger.info(f"キャッシュミス数: {self._download_stats.get('cache_misses', 0)}")

        # AI画像分類統計
        logger.info("\n--- AI画像分類統計 ---")
        ai_stats = self._download_stats['ai_classification_stats']
        logger.info(f"有効な画像数: {ai_stats['valid_images']}")
        logger.info(f"無効な画像数: {ai_stats['invalid_images']}")
        logger.info(f"高解像度画像数: {ai_stats['high_resolution_images']}")
        logger.info(f"低解像度画像数: {ai_stats['low_resolution_images']}")
        logger.info(f"潜在的にNSFWな画像数: {ai_stats['potentially_nsfw_images']}")

        # 画像タイプ分布
        logger.info("\n画像タイプ分布:")
        for type_name, type_data in ai_stats['image_type_distribution'].items():
            avg_confidence = (type_data['total_confidence'] / type_data['count']) if type_data['count'] > 0 else 0
            logger.info(f"{type_name}: 数 = {type_data['count']}, 平均信頼度 = {avg_confidence:.2f}")

        metrics_summary = ai_stats.get('metrics_summary', {})
        if metrics_summary:
            logger.info("\nメトリクスサマリー:")
            for metric_name, metric_data in metrics_summary.items():
                count = metric_data.get('count', 0)
                average = (metric_data['sum'] / count) if count else 0.0
                logger.info(
                    f"{metric_name}: 件数 = {count}, 平均 = {average:.4f}, "
                    f"最小 = {metric_data['min']:.4f}, 最大 = {metric_data['max']:.4f}"
                )

        return self._download_stats

    def generate_image_thumbnail(self, image_path, size=DEFAULT_THUMBNAIL_SIZE, quality=DEFAULT_QUALITY):
        """
        画像のサムネイルを自動生成

        Args:
            image_path (str): 元画像のパス
            size (tuple): サムネイルのサイズ (幅, 高さ)
            quality (int): JPEGエンコーディングの画質 (1-95)

        Returns:
            Path: 生成されたサムネイルのパス
        """
        from PIL import Image
        import hashlib

        try:
            with Image.open(image_path) as img:
                # アスペクト比を維持してリサイズ
                img.thumbnail(size, Image.LANCZOS)
                
                # サムネイル保存パスを生成
                filename = os.path.basename(image_path)
                base, ext = os.path.splitext(filename)
                
                # ハッシュを使用してユニークな名前を作成
                hash_object = hashlib.md5(open(image_path, 'rb').read())
                thumbnail_filename = f'{base}_thumb_{hash_object.hexdigest()[:8]}.jpg'
                thumbnail_path = self.output_folder / 'thumbnails' / thumbnail_filename
                
                # ディレクトリ作成
                thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
                
                # サムネイル保存
                img.convert('RGB').save(thumbnail_path, 'JPEG', quality=quality)
                
                return thumbnail_path
        
        except Exception as e:
            logger.error(f"サムネイル生成エラー: {image_path} - {e}")
            return None

    def evaluate_image_quality(self, image_path, min_resolution=(800, 600), min_entropy=4.0):
        """
        画像の品質を評価

        Args:
            image_path (str): 評価する画像のパス
            min_resolution (tuple): 最小許容解像度
            min_entropy (float): 最小エントロピー関値

        Returns:
            dict: 画像品質評価結果
        """
        from PIL import Image
        if not _NUMPY_AVAILABLE:
            return []
        import math

        try:
            with Image.open(image_path) as img:
                # 解像度チェック
                width, height = img.size
                resolution_score = (
                    width >= min_resolution[0] and 
                    height >= min_resolution[1]
                )

                # エントロピー計算
                img_array = np.array(img.convert('L'))
                hist, _ = np.histogram(img_array, bins=256, range=(0, 256))
                hist = hist[hist > 0] / img_array.size
                entropy = -np.sum(hist * np.log2(hist))

                # 画像の鮮明さ評価
                laplacian = cv2.Laplacian(img_array, cv2.CV_64F).var()

                quality_metrics = {
                    'resolution': {
                        'width': width,
                        'height': height,
                        'meets_minimum': resolution_score
                    },
                    'entropy': {
                        'value': entropy,
                        'meets_threshold': entropy >= min_entropy
                    },
                    'sharpness': {
                        'laplacian_variance': laplacian,
                        'is_sharp': laplacian > 100  # 経験的な関値
                    },
                    'overall_quality': resolution_score and (entropy >= min_entropy) and (laplacian > 100)
                }

                return quality_metrics

        except Exception as e:
            logger.error(f"画像品質評価エラー: {image_path} - {e}")
            return None

    def detect_image_duplicates(self, image_paths, hash_size=8, threshold=0.9):
        """
        知覚的ハッシュを使用した画像重複検出

        Args:
            image_paths (list): 重複を検出する画像パスのリスト
            hash_size (int): ハッシュのサイズ
            threshold (float): 類似度の関値

        Returns:
            list: 重複グループのリスト
        """
        import imagehash
        from PIL import Image

        def calculate_hash(image_path):
            try:
                with Image.open(image_path) as img:
                    return imagehash.phash(img, hash_size=hash_size)
            except Exception as e:
                logger.error(f"ハッシュ計算エラー: {image_path} - {e}")
                return None

        # ハッシュ計算
        image_hashes = {path: calculate_hash(path) for path in image_paths if calculate_hash(path) is not None}

        # 重複検出
        duplicates = []
        processed = set()

        for path1, hash1 in image_hashes.items():
            if path1 in processed:
                continue

            duplicate_group = [path1]
            processed.add(path1)

            for path2, hash2 in image_hashes.items():
                if path1 != path2 and path2 not in processed:
                    similarity = 1 - (hash1 - hash2) / len(hash1.hash)**2
                    if similarity >= threshold:
                        duplicate_group.append(path2)
                        processed.add(path2)

            if len(duplicate_group) > 1:
                duplicates.append(duplicate_group)

        return duplicates

    def configure_network_proxy(self, proxy_config=None):
        """
        ネットワークプロキシの詳細設定

        Args:
            proxy_config (dict, optional): プロキシ設定辞書
                {
                    'type': 'http/https/socks4/socks5',
                    'host': 'プロキシホスト',
                    'port': プロキシポート,
                    'username': オプションのユーザー名,
                    'password': オプションのパスワード
                }

        Returns:
            dict: 設定されたプロキシ情報
        """
        import requests
        from urllib.parse import urlparse

        # デフォルトプロキシ設定
        default_proxy = {
            'type': None,
            'host': None,
            'port': None,
            'username': None,
            'password': None,
            'use_proxy': False
        }

        # プロキシ設定がない場合は対話型設定
        if proxy_config is None:
            try:
                use_proxy = input("プロキシを使用しますか？ (y/N): ").lower() == 'y'
                if use_proxy:
                    proxy_type = input("プロキシタイプ (http/https/socks4/socks5): ").lower()
                    proxy_host = input("プロキシホスト (例: 127.0.0.1): ")
                    proxy_port = int(input("プロキシポート (例: 8080): "))
                    
                    use_auth = input("プロキシ認証が必要ですか？ (y/N): ").lower() == 'y'
                    username = password = None
                    
                    if use_auth:
                        username = input("プロキシユーザー名: ")
                        password = getpass.getpass("プロキシパスワード: ")

                    proxy_config = {
                        'type': proxy_type,
                        'host': proxy_host,
                        'port': proxy_port,
                        'username': username,
                        'password': password
                    }
                else:
                    return default_proxy
            except Exception as e:
                logger.error(f"プロキシ設定エラー: {e}")
                return default_proxy

        # プロキシURLを構築
        proxy_url = f"{proxy_config['type']}://"
        if proxy_config.get('username') and proxy_config.get('password'):
            proxy_url += f"{proxy_config['username']}:{proxy_config['password']}@"
        proxy_url += f"{proxy_config['host']}:{proxy_config['port']}"

        # プロキシ設定を検証
        try:
            proxies = {
                'http': proxy_url,
                'https': proxy_url
            }
            
            # プロキシ接続テスト
            test_url = 'https://www.example.com'
            response = requests.get(test_url, proxies=proxies, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"プロキシ接続成功: {proxy_url}")
                proxy_config['use_proxy'] = True
                return proxy_config
            else:
                logger.warning(f"プロキシ接続テスト失敗: {response.status_code}")
                return default_proxy

        except requests.exceptions.RequestException as e:
            logger.error(f"プロキシ接続エラー: {e}")
            return default_proxy

    def advanced_connection_settings(self, timeout=DEFAULT_TIMEOUT_SECONDS, max_retries=DEFAULT_RETRY_ATTEMPTS, backoff_factor=DEFAULT_BACKOFF_FACTOR):
        """
        高度な接続設定とエラーハンドリング

        Args:
            timeout (int): デフォルト接続タイムアウト（秒）
            max_retries (int): 最大再試行回数
            backoff_factor (float): 再試行間のバックオフ促数

        Returns:
            requests.adapters.HTTPAdapter: カスタマイズされた接続アダプター
        """
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=backoff_factor
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        
        logger.info(f"接続設定: タイムアウト={timeout}秒, 最大再試行={max_retries}, バックオフ促数={backoff_factor}")
        
        return adapter

    def multi_blog_search(self, blogs=None, tags=None, search_params=None):
        """複数ブログ/タグを横断し、条件に合致した画像URLリストを取得"""
        self._ensure_license_for_feature("multi_blog_search")
        from datetime import datetime, timedelta

        default_params = {
            'min_likes': 0,
            'min_notes': 0,
            'date_range': {
                'start': datetime.now() - timedelta(days=DEFAULT_DAYS_BACK),
                'end': datetime.now()
            },
            'content_type': ['photo'],
            'nsfw_filter': True,
            'limit': DEFAULT_PAGE_LIMIT,
            'max_pages': 10
        }

        if search_params:
            default_params.update(search_params)

        blogs = blogs or [self.tumblr_blog]
        tags = tags or []
        results = []
        seen_urls = set()

        def process_posts(posts, params):
            nonlocal results, seen_urls
            for post in posts:
                post_date = post.get('timestamp')
                if post_date is None:
                    continue

                post_datetime = datetime.fromtimestamp(post_date)
                if not (params['date_range']['start'] <= post_datetime <= params['date_range']['end']):
                    continue

                note_count = post.get('note_count', 0)
                if note_count < params['min_notes']:
                    continue

                like_metric = post.get('likes', note_count)
                if like_metric < params['min_likes']:
                    continue

                if params['nsfw_filter'] and post.get('is_nsfw', False):
                    continue

                for photo in post.get('photos', []):
                    url = photo.get('original_size', {}).get('url')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append(url)

        try:
            client = pytumblr.TumblrRestClient(
                self.consumer_key,
                self.consumer_secret,
                self.oauth_token,
                self.oauth_token_secret
            )

            content_types = default_params['content_type'] or ['photo']

            if tags:
                for tag in tags:
                    before_timestamp = int(default_params['date_range']['end'].timestamp())

                    for _ in range(default_params['max_pages']):
                        posts = client.tagged(
                            tag=tag,
                            before=before_timestamp,
                            limit=default_params['limit']
                        )

                        if not posts:
                            break

                        earliest_post = min(posts, key=lambda p: p.get('timestamp', before_timestamp))
                        before_timestamp = earliest_post.get('timestamp', before_timestamp)
                        if before_timestamp:
                            before_timestamp -= 1
                        else:
                            break
            else:
                for blog in blogs:
                    for page in range(default_params['max_pages']):
                        posts = client.posts(
                            blogname=blog,
                            type='photo',
                            limit=default_params['limit'],
                            offset=page * default_params['limit']
                        )
                        posts = posts.get('posts', []) if isinstance(posts, dict) else posts

                        if not posts:
                            break

                        filtered_posts = [
                            post for post in posts
                            if post.get('type', 'photo') in content_types
                        ]

                        process_posts(filtered_posts, default_params)

            logger.info(f"検索結果: {len(results)}件の画像を発見")
            return results

        except Exception as exc:  # pragma: no cover - network dependent
            logger.error(f"マルチブログ検索エラー: {exc}")
            return []

    def advanced_image_search(self, query, search_type='semantic', max_results=100):
        """
        AI支援による高度な画像検索

        Args:
            query (str): 検索クエリ
            search_type (str): 検索タイプ ('semantic', 'tag', 'color', 'style')
            max_results (int): 最大検索結果数

        Returns:
            list: 検索結果の画像ウエブリ
        """
        from PIL import Image
        if not _NUMPY_AVAILABLE:
            return []
        import torch
        from transformers import CLIPProcessor, CLIPModel

        try:
            # CLIP（Contrastive Language-Image Pre-training）モデルの読み込み
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            # 検索タイプに応じた処理
            if search_type == 'semantic':
                # テキストからの意味的検索
                inputs = processor(text=[query], return_tensors="pt", padding=True)
                text_features = model.get_text_features(**inputs)

                # 画像特徴量との類似度計算
                similarity_scores = []
                for image_path in self._get_local_images():
                    img = Image.open(image_path)
                    image_inputs = processor(images=img, return_tensors="pt", padding=True)
                    image_features = model.get_image_features(**image_inputs)
                    
                    # コサイン類似度計算
                    similarity = torch.nn.functional.cosine_similarity(text_features, image_features)
                    similarity_scores.append((image_path, similarity.item()))

                # 類似度でソートして上位を返す
                return [path for path, score in sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:max_results]]

            elif search_type == 'color':
                # 色ベースの検索
                from colorthief import ColorThief
                
                query_color = self._extract_dominant_color(query)
                color_matches = []

                for image_path in self._get_local_images():
                    dominant_color = self._extract_dominant_color(image_path)
                    color_distance = self._color_distance(query_color, dominant_color)
                    color_matches.append((image_path, color_distance))

                return [path for path, distance in sorted(color_matches, key=lambda x: x[1])[:max_results]]

            elif search_type == 'style':
                # スタイル分類による検索
                style_model = self._load_style_classification_model()
                style_matches = []

                for image_path in self._get_local_images():
                    predicted_style = style_model.predict(image_path)
                    if predicted_style == query:
                        style_matches.append(image_path)

                return style_matches[:max_results]

            else:
                logger.warning(f"サポートされていない検索タイプ: {search_type}")
                return []

        except Exception as e:
            logger.error(f"高度な画像検索エラー: {e}")
            return []

    def _get_local_images(self, directory=None, extensions=None):
        """
        ローカルの画像ファイルを取得

        Args:
            directory (str, optional): 検索するディレクトリ。指定しない場合はデフォルトのダウンロードディレクトリ
            extensions (list, optional): 対象とする画像拡張子

        Returns:
            list: 画像ファイルのパスリスト
        """
        import os
        import glob

        # デフォルトディレクトリ設定
        if directory is None:
            directory = os.path.join(os.getcwd(), 'downloads')

        # デフォルト画像拡張子
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']

        # 画像ファイルを再帰的に検索
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(directory, f'**/*{ext}'), recursive=True))

        logger.info(f"{len(image_paths)}件の画像ファイルを発見")
        return image_paths

    def _load_style_classification_model(self):
        """
        スタイル分類のための機械学習モデルを読み込む

        Returns:
            object: 学習済みスタイル分類モデル
        """
        import tensorflow as tf
        if not _NUMPY_AVAILABLE:
            return []
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

        class StyleClassifier:
            def __init__(self, num_classes=10):
                """
                スタイル分類モデルの初期化

                Args:
                    num_classes (int): スタイルカテゴリの数
                """
                base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(DEFAULT_MODEL_INPUT_SIZE, DEFAULT_MODEL_INPUT_SIZE, 3))
                x = base_model.output
                x = GlobalAveragePooling2D()(x)
                x = Dense(DEFAULT_DENSE_LAYER_SIZE, activation='relu')(x)
                output = Dense(num_classes, activation='softmax', name='style_output')(x)
                
                self.model = Model(inputs=base_model.input, outputs=output)
                
                # 基本モデルの重みを凍結
                for layer in base_model.layers:
                    layer.trainable = False

                self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                
                # スタイルラベル
                self.style_labels = [
                    'anime', 'photorealistic', 'illustration', 
                    'digital_art', 'watercolor', 'sketch', 
                    'pixel_art', 'comic', 'abstract', 'minimalist'
                ]

            def preprocess_image(self, image_path):
                """
                画像の前処理

                Args:
                    image_path (str): 画像ファイルのパス

                Returns:
                    numpy.ndarray: 前処理された画像データ
                """
                from PIL import Image
                if not _NUMPY_AVAILABLE:
                    return np.array([])

                img = Image.open(image_path).convert('RGB')
                img = img.resize((DEFAULT_MODEL_INPUT_SIZE, DEFAULT_MODEL_INPUT_SIZE))
                img_array = np.array(img)
                img_array = preprocess_input(img_array)
                return np.expand_dims(img_array, axis=0)

            def predict(self, image_path):
                """
                画像のスタイルを予測

                Args:
                    image_path (str): 画像ファイルのパス

                Returns:
                    str: 予測されたスタイルラベル
                """
                try:
                    preprocessed_img = self.preprocess_image(image_path)
                    predictions = self.model.predict(preprocessed_img)
                    predicted_class_index = np.argmax(predictions)
                    return self.style_labels[predicted_class_index]
                except Exception as e:
                    logger.error(f"スタイル分類エラー: {e}")
                    return None

        # モデルのインスタンス化とキャッシュ
        if not hasattr(self, '_style_classifier'):
            self._style_classifier = StyleClassifier()

        return self._style_classifier

    def advanced_configuration_wizard(self, config_type='full'):
        """
        高度な設定ウィザード

        Args:
            config_type (str): 設定タイプ ('full', 'download', 'processing', 'network')

        Returns:
            dict: 設定された詳細な設定情報
        """
        import os
        import json
        import getpass
        from typing import Dict, Any

        # デフォルト設定テンプレート
        default_config = {
            'tumblr_credentials': {
                'consumer_key': None,
                'consumer_secret': None,
                'oauth_token': None,
                'oauth_token_secret': None
            },
            'download_settings': {
                'output_directory': os.path.join(os.getcwd(), 'downloads'),
                'max_concurrent_downloads': 5,
                'retry_attempts': DEFAULT_RETRY_ATTEMPTS,
                'download_timeout': 60,
                'file_naming_pattern': '{blog}_{timestamp}_{index}'
            },
            'image_processing': {
                'generate_thumbnails': True,
                'thumbnail_size': DEFAULT_THUMBNAIL_SIZE,
                'quality_threshold': {
                    'min_resolution': (800, 600),
                    'min_entropy': 2.0,
                    'max_blur': BLUR_THRESHOLD
                },
                'duplicate_detection': {
                    'method': 'perceptual_hash',
                    'similarity_threshold': 0.9
                }
            },
            'network_settings': {
                'use_proxy': False,
                'proxy_config': {
                    'type': None,
                    'host': None,
                    'port': None,
                    'username': None,
                    'password': None
                },
                'connection_timeout': DEFAULT_TIMEOUT_SECONDS,
                'max_retries': DEFAULT_RETRY_ATTEMPTS
            },
            'logging': {
                'level': 'INFO',
                'file_path': os.path.join(os.getcwd(), 'logs', 'tumblr_collector.log')
            }
        }

        def prompt_tumblr_credentials() -> Dict[str, str]:
            print("\n🔐 Tumblr API 認証情報の設定")
            credentials = {}
            credentials['consumer_key'] = input("Consumer Key を入力: ")
            credentials['consumer_secret'] = getpass.getpass("Consumer Secret を入力: ")
            credentials['oauth_token'] = input("OAuth Token を入力: ")
            credentials['oauth_token_secret'] = getpass.getpass("OAuth Token Secret を入力: ")
            return credentials

        def prompt_download_settings() -> Dict[str, Any]:
            print("\nダウンロード設定")
            settings = {}
            settings['output_directory'] = input(f"出力ディレクトリ (デフォルト: {default_config['download_settings']['output_directory']}): ") or default_config['download_settings']['output_directory']
            settings['max_concurrent_downloads'] = int(input(f"最大同時ダウンロード数 (デフォルト: {default_config['download_settings']['max_concurrent_downloads']}): ") or default_config['download_settings']['max_concurrent_downloads'])
            settings['retry_attempts'] = int(input(f"ダウンロード再試行回数 (デフォルト: {default_config['download_settings']['retry_attempts']}): ") or default_config['download_settings']['retry_attempts'])
            return settings

        def prompt_image_processing() -> Dict[str, Any]:
            print("\n画像処理設定")
            settings = {}
            settings['generate_thumbnails'] = input("サムネイル生成を有効にしますか？ (y/N): ").lower() == 'y'
            if settings['generate_thumbnails']:
                settings['thumbnail_size'] = tuple(map(int, input("サムネイルサイズ (幅,高さ) (デフォルト: 200,200): ").split(',') or DEFAULT_THUMBNAIL_SIZE))
            return settings

        def prompt_network_settings() -> Dict[str, Any]:
            print("\nネットワーク設定")
            settings = {}
            settings['use_proxy'] = input("プロキシを使用しますか？ (y/N): ").lower() == 'y'
            if settings['use_proxy']:
                settings['proxy_config'] = {
                    'type': input("プロキシタイプ (http/https/socks4/socks5): "),
                    'host': input("プロキシホスト: "),
                    'port': input("プロキシポート: "),
                    'username': input("プロキシユーザー名 (オプション): ") or None,
                    'password': getpass.getpass("プロキシパスワード (オプション): ") or None
                }
            return settings

        # 設定タイプに応じた処理
        if config_type == 'full':
            default_config['tumblr_credentials'] = prompt_tumblr_credentials()
            default_config['download_settings'].update(prompt_download_settings())
            default_config['image_processing'].update(prompt_image_processing())
            default_config['network_settings'].update(prompt_network_settings())

        elif config_type == 'download':
            default_config['download_settings'].update(prompt_download_settings())

        elif config_type == 'processing':
            default_config['image_processing'].update(prompt_image_processing())

        elif config_type == 'network':
            default_config['network_settings'].update(prompt_network_settings())

        # 設定の保存
        config_dir = os.path.join(os.getcwd(), 'config')
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, 'tumblr_collector_config.json')

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4, ensure_ascii=False)

        logger.info(f"設定を {config_path} に保存しました。")
        return default_config

    def validate_configuration(self, config_path=None):
        """
        設定ファイルの検証と修復

        Args:
            config_path (str, optional): 設定ファイルのパス

        Returns:
            bool: 設定が有効かどうか
        """
        import os
        import json
        import jsonschema

        # 設定スキーマ定義
        config_schema = {
            "type": "object",
            "properties": {
                "tumblr_credentials": {
                    "type": "object",
                    "required": ["consumer_key", "consumer_secret", "oauth_token", "oauth_token_secret"]
                },
                "download_settings": {
                    "type": "object",
                    "properties": {
                        "max_concurrent_downloads": {"type": "number", "minimum": 1, "maximum": 20},
                        "retry_attempts": {"type": "number", "minimum": 0, "maximum": 10}
                    }
                }
            },
            "required": ["tumblr_credentials", "download_settings"]
        }

        # デフォルトの設定ファイルパス
        if not config_path:
            config_path = os.path.join(os.getcwd(), 'config', 'tumblr_collector_config.json')

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 設定スキーマの検証
            jsonschema.validate(instance=config, schema=config_schema)

            # 追加の検証ロジック
            if not config['tumblr_credentials']['consumer_key']:
                logger.error("Consumer Key が設定されていません。")
                return False

            # ネットワーク設定の検証
            if config.get('network_settings', {}).get('use_proxy'):
                proxy_config = config['network_settings']['proxy_config']
                if not all([proxy_config['type'], proxy_config['host'], proxy_config['port']]):
                    logger.warning("プロキシ設定が不完全です。")

            return True

        except FileNotFoundError:
            logger.error(f"設定ファイルが見つかりません: {config_path}")
            return False
        except jsonschema.exceptions.ValidationError as e:
            logger.error(f"設定ファイルの検証エラー: {e}")
            return False
        except json.JSONDecodeError:
            logger.error(f"設定ファイルの形式が不正です: {config_path}")
            return False

    def auto_image_collection(self, collection_params=None):
        """
        高度な自動画像収集メソッド

        Args:
            collection_params (dict): 画像収集のパラメータ
                {
                    'blogs': ブログリスト,
                    'tags': タグリスト,
                    'max_images': 最大収集画像数,
                    'min_resolution': 最小解像度 (幅, 高さ),
                    'min_likes': 最小いいね数,
                    'min_notes': 最小ノート数,
                    'date_range': {'start': 開始日, 'end': 終了日},
                    'content_type': ['photo', 'illustration', 'art'],
                    'nsfw_filter': ブール値,
                    'style_filters': ['anime', 'photorealistic', 'digital_art'],
                    'download_options': {
                        'output_directory': 出力ディレクトリ,
                        'naming_pattern': ファイル名パターン,
                        'overwrite': 上書きフラグ
                    },
                    'advanced_filters': {
                        'color_palette': 色パレット,
                        'entropy_threshold': エントロピー閾値,
                        'aspect_ratio_range': (最小比率, 最大比率)
                    }
                }

        Returns:
            dict: 収集結果の詳細情報
        """
        import os
        import datetime
        import concurrent.futures
        from urllib.parse import urlparse

        # デフォルトパラメータ
        default_params = {
            'blogs': [self.tumblr_blog],
            'tags': [],
            'max_images': 100,
            'min_resolution': (800, 600),
            'min_likes': 5,
            'min_notes': DEFAULT_COLOR_CLUSTERS,
            'date_range': {
                'start': datetime.datetime.now() - datetime.timedelta(days=DEFAULT_DAYS_BACK),
                'end': datetime.datetime.now()
            },
            'content_type': ['photo'],
            'nsfw_filter': True,
            'style_filters': [],
            'download_options': {
                'output_directory': os.path.join(os.getcwd(), 'downloads', 'auto_collection'),
                'naming_pattern': '{blog}_{timestamp}_{index}',
                'overwrite': False
            },
            'advanced_filters': {
                'color_palette': None,
                'entropy_threshold': 2.0,
                'aspect_ratio_range': (0.5, 2.0)
            }
        }

        # パラメータのマージ
        if collection_params:
            default_params = self._deep_merge(default_params, collection_params)

        # 出力ディレクトリ作成
        os.makedirs(default_params['download_options']['output_directory'], exist_ok=True)

        # 収集結果の初期化
        collection_results = {
            'total_found': 0,
            'downloaded_images': [],
            'skipped_images': [],
            'errors': []
        }

        try:
            # マルチブログ検索
            search_results = self.multi_blog_search(
                blogs=default_params['blogs'],
                tags=default_params['tags'],
                search_params={
                    'min_likes': default_params['min_likes'],
                    'min_notes': default_params['min_notes'],
                    'date_range': default_params['date_range'],
                    'content_type': default_params['content_type'],
                    'nsfw_filter': default_params['nsfw_filter']
                }
            )

            collection_results['total_found'] = len(search_results)

            # 画像処理と並列ダウンロード
            def process_image(image_url):
                try:
                    # 画像の詳細情報取得
                    image_info = self._analyze_image_details(image_url)

                    # 高度なフィルタリング
                    if not self._apply_advanced_filters(image_info, default_params):
                        collection_results['skipped_images'].append(image_url)
                        return None

                    # スタイルフィルタリング
                    if default_params['style_filters']:
                        style_model = self._load_style_classification_model()
                        predicted_style = style_model.predict(image_info['local_path'])
                        if predicted_style not in default_params['style_filters']:
                            collection_results['skipped_images'].append(image_url)
                            return None

                    # ダウンロードと保存
                    filename = self._generate_output_filename(
                        temp_path,
                        metadata,
                        image_url=image_url,
                        post_data={'blog_name': default_params['blog_name']}
                    )

                    if not default_params['download_options']['overwrite'] and os.path.exists(filename):
                        collection_results['skipped_images'].append(image_url)
                        return None

                    # 画像をダウンロード
                    downloaded_path = self.download_image(
                        image_url, 
                        filename, 
                        timeout=DEFAULT_TIMEOUT_SECONDS
                    )

                    collection_results['downloaded_images'].append(downloaded_path)
                    return downloaded_path

                except Exception as e:
                    logger.error(f"画像処理エラー: {image_url}, エラー: {e}")
                    collection_results['errors'].append({
                        'url': image_url,
                        'error': str(e)
                    })
                    return None

            # 並列処理
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(process_image, image_url) 
                    for image_url in search_results[:default_params['max_images']]
                ]
                concurrent.futures.wait(futures)

            return collection_results

        except Exception as e:
            logger.error(f"自動画像収集エラー: {e}")
            return {
                'total_found': 0,
                'downloaded_images': [],
                'skipped_images': [],
                'errors': [{'error': str(e)}]
            }

    def _analyze_image_details(self, image_url):
        import requests
        from PIL import Image
        import io
        if not _NUMPY_AVAILABLE:
            return []

        try:
            response = requests.get(image_url, timeout=10)
            img = Image.open(io.BytesIO(response.content))

            img_array = np.array(img)
            width, height = img.size

            return {
                'url': image_url,
                'width': width,
                'height': height,
                'aspect_ratio': width / height,
                'mode': img.mode,
                'format': img.format,
                'local_path': self._save_temp_image(img)
            }

        except Exception as e:
            logger.error(f"画像詳細分析エラー: {image_url}, エラー: {e}")
            return None

    def _save_temp_image(self, img):
        import os
        import tempfile

        temp_dir = os.path.join(tempfile.gettempdir(), 'tumblr_collector')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f'temp_image_{hash(img)}.png')
        img.save(temp_path)
        return temp_path

    def _apply_advanced_filters(self, image_info, params):
        # 解像度フィルタ
        min_width, min_height = params['min_resolution']
        if image_info['width'] < min_width or image_info['height'] < min_height:
            return False

        # アスペクト比フィルタ
        min_ratio, max_ratio = params['advanced_filters']['aspect_ratio_range']
        if not (min_ratio <= image_info['aspect_ratio'] <= max_ratio):
            return False

        # エントロピーフィルタ
        if params['advanced_filters']['entropy_threshold']:
            entropy = self._calculate_image_entropy(image_info['local_path'])
            if entropy < params['advanced_filters']['entropy_threshold']:
                return False

        # 色パレットフィルタ
        if params['advanced_filters']['color_palette']:
            dominant_color = self._extract_dominant_color(image_info['local_path'])
            if not self._color_matches_palette(dominant_color, params['advanced_filters']['color_palette']):
                return False

        return True

    def _generate_filename(self, image_url, naming_pattern, output_directory):
        import os
        from urllib.parse import urlparse
        from datetime import datetime

        # URLからブログ名を抽出
        parsed_url = urlparse(image_url)
        blog_name = parsed_url.netloc.split('.')[0]

        # パターンを置換
        filename = naming_pattern.format(
            blog=blog_name,
            timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'),
            index=hash(image_url) % 10000
        )

        # 拡張子を追加
        filename = f"{filename}.{image_url.split('.')[-1]}"

        return os.path.join(output_directory, filename)

    def _calculate_image_entropy(self, image_path):
        from PIL import Image
        if not _NUMPY_AVAILABLE:
            return []
        from scipy.stats import entropy

        img = Image.open(image_path)
        img_array = np.array(img)
        
        # グレースケールに変換
        if len(img_array.shape) == 3:
            img_array = np.mean(img_array, axis=2)
        
        # ヒストグラムを計算
        hist, _ = np.histogram(img_array, bins=256)
        hist = hist[hist > 0]
        
        # エントロピーを計算
        return entropy(hist / hist.sum())

    def _extract_dominant_color(self, image_path):
        from PIL import Image
        if not _NUMPY_AVAILABLE:
            return []
        from sklearn.cluster import KMeans

        img = Image.open(image_path)
        img_array = np.array(img)
        
        # 画像をリサイズして処理を高速化
        img_array = img_array.reshape(-1, 3)
        
        # K-meansで支配的な色を抽出
        kmeans = KMeans(n_clusters=1)
        kmeans.fit(img_array)
        
        return tuple(map(int, kmeans.cluster_centers_[0]))

    def _color_matches_palette(self, color, palette, tolerance=COLOR_TOLERANCE):
        if not _NUMPY_AVAILABLE:
            return []

        for palette_color in palette:
            distance = np.sqrt(np.sum((np.array(color) - np.array(palette_color))**2))
            if distance <= tolerance:
                return True
        
        return False

    def resume_image_collection(self, previous_collection_results=None, additional_params=None):
        """
        前回の画像収集を再開または継続するメソッド

        Args:
            previous_collection_results (dict): 前回の収集結果
            additional_params (dict): 追加または上書きするパラメータ

        Returns:
            dict: 更新された収集結果
        """
        import os
        import json
        from datetime import datetime, timedelta

        # デフォルトの再開パラメータ
        resume_params = {
            'resume_from_last_collection': True,
            'skip_downloaded_images': True,
            'extend_date_range': True,
            'max_retry_count': DEFAULT_RETRY_ATTEMPTS,
            'retry_delay': 60  # 秒
        }

        # ユーザーパラメータでデフォルトを上書き
        if additional_params:
            resume_params.update(additional_params)

        # 前回の収集結果がない場合は新規収集を実行
        if not previous_collection_results:
            last_state = self._load_last_collection_state()
            if last_state:
                previous_collection_results = last_state
                self._restore_cli_filters(last_state.get('cli_filters'))
                self._restore_resume_offsets(last_state.get('offsets'))

        if not previous_collection_results:
            logger.warning("前回の収集結果が見つかりません。新規収集を開始します。")
            return self.auto_image_collection()

        # CLIフィルタを復元
        self._restore_cli_filters(previous_collection_results.get('cli_filters'))

        # 収集パラメータを復元
        raw_collection_params = previous_collection_results.get('collection_params', {}) or {}
        collection_params = dict(raw_collection_params)

        if 'date_range' in raw_collection_params and raw_collection_params['date_range']:
            collection_params['date_range'] = dict(raw_collection_params['date_range'])

        stored_tags = collection_params.get('tags') or []
        if stored_tags:
            collection_params['tags'] = [str(tag).lower() for tag in stored_tags]
        elif self._cli_tags:
            collection_params['tags'] = list(self._cli_tags)

        date_range = collection_params.get('date_range')
        if date_range:
            start_value = date_range.get('start')
            end_value = date_range.get('end')

            if isinstance(start_value, str):
                try:
                    date_range['start'] = datetime.datetime.fromisoformat(start_value)
                except ValueError:
                    date_range['start'] = self._cli_start_date
            if isinstance(end_value, str):
                try:
                    date_range['end'] = datetime.datetime.fromisoformat(end_value) if end_value else None
                except ValueError:
                    date_range['end'] = self._cli_end_date
        elif self._cli_start_date or self._cli_end_date:
            collection_params['date_range'] = {
                'start': self._cli_start_date,
                'end': self._cli_end_date
            }

        if 'include_likes' not in collection_params:
            collection_params['include_likes'] = self._include_likes

        # 日付範囲を拡張
        if resume_params['extend_date_range']:
            collection_params['date_range'] = {
                'start': previous_collection_results.get('end_date', 
                    datetime.now() - timedelta(days=DEFAULT_DAYS_BACK)),
                'end': datetime.now()
            }

        # スキップするイメージのリストを作成
        if resume_params['skip_downloaded_images']:
            collection_params['skip_images'] = set(
                previous_collection_results.get('downloaded_images', []) +
                previous_collection_results.get('skipped_images', [])
            )

        # 再試行メカニズム
        retry_count = 0
        while retry_count < resume_params['max_retry_count']:
            try:
                # 画像収集を実行
                new_collection_results = self.auto_image_collection(collection_params)

                # 結果をマージ
                merged_results = self._merge_collection_results(
                    previous_collection_results, 
                    new_collection_results
                )

                # 状態を保存
                self._save_collection_state(merged_results)

                return merged_results

            except Exception as e:
                logger.error(f"収集中にエラーが発生: {e}")
                retry_count += 1
                
                if retry_count < resume_params['max_retry_count']:
                    logger.info(f"再試行 {retry_count}/{resume_params['max_retry_count']}")
                    time.sleep(resume_params['retry_delay'])
                else:
                    logger.error("最大再試行回数に達しました。")
                    return previous_collection_results

        return previous_collection_results

    def _load_last_collection_state(self, state_file='last_collection_state.json'):
        """
        最後の収集状態を読み込む

        Args:
            state_file (str): 状態ファイルのパス

        Returns:
            dict: 前回の収集状態、見つからない場合はNone
        """
        import os
        import json

        state_path = os.path.join(os.getcwd(), 'downloads', state_file)
        
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"状態ファイルの読み込みエラー: {e}")
        
        return None

    def _save_collection_state(self, collection_results, state_file='last_collection_state.json'):
        """
        収集状態を保存する

        Args:
            collection_results (dict): 収集結果
            state_file (str): 状態ファイルのパス
        """
        import os
        import json
        from datetime import datetime

        # ダウンロードディレクトリを作成
        os.makedirs(os.path.join(os.getcwd(), 'downloads'), exist_ok=True)
        
        state_path = os.path.join(os.getcwd(), 'downloads', state_file)
        
        try:
            # 追加のメタデータを含める
            collection_results['saved_timestamp'] = datetime.now().isoformat()
            
            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(collection_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"収集状態を {state_path} に保存しました。")
        
        except Exception as e:
            logger.error(f"状態ファイルの保存エラー: {e}")

    def _merge_collection_results(self, previous_results, new_results):
        """
        複数の収集結果をマージする

        Args:
            previous_results (dict): 以前の収集結果
            new_results (dict): 新しい収集結果

        Returns:
            dict: マージされた収集結果
        """
        merged_results = {
            'total_found': previous_results.get('total_found', 0) + new_results.get('total_found', 0),
            'downloaded_images': list(set(
                previous_results.get('downloaded_images', []) + 
                new_results.get('downloaded_images', [])
            )),
            'skipped_images': list(set(
                previous_results.get('skipped_images', []) + 
                new_results.get('skipped_images', [])
            )),
            'errors': (
                previous_results.get('errors', []) + 
                new_results.get('errors', [])
            ),
            'collection_params': previous_results.get('collection_params', {})
        }

        # 最新の収集パラメータで更新
        merged_results['collection_params'].update(new_results.get('collection_params', {}))

        merged_results['cli_filters'] = (
            new_results.get('cli_filters')
            if new_results.get('cli_filters') is not None
            else previous_results.get('cli_filters')
        )

        merged_results['offsets'] = {
            **(previous_results.get('offsets') or {}),
            **(new_results.get('offsets') or {})
        }

        return merged_results

    def export_metadata(self, output_format='json'):
        """画像メタデータをエクスポートする

        Args:
            output_format (str, optional): エクスポート形式. デフォルトは 'json'.
                サポートされる形式: 'json', 'csv'

        Returns:
            Path: 生成されたメタデータファイルのパス
        """
        import json
        import csv
        from pathlib import Path
        from PIL import Image
        import datetime
        import os

        # メタデータディレクトリを作成
        metadata_dir = self.output_folder / 'metadata'
        metadata_dir.mkdir(exist_ok=True)

        # メタデータファイル名を生成
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        metadata_filename = f'tumblr_metadata_{timestamp}'

        # メタデータリストを作成
        metadata_list = []
        for filepath in self.output_folder.glob('*.*'):
            if filepath.is_file() and filepath.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                try:
                    # 画像メタデータを抽出
                    metadata = self._extract_image_metadata(str(filepath))
                    
                    if metadata:
                        # 追加のファイル情報を含める
                        metadata['local_path'] = str(filepath)
                        metadata['file_size'] = filepath.stat().st_size
                        metadata['last_modified'] = filepath.stat().st_mtime
                        
                        # 画像の追加情報
                        with Image.open(filepath) as img:
                            metadata['width'], metadata['height'] = img.size
                            metadata['format'] = img.format
                        
                        metadata_list.append(metadata)
                
                except Exception as e:
                    logger.error(f"メタデータ抽出エラー: {filepath} - {e}")

        # エクスポート形式を選択
        if output_format == 'json':
            output_file = metadata_dir / f'{metadata_filename}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, ensure_ascii=False, indent=4)
            logger.info(f"メタデータをJSONファイルにエクスポート: {output_file}")
            return output_file

        elif output_format == 'csv':
            output_file = metadata_dir / f'{metadata_filename}.csv'
            # CSVに出力可能な形式に変換
            if metadata_list:
                # すべてのキーを取得（フラット化）
                keys = set()
                for item in metadata_list:
                    keys.update(item.keys())
                
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=sorted(list(keys)))
                    writer.writeheader()
                    writer.writerows(metadata_list)
            
            logger.info(f"メタデータをCSVファイルにエクスポート: {output_file}")
            return output_file

        else:
            logger.warning(f"サポートされていないフォーマット: {output_format}. JSONにフォールバックします。")
            output_file = metadata_dir / f'{metadata_filename}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, ensure_ascii=False, indent=4)
            return output_file

    def download_tagged_images(self, tag, count=DEFAULT_COLOR_CLUSTERS):
        """指定されたタグの画像と動画をダウンロードする (高度な並列ダウンロード)"""
        if not self.executor:
            self._setup_executor()

        rate_limit_suspected = False
        download_results = {
            'total_images': 0,
            'total_videos': 0,
            'successful_downloads': 0,
            'failed_downloads': 0
        }
        results_lock = threading.Lock()

        try:
            limit = min(max(1, int(count or 1)), self.max_download_limit)
            posts = self.client.tagged(tag, limit=limit)

            download_queue = queue.Queue()
            for post in posts:
                download_queue.put(post)

            if download_queue.empty():
                logger.info(f"No posts found for tag '{tag}'.")
                return True

            def download_worker():
                while True:
                    try:
                        post = download_queue.get_nowait()
                    except queue.Empty:
                        break

                    try:
                        post_type = post.get('type')
                        if post_type == 'photo':
                            # 画像ダウンロード
                            for photo in post.get('photos', []):
                                image_url = photo.get('original_size', {}).get('url')
                                if image_url:
                                    success = self.download_image(image_url, post)
                                    with results_lock:
                                        download_results['total_images'] += 1
                                        if success:
                                            download_results['successful_downloads'] += 1
                                        else:
                                            download_results['failed_downloads'] += 1
                        elif post_type == 'video':
                            # 動画ダウンロード
                            video_url = post.get('video_url')
                            if video_url:
                                success = self.download_video(video_url, post)
                                with results_lock:
                                    download_results['total_videos'] += 1
                                    if success:
                                        download_results['successful_downloads'] += 1
                                    else:
                                        download_results['failed_downloads'] += 1
                            else:
                                # video_urlがない場合、player配列から取得を試行
                                players = post.get('player', [])
                                if players:
                                    # 最も高画質のプレイヤーを選択
                                    player = max(players, key=lambda p: p.get('width', 0))
                                    embed_code = player.get('embed_code', '')
                                    # embed_codeからvideo_urlを抽出（正規表現使用）
                                    import re
                                    video_match = re.search(r'src="([^"]*\.mp4[^"]*)"', embed_code)
                                    if video_match:
                                        video_url = video_match.group(1)
                                        success = self.download_video(video_url, post)
                                        with results_lock:
                                            download_results['total_videos'] += 1
                                            if success:
                                                download_results['successful_downloads'] += 1
                                            else:
                                                download_results['failed_downloads'] += 1
                    except Exception as worker_error:
                        logger.error(f"Error in download worker: {worker_error}")
                    finally:
                        download_queue.task_done()

            workers = []
            worker_count = min(self.max_workers, max(1, download_queue.qsize()))
            for _ in range(worker_count):
                thread = threading.Thread(target=download_worker, daemon=True)
                thread.start()
                workers.append(thread)

            download_queue.join()
            for thread in workers:
                thread.join()

            logger.info(
                f"Tag '{tag}' download summary: images={download_results['total_images']}, "
                f"videos={download_results['total_videos']}, "
                f"success={download_results['successful_downloads']}, "
                f"failed={download_results['failed_downloads']}"
            )

        except Exception as e:
            logger.error(f"Error during tagged post fetch/submission for tag '{tag}': {e}")
            if "limit" in str(e).lower() or "429" in str(e) or "too many requests" in str(e).lower():
                logger.warning(f"Rate limit likely hit while fetching tagged posts for '{tag}'.")
                rate_limit_suspected = True

        return not rate_limit_suspected

    def _apply_cli_post_filters(self, posts):
        """CLIオプションに基づき投稿リストをフィルタリング"""
        if not posts:
            return []

        filtered = []
        for post in posts:
            if self._post_matches_cli_filters(post):
                filtered.append(post)
        return filtered

    def _post_matches_cli_filters(self, post):
        """CLIフィルタ条件を満たすか判定"""
        if not isinstance(post, dict):
            return False

        if self._cli_tags:
            post_tags = [str(tag).lower() for tag in post.get('tags', [])]
            if not any(tag in post_tags for tag in self._cli_tags):
                return False

        timestamp = post.get('timestamp')
        if (self._cli_start_date or self._cli_end_date) and timestamp is not None:
            post_datetime = datetime.datetime.fromtimestamp(timestamp)
            if self._cli_start_date and post_datetime < self._cli_start_date:
                return False
            if self._cli_end_date and post_datetime > self._cli_end_date:
                return False
        elif (self._cli_start_date or self._cli_end_date) and timestamp is None:
            # タイムスタンプが無い場合はフィルタに一致しないと見なす
            return False

        return True

    def process_posts(self, posts):
        """投稿から画像や動画をダウンロードし、関連タグの画像もダウンロードする (並列ダウンロード)"""
        if not self.executor:
            logger.error("Executor not available for processing posts.")
            return True

        if not posts:
            logger.debug("No posts to process after applying filters.")
            return True

        initial_media_futures = []
        rate_limit_hit = False

        # Submit initial media downloads
        submitted_count = 0
        max_initial_downloads = min(len(posts), self.max_download_limit)

        # Submit initial media downloads
        for post in posts:
            if submitted_count >= max_initial_downloads:
                logger.debug("Reached max_download_limit for this batch; skipping remaining posts.")
                break
            
            post_type = post.get('type')
            if post_type == 'photo':
                # 画像投稿の処理
                for photo in post.get('photos', []):
                    image_url = photo.get('original_size', {}).get('url')
                    if image_url:
                        future = self.executor.submit(self.download_image, image_url, post)
                        initial_media_futures.append((future, post.get('tags', []), 'image'))
                        submitted_count += 1
            elif post_type == 'video':
                # 動画投稿の処理
                video_url = post.get('video_url')
                if video_url:
                    future = self.executor.submit(self.download_video, video_url, post)
                    initial_media_futures.append((future, post.get('tags', []), 'video'))
                    submitted_count += 1
                else:
                    # video_urlがない場合、player配列から取得を試行
                    players = post.get('player', [])
                    if players:
                        # 最も高画質のプレイヤーを選択
                        player = max(players, key=lambda p: p.get('width', 0))
                        embed_code = player.get('embed_code', '')
                        # embed_codeからvideo_urlを抽出（正規表現使用）
                        import re
                        video_match = re.search(r'src="([^"]*\.mp4[^"]*)"', embed_code)
                        if video_match:
                            video_url = video_match.group(1)
                            future = self.executor.submit(self.download_video, video_url, post)
                            initial_media_futures.append((future, post.get('tags', []), 'video'))
                            submitted_count += 1
        
        logger.debug(f"Submitted {submitted_count} initial media download tasks for this batch.")

        # Process results as they complete
        processed_count = 0
        for future, tags, media_type in initial_media_futures:
            if rate_limit_hit:
                future.cancel()
                continue

            try:
                media_filename = future.result()  # Wait for initial download
                processed_count += 1

                if media_filename:
                    logger.info(f"({processed_count}/{submitted_count}) Downloaded initial {media_type}: {media_filename}")

                    # タグ処理の最適化と並列化
                    related_tags_successful = self._process_related_tags(media_filename, tags)

                    if not related_tags_successful:
                        rate_limit_hit = True
                        continue

            except concurrent.futures.CancelledError:
                logger.debug("Initial media download cancelled.")
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing result for an initial media download: {e}")
                processed_count += 1  # Count as processed even if errored

        logger.debug(f"Finished processing results for {submitted_count} initial media.")
        return not rate_limit_hit

    def _collect_liked_posts(self, batch_size=20):
        """認証ユーザーのLike投稿を収集"""
        if not hasattr(self.client, 'likes'):
            logger.warning("Tumblr client does not support likes API. Skipping like collection.")
            return

        logger.info("Starting liked-post collection...")

        offset = 0
        batch_size = min(max(1, int(batch_size)), self.max_download_limit)

        while True:
            try:
                response = self.client.likes(limit=batch_size, offset=offset)
            except Exception as exc:
                logger.error(f"Error fetching liked posts: {exc}")
                break

            liked_posts = []
            if isinstance(response, dict):
                liked_posts = response.get('liked_posts', []) or []
            elif isinstance(response, (list, tuple)):
                liked_posts = list(response)

            if not liked_posts:
                logger.info("No more liked posts to process.")
                break

            filtered_posts = self._apply_cli_post_filters(liked_posts)

            if filtered_posts:
                logger.info(f"Processing {len(filtered_posts)} liked posts (offset {offset}).")
                if not self.process_posts(filtered_posts):
                    self.wait_and_resume()
                    continue
            else:
                logger.debug(f"All liked posts in batch {offset} skipped by filters.")

            offset += len(liked_posts)

            if len(liked_posts) < batch_size:
                logger.info("Reached end of liked posts (last batch smaller than requested).")
                break

        logger.info("Finished liked-post collection.")
        self._persist_runtime_state()

    def _process_related_tags(self, image_filename, tags):
        """関連タグの処理を分離したメソッド"""
        if not tags:
            logger.info(f"No tags found for post containing image: {image_filename}")
            return True

        logger.debug(f"Processing {len(tags)} related tags for {image_filename}...")

        # 関連タグの画像をダウンロード
        tag_processing_results = {'successful_tags': 0, 'failed_tags': 0}

        for tag in tags[:DEFAULT_COLOR_CLUSTERS]:  # 最大3個のタグを処理
            try:
                result = self.download_tagged_images(tag, count=DEFAULT_COLOR_CLUSTERS)
                if result:
                    tag_processing_results['successful_tags'] += 1
                    logger.debug(f"Successfully processed tag: {tag}")
                else:
                    tag_processing_results['failed_tags'] += 1
                    logger.warning(f"Failed to process tag: {tag}")

            except Exception as e:
                tag_processing_results['failed_tags'] += 1
                logger.warning(f"Error processing tag {tag}: {e}")

        # 処理結果のサマリーをログ出力
        logger.info(
            f"Tag Processing Summary: "
            f"Successful: {tag_processing_results['successful_tags']}, "
            f"Failed: {tag_processing_results['failed_tags']}"
        )

        return True

    def wait_and_resume(self):
        """API制限時の待機処理"""
        sleep_time = int(self.api_wait_hours * 60 * 60)
        logger.warning(f"API rate limit hit. Waiting for {self.api_wait_hours} hours ({sleep_time} seconds)...")

        # クラッシュレポートを生成
        self._generate_crash_report(None, None, "Rate limit hit")

        # 待機中のプログレスバーとカウントダウン
        for remaining in range(sleep_time, 0, -1):
            logger.debug(f"Waiting... {remaining} seconds remaining.")
            logger.info(_("initialization_complete"))
        logger.info(_("starting_download"))  # after rate limit wait

    def batch_blog_download(self, blog_names, common_params=None, max_concurrent_blogs=3):
        """
        複数のブログを並列でダウンロードするバッチ処理
        
        Args:
            blog_names (list): ダウンロードするブログ名のリスト
            common_params (dict): 全ブログに共通するパラメータ
                {
                    'tags': タグリスト,
                    'date_range': {'start': 開始日, 'end': 終了日},
                    'include_likes': ブール値,
                    'workers': ワーカー数,
                    'output_dir': 出力ディレクトリ
                }
            max_concurrent_blogs (int): 同時に処理するブログの最大数
            
        Returns:
            dict: バッチ処理結果
        """
        if not blog_names:
            logger.warning("ブログ名が指定されていません")
            return {'error': 'No blogs specified'}
        
        # デフォルトパラメータ
        default_params = {
            'tags': [],
            'date_range': None,
            'include_likes': False,
            'workers': 5,
            'output_dir': None
        }
        
        if common_params:
            default_params.update(common_params)
        
        logger.info(f"バッチダウンロード開始: {len(blog_names)}ブログ, 最大同時処理数: {max_concurrent_blogs}")
        
        results = {}
        semaphore = threading.Semaphore(max_concurrent_blogs)
        
        def download_single_blog(blog_name):
            """単一ブログのダウンロード処理"""
            with semaphore:
                try:
                    logger.info(f"ブログ '{blog_name}' の処理を開始")
                    
                    # ブログ固有の出力ディレクトリ
                    output_dir = default_params.get('output_dir')
                    if output_dir:
                        blog_output_dir = Path(output_dir) / blog_name
                        blog_output_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        blog_output_dir = None
                    
                    # コレクターの初期化（ブログ固有の設定）
                    blog_collector = TumblrImageCollector(
                        output_dir_override=str(blog_output_dir) if blog_output_dir else None,
                        workers_override=default_params.get('workers', 5)
                    )
                    
                    # ブログの実行
                    blog_collector.run(
                        blog_name=blog_name,
                        tags=default_params.get('tags', []),
                        date_range=default_params.get('date_range'),
                        include_likes=default_params.get('include_likes', False)
                    )
                    
                    results[blog_name] = {
                        'status': 'completed',
                        'stats': blog_collector._download_stats,
                        'error': None
                    }
                    logger.info(f"ブログ '{blog_name}' の処理が完了")
                    
                except Exception as e:
                    logger.error(f"ブログ '{blog_name}' の処理中にエラー: {e}")
                    results[blog_name] = {
                        'status': 'failed',
                        'stats': None,
                        'error': str(e)
                    }
        
        # 並列実行
        threads = []
        for blog_name in blog_names:
            thread = threading.Thread(target=download_single_blog, args=(blog_name,))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # 全スレッドの完了を待つ
        for thread in threads:
            thread.join()
        
        # 結果の集計
        successful = sum(1 for r in results.values() if r['status'] == 'completed')
        failed = sum(1 for r in results.values() if r['status'] == 'failed')
        
        logger.info(f"バッチダウンロード完了: 成功 {successful}, 失敗 {failed}")
        
        return {
            'total_blogs': len(blog_names),
            'successful': successful,
            'failed': failed,
            'results': results
        }
        """メインの実行ループ"""
        self._cli_tags = [str(tag).lower() for tag in (tags or [])]
        self._cli_start_date = (date_range or {}).get('start')
        self._cli_end_date = (date_range or {}).get('end')
        self._include_likes = bool(include_likes)

        offset_key = f"offset_{blog_name}" if blog_name else None
        offset = self.config.get(offset_key, 0) if offset_key else 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as self.executor:
            if blog_name:
                while True:
                    try:
                        logger.info(f"Fetching posts for '{blog_name}' with offset {offset}...")
                        posts = self.get_blog_posts(blog_name, limit=20, offset=offset)

                        if posts is None:  # Rate limit suspected
                            self.wait_and_resume()
                            continue

                        if not posts:  # No more posts or non-rate-limit error
                            logger.info("No more posts retrieved or error fetching posts.")
                            if offset_key:
                                self.config[offset_key] = offset  # Save current offset on clean exit
                                self._save_config()
                            break

                        filtered_posts = self._apply_cli_post_filters(posts)

                        # Process posts
                        if not self.process_posts(filtered_posts):
                            # Rate limit hit during processing
                            self.wait_and_resume()
                            continue  # Retry same offset

                        # Success, advance offset
                        processed_count = len(posts)
                        offset += processed_count
                        if offset_key:
                            self.config[offset_key] = offset  # Save progress
                            self._save_config()
                        self._persist_runtime_state(blog_name=blog_name, offset=offset)
                        logger.info(f"Successfully processed batch. New offset for '{blog_name}': {offset}")

                        # Wait before fetching next batch
                        logger.debug(f"Sleeping for {self.api_batch_sleep} seconds before next batch...")
                        time.sleep(self.api_batch_sleep)

                    except Exception as e:
                        logger.error(f"Unexpected error in download process: {e}")
                        self._generate_crash_report(type(e), e, sys.exc_info()[2])
                        break
                    except KeyboardInterrupt:
                        logger.info("ユーザーによる中断を検出しました")
                        if offset_key:
                            self.config[offset_key] = offset  # 現在のオフセットを保存
                            self._save_config()
                        break

            if self._include_likes:
                self._collect_liked_posts(batch_size=20)
                self._persist_runtime_state(blog_name=blog_name, offset=offset)

        self.executor = None

        # クリーンアップ
        self._cleanup_resources()
        if blog_name:
            logger.info(f"Download process finished for blog '{blog_name}'.")
        if self._include_likes and not blog_name:
            logger.info("Download process finished for liked posts.")

        # 最終統計情報を表示
        self._display_final_statistics()

    def _persist_runtime_state(self, blog_name=None, offset=None):
        current_time_iso = datetime.datetime.now().isoformat()

        collection_params = {
            'blogs': [blog_name] if blog_name else [],
            'tags': list(self._cli_tags),
            'date_range': {
                'start': self._cli_start_date.isoformat() if self._cli_start_date else None,
                'end': self._cli_end_date.isoformat() if self._cli_end_date else None
            },
            'include_likes': self._include_likes
        }

        state_payload = {
            'total_found': self._download_stats.get('total_images_processed', 0),
            'downloaded_images': sorted(self.downloaded_files),
            'skipped_images': [],
            'errors': [],
            'collection_params': collection_params,
            'offsets': {blog_name: offset} if blog_name is not None else {},
            'cli_filters': self._serialize_cli_filters(),
            'end_date': current_time_iso
        }

        self._save_collection_state(state_payload)
        if blog_name and offset is not None:
            self.config[f"offset_{blog_name}"] = offset
            self._save_config()

    def _restore_resume_offsets(self, offsets):
        if not offsets:
            return
        for blog_name, offset in offsets.items():
            if blog_name:
                self.config[f"offset_{blog_name}"] = offset

    def _display_final_statistics(self):
        """最終統計情報を表示"""
        stats = self._download_stats
        logger.info("-" * 60)
        logger.info("ダウンロード統計:")
        logger.info(f"  総処理画像数: {stats['total_images_processed']}")
        logger.info(f"  成功ダウンロード数: {stats['successful_downloads']}")
        logger.info(f"  スキップ数 (重複): {stats['skipped_duplicates']}")
        logger.info(f"  失敗数: {stats['failed_downloads']}")

        if self.image_classifier and stats.get('ai_classification_stats'):
            ai_stats = stats['ai_classification_stats']
            logger.info("画像分類統計:")
            logger.info(f"  有効画像: {ai_stats['valid_images']}")
            logger.info(f"  無効画像: {ai_stats['invalid_images']}")
            logger.info(f"  高解像度画像: {ai_stats['high_resolution_images']}")
        logger.info("-" * 60)


def main():
    """Parses arguments, sets up logging, creates collector, and runs it."""
    parser = argparse.ArgumentParser(description="Download images from Tumblr blogs, likes, tags, and date ranges.")
    parser.add_argument("blog_name", nargs='?', help="The name of the Tumblr blog (e.g., staff)")
    parser.add_argument("-c", "--config", default=constants.DEFAULT_CONFIG_FILE,
                        help=f"Path to the configuration file (default: {constants.DEFAULT_CONFIG_FILE})")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory name or path (overrides config file setting)")
    parser.add_argument("-w", "--workers", type=int, default=None,
                        help="Number of download workers (overrides config file setting)")
    parser.add_argument("-l", "--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (default: INFO)")
    parser.add_argument("--log-file", default=constants.DEFAULT_LOG_FILE,
                        help=f"Path to the log file (default: {constants.DEFAULT_LOG_FILE})")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--list-plans", action="store_true",
                        help="List configured Stripe billing plans and exit")
    parser.add_argument("--create-checkout", metavar="PLAN_KEY",
                        help="Create a Stripe Checkout session for the specified plan key")
    parser.add_argument("--checkout-email", metavar="EMAIL",
                        help="Attach customer email to the Stripe Checkout session")
    parser.add_argument("--open-checkout", action="store_true",
                        help="Open the generated Stripe Checkout URL in the default browser")
    parser.add_argument("--include-likes", action="store_true",
                        help="Download liked posts for the authenticated user")
    parser.add_argument("--tags", nargs='*',
                        help="Filter downloads to posts containing the specified tags")
    parser.add_argument("--start-date", type=str,
                        help="ISO date (YYYY-MM-DD). Only posts created on or after this date will be considered")
    parser.add_argument("--end-date", type=str,
                        help="ISO date (YYYY-MM-DD). Only posts created on or before this date will be considered")
    parser.add_argument("--gui", action="store_true",
                        help="Launch graphical user interface")
    parser.add_argument("--batch-blogs", nargs='*',
                        help="Download from multiple blogs in batch mode (space separated blog names)")
    parser.add_argument("--max-concurrent-blogs", type=int, default=3,
                        help="Maximum number of blogs to process concurrently in batch mode (default: 3)")

    args = parser.parse_args()

    # --- Configure Logging ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File Handler
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    logger.setLevel(log_level)

    # --- Interactive Mode ---
    if args.interactive and (args.list_plans or args.create_checkout):
        logger.error("Billing management options cannot be combined with --interactive mode.")
        return

    if args.interactive:
        cli = InteractiveCLI()
        try:
            collector = TumblrImageCollector(
                config_file=args.config,
                output_dir_override=args.output,
                workers_override=args.workers
            )
            cli.run_interactive_mode(collector)
        except KeyboardInterrupt:
            logger.info("Process interrupted by user.")
        except Exception as e:
            logger.exception(f"An unexpected error occurred: {e}")
        finally:
            if 'collector' in locals() and hasattr(collector, 'config'):
                logger.info("Saving final configuration state.")
                collector._save_config()
            logger.info("Exiting.")
        return

    if args.list_plans or args.create_checkout:
        checkout_plan_key = None
        if args.create_checkout is not None:
            checkout_plan_key = args.create_checkout.strip()
            if not checkout_plan_key:
                logger.error("Plan key for --create-checkout must be non-empty.")
                return

        collector = None
        try:
            collector = TumblrImageCollector(
                config_file=args.config,
                output_dir_override=args.output,
                workers_override=args.workers
            )
        except ConfigurationError as exc:
            logger.error(f"Failed to initialize TumblrImageCollector for billing operations: {exc}")
            return
        except Exception as exc:
            logger.exception(f"Unexpected error during billing initialization: {exc}")
            return

        try:
            if args.list_plans:
                try:
                    plans = collector.list_billing_plans()
                except ConfigurationError as exc:
                    logger.error(f"Unable to list billing plans: {exc}")
                    return

                if not plans:
                    print("Stripe課金プランが設定されていません。\nNo Stripe billing plans configured.")
                else:
                    print("利用可能なStripe課金プラン / Available Stripe Billing Plans:")
                    for key, plan in plans.items():
                        print(f"\nプランキー / Key: {key}")
                        print(f"  名称 / Name: {plan.get('name', '-')}")
                        recurring = bool(plan.get('recurring', False))
                        billing_type = "定期課金 / Subscription" if recurring else "単発支払い / One-time"
                        print(f"  課金種別 / Billing Type: {billing_type}")
                        period = plan.get('billing_period')
                        if recurring:
                            print(f"  課金周期 / Billing Period: {period or '不明 / Unspecified'}")
                        features = plan.get('features') or []
                        if features:
                            print("  主な機能 / Features:")
                            for feature in features:
                                print(f"    - {feature}")

            if checkout_plan_key:
                try:
                    session = collector.create_checkout_session(
                        plan_key=checkout_plan_key,
                        customer_email=args.checkout_email
                    )
                except (ConfigurationError, ValueError) as exc:
                    logger.error(f"Failed to create Stripe Checkout session: {exc}")
                    return

                url = session.get('url') or ''
                print("Stripe Checkoutセッションを作成しました。")
                print("Checkout session created.")
                print(f"  セッションID / Session ID: {session.get('id', '-')}")
                print(f"  プランキー / Plan Key: {session.get('plan_key', '-')}")
                if url:
                    print(f"  URL: {url}")
                    if args.open_checkout:
                        try:
                            webbrowser.open(url, new=2)
                        except Exception as exc:
                            logger.warning(f"Failed to open browser for Stripe Checkout: {exc}")
                else:
                    logger.warning("Checkout session URL was not returned by Stripe.")
        finally:
            if collector and hasattr(collector, 'config'):
                logger.info("Saving final configuration state.")
                collector._save_config()
        return

    if args.gui:
        # GUIモード
        try:
            from gui import TumblrCollectorGUI
            import tkinter as tk
            root = tk.Tk()
            app = TumblrCollectorGUI(root)
            app.run()
        except ImportError as e:
            logger.error(f"GUIモードに必要なモジュールがインストールされていません: {e}")
            logger.error("Tkinterが利用できない場合は 'pip install tk' を実行してください")
            return
        except Exception as e:
            logger.exception(f"GUI起動エラー: {e}")
            return

    if not args.gui:
        # CLIモード
        if not args.blog_name and not args.include_likes:
            logger.error("Blog name is required unless --include-likes is specified. Use --interactive for interactive mode.")
            return

        def _parse_date(date_text):
            if not date_text:
                return None
            try:
                return datetime.datetime.strptime(date_text, "%Y-%m-%d")
            except ValueError:
                logger.error(f"Invalid date format for '{date_text}'. Expected YYYY-MM-DD.")
                return None

        start_date = _parse_date(args.start_date)
        end_date = _parse_date(args.end_date)
        if start_date and end_date and start_date > end_date:
            logger.warning("Start date is after end date. Swapping values.")
            start_date, end_date = end_date, start_date

        cli_tags = args.tags or []

        # バッチモードの処理
        if args.batch_blogs:
            logger.info(f"バッチモードで {len(args.batch_blogs)} ブログを処理します")
            
            # バッチ共通パラメータ
            batch_params = {
                'tags': cli_tags,
                'date_range': {'start': start_date, 'end': end_date} if start_date or end_date else None,
                'include_likes': args.include_likes,
                'workers': args.workers,
                'output_dir': args.output
            }
            
            # バッチコレクターの初期化
            batch_collector = TumblrImageCollector(
                config_file=args.config,
                output_dir_override=None,  # バッチではブログごとのディレクトリを作成
                workers_override=1  # バッチレベルでのワーカー制御
            )
            
            # バッチダウンロード実行
            batch_result = batch_collector.batch_blog_download(
                blog_names=args.batch_blogs,
                common_params=batch_params,
                max_concurrent_blogs=args.max_concurrent_blogs
            )
            
            logger.info("バッチダウンロード結果:")
            logger.info(f"  全ブログ数: {batch_result['total_blogs']}")
            logger.info(f"  成功: {batch_result['successful']}")
            logger.info(f"  失敗: {batch_result['failed']}")
            
        else:
            # 通常の単一ブログ処理
            collector = TumblrImageCollector(
                config_file=args.config,
                output_dir_override=args.output,
                workers_override=args.workers
            )

            date_filter = None
            if start_date or end_date:
                date_filter = {'start': start_date, 'end': end_date}

            collector.run(
                blog_name=args.blog_name,
                tags=cli_tags,
                date_range=date_filter,
                include_likes=args.include_likes
            )
        except (ValueError, ConnectionError, IOError) as e:
            logger.error(f"Initialization or runtime error: {e}")
            # No need to save config if initialization failed badly
        except KeyboardInterrupt:
            logger.info("Process interrupted by user.")
        except Exception as e:
            logger.exception(f"An unexpected error occurred: {e}") # Log full traceback
        finally:
            # Ensure config is saved on exit, especially if interrupted during run
            if 'batch_collector' in locals() and batch_collector and hasattr(batch_collector, 'config'):
                logger.info("Saving final configuration state.")
                batch_collector._save_config()
            elif 'collector' in locals() and collector and hasattr(collector, 'config'):
                logger.info("Saving final configuration state.")
                collector._save_config()
            logger.info("Exiting.")


    def _check_robots_txt(self, blog_url):
        """
        robots.txtを確認してクローリングポリシーをチェック

        Args:
            blog_url (str): チェック対象のブログURL

        Returns:
            dict: robots.txtの解析結果
        """
        try:
            from urllib.parse import urlparse
            from urllib.robotparser import RobotFileParser

            parsed = urlparse(blog_url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

            # robots.txtパーサーの初期化
            rp = RobotFileParser()
            rp.set_url(robots_url)

            # タイムアウト付きでrobots.txtを取得
            try:
                rp.read()
            except Exception as e:
                logger.warning(f"robots.txtの取得に失敗: {e}")
                return {
                    'can_fetch': True,  # 取得できない場合は許可とみなす
                    'crawl_delay': None,
                    'disallowed_paths': []
                }

            # Tumblrのユーザーエージェントでチェック
            user_agent = 'TumblrImageCollector/1.0'
            can_fetch = rp.can_fetch(user_agent, blog_url)

            # クロール遅延を取得（秒単位）
            crawl_delay = rp.crawl_delay(user_agent)
            if crawl_delay:
                crawl_delay = float(crawl_delay)

            # 禁止パスの取得
            disallowed_paths = []
            for rule in rp.entries:
                if rule.useragent in ['*', user_agent]:
                    disallowed_paths.extend(rule.rulelist)

            return {
                'can_fetch': can_fetch,
                'crawl_delay': crawl_delay,
                'disallowed_paths': disallowed_paths,
                'robots_url': robots_url
            }

        except Exception as e:
            logger.error(f"robots.txtチェックエラー: {e}")
            return {
                'can_fetch': True,
                'crawl_delay': None,
                'disallowed_paths': []
            }

    def _respect_robots_txt(self, blog_url):
        """
        robots.txtのポリシーを尊重して処理を調整

        Args:
            blog_url (str): チェック対象のブログURL

        Returns:
            bool: クローリングを継続可能かどうか
        """
        try:
            robots_info = self._check_robots_txt(blog_url)

            if not robots_info['can_fetch']:
                logger.warning(f"robots.txtによりクローリングが禁止されています: {blog_url}")
                return False

            # クロール遅延を適用
            if robots_info['crawl_delay']:
                delay = robots_info['crawl_delay']
                logger.info(f"robots.txtで指定されたクロール遅延を適用: {delay}秒")
                # レート制限にクロール遅延を反映
                if hasattr(self, '_rate_limiter'):
                    self._rate_limiter['requests_per_minute'] = min(
                        self._rate_limiter['requests_per_minute'],
                        60 / delay  # 遅延に基づく1分あたりの最大リクエスト数
                    )

            # 禁止パスのログ出力
            if robots_info['disallowed_paths']:
                logger.info(f"robots.txtで禁止されているパス: {robots_info['disallowed_paths']}")

            return True

        except Exception as e:
            logger.error(f"robots.txt尊重処理エラー: {e}")
            return True  # エラー時は継続を許可

    def _add_ethical_headers(self):
        """
        リクエストに倫理的なヘッダーを追加

        Returns:
            dict: 追加するヘッダー
        """
        return {
            'User-Agent': 'TumblrImageCollector/1.0 (Educational/Research purposes only)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    def _log_ethical_scraping_info(self, blog_url):
        """
        倫理的スクレイピング情報をログに記録

        Args:
            blog_url (str): 対象ブログURL
        """
        logger.info("=== 倫理的スクレイピング情報 ===")
        logger.info(f"対象ブログ: {blog_url}")
        logger.info("目的: 教育・研究目的の画像収集")
        logger.info("遵守事項:")
        logger.info("- Tumblrの利用規約を尊重")
        logger.info("- robots.txtのポリシーを遵守")
        logger.info("- 過度な負荷をかけない")
        logger.info("- 個人情報の保護を徹底")
        logger.info("- 著作権を尊重")
        logger.info("================================")

        # robots.txt確認
        robots_info = self._check_robots_txt(blog_url)
        if not robots_info['can_fetch']:
            logger.error(f"robots.txtによりクローリングが禁止されています: {blog_url}")
            return False

        return True


    def _detect_video_content(self, media_url):
        """
        動画コンテンツを検出して分析

        Args:
            media_url (str): メディアURL

        Returns:
            dict: 動画分析結果
        """
        try:
            # 動画ファイルの拡張子チェック
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
            url_lower = media_url.lower()

            if any(ext in url_lower for ext in video_extensions):
                return {
                    'is_video': True,
                    'video_type': 'detected',
                    'duration': None,  # 実際の動画解析が必要
                    'resolution': None,
                    'fps': None,
                    'codec': None
                }

            # 動画のメタデータ解析（オプション）
            if _CV2_AVAILABLE:
                try:
                    import cv2
                    cap = cv2.VideoCapture(media_url)

                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / fps if fps > 0 else None

                        # 最初のフレームから解像度を取得
                        ret, frame = cap.read()
                        if ret:
                            height, width = frame.shape[:2]
                            resolution = f"{width}x{height}"
                        else:
                            resolution = None

                        cap.release()

                        return {
                            'is_video': True,
                            'video_type': 'analyzed',
                            'duration': duration,
                            'resolution': resolution,
                            'fps': fps,
                            'frame_count': frame_count
                        }
                except Exception as video_error:
                    logger.warning(f"動画解析エラー: {video_error}")

            return {
                'is_video': False,
                'video_type': None,
                'duration': None,
                'resolution': None,
                'fps': None
            }

        except Exception as e:
            logger.error(f"動画検出エラー: {e}")
            return {
                'is_video': False,
                'video_type': None,
                'duration': None,
                'resolution': None,
                'fps': None
            }

    def _extract_video_frames(self, video_path, max_frames=10):
        """
        動画から代表的なフレームを抽出

        Args:
            video_path (str): 動画ファイルパス
            max_frames (int): 抽出する最大フレーム数

        Returns:
            list: 抽出されたフレーム画像のパスリスト
        """
        try:
            if not _CV2_AVAILABLE:
                logger.warning("動画フレーム抽出にOpenCVが必要です")
                return []

            import cv2

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"動画ファイルを開けません: {video_path}")
                return []

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # フレームを均等に抽出
            frame_indices = []
            if frame_count > 0:
                step = max(1, frame_count // max_frames)
                for i in range(0, frame_count, step):
                    if len(frame_indices) >= max_frames:
                        break
                    frame_indices.append(i)

            extracted_frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    # フレームを画像として保存
                    frame_filename = f"frame_{frame_idx:06d}.jpg"
                    frame_path = Path(video_path).parent / frame_filename

                    # フレームサイズを制限
                    height, width = frame.shape[:2]
                    if max(height, width) > 1000:
                        scale = 1000 / max(height, width)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))

                    cv2.imwrite(str(frame_path), frame)
                    extracted_frames.append(str(frame_path))

            cap.release()
            logger.info(f"{len(extracted_frames)}フレームを抽出しました")
            return extracted_frames

        except Exception as e:
            logger.error(f"動画フレーム抽出エラー: {e}")
            return []

    def _analyze_video_quality(self, video_path):
        """
        動画の品質を分析

        Args:
            video_path (str): 動画ファイルパス

        Returns:
            dict: 動画品質分析結果
        """
        try:
            if not _CV2_AVAILABLE:
                return {}

            import cv2

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {}

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0

            # 複数のフレームから品質を評価
            quality_scores = []
            frame_indices = [0, frame_count // 4, frame_count // 2, frame_count * 3 // 4, frame_count - 1]

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    # フレームのシャープネスを評価
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

                    # フレームの明るさを評価
                    brightness = np.mean(gray)

                    quality_scores.append({
                        'sharpness': sharpness,
                        'brightness': brightness,
                        'frame_index': frame_idx
                    })

            cap.release()

            if quality_scores:
                avg_sharpness = np.mean([q['sharpness'] for q in quality_scores])
                avg_brightness = np.mean([q['brightness'] for q in quality_scores])

                return {
                    'duration': duration,
                    'fps': fps,
                    'frame_count': frame_count,
                    'average_sharpness': avg_sharpness,
                    'average_brightness': avg_brightness,
                    'quality_score': min(1.0, (avg_sharpness / 1000.0 + avg_brightness / 255.0) / 2),
                    'frame_analysis': quality_scores
                }
            else:
                return {}

        except Exception as e:
            logger.error(f"動画品質分析エラー: {e}")
            return {}

    def _process_video_content(self, video_url, output_path):
        """
        動画コンテンツを処理してフレームを抽出

        Args:
            video_url (str): 動画URL
            output_path (str): 出力ディレクトリ

        Returns:
            list: 処理されたフレームのパスリスト
        """
        try:
            import tempfile
            import requests

            # 一時ファイルに動画をダウンロード
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_video_path = temp_file.name

            # 動画をダウンロード
            response = requests.get(video_url, timeout=30)
            response.raise_for_status()

            with open(temp_video_path, 'wb') as f:
                f.write(response.content)

            # 動画を分析
            video_analysis = self._analyze_video_quality(temp_video_path)

            # フレームを抽出
            extracted_frames = self._extract_video_frames(temp_video_path, max_frames=5)

            # 一時ファイルを削除
            try:
                os.unlink(temp_video_path)
            except:
                pass

            # フレームを指定の出力ディレクトリに移動
            output_dir = Path(output_path)
            output_dir.mkdir(exist_ok=True)

            processed_frames = []
            for frame_path in extracted_frames:
                frame_name = Path(frame_path).name
                new_frame_path = output_dir / f"video_frame_{frame_name}"

                try:
                    shutil.move(frame_path, new_frame_path)
                    processed_frames.append(str(new_frame_path))
                except Exception as move_error:
                    logger.warning(f"フレーム移動エラー: {move_error}")

            logger.info(f"動画処理完了: {len(processed_frames)}フレーム抽出")
            return processed_frames

        except Exception as e:
            logger.error(f"動画処理エラー: {e}")
            return []

    def _classify_video_content(self, video_path):
        """
        機械学習による動画コンテンツの分類

        Args:
            video_path (str): 動画ファイルのパス

        Returns:
            dict: 動画分類結果
        """
        try:
            if not _CV2_AVAILABLE or not _NUMPY_AVAILABLE:
                logger.warning("動画分類に必要なライブラリが利用できません")
                return {}

            import cv2
            import torch
            from torchvision import models, transforms

            # 動画からフレームを抽出して分類
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {}

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # 動画の中央フレームを抽出して分類
            target_frame = max(1, frame_count // 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                return {}

            # フレームをRGBに変換
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 画像分類のための前処理
            preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            input_tensor = preprocess(frame_rgb)
            input_batch = input_tensor.unsqueeze(0)

            # 分類モデル（ResNet50）の読み込み
            try:
                model = models.resnet50(pretrained=True)
                model.eval()

                # 分類実行
                with torch.no_grad():
                    output = model(input_batch)

                # 結果を処理
                probabilities = torch.nn.functional.softmax(output[0], dim=0)

                # ImageNetクラスラベル（上位5つ）
                with open('imagenet_classes.txt', 'r') as f:
                    classes = [line.strip() for line in f.readlines()]

                top5_prob, top5_catid = torch.topk(probabilities, 5)

                classifications = []
                for i in range(top5_prob.size(0)):
                    class_id = top5_catid[i].item()
                    probability = top5_prob[i].item()
                    class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
                    classifications.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': probability
                    })

                return {
                    'video_classification': classifications,
                    'dominant_category': classifications[0]['class_name'],
                    'confidence_score': classifications[0]['confidence'],
                    'frame_analyzed': target_frame,
                    'total_frames': frame_count,
                    'fps': fps
                }

            except Exception as model_error:
                logger.warning(f"動画分類モデルエラー: {model_error}")
                return {}

        except Exception as e:
            logger.error(f"動画分類エラー: {e}")
            return {}

    def _generate_video_tags(self, video_path, classification_result=None):
        """
        動画から自動タグを生成

        Args:
            video_path (str): 動画ファイルのパス
            classification_result (dict): 分類結果（オプション）

        Returns:
            list: 生成されたタグのリスト
        """
        try:
            tags = []

            # 動画の基本情報からタグを生成
            video_analysis = self._analyze_video_quality(video_path)

            if video_analysis:
                # 解像度ベースのタグ
                if video_analysis.get('frame_count', 0) > 1000:
                    tags.append('long_video')
                elif video_analysis.get('frame_count', 0) < 100:
                    tags.append('short_video')

                # 品質ベースのタグ
                quality_score = video_analysis.get('quality_score', 0)
                if quality_score > 0.8:
                    tags.append('high_quality')
                elif quality_score < 0.4:
                    tags.append('low_quality')

                # FPSベースのタグ
                fps = video_analysis.get('fps', 0)
                if fps > 50:
                    tags.append('high_fps')
                elif fps < 20:
                    tags.append('low_fps')

            # 分類結果からタグを生成
            if classification_result and 'video_classification' in classification_result:
                classifications = classification_result['video_classification']

                # 上位分類からタグを生成
                for cls in classifications[:3]:
                    class_name = cls['class_name'].lower()
                    confidence = cls['confidence']

                    if confidence > 0.7:
                        # 一般的なカテゴリマッピング
                        category_mapping = {
                            'animal': ['animal', 'wildlife', 'pet'],
                            'vehicle': ['car', 'vehicle', 'transport'],
                            'person': ['person', 'human', 'portrait'],
                            'nature': ['nature', 'landscape', 'outdoor'],
                            'building': ['building', 'architecture', 'structure'],
                            'food': ['food', 'cooking', 'recipe'],
                            'sport': ['sport', 'athletic', 'exercise'],
                            'music': ['music', 'instrument', 'performance']
                        }

                        for category, keywords in category_mapping.items():
                            if any(keyword in class_name for keyword in keywords):
                                tags.append(category)
                                break

                        # 具体的なクラス名も追加（信頼度が高い場合）
                        if confidence > 0.9:
                            tags.append(class_name.replace(' ', '_'))

            # 重複を除去して返す
            return list(set(tags))

        except Exception as e:
            logger.error(f"動画タグ生成エラー: {e}")
            return []

    def _process_video_with_ml(self, video_url, output_path, enable_classification=True):
        """
        機械学習を活用した動画処理

        Args:
            video_url (str): 動画URL
            output_path (str): 出力ディレクトリ
            enable_classification (bool): 分類を有効にするかどうか

        Returns:
            dict: 処理結果
        """
        try:
            import tempfile

            # 一時ファイルに動画をダウンロード
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_video_path = temp_file.name

            # 動画をダウンロード
            response = requests.get(video_url, timeout=30)
            response.raise_for_status()

            with open(temp_video_path, 'wb') as f:
                f.write(response.content)

            # 動画を分析
            video_analysis = self._analyze_video_quality(temp_video_path)

            # 機械学習による分類（オプション）
            classification_result = None
            if enable_classification:
                classification_result = self._classify_video_content(temp_video_path)

            # タグを生成
            video_tags = self._generate_video_tags(temp_video_path, classification_result)

            # フレームを抽出
            extracted_frames = self._extract_video_frames(temp_video_path, max_frames=5)

            # 一時ファイルを削除
            try:
                os.unlink(temp_video_path)
            except:
                pass

            # フレームを指定の出力ディレクトリに移動
            output_dir = Path(output_path)
            output_dir.mkdir(exist_ok=True)

            processed_frames = []
            for frame_path in extracted_frames:
                frame_name = Path(frame_path).name
                new_frame_path = output_dir / f"ml_video_frame_{frame_name}"

                try:
                    shutil.move(frame_path, new_frame_path)
                    processed_frames.append(str(new_frame_path))
                except Exception as move_error:
                    logger.warning(f"フレーム移動エラー: {move_error}")

            # 処理結果をまとめる
            result = {
                'video_url': video_url,
                'output_path': str(output_dir),
                'video_analysis': video_analysis,
                'classification_result': classification_result,
                'generated_tags': video_tags,
                'processed_frames': processed_frames,
                'processing_timestamp': datetime.datetime.now().isoformat()
            }

            logger.info(f"機械学習動画処理完了: {len(processed_frames)}フレーム, タグ: {len(video_tags)}個")
            return result

        except Exception as e:
            logger.error(f"機械学習動画処理エラー: {e}")
            return {}

    def _create_video_summary_report(self, video_results):
        """
        動画処理結果のサマリレポートを作成

        Args:
            video_results (list): 動画処理結果のリスト

        Returns:
            dict: サマリレポート
        """
        try:
            total_videos = len(video_results)
            total_frames = sum(len(result.get('processed_frames', [])) for result in video_results)

            # タグの統計
            all_tags = []
            for result in video_results:
                all_tags.extend(result.get('generated_tags', []))

            tag_counts = {}
            for tag in all_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

            # 品質スコアの統計
            quality_scores = []
            for result in video_results:
                analysis = result.get('video_analysis', {})
                if 'quality_score' in analysis:
                    quality_scores.append(analysis['quality_score'])

            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

            # 分類結果の統計
            classification_stats = {
                'total_classified': sum(1 for result in video_results if result.get('classification_result')),
                'top_categories': sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            }

            return {
                'summary': {
                    'total_videos_processed': total_videos,
                    'total_frames_extracted': total_frames,
                    'average_quality_score': avg_quality,
                    'unique_tags_count': len(tag_counts),
                    'most_common_tags': classification_stats['top_categories']
                },
                'detailed_results': video_results,
                'generated_at': datetime.datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"動画サマリレポート作成エラー: {e}")
            return {}

    def _create_batch_processing_queue(self, image_urls, batch_size=10):
        """
        大規模画像処理のためのバッチキューを作成

        Args:
            image_urls (list): 処理対象の画像URLリスト
            batch_size (int): バッチサイズ

        Returns:
            list: バッチ処理キュー
        """
        try:
            batches = []
            for i in range(0, len(image_urls), batch_size):
                batch = image_urls[i:i + batch_size]
                batches.append({
                    'batch_id': f"batch_{len(batches)}",
                    'urls': batch,
                    'size': len(batch),
                    'status': 'pending',
                    'created_at': datetime.datetime.now().isoformat()
                })

            logger.info(f"バッチ処理キューを作成: {len(batches)}バッチ, 総{len(image_urls)}画像")
            return batches

        except Exception as e:
            logger.error(f"バッチキュー作成エラー: {e}")
            return []

    def _process_image_batch(self, batch_data, output_dir, enable_ml_analysis=True):
        """
        画像バッチを処理

        Args:
            batch_data (dict): バッチデータ
            output_dir (str): 出力ディレクトリ
            enable_ml_analysis (bool): ML分析を有効にするかどうか

        Returns:
            dict: バッチ処理結果
        """
        try:
            batch_id = batch_data['batch_id']
            urls = batch_data['urls']

            logger.info(f"バッチ処理開始: {batch_id} ({len(urls)}画像)")

            batch_results = []
            processed_count = 0
            failed_count = 0

            for url in urls:
                try:
                    # 画像をダウンロードして処理
                    result = self._download_and_store_image(url, post_data={'batch_id': batch_id})

                    if result:
                        image_path = result
                        processed_count += 1

                        # ML分析（オプション）
                        if enable_ml_analysis and self.image_classifier:
                            try:
                                analysis_result = self.image_classifier.analyze_image(image_path)
                                if analysis_result:
                                    # 分析結果をメタデータファイルに追加
                                    metadata_path = Path(image_path).with_suffix('.json')
                                    if metadata_path.exists():
                                        with open(metadata_path, 'r') as f:
                                            metadata = json.load(f)
                                        metadata['ml_analysis'] = analysis_result
                                        with open(metadata_path, 'w') as f:
                                            json.dump(metadata, f, ensure_ascii=False, indent=2)
                            except Exception as analysis_error:
                                logger.warning(f"画像分析エラー: {analysis_error}")

                        batch_results.append({
                            'url': url,
                            'status': 'success',
                            'image_path': str(image_path)
                        })
                    else:
                        failed_count += 1
                        batch_results.append({
                            'url': url,
                            'status': 'failed',
                            'error': 'ダウンロード失敗'
                        })

                except Exception as e:
                    failed_count += 1
                    logger.error(f"バッチ内画像処理エラー: {url} - {e}")
                    batch_results.append({
                        'url': url,
                        'status': 'error',
                        'error': str(e)
                    })

            # バッチ結果をまとめる
            batch_result = {
                'batch_id': batch_id,
                'total_images': len(urls),
                'processed_count': processed_count,
                'failed_count': failed_count,
                'success_rate': processed_count / len(urls) if urls else 0,
                'results': batch_results,
                'completed_at': datetime.datetime.now().isoformat()
            }

            # バッチ結果をファイルに保存
            batch_output_dir = Path(output_dir) / f"batch_{batch_id}"
            batch_output_dir.mkdir(exist_ok=True)

            batch_result_path = batch_output_dir / "batch_result.json"
            with open(batch_result_path, 'w', encoding='utf-8') as f:
                json.dump(batch_result, f, ensure_ascii=False, indent=2)

            logger.info(f"バッチ処理完了: {batch_id} - 成功率: {batch_result['success_rate']:.2%}")
            return batch_result

        except Exception as e:
            logger.error(f"バッチ処理エラー: {e}")
            return {
                'batch_id': batch_data.get('batch_id', 'unknown'),
                'total_images': len(batch_data.get('urls', [])),
                'processed_count': 0,
                'failed_count': len(batch_data.get('urls', [])),
                'success_rate': 0,
                'results': [],
                'error': str(e)
            }

    def _implement_distributed_processing(self, batches, worker_nodes=None, max_workers=5):
        """
        分散処理を実装

        Args:
            batches (list): バッチリスト
            worker_nodes (list): ワーカーノードリスト（オプション）
            max_workers (int): 最大ワーカー数

        Returns:
            dict: 分散処理結果
        """
        try:
            import concurrent.futures
            import multiprocessing

            # ワーカーノードが指定されていない場合はローカルワーカーを使用
            if worker_nodes is None:
                num_workers = min(max_workers, multiprocessing.cpu_count())
            else:
                num_workers = min(max_workers, len(worker_nodes))

            logger.info(f"分散処理開始: {len(batches)}バッチ, {num_workers}ワーカー")

            # バッチ処理の実行
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                # 各バッチの処理を並列実行
                future_to_batch = {
                    executor.submit(self._process_image_batch, batch, self.output_folder): batch
                    for batch in batches
                }

                # 結果を収集
                batch_results = []
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        result = future.result()
                        batch_results.append(result)
                    except Exception as exc:
                        logger.error(f"バッチ処理エラー: {batch['batch_id']} - {exc}")
                        batch_results.append({
                            'batch_id': batch['batch_id'],
                            'error': str(exc),
                            'total_images': len(batch.get('urls', [])),
                            'processed_count': 0,
                            'failed_count': len(batch.get('urls', [])),
                            'success_rate': 0
                        })

            # 全体結果をまとめる
            total_batches = len(batch_results)
            total_images = sum(batch['total_images'] for batch in batch_results)
            total_processed = sum(batch['processed_count'] for batch in batch_results)
            total_failed = sum(batch['failed_count'] for batch in batch_results)
            overall_success_rate = total_processed / total_images if total_images > 0 else 0

            distributed_result = {
                'total_batches': total_batches,
                'total_images': total_images,
                'total_processed': total_processed,
                'total_failed': total_failed,
                'overall_success_rate': overall_success_rate,
                'batch_results': batch_results,
                'processing_completed_at': datetime.datetime.now().isoformat()
            }

            # 分散処理結果をファイルに保存
            result_path = self.output_folder / f"distributed_processing_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(distributed_result, f, ensure_ascii=False, indent=2)

            logger.info(f"分散処理完了: 全体成功率 {overall_success_rate:.2%}")
            return distributed_result

        except Exception as e:
            logger.error(f"分散処理エラー: {e}")
            return {}

    def _optimize_processing_pipeline(self, image_urls, optimization_level='auto'):
        """
        処理パイプラインを最適化

        Args:
            image_urls (list): 画像URLリスト
            optimization_level (str): 最適化レベル（'low', 'medium', 'high', 'auto'）

        Returns:
            dict: 最適化された処理結果
        """
        try:
            # システムリソースの確認
            import psutil

            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)

            # 最適化レベルの決定
            if optimization_level == 'auto':
                if cpu_count >= 8 and memory_gb >= 16:
                    optimization_level = 'high'
                elif cpu_count >= 4 and memory_gb >= 8:
                    optimization_level = 'medium'
                else:
                    optimization_level = 'low'

            # 最適化設定の適用
            optimization_settings = {
                'low': {
                    'batch_size': 5,
                    'max_workers': 2,
                    'enable_ml_analysis': False,
                    'cache_enabled': True
                },
                'medium': {
                    'batch_size': 10,
                    'max_workers': min(4, cpu_count // 2),
                    'enable_ml_analysis': True,
                    'cache_enabled': True
                },
                'high': {
                    'batch_size': 20,
                    'max_workers': min(8, cpu_count),
                    'enable_ml_analysis': True,
                    'cache_enabled': True
                }
            }

            settings = optimization_settings.get(optimization_level, optimization_settings['medium'])

            logger.info(f"処理パイプライン最適化レベル: {optimization_level}")
            logger.info(f"設定: バッチサイズ={settings['batch_size']}, ワーカー数={settings['max_workers']}")

            # バッチ処理キューを作成
            batches = self._create_batch_processing_queue(image_urls, settings['batch_size'])

            # 分散処理を実行
            result = self._implement_distributed_processing(
                batches,
                max_workers=settings['max_workers']
            )

            # 追加の最適化メトリクス
            processing_time = (datetime.datetime.now() - datetime.datetime.fromisoformat(result.get('processing_completed_at', datetime.datetime.now().isoformat()))).total_seconds()

            optimization_metrics = {
                'optimization_level': optimization_level,
                'system_resources': {
                    'cpu_cores': cpu_count,
                    'memory_gb': memory_gb
                },
                'processing_settings': settings,
                'performance_metrics': {
                    'total_processing_time_seconds': processing_time,
                    'images_per_second': result.get('total_images', 0) / processing_time if processing_time > 0 else 0,
                    'success_rate': result.get('overall_success_rate', 0)
                }
            }

            # 最適化レポートを保存
            report_path = self.output_folder / f"optimization_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(optimization_metrics, f, ensure_ascii=False, indent=2)

            logger.info(f"最適化処理完了: {optimization_metrics['performance_metrics']['images_per_second']:.2f}画像/秒")
            return result

        except Exception as e:
            logger.error(f"処理パイプライン最適化エラー: {e}")
            return {}

    def _create_processing_dashboard(self, processing_results):
        """
        処理結果のダッシュボードを作成

        Args:
            processing_results (dict): 処理結果データ

        Returns:
            str: ダッシュボードファイルのパス
        """
        try:
            # HTMLダッシュボードテンプレート
            dashboard_html = """
            <!DOCTYPE html>
            <html lang="ja">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Tumblr Image Collector - Processing Dashboard</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background: #f5f5f5;
                    }
                    .dashboard {
                        max-width: 1200px;
                        margin: 0 auto;
                        background: white;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }
                    .metrics-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 20px;
                        margin: 20px 0;
                    }
                    .metric-card {
                        background: #f8f9fa;
                        padding: 20px;
                        border-radius: 8px;
                        text-align: center;
                        border-left: 4px solid #007bff;
                    }
                    .metric-value {
                        font-size: 2em;
                        font-weight: bold;
                        color: #007bff;
                    }
                    .metric-label {
                        color: #6c757d;
                        margin-top: 5px;
                    }
                    .batch-results {
                        margin-top: 30px;
                    }
                    .batch-item {
                        background: #e9ecef;
                        margin: 10px 0;
                        padding: 15px;
                        border-radius: 5px;
                    }
                    .success { border-left: 4px solid #28a745; }
                    .warning { border-left: 4px solid #ffc107; }
                    .error { border-left: 4px solid #dc3545; }
                </style>
            </head>
            <body>
                <div class="dashboard">
                    <h1>Tumblr Image Collector - Processing Dashboard</h1>

                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">{total_images}</div>
                            <div class="metric-label">総処理画像数</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{success_rate:.1%}</div>
                            <div class="metric-label">成功率</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{total_batches}</div>
                            <div class="metric-label">バッチ数</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{avg_processing_time:.2f}s</div>
                            <div class="metric-label">平均処理時間</div>
                        </div>
                    </div>

                    <div class="batch-results">
                        <h2>バッチ処理結果</h2>
                        {batch_items}
                    </div>
                </div>
            </body>
            </html>
            """

            # メトリクスの計算
            total_batches = processing_results.get('total_batches', 0)
            total_images = processing_results.get('total_images', 0)
            overall_success_rate = processing_results.get('overall_success_rate', 0)

            # バッチアイテムの生成
            batch_items = ""
            for batch_result in processing_results.get('batch_results', []):
                batch_id = batch_result.get('batch_id', 'unknown')
                processed = batch_result.get('processed_count', 0)
                failed = batch_result.get('failed_count', 0)
                success_rate = batch_result.get('success_rate', 0)

                status_class = 'success' if success_rate > 0.8 else 'warning' if success_rate > 0.5 else 'error'

                batch_items += f"""
                <div class="batch-item {status_class}">
                    <h3>バッチ {batch_id}</h3>
                    <p>処理画像数: {processed + failed}</p>
                    <p>成功: {processed}, 失敗: {failed}</p>
                    <p>成功率: {success_rate:.1%}</p>
                </div>
                """

            # HTMLを生成
            html_content = dashboard_html.format(
                total_images=total_images,
                success_rate=overall_success_rate,
                total_batches=total_batches,
                avg_processing_time=0,  # 実際の計算が必要
                batch_items=batch_items
            )

            # ダッシュボードファイルを保存
            dashboard_path = self.output_folder / f"processing_dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(dashboard_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"処理ダッシュボードを作成: {dashboard_path}")
            return str(dashboard_path)

        except Exception as e:
            logger.error(f"ダッシュボード作成エラー: {e}")
            return ""

    def _setup_headless_browser(self, browser_type='chrome'):
        """
        ヘッドレスブラウザを設定

        Args:
            browser_type (str): ブラウザタイプ（'chrome', 'firefox'）

        Returns:
            WebDriver: 設定されたブラウザドライバー
        """
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            from selenium.webdriver.firefox.options import Options as FirefoxOptions

            if browser_type == 'chrome':
                options = ChromeOptions()
                options.add_argument('--headless')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-gpu')
                options.add_argument('--window-size=1920,1080')
                options.add_argument('--user-agent=TumblrImageCollector/1.0')

                try:
                    driver = webdriver.Chrome(options=options)
                except Exception as chrome_error:
                    logger.warning(f"Chromeドライバーエラー: {chrome_error}")
                    # Chromeが利用できない場合はFirefoxを試行
                    browser_type = 'firefox'

            if browser_type == 'firefox':
                options = FirefoxOptions()
                options.add_argument('--headless')

                try:
                    driver = webdriver.Firefox(options=options)
                except Exception as firefox_error:
                    logger.error(f"Firefoxドライバーエラー: {firefox_error}")
                    return None

            # 追加の設定
            driver.implicitly_wait(10)
            logger.info(f"ヘッドレスブラウザを設定: {browser_type}")
            return driver

        except ImportError as import_error:
            logger.warning(f"Seleniumがインストールされていません: {import_error}")
            return None
        except Exception as e:
            logger.error(f"ヘッドレスブラウザ設定エラー: {e}")
            return None

    def _extract_dynamic_content(self, blog_url, content_selectors=None):
        """
        動的コンテンツを抽出

        Args:
            blog_url (str): ブログURL
            content_selectors (dict): CSSセレクター設定

        Returns:
            dict: 抽出されたコンテンツデータ
        """
        try:
            driver = self._setup_headless_browser()
            if not driver:
                return {}

            driver.get(blog_url)

            # デフォルトのコンテンツセレクター
            default_selectors = {
                'posts': '.post',
                'images': 'img',
                'videos': 'video',
                'links': 'a[href]',
                'titles': 'h1, h2, h3'
            }

            selectors = content_selectors or default_selectors

            extracted_data = {}

            for content_type, selector in selectors.items():
                try:
                    elements = driver.find_elements_by_css_selector(selector)
                    content_list = []

                    for element in elements[:10]:  # 上位10要素のみ処理
                        if content_type == 'images':
                            src = element.get_attribute('src')
                            alt = element.get_attribute('alt')
                            if src:
                                content_list.append({
                                    'url': src,
                                    'alt': alt,
                                    'type': 'image'
                                })
                        elif content_type == 'videos':
                            src = element.get_attribute('src')
                            if src:
                                content_list.append({
                                    'url': src,
                                    'type': 'video'
                                })
                        elif content_type == 'links':
                            href = element.get_attribute('href')
                            text = element.text
                            if href:
                                content_list.append({
                                    'url': href,
                                    'text': text,
                                    'type': 'link'
                                })
                        else:
                            text = element.text.strip()
                            if text:
                                content_list.append({
                                    'text': text,
                                    'type': content_type
                                })

                    extracted_data[content_type] = content_list

                except Exception as selector_error:
                    logger.warning(f"コンテンツ抽出エラー ({content_type}): {selector_error}")

            driver.quit()

            logger.info(f"動的コンテンツ抽出完了: {blog_url}")
            return extracted_data

        except Exception as e:
            logger.error(f"動的コンテンツ抽出エラー: {e}")
            return {}

    def _implement_anti_detection_measures(self, driver):
        """
        検出回避策を実装

        Args:
            driver: WebDriverインスタンス
        """
        try:
            # ユーザーエージェントの設定
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            driver.execute_script(f"Object.defineProperty(navigator, 'userAgent', {{get: () => '{user_agent}'}})")

            # WebGLを無効化（検出回避）
            driver.execute_script("Object.defineProperty(navigator, 'webgl', {get: () => undefined})")

            # プラグインを偽装
            driver.execute_script("""
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [
                        {name: 'Chrome PDF Plugin', description: 'Portable Document Format'},
                        {name: 'Chrome PDF Viewer', description: ''},
                        {name: 'Native Client', description: ''}
                    ]
                })
            """)

            # 画面サイズとビューポートの設定
            driver.set_window_size(1920, 1080)
            driver.execute_script("Object.defineProperty(screen, 'width', {get: () => 1920})")
            driver.execute_script("Object.defineProperty(screen, 'height', {get: () => 1080})")

            logger.info("検出回避策を適用")

        except Exception as e:
            logger.warning(f"検出回避策適用エラー: {e}")

    def _handle_javascript_rendering(self, blog_url, wait_time=5):
        """
        JavaScriptレンダリングを処理

        Args:
            blog_url (str): ブログURL
            wait_time (int): 待機時間（秒）

        Returns:
            dict: レンダリング結果
        """
        try:
            driver = self._setup_headless_browser()
            if not driver:
                return {}

            driver.get(blog_url)

            # 検出回避策を適用
            self._implement_anti_detection_measures(driver)

            # ページが完全にロードされるまで待機
            import time
            time.sleep(wait_time)

            # JavaScriptが実行されるまで追加で待機
            try:
                # ページの高さをチェックして動的コンテンツがロードされたか確認
                last_height = driver.execute_script("return document.body.scrollHeight")
                for _ in range(3):
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)
                    new_height = driver.execute_script("return document.body.scrollHeight")
                    if new_height == last_height:
                        break
                    last_height = new_height
            except:
                pass

            # 最終的なページコンテンツを取得
            page_title = driver.title
            page_source = driver.page_source
            current_url = driver.current_url

            # 画像と動画のURLを抽出
            image_urls = []
            video_urls = []

            try:
                # 画像要素からURLを抽出
                img_elements = driver.find_elements_by_tag_name('img')
                for img in img_elements:
                    src = img.get_attribute('src')
                    if src and src.startswith('http'):
                        image_urls.append(src)

                # 動画要素からURLを抽出
                video_elements = driver.find_elements_by_tag_name('video')
                for video in video_elements:
                    src = video.get_attribute('src')
                    if src and src.startswith('http'):
                        video_urls.append(src)

            except Exception as extract_error:
                logger.warning(f"メディア抽出エラー: {extract_error}")

            driver.quit()

            return {
                'page_title': page_title,
                'final_url': current_url,
                'page_length': len(page_source),
                'image_urls': list(set(image_urls))[:50],  # 重複除去して上位50件
                'video_urls': list(set(video_urls))[:20],  # 重複除去して上位20件
                'rendering_successful': True,
                'processing_timestamp': datetime.datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"JavaScriptレンダリング処理エラー: {e}")
            return {
                'rendering_successful': False,
                'error': str(e)
            }

    def _create_scraping_session_manager(self):
        """
        高度なスクレイピングセッションマネージャーを作成

        Returns:
            dict: セッションマネージャー設定
        """
        try:
            session_config = {
                'user_agents': [
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                ],
                'proxies': self.proxies,
                'headers': self._add_ethical_headers(),
                'timeout': self.download_timeout,
                'retry_strategy': {
                    'max_retries': 3,
                    'backoff_factor': 1.5,
                    'status_forcelist': [429, 500, 502, 503, 504]
                }
            }

            logger.info("高度なスクレイピングセッションマネージャーを作成")
            return session_config

        except Exception as e:
            logger.error(f"セッションマネージャー作成エラー: {e}")
            return {}

    def _implement_advanced_scraping_pipeline(self, blog_urls, use_dynamic_scraping=True):
        """
        高度なスクレイピングパイプラインを実装

        Args:
            blog_urls (list): ブログURLリスト
            use_dynamic_scraping (bool): 動的スクレイピングを使用するかどうか

        Returns:
            dict: スクレイピング結果
        """
        try:
            pipeline_results = {
                'total_blogs': len(blog_urls),
                'successful_scrapes': 0,
                'failed_scrapes': 0,
                'extracted_images': [],
                'extracted_videos': [],
                'dynamic_content_results': [],
                'static_content_results': [],
                'processing_timestamp': datetime.datetime.now().isoformat()
            }

            for blog_url in blog_urls:
                try:
                    logger.info(f"高度なスクレイピング開始: {blog_url}")

                    # 倫理的ガイドラインを確認
                    if not self._respect_ethical_guidelines(blog_url):
                        logger.warning(f"倫理的ガイドラインによりスキップ: {blog_url}")
                        pipeline_results['failed_scrapes'] += 1
                        continue

                    # 動的コンテンツ抽出（オプション）
                    if use_dynamic_scraping:
                        try:
                            dynamic_result = self._handle_javascript_rendering(blog_url)
                            if dynamic_result.get('rendering_successful', False):
                                pipeline_results['dynamic_content_results'].append({
                                    'blog_url': blog_url,
                                    'result': dynamic_result
                                })
                                pipeline_results['extracted_images'].extend(dynamic_result.get('image_urls', []))
                                pipeline_results['extracted_videos'].extend(dynamic_result.get('video_urls', []))
                                pipeline_results['successful_scrapes'] += 1
                            else:
                                logger.warning(f"動的スクレイピング失敗: {blog_url}")
                        except Exception as dynamic_error:
                            logger.warning(f"動的スクレイピングエラー: {dynamic_error}")

                    # 静的コンテンツ抽出（フォールバック）
                    try:
                        static_result = self._extract_dynamic_content(blog_url)
                        if static_result:
                            pipeline_results['static_content_results'].append({
                                'blog_url': blog_url,
                                'result': static_result
                            })
                    except Exception as static_error:
                        logger.warning(f"静的コンテンツ抽出エラー: {static_error}")

                except Exception as e:
                    logger.error(f"ブログ処理エラー: {blog_url} - {e}")
                    pipeline_results['failed_scrapes'] += 1

            # 重複を除去
            pipeline_results['extracted_images'] = list(set(pipeline_results['extracted_images']))
            pipeline_results['extracted_videos'] = list(set(pipeline_results['extracted_videos']))

            # 結果を保存
            result_path = self.output_folder / f"advanced_scraping_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(pipeline_results, f, ensure_ascii=False, indent=2)

            logger.info(f"高度なスクレイピング完了: 成功率 {(pipeline_results['successful_scrapes']/pipeline_results['total_blogs'])*100:.1f}%")
            return pipeline_results

        except Exception as e:
            logger.error(f"高度なスクレイピングパイプラインエラー: {e}")
            return {}