"""
Tumblr Image Collector - Core Module

メインのコレクタークラスと統合機能
"""

import logging
import time
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse
import concurrent.futures
import requests
from PIL import Image
import threading
import atexit
import signal

from .api_manager import ApiManager
from .cache_manager import CacheManager
from .processor import ImageProcessor
from .download_manager import DownloadManager

# 必要なモジュールのインポート
from billing import LicenseManager, LicenseStatus
from ui import InteractiveCLI, ProgressDisplay
from config import ConfigWizard
from exceptions import (
    TumblrCollectorError, NetworkError, ValidationError, ConfigurationError,
    DownloadError, URLValidationError, DiskSpaceError
)
from logging_utils import LoggingConfig, SensitiveDataFilter, create_context_logger
from resource_manager import (
    ResourceMonitor, ResourceLimits, ResourceGuard,
    get_file_handle_manager, get_cleanup_manager,
    resource_guarded_operation
)

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
        'total_images_processed': 0,
        'total_images_downloaded': 0,
        'total_images_skipped': 0,
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

    def __init__(self, config_file: str = "config.json",
                 output_dir_override: Optional[str] = None,
                 workers_override: Optional[int] = None,
                 proxy_config: Optional[Dict[str, Any]] = None):

        # 基本設定
        self.config_file = Path(config_file).resolve()
        self.config = self._load_config()
        self.script_dir = Path(__file__).parent.parent.resolve()

        # ロギング初期化
        self._setup_logging()

        # リソース管理
        self.cleanup_manager = get_cleanup_manager()

        # プロキシ設定
        self.proxy_config = proxy_config or self.config.get('proxy', {})

        # 出力ディレクトリ設定
        default_values = ConfigWizard.default_config_values()
        output_folder_name = output_dir_override or self.config.get("output_folder_name", default_values["output_folder_name"])
        if Path(output_folder_name).is_absolute():
            self.output_folder = Path(output_folder_name)
        else:
            self.output_folder = self.script_dir / output_folder_name

        # ワーカー数設定
        if workers_override:
            self.max_workers = workers_override
        elif "max_download_workers" in self.config:
            self.max_workers = self.config["max_download_workers"]
        else:
            self.max_workers = ResourceMonitor.get_optimal_worker_count(max_workers=15)
            logger.info(f"Dynamically calculated worker count: {self.max_workers}")

        # コンポーネント初期化
        self.api_manager = ApiManager(self.config)
        self.cache_manager = CacheManager(self.output_folder / "cache", self.config)
        self.processor = ImageProcessor(self.config)

        # HTTPセッションとダウンロードマネージャー
        self.session = self._create_requests_session()
        self.download_manager = DownloadManager(self.config, self.session)

        # 出力ディレクトリ初期化
        self._setup_output_directory()

        # ダウンロード済みファイルの読み込み
        self._load_downloaded_files()

        # 設定検証
        self._validate_configuration()

        # 課金管理
        self.billing_manager = self._initialize_billing_manager()
        self.license_manager = LicenseManager(self.script_dir / "licenses" / "license.json")
        self._load_license_from_config()

        # クリーンアップ登録
        atexit.register(self._cleanup_resources)
        self._setup_signal_handlers()

        logger.info("TumblrImageCollector initialized successfully")

    def _setup_logging(self) -> None:
        """ロギングシステムを設定"""
        log_dir = self.script_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        log_filename = log_dir / "tumblr_collector.log"

        LoggingConfig.setup_logging(
            config=self.config,
            log_file=str(log_filename),
            enable_colors=True,
            enable_sanitization=True
        )

    def _create_requests_session(self) -> requests.Session:
        """HTTPセッションを作成"""
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()

        # リトライ設定
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # プロキシ設定
        if self.proxy_config:
            proxies = self._build_requests_proxies(self.proxy_config)
            session.proxies.update(proxies)

        # タイムアウト設定
        session.timeout = 30

        return session

    def _build_requests_proxies(self, proxy_config: Dict[str, Any]) -> Dict[str, str]:
        """プロキシ設定をrequests形式に変換"""
        proxies = {}

        if not proxy_config or not proxy_config.get('host'):
            return proxies

        proxy_type = proxy_config.get('type', 'http')
        host = proxy_config['host']
        port = proxy_config['port']

        proxy_url = f"{proxy_type}://{host}:{port}"

        # 認証情報
        username = proxy_config.get('username')
        password = proxy_config.get('password')
        if username and password:
            proxy_url = f"{proxy_type}://{username}:{password}@{host}:{port}"

        proxies = {
            'http': proxy_url,
            'https': proxy_url
        }

        return proxies

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込む"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"{self.config_file} is corrupted. Using defaults.")
                return {}
            except Exception as e:
                logger.error(f"Error loading config file {self.config_file}: {e}")
                return {}
        logger.info(f"Config file {self.config_file} not found. Using defaults.")
        return {}

    def _save_config(self) -> None:
        """設定ファイルを保存"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, sort_keys=True, ensure_ascii=False)
            logger.debug(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving config file {self.config_file}: {e}")

    def _setup_output_directory(self) -> None:
        """出力ディレクトリを作成"""
        try:
            self.output_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory set to: {self.output_folder}")
        except Exception as e:
            logger.error(f"Failed to create output directory {self.output_folder}: {e}")
            raise IOError(f"Cannot create output directory: {self.output_folder}") from e

    def _load_downloaded_files(self) -> None:
        """既存のダウンロード済みファイルを読み込み"""
        if not self.output_folder.exists():
            logger.warning(f"Output folder {self.output_folder} does not exist yet.")
            return

        try:
            existing_files = {f.name for f in self.output_folder.iterdir()
                            if f.is_file() and f.suffix.lower() not in {'.json', '.log'}}

            # メタデータからハッシュを読み込み
            hash_file = self.output_folder / "downloaded_hashes.txt"
            self.processor.load_downloaded_hashes(hash_file)

            logger.info(f"Loaded {len(existing_files)} existing files")
        except Exception as e:
            logger.error(f"Error reading existing files from {self.output_folder}: {e}")

    def _validate_configuration(self) -> None:
        """設定の検証"""
        # 基本的な設定検証
        if self.max_workers < 1 or self.max_workers > 50:
            logger.warning(f"Invalid max_workers: {self.max_workers}, setting to 4")
            self.max_workers = 4

        # 出力ディレクトリの書き込み権限チェック
        if not os.access(self.output_folder, os.W_OK):
            raise ConfigurationError(f"No write permission for output directory: {self.output_folder}")

    def _initialize_billing_manager(self):
        """課金マネージャーの初期化"""
        from billing import StripeBillingManager

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

        payload = None

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

        if payload:
            self._apply_license_payload(payload)

    def _apply_license_payload(self, payload: Dict[str, Any]) -> None:
        """ライセンス情報を適用"""
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
                f"機能 '{feature_name}' の利用には有効なライセンスが必要です。"
            )

    def _cleanup_resources(self) -> None:
        """リソースのクリーンアップ"""
        try:
            # HTTPセッションのクリーンアップ
            if hasattr(self, 'session') and self.session:
                self.session.close()

            # 統計情報の保存
            self._save_statistics()

            logger.info("リソースをクリーンアップしました")
        except Exception as e:
            logger.error(f"リソースクリーンアップ中にエラー: {e}")

    def _setup_signal_handlers(self) -> None:
        """シグナルハンドラーの設定"""
        def signal_handler(signum, frame):
            logger.info(f"Signal {signum} received, cleaning up...")
            self._cleanup_resources()
            exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _save_statistics(self) -> None:
        """統計情報を保存"""
        stats_path_json = self.script_dir / 'download_statistics.json'
        stats_path_csv = self.script_dir / 'download_statistics.csv'

        try:
            # JSON保存
            with open(stats_path_json, 'w', encoding='utf-8') as f:
                json.dump(self._download_stats, f, ensure_ascii=False, indent=2)

            # CSV保存
            with open(stats_path_csv, 'w', newline='', encoding='utf-8') as csvfile:
                import csv
                writer = csv.writer(csvfile)
                writer.writerow(['metric', 'value'])
                writer.writerow(['total_attempts', self._download_stats['total_attempts']])
                writer.writerow(['successful_downloads', self._download_stats['successful_downloads']])
                writer.writerow(['failed_downloads', self._download_stats['failed_downloads']])
                writer.writerow(['skipped_duplicates', self._download_stats['skipped_duplicates']])

            logger.info(f"統計情報を保存しました: {stats_path_json}, {stats_path_csv}")
        except Exception as e:
            logger.error(f"統計情報の保存に失敗: {e}")

    # 以下に主要な公開メソッドを実装
    def run(self, blog_name: str, tags: Optional[List[str]] = None,
            date_range: Optional[Dict[str, str]] = None,
            include_likes: bool = False) -> None:
        """
        メインの実行メソッド

        Args:
            blog_name: ブログ名
            tags: 検索タグリスト
            date_range: 日付範囲
            include_likes: いいねを含むかどうか
        """
        try:
            # ライセンスチェック
            self._ensure_license_for_feature("image_collection")

            # ブログ投稿の取得
            posts = self.api_manager.get_blog_posts(blog_name, limit=50)
            if not posts:
                logger.warning(f"No posts found for blog: {blog_name}")
                return

            # 画像URLの抽出とダウンロード
            image_urls = self._extract_image_urls(posts, tags, date_range, include_likes)
            if not image_urls:
                logger.info("No images found matching criteria")
                return

            # ダウンロード実行
            results = self.download_manager.download_images_batch(
                image_urls, self.output_folder, self._create_progress_callback()
            )

            logger.info(f"Download completed: {results['successful']}/{results['total']} images")

        except Exception as e:
            logger.error(f"Error during run: {e}")
            raise

    def _extract_image_urls(self, posts: List[Dict[str, Any]],
                           tags: Optional[List[str]] = None,
                           date_range: Optional[Dict[str, str]] = None,
                           include_likes: bool = False) -> List[str]:
        """投稿から画像URLを抽出"""
        image_urls = []

        for post in posts:
            # 投稿タイプチェック
            if post.get('type') != 'photo':
                continue

            # 日付フィルタ
            if date_range:
                post_timestamp = post.get('timestamp')
                if post_timestamp:
                    post_date = time.strftime('%Y-%m-%d', time.localtime(post_timestamp))
                    if not self._date_in_range(post_date, date_range):
                        continue

            # タグフィルタ
            if tags:
                post_tags = [tag.lower() for tag in post.get('tags', [])]
                if not any(tag.lower() in post_tags for tag in tags):
                    continue

            # 画像URL抽出
            photos = post.get('photos', [])
            for photo in photos:
                original_size = photo.get('original_size')
                if original_size and 'url' in original_size:
                    image_urls.append(original_size['url'])

        return image_urls

    def _date_in_range(self, date_str: str, date_range: Dict[str, str]) -> bool:
        """日付が範囲内かどうかチェック"""
        try:
            from datetime import datetime
            post_date = datetime.strptime(date_str, '%Y-%m-%d').date()

            start_date = None
            end_date = None

            if 'start' in date_range and date_range['start']:
                start_date = datetime.strptime(date_range['start'], '%Y-%m-%d').date()
            if 'end' in date_range and date_range['end']:
                end_date = datetime.strptime(date_range['end'], '%Y-%m-%d').date()

            if start_date and post_date < start_date:
                return False
            if end_date and post_date > end_date:
                return False

            return True
        except Exception as e:
            logger.error(f"Date range check error: {e}")
            return True  # エラーの場合は許可

    def _create_progress_callback(self):
        """進捗表示コールバックを作成"""
        display = ProgressDisplay()
        return lambda current, total: display.update_progress(current, total)

    # その他のメソッドは必要に応じて実装
    def get_blog_posts(self, blog_name: str, limit: int = 20, offset: int = 0):
        """ApiManagerに委譲"""
        return self.api_manager.get_blog_posts(blog_name, limit, offset)

    def list_billing_plans(self):
        """課金プラン一覧"""
        if not self.billing_manager:
            raise ConfigurationError("Stripe課金が設定されていません")
        return self.billing_manager.list_products()

    def create_checkout_session(self, plan_key: str, customer_email: Optional[str] = None):
        """Checkoutセッション作成"""
        if not self.billing_manager:
            raise ConfigurationError("Stripe課金が初期化されていないためCheckoutを作成できません")
        return self.billing_manager.create_checkout_session(plan_key, customer_email)

    def apply_license_file(self, license_file: str) -> None:
        """ライセンスファイル適用"""
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
        """ライセンス情報エクスポート"""
        info = self.license_manager.get_license()
        return {
            'status': info.status.value,
            'plan_key': info.plan_key,
            'current_period_end': info.current_period_end,
            'customer_email': info.customer_email,
            'stripe_subscription_id': info.stripe_subscription_id,
            'metadata': info.metadata or {}
        }
