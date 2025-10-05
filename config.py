"""Configuration wizard and validator for Tumblr Image Collector.

This module provides interactive configuration setup with validation
and environment variable support.
"""

import json
import os
import sys
import getpass
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

class ConfigWizard:
    """Interactive configuration wizard with validation.

    Attributes:
        config_path (Path): Path to configuration file
        config (Dict): Current configuration dictionary
    """

    def __init__(self, config_path: str = 'config.json'):
        """Initialize configuration wizard.

        Args:
            config_path: Path to configuration file (default: config.json)
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = self._load_existing_config()
        self._apply_defaults()

    @staticmethod
    def default_config_values() -> Dict[str, Any]:
        """Return default configuration values.

        Returns:
            Dictionary containing default configuration
        """
        return {
            'output_folder_name': 'tumblr_images',
            'max_download_workers': 5,
            'enable_deep_model': False,
            'filters': {
                'max_file_size_mb': 10,
                'nsfw_threshold': 0.35
            },
            'network': {
                'download_timeout_seconds': 30,
                'max_retries': 3,
                'backoff_factor': 0.5,
                'max_backoff_seconds': 30
            },
            'logging': {
                'level': 'INFO',
                'max_bytes': 5 * 1024 * 1024,
                'backup_count': 5
            },
            'cache': {
                'enabled': True,
                'ttl_seconds': 24 * 60 * 60,
                'max_entries': 2048
            }
        }

    def _apply_defaults(self) -> None:
        """Merge user config with default values.

        Ensures all required configuration keys exist with sensible defaults.
        """
        defaults = self.default_config_values()
        merged = defaults.copy()
        merged.update(self.config)

        # Merge nested dictionaries
        for nested_key in ('filters', 'network', 'cache', 'logging'):
            default_nested = defaults.get(nested_key, {}).copy()
            user_nested = self.config.get(nested_key, {})
            default_nested.update(user_nested)
            merged[nested_key] = default_nested

        self.config = merged

    @staticmethod
    def _sanitize_config_for_display(config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive information from config for display.

        Args:
            config: Configuration dictionary

        Returns:
            Sanitized configuration with secrets redacted
        """
        sanitized = deepcopy(config)

        # Redact API credentials
        for key in ('consumer_key', 'consumer_secret'):
            if key in sanitized:
                sanitized[key] = '***REDACTED***'

        # Redact proxy password
        proxy = sanitized.get('proxy')
        if isinstance(proxy, dict) and proxy.get('password'):
            proxy['password'] = '***REDACTED***'

        return sanitized


    @staticmethod
    def _ask_int(prompt: str, default: int) -> int:
        """Prompt user for integer input.

        Args:
            prompt: Question to ask user
            default: Default value if user presses Enter

        Returns:
            Integer value from user or default
        """
        while True:
            raw = input(f"{prompt} (default: {default}): ").strip()
            if not raw:
                return default
            if raw.isdigit():
                return int(raw)
            print("Please enter a valid integer.")

    @staticmethod
    def _ask_float(prompt: str, default: float) -> float:
        """Prompt user for float input.

        Args:
            prompt: Question to ask user
            default: Default value if user presses Enter

        Returns:
            Float value from user or default
        """
        while True:
            raw = input(f"{prompt} (default: {default}): ").strip()
            if not raw:
                return default
            try:
                return float(raw)
            except ValueError:
                print("Please enter a valid number.")

    @staticmethod
    def _ask_yes_no(prompt: str, default: bool = False) -> bool:
        """Prompt user for yes/no input.

        Args:
            prompt: Question to ask user
            default: Default value if user presses Enter

        Returns:
            True for yes, False for no
        """
        default_text = 'y' if default else 'n'
        while True:
            raw = input(f"{prompt} (y/n, default: {default_text.upper()}): ").strip().lower()
            if not raw:
                return default
            if raw in ('y', 'yes'):
                return True
            if raw in ('n', 'no'):
                return False
            print("Please answer with 'y' or 'n'.")

    def _load_existing_config(self) -> Dict[str, Any]:
        """Load existing configuration file if it exists.

        Returns:
            Configuration dictionary or empty dict if file doesn't exist
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in config file: {e}")
                return {}
            except Exception as e:
                print(f"Warning: Failed to load config: {e}")
                return {}
        return {}

    def _validate_tumblr_credentials(self, consumer_key: str, consumer_secret: str) -> bool:
        """Validate Tumblr API credentials format.

        Args:
            consumer_key: Tumblr API consumer key
            consumer_secret: Tumblr API consumer secret

        Returns:
            True if credentials pass validation, False otherwise
        """
        if not consumer_key or not consumer_secret:
            return False

        # 型チェック
        if not isinstance(consumer_key, str) or not isinstance(consumer_secret, str):
            return False

        # 長さチェック（最小20文字、最大512文字）
        if len(consumer_key) < 20 or len(consumer_key) > 512:
            return False
        if len(consumer_secret) < 20 or len(consumer_secret) > 512:
            return False

        # 許可される文字のみチェック（英数字、アンダースコア、ハイフン）
        pattern = re.compile(r"^[A-Za-z0-9_-]+$")
        if not pattern.fullmatch(consumer_key):
            return False
        if not pattern.fullmatch(consumer_secret):
            return False

        # 同一でないことを確認
        if consumer_key == consumer_secret:
            return False

        return True

    def _get_proxy_config(self):
        """プロキシ設定を対話的に取得"""
        print("\nプロキシ設定 (オプション)")
        use_proxy = self._ask_yes_no("プロキシを使用しますか？", False)

        if not use_proxy:
            return None

        proxy_types = ['http', 'https', 'socks4', 'socks5']
        while True:
            proxy_type = input(f"プロキシタイプを選択 {proxy_types}: ").strip().lower()
            if proxy_type in proxy_types:
                break
            print("無効なプロキシタイプです。")

        # ホストの入力と検証
        while True:
            host = input("プロキシホスト (例: 127.0.0.1): ").strip()
            if host and len(host) <= 255 and re.match(r'^[a-zA-Z0-9.-]+$', host):
                break
            print("無効なホスト名です。")

        # ポートの入力と検証
        while True:
            port_str = input("プロキシポート (例: 8080): ").strip()
            try:
                port = int(port_str)
                if 1 <= port <= 65535:
                    break
                print("ポート番号は1〜65535の範囲で入力してください。")
            except ValueError:
                print("無効なポート番号です。")

        use_auth = self._ask_yes_no("認証が必要ですか？", False)
        username = password = None

        if use_auth:
            username = input("ユーザー名: ").strip()
            password = getpass.getpass("パスワード: ")

            # 認証情報の検証
            if not username or len(username) > 256:
                print("警告: ユーザー名が無効です")
                username = None
                password = None

        return {
            'type': proxy_type,
            'host': host,
            'port': port,
            'username': username,
            'password': password
        }

    def run(self):
        """設定ウィザードを実行"""
        print("Tumblr Image Collector 設定ウィザード")
        
        # Tumblr API認証情報
        print("\nTumblr API 認証情報")
        consumer_key, consumer_secret = self._obtain_tumblr_credentials()
        self.config['consumer_key'] = consumer_key
        self.config['consumer_secret'] = consumer_secret

        # 出力フォルダ
        print("\n出力設定")
        current_output = self.config.get('output_folder_name', 'tumblr_images')
        output_folder = input(f"画像の保存先フォルダ (デフォルト: {current_output}): ") or current_output
        self.config['output_folder_name'] = output_folder

        # 深層学習モデルの利用設定
        print("\n画像分類オプション")
        current_deep_model = self.config.get('enable_deep_model', False)
        self.config['enable_deep_model'] = self._ask_yes_no(
            "TensorFlowを利用した詳細な画像分類を有効にしますか？",
            current_deep_model
        )

        # プロキシ設定
        proxy_config = self._get_proxy_config()
        if proxy_config:
            self.config['proxy'] = proxy_config

        # 高度なフィルタリング設定
        print("\n画像フィルタリング設定")
        filters_defaults = self.config.get('filters', {})
        current_max_size = filters_defaults.get('max_file_size_mb', 10)
        current_nsfw_threshold = filters_defaults.get('nsfw_threshold', 0.35)

        max_file_size = input(f"最大ファイルサイズ (MB, デフォルト: {current_max_size}): ") or current_max_size

        while True:
            nsfw_threshold_raw = input(
                f"NSFW判定閾値 (0.0〜1.0, デフォルト: {current_nsfw_threshold}): "
            ).strip()
            if not nsfw_threshold_raw:
                nsfw_threshold = current_nsfw_threshold
                break
            try:
                nsfw_threshold = float(nsfw_threshold_raw)
                if 0.0 <= nsfw_threshold <= 1.0:
                    break
                print("0.0〜1.0の範囲で入力してください。")
            except ValueError:
                print("数値を入力してください。")

        self.config['filters'] = {
            'max_file_size_mb': int(max_file_size),
            'nsfw_threshold': float(nsfw_threshold)
        }

        # ネットワーク設定
        print("\nネットワーク設定")
        network_defaults = self.config.get('network', {})
        download_timeout = self._ask_int("ダウンロードタイムアウト秒", network_defaults.get('download_timeout_seconds', 30))
        max_retries = self._ask_int("最大再試行回数", network_defaults.get('max_retries', 3))
        backoff_factor = self._ask_float("バックオフ係数", network_defaults.get('backoff_factor', 0.5))
        max_backoff_seconds = self._ask_int("バックオフ最大待機秒", network_defaults.get('max_backoff_seconds', 30))

        self.config['network'] = {
            'download_timeout_seconds': download_timeout,
            'max_retries': max(0, max_retries),
            'backoff_factor': max(0.0, backoff_factor),
            'max_backoff_seconds': max(1, max_backoff_seconds)
        }

        # ログ設定
        print("\nログ設定")
        logging_defaults = self.config.get('logging', {})
        if self._ask_yes_no("ログローテーション設定をカスタマイズしますか？", False):
            current_max_bytes_mb = max(1, int(logging_defaults.get('max_bytes', 5 * 1024 * 1024) / (1024 * 1024)))
            max_bytes_mb = self._ask_int("ログファイル1個あたりの最大サイズ (MB)", current_max_bytes_mb)
            backup_count = self._ask_int("ログファイルの世代数", logging_defaults.get('backup_count', 5))
            current_level = logging_defaults.get('level', 'INFO').upper()
            level_prompt = input(
                f"ログレベル (DEBUG/INFO/WARNING/ERROR, デフォルト: {current_level}): "
            ).strip().upper()
            valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
            log_level = level_prompt if level_prompt in valid_levels else current_level

            self.config['logging'] = {
                'level': log_level,
                'max_bytes': max(1, max_bytes_mb) * 1024 * 1024,
                'backup_count': max(1, backup_count)
            }
        else:
            self.config['logging'] = logging_defaults

        # キャッシュ設定
        print("\nキャッシュ設定")
        cache_defaults = self.config.get('cache', {})
        cache_enabled = self._ask_yes_no("ダウンロードキャッシュを有効にしますか？", cache_defaults.get('enabled', True))
        ttl_seconds_default = int(cache_defaults.get('ttl_seconds', 24 * 60 * 60))
        max_entries_default = int(cache_defaults.get('max_entries', 2048))

        ttl_seconds_value = max(60, self._ask_int("キャッシュの有効期限 (秒)", ttl_seconds_default))
        max_entries_value = max(1, self._ask_int("キャッシュの最大保持件数", max_entries_default))

        self.config['cache'] = {
            'enabled': cache_enabled,
            'ttl_seconds': ttl_seconds_value,
            'max_entries': max_entries_value
        }

        # 設定を保存
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

        print("\n設定が正常に保存されました。")
        return self.config

    def _obtain_tumblr_credentials(self):
        env_consumer_key = os.getenv('TUMBLR_CONSUMER_KEY')
        env_consumer_secret = os.getenv('TUMBLR_CONSUMER_SECRET')

        if env_consumer_key and env_consumer_secret:
            if self._validate_tumblr_credentials(env_consumer_key, env_consumer_secret):
                print("環境変数からTumblr API認証情報を読み込みました。")
                return env_consumer_key, env_consumer_secret
            print("環境変数から読み込んだTumblr API認証情報が無効です。再入力してください。")

        existing_key = self.config.get('consumer_key')
        existing_secret = self.config.get('consumer_secret')
        if existing_key and existing_secret:
            if self._validate_tumblr_credentials(existing_key, existing_secret):
                if self._ask_yes_no("既存の設定ファイルに保存された認証情報を使用しますか？", True):
                    return existing_key, existing_secret
                print("認証情報を再入力します。")
            else:
                print("既存の設定ファイルの認証情報が無効です。再入力してください。")

        return self._prompt_for_tumblr_credentials()

    def _prompt_for_tumblr_credentials(self):
        while True:
            consumer_key = input("Consumer Key: ").strip()
            consumer_secret = getpass.getpass("Consumer Secret: ").strip()

            if self._validate_tumblr_credentials(consumer_key, consumer_secret):
                return consumer_key, consumer_secret
            print("無効な認証情報です。再入力してください。")

def main():
    wizard = ConfigWizard()
    config = wizard.run()
    print("\n設定の詳細:")
    sanitized = ConfigWizard._sanitize_config_for_display(config)
    print(json.dumps(sanitized, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
