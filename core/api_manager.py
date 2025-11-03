"""
Tumblr API Manager

Tumblr APIとの通信、認証、レート制限を管理するモジュール
"""

import pytumblr
import time
import logging
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
import webbrowser
import os

logger = logging.getLogger(__name__)

class ApiManager:
    """
    Tumblr APIとの通信を管理するクラス
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.consumer_key: Optional[str] = None
        self.consumer_secret: Optional[str] = None
        self.token: Optional[str] = None
        self.token_secret: Optional[str] = None
        self.client: Optional[pytumblr.TumblrRestClient] = None

        # レート制限設定
        self.rate_limiter = {
            'requests_per_minute': self.config.get('rate_limit', {}).get('requests_per_minute', 30),
            'burst_limit': self.config.get('rate_limit', {}).get('burst_limit', 5),
            'window_seconds': 60
        }
        self._request_timestamps = []

        # 初期化
        self._setup_credentials()
        self._initialize_client()

    def _setup_credentials(self) -> None:
        """APIキーとOAuthトークンを設定"""
        # 環境変数からの読み込み
        env_consumer_key = os.environ.get("TUMBLR_CONSUMER_KEY")
        env_consumer_secret = os.environ.get("TUMBLR_CONSUMER_SECRET")

        if env_consumer_key and env_consumer_secret:
            self.consumer_key = env_consumer_key.strip()
            self.consumer_secret = env_consumer_secret.strip()
            logger.info("Using Tumblr consumer credentials from environment variables.")
        else:
            self.consumer_key = self.config.get("consumer_key")
            self.consumer_secret = self.config.get("consumer_secret")

            if not (self.consumer_key and self.consumer_secret):
                logger.info("Tumblr API keys not found. Please enter them:")
                self.consumer_key = input("Enter your Tumblr Consumer Key: ").strip()
                self.consumer_secret = input("Enter your Tumblr Consumer Secret: ").strip()
                if not (self.consumer_key and self.consumer_secret):
                    logger.error("Consumer Key and Secret are required.")
                    raise ValueError("Missing API credentials")
                self.config["consumer_key"] = self.consumer_key
                self.config["consumer_secret"] = self.consumer_secret

        # OAuthトークンの設定
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
                    raise ValueError("Missing OAuth credentials")
                self.config["token"] = self.token
                self.config["token_secret"] = self.token_secret

    def _get_oauth_token(self) -> tuple[Optional[str], Optional[str]]:
        """OAuthアクセストークンを取得"""
        if not self.consumer_key or not self.consumer_secret:
            logger.error("Consumer key and secret must be set before getting OAuth token.")
            return None, None

        oauth_client = pytumblr.TumblrRestClient(self.consumer_key, self.consumer_secret)
        try:
            url = oauth_client.get_authorize_url()

            # URL検証
            if not url or not isinstance(url, str):
                logger.error("Invalid OAuth URL received")
                return None, None

            # Tumblrドメイン確認
            parsed = urlparse(url)
            if not parsed.netloc.endswith('tumblr.com'):
                logger.error(f"OAuth URL is not from Tumblr domain: {parsed.netloc}")
                return None, None

            logger.info("Please visit the following URL in your browser to get the OAuth verifier:")
            logger.info(url)

            try:
                webbrowser.open(url)
            except Exception as browser_error:
                logger.warning(f"Could not automatically open browser: {browser_error}")

            # Verifier入力と検証
            verifier = input("Enter the OAuth verifier here: ").strip()
            if not verifier:
                logger.error("OAuth verifier is required.")
                return None, None

            if not verifier.isalnum() or len(verifier) < 6 or len(verifier) > 128:
                logger.error("Invalid OAuth verifier format")
                return None, None

            oauth_client.get_access_token(verifier)

            if not oauth_client.token or not oauth_client.token_secret:
                logger.error("Failed to obtain valid OAuth tokens")
                return None, None

            logger.info("OAuth access token obtained successfully!")
            logger.debug(f"OAuth Token (first 10 chars): {oauth_client.token[:10]}...")
            return oauth_client.token, oauth_client.token_secret

        except KeyboardInterrupt:
            logger.info("OAuth token acquisition cancelled by user.")
            return None, None
        except Exception as e:
            logger.error(f"Error during OAuth token acquisition: {e}")
            return None, None

    def _initialize_client(self) -> None:
        """Tumblr APIクライアントを初期化"""
        if not all([self.consumer_key, self.consumer_secret, self.token, self.token_secret]):
            logger.error("Cannot initialize Tumblr client: Credentials missing.")
            raise ValueError("Cannot initialize client due to missing credentials")

        try:
            self.client = pytumblr.TumblrRestClient(
                self.consumer_key,
                self.consumer_secret,
                self.token,
                self.token_secret
            )
            logger.info("Tumblr client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Tumblr client: {e}")
            raise ConnectionError("Failed to initialize Tumblr client") from e

    def _check_rate_limit(self) -> bool:
        """レート制限をチェック"""
        current_time = time.time()
        rate_limiter = self.rate_limiter

        # 古いタイムスタンプを削除
        self._request_timestamps = [
            ts for ts in self._request_timestamps
            if current_time - ts < rate_limiter['window_seconds']
        ]

        # レート制限チェック
        if len(self._request_timestamps) >= rate_limiter['requests_per_minute']:
            recent_requests = [ts for ts in self._request_timestamps
                             if current_time - ts < 1]

            if len(recent_requests) >= rate_limiter['burst_limit']:
                sleep_time = 1.0
                logger.warning(f"Rate limit exceeded. Sleeping for {sleep_time} seconds.")
                time.sleep(sleep_time)
                return self._check_rate_limit()

        # リクエストタイムスタンプを記録
        self._request_timestamps.append(current_time)
        return True

    def get_blog_posts(self, blog_name: str, limit: int = 20, offset: int = 0) -> Optional[List[Dict[str, Any]]]:
        """
        Tumblrブログの投稿を取得

        Args:
            blog_name: ブログ名
            limit: 取得する投稿数
            offset: オフセット

        Returns:
            投稿データのリスト、またはNone（レート制限時）
        """
        # レート制限チェック
        if not self._check_rate_limit():
            return None

        try:
            # 入力検証
            if not self._validate_blog_name(blog_name):
                logger.error(f"Invalid blog name: {blog_name}")
                return None

            normalized_limit = max(1, min(int(limit or 1), 100))  # Tumblr APIの制限
            posts_data = self.client.posts(blog_name, limit=normalized_limit, offset=offset)

            return posts_data.get('posts', [])
        except Exception as e:
            if "limit" in str(e).lower() or "429" in str(e) or "too many requests" in str(e).lower():
                logger.warning(f"Rate limit likely hit while fetching posts for '{blog_name}'.")
                return None
            logger.error(f"Error fetching posts for {blog_name}: {e}")
            return []

    def _validate_blog_name(self, blog_name: str) -> bool:
        """ブログ名の検証"""
        if not blog_name or not isinstance(blog_name, str):
            return False
        if len(blog_name) > 100:  # 妥当な長さ制限
            return False
        # Tumblrのブログ名は英数字、ハイフン、アンダースコア、ドットのみ
        import re
        if not re.match(r'^[a-zA-Z0-9._-]+$', blog_name):
            return False
        return True

    def save_credentials_to_config(self, config: Dict[str, Any]) -> None:
        """認証情報を設定ファイルに保存"""
        if self.consumer_key:
            config["consumer_key"] = self.consumer_key
        if self.consumer_secret:
            config["consumer_secret"] = self.consumer_secret
        if self.token:
            config["token"] = self.token
        if self.token_secret:
            config["token_secret"] = self.token_secret
