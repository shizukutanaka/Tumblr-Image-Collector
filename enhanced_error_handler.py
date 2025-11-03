#!/usr/bin/env python3
"""
Enhanced Error Handler Module for Tumblr Image Collector
強化されたエラー処理機能を提供するモジュール
"""

import sys
import traceback
import json
from pathlib import Path
from collections import defaultdict, deque
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """エラータイプの列挙型"""
    NETWORK_ERROR = "network_error"
    INVALID_URL = "invalid_url"
    SERVER_ERROR = "server_error"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    AUTHENTICATION_ERROR = "authentication_error"
    PARSING_ERROR = "parsing_error"
    FILE_SYSTEM_ERROR = "file_system_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorHandler:
    """
    強化されたエラー処理を管理するクラス
    """

    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        """
        エラーハンドラーを初期化

        Args:
            max_retries: 最大リトライ回数
            backoff_factor: バックオフ係数
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        # エラー統計
        self.error_stats = {
            'total_errors': 0,
            'errors_by_type': {},
            'retry_successes': 0,
            'permanent_failures': 0
        }

        # 失敗したURLの追跡
        self.failed_urls = set()
        self.temporary_failures = {}  # URL -> 次回リトライ時間

        self.lock = threading.Lock()

    def handle_error(self, error: Exception, url: str = "", context: str = "") -> Dict[str, Any]:
        """
        エラーを処理

        Args:
            error: 発生したエラー
            url: エラーが発生したURL
            context: エラーの文脈情報

        Returns:
            エラー処理結果
        """
        error_type = self._classify_error(error)
        self._record_error(error_type, url, error)

        result = {
            'error_type': error_type,
            'should_retry': self._should_retry(error_type),
            'retry_after': self._calculate_retry_delay(error_type),
            'is_permanent': self._is_permanent_error(error_type),
            'context': context
        }

        # ログ記録
        logger.warning(f"エラー処理: {error_type.value} - {str(error)} - URL: {url} - Context: {context}")

        return result

    def _classify_error(self, error: Exception) -> ErrorType:
        """エラーを分類"""
        error_str = str(error).lower()

        if isinstance(error, requests.exceptions.Timeout):
            return ErrorType.TIMEOUT
        elif isinstance(error, requests.exceptions.ConnectionError):
            return ErrorType.NETWORK_ERROR
        elif isinstance(error, requests.exceptions.HTTPError):
            status_code = getattr(error.response, 'status_code', None)
            if status_code == 401:
                return ErrorType.AUTHENTICATION_ERROR
            elif status_code == 429:
                return ErrorType.RATE_LIMIT
            elif 500 <= status_code < 600:
                return ErrorType.SERVER_ERROR
            else:
                return ErrorType.NETWORK_ERROR
        elif 'invalid url' in error_str or 'malformed' in error_str:
            return ErrorType.INVALID_URL
        elif 'parsing' in error_str or 'decode' in error_str:
            return ErrorType.PARSING_ERROR
        elif 'file system' in error_str or 'permission denied' in error_str:
            return ErrorType.FILE_SYSTEM_ERROR
        else:
            return ErrorType.UNKNOWN_ERROR

    def _should_retry(self, error_type: ErrorType) -> bool:
        """リトライすべきか判定"""
        retryable_types = {
            ErrorType.NETWORK_ERROR,
            ErrorType.TIMEOUT,
            ErrorType.SERVER_ERROR,
            ErrorType.RATE_LIMIT
        }
        return error_type in retryable_types

    def _calculate_retry_delay(self, error_type: ErrorType) -> float:
        """リトライまでの遅延時間を計算"""
        base_delays = {
            ErrorType.NETWORK_ERROR: 1.0,
            ErrorType.TIMEOUT: 2.0,
            ErrorType.SERVER_ERROR: 5.0,
            ErrorType.RATE_LIMIT: 60.0  # レート制限時は1分待機
        }

        base_delay = base_delays.get(error_type, 1.0)

        # 指数バックオフを適用
        delay = base_delay * (self.backoff_factor ** min(self.max_retries, 3))

        # ジッターを追加（±25%のランダム性）
        jitter = delay * 0.25 * (2 * (0.5 - 0.5))  # -0.125 * delay から +0.125 * delay
        delay += jitter

        return max(delay, 0.1)  # 最低0.1秒

    def _is_permanent_error(self, error_type: ErrorType) -> bool:
        """恒久的なエラーか判定"""
        permanent_types = {
            ErrorType.INVALID_URL,
            ErrorType.AUTHENTICATION_ERROR,
            ErrorType.FILE_SYSTEM_ERROR
        }
        return error_type in permanent_types

    def _record_error(self, error_type: ErrorType, url: str, error: Exception):
        """エラー統計を記録"""
        with self.lock:
            self.error_stats['total_errors'] += 1

            if error_type.value not in self.error_stats['errors_by_type']:
                self.error_stats['errors_by_type'][error_type.value] = 0
            self.error_stats['errors_by_type'][error_type.value] += 1

            # 永久エラーの場合は失敗URLとして記録
            if self._is_permanent_error(error_type):
                self.failed_urls.add(url)
                self.error_stats['permanent_failures'] += 1

    def should_retry_url(self, url: str) -> bool:
        """URLをリトライすべきかチェック"""
        with self.lock:
            # 永久失敗URLはリトライしない
            if url in self.failed_urls:
                return False

            # 一時的失敗の場合、次回リトライ時間をチェック
            if url in self.temporary_failures:
                if time.time() >= self.temporary_failures[url]:
                    del self.temporary_failures[url]
                    return True
                return False

            return True

    def mark_retry_success(self, url: str):
        """リトライ成功を記録"""
        with self.lock:
            if url in self.temporary_failures:
                del self.temporary_failures[url]
            self.error_stats['retry_successes'] += 1

    def execute_with_retry(self, func: Callable, url: str = "", context: str = "", *args, **kwargs) -> Any:
        """
        リトライ付きで関数を実行

        Args:
            func: 実行する関数
            url: 処理対象のURL
            context: 文脈情報
            *args, **kwargs: 関数に渡す引数

        Returns:
            関数の実行結果
        """
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                error_result = self.handle_error(e, url, context)

                if not error_result['should_retry'] or not self.should_retry_url(url):
                    # リトライしない場合
                    logger.error(f"最終失敗: {context} - {str(e)}")
                    raise e

                # リトライする場合
                if attempt < self.max_retries:
                    retry_delay = error_result['retry_after']
                    if url:
                        self.temporary_failures[url] = time.time() + retry_delay

                    logger.info(f"リトライ {attempt + 1}/{self.max_retries} まで {retry_delay:.1f}秒待機: {context}")
                    time.sleep(retry_delay)

        # 全てのリトライが失敗
        logger.error(f"全リトライ失敗: {context}")
        raise Exception(f"全リトライ失敗: {context}")

    def get_error_stats(self) -> Dict[str, Any]:
        """エラー統計を取得"""
        with self.lock:
            return self.error_stats.copy()

    def clear_error_history(self):
        """エラー履歴をクリア"""
        with self.lock:
            self.failed_urls.clear()
            self.temporary_failures.clear()
            self.error_stats = {
                'total_errors': 0,
                'errors_by_type': {},
                'retry_successes': 0,
                'permanent_failures': 0
            }
        logger.info("エラー履歴をクリアしました")


# グローバルインスタンス
error_handler = ErrorHandler()

def get_error_handler() -> ErrorHandler:
    """エラーハンドラーのインスタンスを取得"""
    return error_handler
