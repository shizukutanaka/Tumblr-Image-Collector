"""
プログレス表示システム（URL検証機能強化版）
"""

import sys
import time
import re
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import threading
import logging

logger = logging.getLogger(__name__)

# Import localization system
try:
    from localization import get_localization_manager, msg, set_language
    _LOCALIZATION_AVAILABLE = True
except ImportError:
    _LOCALIZATION_AVAILABLE = False
    # Fallback functions if localization is not available
    def msg(key: str, **kwargs) -> str:
        return key
    def set_language(lang: str) -> bool:
        return False

class ProgressDisplay:
    """プログレス表示を管理するクラス（強化版）"""

    def __init__(self):
        self.current_task = ""
        self.total_items = 0
        self.completed_items = 0
        self.start_time = time.time()
        self.display_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.last_update_time = time.time()
        self.items_per_second = 0.0
        self.error_count = 0
        self.warning_count = 0
        self.success_count = 0
        self.status_messages = []

    def start_task(self, task_name: str, total_items: int = 0):
        """タスクを開始する（強化版）"""
        self.current_task = task_name
        self.total_items = total_items
        self.completed_items = 0
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.items_per_second = 0.0
        self.error_count = 0
        self.warning_count = 0
        self.success_count = 0
        self.status_messages = []
        self.is_running = True

        if total_items > 0:
            self._start_enhanced_progress_bar()
        else:
            logger.info(msg("initializing_system", task=task_name))

    def update_progress(self, increment: int = 1, status: str = "success"):
        """プログレスを更新する（強化版）"""
        self.completed_items += increment

        # ステータス統計を更新
        if status == "success":
            self.success_count += increment
        elif status == "error":
            self.error_count += increment
        elif status == "warning":
            self.warning_count += increment

        # 速度計算
        current_time = time.time()
        time_diff = current_time - self.last_update_time
        if time_diff > 0:
            self.items_per_second = increment / time_diff

        self.last_update_time = current_time
        self._update_enhanced_progress_bar()

    def add_status_message(self, message: str, level: str = "info"):
        """ステータスメッセージを追加"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        status_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message
        }
        self.status_messages.append(status_entry)

        # 最新50件のみ保持
        if len(self.status_messages) > 50:
            self.status_messages = self.status_messages[-50:]

        # ログにも記録
        if level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
        else:
            logger.info(message)

    def complete_task(self, success: bool = True):
        """タスクを完了する（強化版）"""
        self.is_running = False

        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=1.0)

        elapsed_time = time.time() - self.start_time

        # 最終統計を表示
        status_summary = self._generate_status_summary(elapsed_time, success)
        logger.info(f"\n{'='*80}")
        logger.info(msg("task_completion_summary"))
        logger.info(f"{'='*80}")
        logger.info(status_summary)
        logger.info(f"{'='*80}")

        if success:
            if self.total_items > 0:
                logger.info(msg("task_completed_with_items", task=self.current_task, completed=self.completed_items, total=self.total_items, time=elapsed_time))
            else:
                logger.info(msg("task_completed", task=self.current_task, time=elapsed_time))
        else:
            logger.error(msg("task_failed", task=self.current_task, time=elapsed_time))

    def _generate_status_summary(self, elapsed_time: float, success: bool) -> str:
        """ステータスサマリーを生成"""
        success_rate = (self.success_count / max(self.completed_items, 1)) * 100

        summary_lines = [
            msg("task_name", task=self.current_task),
            msg("execution_time", time=elapsed_time),
            msg("processed_items", completed=self.completed_items, total=self.total_items),
            msg("success_count", count=self.success_count, rate=success_rate),
            msg("errors_occurred", count=self.error_count),
            msg("warnings_issued", count=self.warning_count)
        ]

        if self.completed_items > 0 and elapsed_time > 0:
            avg_items_per_sec = self.completed_items / elapsed_time
            summary_lines.append(msg("avg_processing_speed", speed=avg_items_per_sec))

        if self.status_messages:
            summary_lines.append(msg("latest_message", message=self.status_messages[-1]['message']))

        return "\n".join(summary_lines)

    def _start_enhanced_progress_bar(self):
        """強化プログレスバーを開始する"""
        if not self.is_running:
            return

        self.display_thread = threading.Thread(target=self._enhanced_progress_bar_worker, daemon=True)
        self.display_thread.start()

    def _enhanced_progress_bar_worker(self):
        """強化プログレスバーワーカー"""
        while self.is_running and self.completed_items < self.total_items:
            self._update_enhanced_progress_bar()
            time.sleep(0.1)  # より滑らかな更新

    def _update_enhanced_progress_bar(self):
        """強化プログレスバーを更新する"""
        if self.total_items == 0:
            return

        percentage = min(100, (self.completed_items / self.total_items) * 100)
        elapsed_time = time.time() - self.start_time

        # ETA計算
        eta = 0
        if self.completed_items > 0 and self.items_per_second > 0:
            remaining_items = self.total_items - self.completed_items
            eta = remaining_items / self.items_per_second

        # プログレスバーの表示
        bar_length = 40
        filled_length = int(bar_length * percentage / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)

        # ステータス情報
        status_info = []
        if self.items_per_second > 0:
            status_info.append(msg("items_per_second", speed=self.items_per_second))
        if self.error_count > 0:
            status_info.append(msg("errors", count=self.error_count))
        if eta > 0:
            status_info.append(msg("remaining_time", time=eta))

        status_str = ", ".join(status_info)

        progress_text = msg("progress_display", task=self.current_task, bar=bar, percentage=percentage, completed=self.completed_items, total=self.total_items)

        if status_str:
            progress_text += f" | {status_str}"

        sys.stdout.write(progress_text)
        sys.stdout.flush()

    def show_final_stats(self, stats: Dict[str, Any]):
        """最終統計を表示する"""
        # 動的な幅で区切り線を生成（テキスト拡張対応）
        separator_width = 80  # デフォルト幅
        max_key_length = max(len(str(key)) for key in stats.keys()) if stats else 20
        separator_width = max(separator_width, max_key_length + 10)

        separator = "=" * separator_width

        logger.info("\n" + separator)
        logger.info(msg("final_statistics"))
        logger.info(separator)

        for key, value in stats.items():
            if isinstance(value, dict):
                logger.info(f"{key}:")
                for sub_key, sub_value in value.items():
                    # 長いテキストを適切に折り返す
                    wrapped_value = self._wrap_text(str(sub_value), separator_width - 4)
                    logger.info(f"  {sub_key}: {wrapped_value}")
            else:
                # 長いテキストを適切に折り返す
                wrapped_value = self._wrap_text(str(value), separator_width - len(str(key)) - 2)
                logger.info(f"{key}: {wrapped_value}")

        logger.info(separator)


class InteractiveCLI:
    """対話型CLIを提供するクラス（エラーメッセージ改善版）"""

    def __init__(self):
        self.progress = ProgressDisplay()
        self._command_history = []

        # URLホワイトリスト設定
        self.url_whitelist = {
            'tumblr.com',
            'www.tumblr.com',
            'assets.tumblr.com',
            'static.tumblr.com',
            'media.tumblr.com',
            '64.media.tumblr.com',
            'va.media.tumblr.com'
        }

        # 管理者権限でホワイトリストを変更可能にする設定
        self.whitelist_strict_mode = True

        # HTTPS強制設定
        self.https_enforcement = True
        self.allowed_http_domains = {'localhost', '127.0.0.1', '0.0.0.0'}

        # エラーメッセージ改善設定
        self.error_message_templates = self._load_error_templates()

        # 並列ダウンロード最適化設定
        self.optimal_workers = self._calculate_optimal_workers()
        self.system_info = self._get_system_info()

        # RTL言語サポート設定
        self.rtl_support_enabled = True

    def _get_text_alignment(self) -> str:
        """現在の言語に応じたテキスト整列を取得"""
        return "right" if self.progress.is_rtl_language() else "left"

    def _get_progress_bar_format(self) -> str:
        """RTL言語用のプログレスバーフォーマットを調整"""
        if self._get_text_alignment() == "right":
            return "\r{task}: {percentage:5.1f}% [{bar}] ({completed}/{total})"
        else:
            return "\r{task}: [{bar}] {percentage:5.1f}% ({completed}/{total})"

    def _format_menu_text(self, menu_text: str) -> str:
        """RTL言語用のメニューテキストフォーマットを調整"""
        if self._get_text_alignment() == "right":
            # RTL言語では番号を右側に配置
            lines = menu_text.split('\n')
            formatted_lines = []
            for line in lines:
                if line.strip() and any(line.strip().startswith(str(i)) for i in range(1, 10)):
                    # 番号付き行をRTL用に調整
                    parts = line.split('.', 1)
                    if len(parts) == 2:
                        number = parts[0].strip()
                        text = parts[1].strip()
                        formatted_lines.append(f"{text} {number}.")
                    else:
                        formatted_lines.append(line)
                else:
                    formatted_lines.append(line)
            return '\n'.join(formatted_lines)
        return menu_text

    def _create_rtl_aware_table(self, headers: list, rows: list) -> str:
        """RTL言語対応のテーブルを作成"""
        if not rows:
            return ""

        alignment = self._get_text_alignment()
        col_widths = [len(header) for header in headers]

        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        # 最小幅と最大幅を設定
        min_width = 8
        max_width = 40
        col_widths = [max(min_width, min(width, max_width)) for width in col_widths]

        # セパレーターの作成
        separator = "+" + "+".join("-" * (width + 2) for width in col_widths) + "+"

        # テーブルの作成
        table_lines = [separator]

        # ヘッダー行（RTL言語では右寄せ）
        if alignment == "right":
            header_line = "|" + "|".join(f" {header:>{width}} " for header, width in zip(headers, col_widths)) + "|"
        else:
            header_line = "|" + "|".join(f" {header:<{width}} " for header, width in zip(headers, col_widths)) + "|"

        table_lines.append(header_line)
        table_lines.append(separator)

        # データ行
        for row in rows:
            if alignment == "right":
                data_line = "|" + "|".join(f" {str(cell):>{width}} " for cell, width in zip(row, col_widths)) + "|"
            else:
                data_line = "|" + "|".join(f" {str(cell):<{width}} " for cell, width in zip(row, col_widths)) + "|"
            table_lines.append(data_line)

        table_lines.append(separator)

        return "\n".join(table_lines)

    def get_download_config(self) -> Dict[str, Any]:
        """ダウンロード設定を取得"""
        return {
            'optimal_workers': self.optimal_workers,
            'system_info': self.system_info,
            'recommendations': self._get_download_recommendations()
        }

    def _get_system_info(self) -> Dict[str, Any]:
        """システム情報を取得"""
        try:
            import psutil
            import os

            info = {
                'cpu_count': os.cpu_count() or 1,
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'platform': os.sys.platform,
                'python_version': os.sys.version
            }

            # ディスク情報
            disk = psutil.disk_usage('/')
            info['disk_total'] = disk.total
            info['disk_free'] = disk.free

            return info

        except ImportError:
            # psutilが利用できない場合のフォールバック
            import os
            return {
                'cpu_count': os.cpu_count() or 1,
                'cpu_count_logical': os.cpu_count() or 1,
                'memory_total': 0,
                'memory_available': 0,
                'platform': os.sys.platform,
                'python_version': os.sys.version,
                'disk_total': 0,
                'disk_free': 0,
                'note': 'psutil not available, limited system info'
            }

    def _calculate_optimal_workers(self) -> int:
        """最適なワーカー数を計算"""
        try:
            import psutil
            import os

            # CPUコア数を取得
            cpu_count = os.cpu_count() or 1

            # メモリサイズに基づいて調整（GB単位）
            memory_gb = psutil.virtual_memory().total / (1024**3)

            # 基本的な最適化ロジック
            if cpu_count <= 2:
                # 低スペックシステム
                base_workers = 2
            elif cpu_count <= 4:
                # 中スペックシステム
                base_workers = cpu_count
            elif cpu_count <= 8:
                # 高スペックシステム
                base_workers = cpu_count - 1
            else:
                # 超高スペックシステム
                base_workers = min(cpu_count - 2, 16)

            # メモリサイズによる調整
            if memory_gb < 4:
                # メモリ不足の場合
                base_workers = min(base_workers, 2)
            elif memory_gb < 8:
                # 標準メモリ
                base_workers = min(base_workers, 4)
            elif memory_gb > 16:
                # 大容量メモリ
                base_workers = min(base_workers * 2, 32)

            # ネットワーク考慮（ダウンロードタスクの場合）
            # ネットワークI/Oバウンドなので、少し多めに設定
            network_factor = 1.5
            optimal = max(1, int(base_workers * network_factor))

            logger.info(f"Calculated optimal workers: CPU={cpu_count}, Memory={memory_gb:.1f}GB, Optimal={optimal}")
            return optimal

        except ImportError:
            # psutilが利用できない場合
            import os
            fallback_count = max(1, (os.cpu_count() or 1) - 1)
            logger.info(f"Using fallback worker count: {fallback_count}")
            return fallback_count

    def get_download_config(self) -> Dict[str, Any]:
        """ダウンロード設定を取得"""
        return {
            'optimal_workers': self.optimal_workers,
            'system_info': self.system_info,
            'recommendations': self._get_download_recommendations()
        }

    def _get_download_recommendations(self) -> List[str]:
        """ダウンロード設定の推奨事項を取得"""
        recommendations = []

        try:
            import psutil

            # CPUベースの推奨
            cpu_count = self.system_info['cpu_count']
            if cpu_count > 8:
                recommendations.append(f"高性能CPU検出 ({cpu_count}コア) - 大規模ダウンロードに適しています")
            elif cpu_count < 4:
                recommendations.append(f"低スペックCPU検出 ({cpu_count}コア) - 小規模ダウンロードを推奨")

            # メモリベースの推奨
            memory_gb = self.system_info['memory_total'] / (1024**3)
            if memory_gb < 4:
                recommendations.append(f"メモリ容量が少ない ({memory_gb:.1f}GB) - バッチサイズを小さくしてください")
            elif memory_gb > 16:
                recommendations.append(f"大容量メモリ検出 ({memory_gb:.1f}GB) - 大規模ダウンロードが可能")

            # ディスクスペースの推奨
            disk_free_gb = self.system_info['disk_free'] / (1024**3)
            if disk_free_gb < 10:
                recommendations.append(f"ディスク容量が少ない ({disk_free_gb:.1f}GB) - 不要ファイルを削除してください")
            elif disk_free_gb > 100:
                recommendations.append(f"十分なディスク容量 ({disk_free_gb:.1f}GB) - 大規模ダウンロードが可能")

        except ImportError:
            recommendations.append("システム情報の取得に制限があります")

        return recommendations

    def _load_error_templates(self) -> Dict[str, Dict[str, str]]:
        """エラーメッセージテンプレートを読み込み"""
        return {
            # ネットワーク関連エラー
            'network_error': {
                'title': 'ネットワーク接続エラー',
                'user_message': 'インターネット接続を確認してください。ファイアウォールやプロキシの設定もチェックしてください。',
                'technical_details': 'ネットワークタイムアウトまたは接続拒否が発生しました。',
                'suggestion': 'しばらく待ってから再試行するか、ネットワーク設定を確認してください。'
            },

            # 認証関連エラー
            'auth_error': {
                'title': '認証エラー',
                'user_message': 'Tumblr APIの認証に失敗しました。',
                'technical_details': 'APIキーまたはOAuthトークンが無効です。',
                'suggestion': '設定画面からAPIキーを確認・更新してください。'
            },

            # ブログ関連エラー
            'blog_not_found': {
                'title': 'ブログが見つかりません',
                'user_message': '指定されたブログは存在しないか、非公開設定になっています。',
                'technical_details': 'HTTP 404エラーまたはブログのプライバシー設定によりアクセスできません。',
                'suggestion': 'ブログ名が正しいか確認し、公開設定になっているかチェックしてください。'
            },

            # レート制限エラー
            'rate_limit': {
                'title': 'レート制限に達しました',
                'user_message': 'APIの使用制限に達しました。',
                'technical_details': 'Tumblr APIのレート制限により一時的にアクセスが制限されています。',
                'suggestion': 'しばらく待ってから再試行してください。通常、数分で制限が解除されます。'
            },

            # ファイルシステムエラー
            'filesystem_error': {
                'title': 'ファイルシステムエラー',
                'user_message': 'ファイルの保存に失敗しました。',
                'technical_details': 'ディスク容量不足またはアクセス権限の問題が発生しました。',
                'suggestion': 'ディスクの空き容量を確認し、書き込み権限があるかチェックしてください。'
            },

            # 一般的なエラー
            'generic_error': {
                'title': 'エラーが発生しました',
                'user_message': '予期せぬエラーが発生しました。',
                'technical_details': 'システム内部でエラーが発生しました。',
                'suggestion': 'ログファイルを確認し、必要に応じて開発者に連絡してください。'
            }
        }

    def show_welcome(self):
        """ウェルカムメッセージを表示"""
        welcome_text = msg("welcome_banner")
        logger.info(msg("displaying_welcome"))
        print(welcome_text)

    def show_menu(self):
        """メインメニューを表示"""
        menu_text = self._format_menu_text(msg("main_menu"))
        logger.info(msg("displaying_main_menu"))
        print(menu_text)

    def get_user_choice(self, prompt: str, valid_choices: list) -> str:
        """ユーザーからの選択を取得"""
        while True:
            choice = input(prompt).strip().lower()

            if choice in valid_choices:
                return choice
            elif choice in ['q', 'quit', 'exit']:
                return 'quit'
            else:
                self.show_warning(msg("invalid_choice", choices=valid_choices))

    def prompt_blog_name(self, prompt: str = "収集するブログ名を入力してください") -> Optional[str]:
        """ブログ名を入力し、即時検証する"""
        while True:
            blog_input = input(f"{prompt}: ").strip()

            if not blog_input:
                self.show_warning("ブログ名が入力されませんでした。")
                continue

            # 基本的な検証
            if len(blog_input) > 63:
                self.show_warning("ブログ名は63文字以下でなければなりません。")
                continue

            # 文字種の検証（英数字とハイフンのみ）
            if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$', blog_input):
                self.show_warning("ブログ名は英数字とハイフン（-）のみ使用できます。ハイフンは先頭・末尾に使用できません。")
                continue

            # URL形式での入力も許可
            if blog_input.startswith(('http://', 'https://')):
                if URL_VALIDATOR_AVAILABLE:
                    validator = get_url_validator()
                    is_valid, url_type, message = validator.validate_url_format(blog_input)

                    if not is_valid:
                        self.show_warning(f"無効なURLです: {message}")
                        continue
                    elif url_type != 'blog':
                        self.show_warning("ブログのURLを入力してください。")
                        continue
                    else:
                        self.show_success(f"有効なブログURLです: {message}")
                        # URLからブログ名を抽出
                        blog_name = validator.extract_blog_name(blog_input)
                        if blog_name:
                            return blog_name
                        else:
                            return blog_input.replace('.tumblr.com', '').split('/')[-1]
                else:
                    # 基本的なURL検証
                    if 'tumblr.com' not in blog_input:
                        self.show_warning("TumblrのURLを入力してください。")
                        continue
                    self.show_info("URL検証機能が利用できないため、基本的な検証のみ行います。")

            # ブログ存在チェック（オプション）
            if self.confirm_action(f"ブログ '{blog_input}' を検証しますか？"):
                if URL_VALIDATOR_AVAILABLE:
                    try:
                        validator = get_url_validator()
                        exists, message = validator.validate_blog_exists(blog_input)

                        if exists:
                            self.show_success(f"ブログが存在することを確認しました: {message}")
                            return blog_input
                        else:
                            self.show_warning(f"ブログが存在しないか、非公開です: {message}")
                            if not self.confirm_action("続行しますか？"):
                                continue
                            return blog_input
                    except Exception as e:
                        self.show_warning(f"ブログ検証中にエラーが発生しました: {e}")
                        if self.confirm_action("続行しますか？"):
                            return blog_input
                else:
                    self.show_info("ブログ検証機能が利用できません。")
                    return blog_input
            else:
                return blog_input

    def prompt_url_input(self, prompt: str = "URLを入力してください", url_type: str = "any") -> Optional[str]:
        """URLを入力し、即時検証する"""
        while True:
            url_input = input(f"{prompt}: ").strip()

            if not url_input:
                self.show_warning("URLが入力されませんでした。")
                continue

            # URL長の検証
            if len(url_input) > 2048:
                self.show_warning("URLが長すぎます（2048文字以下）。")
                continue

            if URL_VALIDATOR_AVAILABLE:
                validator = get_url_validator()
                is_valid, validated_type, message = validator.validate_url_format(url_input)

                if not is_valid:
                    self.show_warning(f"無効なURLです: {message}")
                    continue

                # URLタイプの検証
                if url_type != "any" and validated_type != url_type:
                    self.show_warning(f"{url_type}のURLを入力してください。現在: {validated_type}")
                    continue

                self.show_success(f"有効なURLです: {message}")

                # アクセシビリティチェック（オプション）
                if self.confirm_action("URLのアクセシビリティをチェックしますか？"):
                    try:
                        accessible, status_code, access_message = validator.check_url_accessibility(url_input)
                        if accessible:
                            self.show_success(f"URLにアクセス可能です: {access_message}")
                            return url_input
                        else:
                            self.show_warning(f"URLにアクセスできません: {access_message}")
                            if not self.confirm_action("続行しますか？"):
                                continue
                            return url_input
                    except Exception as e:
                        self.show_warning(f"アクセシビリティチェック中にエラーが発生しました: {e}")
                        if self.confirm_action("続行しますか？"):
                            return url_input
                else:
                    return url_input
            else:
                # 基本的なURL検証
                if not url_input.startswith(('http://', 'https://')):
                    self.show_warning("http:// または https:// で始まるURLを入力してください。")
                    continue

    def cleanup_url_input(self, url: str) -> Tuple[str, List[str]]:
        """入力URLをクリーンアップし、警告を返す"""
        warnings = []
        original_url = url

        # 危険な文字やパターンの除去
        # SQLインジェクション対策
        dangerous_patterns = [
            ('script', 'script'), ('javascript:', 'javascript:'),
            ('vbscript:', 'vbscript:'), ('data:', 'data:'),
            ('file:', 'file:'), ('ftp:', 'ftp:'),
            ('--', '--'), ('/*', '/*'), ('*/', '*/'),
            ('\'', '\''), ('"', '"'), (';', ';'),
            ('<', '<'), ('>', '>'), ('{', '{'), ('}', '}'),
            ('(', '('), (')', ')')
        ]

        for pattern, replacement in dangerous_patterns:
            if pattern in url.lower():
                url = url.replace(pattern, '')
                warnings.append(f"危険なパターン '{pattern}' を除去しました")

        # 余分な空白の除去
        url = url.strip()

        # URLエンコードの修正
        try:
            from urllib.parse import unquote
            url = unquote(url)
        except:
            pass

        # ログに記録
        if warnings:
            logger.warning(f"URL cleaned up: {original_url} -> {url}")
            logger.warning(f"Warnings: {warnings}")

    def _wrap_text(self, text: str, max_width: int) -> str:
        """テキストを指定幅で折り返す（テキスト拡張対応）"""
        if not text or max_width <= 0:
            return text

        import textwrap
        return textwrap.fill(text, width=max_width, subsequent_indent="  ")

    def create_table(self, headers: list, rows: list) -> str:
        """テーブル形式でデータを表示（テキスト拡張対応）"""
        if not rows:
            return ""

        # 動的な列幅計算（テキスト拡張考慮）
        col_widths = [len(header) for header in headers]
        for row in rows:
            for i, cell in enumerate(row):
                cell_str = str(cell)
                col_widths[i] = max(col_widths[i], len(cell_str))

        # 最小幅と最大幅を設定（テキスト拡張対応）
        min_width = 8
        max_width = 50
        col_widths = [max(min_width, min(width, max_width)) for width in col_widths]

        # セパレーターの作成
        separator = "+" + "+".join("-" * (width + 2) for width in col_widths) + "+"

        # テーブルの作成
        table_lines = [separator]

        # ヘッダー行
        header_line = "|" + "|".join(f" {header:<{width}} " for header, width in zip(headers, col_widths)) + "|"
        table_lines.append(header_line)
        table_lines.append(separator)

        # データ行
        for row in rows:
            # 長いセル内容を折り返す
            wrapped_rows = [1]  # 各行の倍率
            for cell in row:
                cell_str = str(cell)
                if len(cell_str) > max_width:
                    wrapped_count = (len(cell_str) // max_width) + 1
                    wrapped_rows[0] = max(wrapped_rows[0], wrapped_count)

            # 複数行にまたがる場合の処理
            for line_idx in range(wrapped_rows[0]):
                row_lines = []
                for i, cell in enumerate(row):
                    cell_str = str(cell)
                    if len(cell_str) <= max_width:
                        # 短いセルはそのまま
                        row_lines.append(f" {cell_str:<{col_widths[i]}} ")
                    else:
                        # 長いセルを折り返す
                        import textwrap
                        wrapped_lines = textwrap.wrap(cell_str, width=max_width)
                        if line_idx < len(wrapped_lines):
                            row_lines.append(f" {wrapped_lines[line_idx]:<{col_widths[i]}} ")
                        else:
                            row_lines.append(f" {'':<{col_widths[i]}} ")

                table_lines.append("|" + "|".join(row_lines) + "|")

        table_lines.append(separator)

        return "\n".join(table_lines)

    def prompt_url_with_cleanup(self, prompt: str = "URLを入力してください", url_type: str = "any") -> Optional[str]:
        """URLを入力し、クリーンアップして検証する"""
        while True:
            url_input = input(f"{prompt}: ").strip()

            if not url_input:
                self.show_warning("URLが入力されませんでした。")
                continue

            # URLクリーンアップ
            cleaned_url, warnings = self.cleanup_url_input(url_input)

            # クリーンアップでURLが変更された場合
            if cleaned_url != url_input:
                self.show_info(f"URLがクリーンアップされました: {url_input} -> {cleaned_url}")
                for warning in warnings:
                    self.show_warning(warning)

                # クリーンアップされたURLを使用するか確認
                if not self.confirm_action("クリーンアップされたURLを使用しますか？"):
                    continue

                url_input = cleaned_url

            # 検証処理（prompt_url_inputと同じ）
            if len(url_input) > 2048:
                self.show_warning("URLが長すぎます（2048文字以下）。")
                continue

            if URL_VALIDATOR_AVAILABLE:
                validator = get_url_validator()
                is_valid, validated_type, message = validator.validate_url_format(url_input)

                if not is_valid:
                    self.show_warning(f"無効なURLです: {message}")
                    continue

                # URLタイプの検証
                if url_type != "any" and validated_type != url_type:
                    self.show_warning(f"{url_type}のURLを入力してください。現在: {validated_type}")
                    continue

                self.show_success(f"有効なURLです: {message}")

                # アクセシビリティチェック（オプション）
                if self.confirm_action("URLのアクセシビリティをチェックしますか？"):
                    try:
                        accessible, status_code, access_message = validator.check_url_accessibility(url_input)
                        if accessible:
                            self.show_success(f"URLにアクセス可能です: {access_message}")
                            return url_input
                        else:
                            self.show_warning(f"URLにアクセスできません: {access_message}")
                            if not self.confirm_action("続行しますか？"):
                                continue
                            return url_input
                    except Exception as e:
                        self.show_warning(f"アクセシビリティチェック中にエラーが発生しました: {e}")
                        if self.confirm_action("続行しますか？"):
                            return url_input
                else:
                    return url_input
            else:
                # 基本的なURL検証
                if not url_input.startswith(('http://', 'https://')):
                    self.show_warning("http:// または https:// で始まるURLを入力してください。")
                    continue

    def validate_url_whitelist(self, url: str) -> Tuple[bool, str]:
        """URLがホワイトリストに含まれているか検証"""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            if not parsed.netloc:
                return False, "無効なURL形式です"

            domain = parsed.netloc.lower()

            # サブドメインも許可（例: xxx.tumblr.com）
            for allowed_domain in self.url_whitelist:
                if domain == allowed_domain or domain.endswith('.' + allowed_domain):
                    return True, f"許可されたドメインです: {domain}"

            return False, f"許可されていないドメインです: {domain}"

    def validate_https_enforcement(self, url: str) -> Tuple[bool, str]:
        """HTTPS強制を検証"""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            if not parsed.scheme:
                return False, "URLにスキームが指定されていません"

            # HTTPスキームの場合
            if parsed.scheme == 'http':
                # 許可されたHTTPドメインかチェック
                domain = parsed.netloc.lower()
                for allowed_domain in self.allowed_http_domains:
                    if domain == allowed_domain or domain.endswith('.' + allowed_domain):
                        return True, "ローカルドメインのためHTTP許可"

                return False, f"HTTPSが必要です。HTTPは許可されていません: {domain}"

            # HTTPSスキームの場合
            elif parsed.scheme == 'https':
                return True, "HTTPSが使用されています"

            else:
                return False, f"サポートされていないスキームです: {parsed.scheme}"

        except Exception as e:
            return False, f"HTTPS検証エラー: {str(e)}"

    def enforce_https_on_url(self, url: str) -> str:
        """URLをHTTPSに強制変換"""
        try:
            from urllib.parse import urlparse, urlunparse

            parsed = urlparse(url)

            # 既にHTTPSの場合
            if parsed.scheme == 'https':
                return url

            # HTTPの場合のみHTTPSに変換
            if parsed.scheme == 'http':
                domain = parsed.netloc.lower()
                # 許可されたHTTPドメインは変換しない
                for allowed_domain in self.allowed_http_domains:
                    if domain == allowed_domain or domain.endswith('.' + allowed_domain):
                        return url

                # HTTPSに変換
                https_parts = parsed._replace(scheme='https')
                return urlunparse(https_parts)

            return url

        except Exception as e:
            logger.warning(f"HTTPS変換エラー: {e}")
            return url

    def prompt_url_with_whitelist(self, prompt: str = "URLを入力してください", url_type: str = "any") -> Optional[str]:
        """ホワイトリストを強制してURLを入力・検証する"""
        while True:
            url_input = input(f"{prompt}: ").strip()

            if not url_input:
                self.show_warning("URLが入力されませんでした。")
                continue

            # URLクリーンアップ
            cleaned_url, warnings = self.cleanup_url_input(url_input)

            # クリーンアップでURLが変更された場合
            if cleaned_url != url_input:
                self.show_info(f"URLがクリーンアップされました: {url_input} -> {cleaned_url}")
                for warning in warnings:
                    self.show_warning(warning)

                if not self.confirm_action("クリーンアップされたURLを使用しますか？"):
                    continue

                url_input = cleaned_url

            # ホワイトリスト検証（最優先）
            if self.whitelist_strict_mode:
                is_whitelisted, whitelist_message = self.validate_url_whitelist(url_input)

                if not is_whitelisted:
                    self.show_error(f"セキュリティ違反: {whitelist_message}")
                    self.show_info(f"許可されているドメイン: {', '.join(sorted(self.url_whitelist))}")

                    if not self.confirm_action("本当にこのURLを使用しますか？（推奨しません）"):
                        continue
                else:
                    self.show_success(f"セキュリティチェック通過: {whitelist_message}")

            # HTTPS強制検証
            if self.https_enforcement:
                is_https_valid, https_message = self.validate_https_enforcement(url_input)

                if not is_https_valid:
                    self.show_warning(f"HTTPSセキュリティ警告: {https_message}")

                    # HTTPSに自動変換
                    converted_url = self.enforce_https_on_url(url_input)
                    if converted_url != url_input:
                        self.show_info(f"URLをHTTPSに自動変換: {url_input} -> {converted_url}")
                        url_input = converted_url
                        self.show_success("HTTPS変換完了")
                    else:
                        if not self.confirm_action("HTTPSを使用できないURLですが続行しますか？"):
                            continue
                else:
                    self.show_success(f"HTTPSチェック通過: {https_message}")

            # 基本的なURL検証
            if len(url_input) > 2048:
                self.show_warning("URLが長すぎます（2048文字以下）。")
                continue

            if URL_VALIDATOR_AVAILABLE:
                validator = get_url_validator()
                is_valid, validated_type, message = validator.validate_url_format(url_input)

                if not is_valid:
                    self.show_warning(f"無効なURLです: {message}")
                    continue

                # URLタイプの検証
                if url_type != "any" and validated_type != url_type:
                    self.show_warning(f"{url_type}のURLを入力してください。現在: {validated_type}")
                    continue

                self.show_success(f"有効なURLです: {message}")

                # アクセシビリティチェック（オプション）
                if self.confirm_action("URLのアクセシビリティをチェックしますか？"):
                    try:
                        accessible, status_code, access_message = validator.check_url_accessibility(url_input)
                        if accessible:
                            self.show_success(f"URLにアクセス可能です: {access_message}")
                            return url_input
                        else:
                            self.show_warning(f"URLにアクセスできません: {access_message}")
                            if not self.confirm_action("続行しますか？"):
                                continue
                            return url_input
                    except Exception as e:
                        self.show_warning(f"アクセシビリティチェック中にエラーが発生しました: {e}")
                        if self.confirm_action("続行しますか？"):
                            return url_input
                else:
                    return url_input
            else:
                # 基本的なURL検証
                if not url_input.startswith(('http://', 'https://')):
                    self.show_warning("http:// または https:// で始まるURLを入力してください。")
                    continue

                self.show_info("高度なURL検証機能が利用できないため、基本的な検証のみ行います。")
                return url_input

    def add_to_whitelist(self, domain: str) -> bool:
        """ホワイトリストにドメインを追加（管理者機能）"""
        if not self.confirm_action(f"ドメイン '{domain}' をホワイトリストに追加しますか？"):
            return False

        try:
            # ドメイン形式の検証
            if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$', domain):
                self.show_error("無効なドメイン形式です。")
                return False

            self.url_whitelist.add(domain.lower())
            self.show_success(f"ドメイン '{domain}' をホワイトリストに追加しました。")
            logger.info(f"Added domain to whitelist: {domain}")
            return True

        except Exception as e:
            self.show_error(f"ホワイトリスト追加エラー: {e}")
            return False

    def remove_from_whitelist(self, domain: str) -> bool:
        """ホワイトリストからドメインを削除（管理者機能）"""
        if not self.confirm_action(f"ドメイン '{domain}' をホワイトリストから削除しますか？"):
            return False

        try:
            removed = self.url_whitelist.discard(domain.lower())
            if removed:
                self.show_success(f"ドメイン '{domain}' をホワイトリストから削除しました。")
                logger.info(f"Removed domain from whitelist: {domain}")
                return True
            else:
                self.show_warning(f"ドメイン '{domain}' はホワイトリストに含まれていませんでした。")
                return False

        except Exception as e:
            self.show_error(f"ホワイトリスト削除エラー: {e}")
            return False

    def prompt_int(self, prompt: str, default: int = 0, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
        """整数入力を求める"""
        while True:
            try:
                if default is not None and default != 0:
                    result = input(f"{prompt} [{default}]: ").strip()
                    value = int(result) if result else default
                else:
                    result = input(f"{prompt}: ").strip()
                    value = int(result)

                if min_val is not None and value < min_val:
                    self.show_warning(f"値は{min_val}以上でなければなりません。")
                    continue
                if max_val is not None and value > max_val:
                    self.show_warning(f"値は{max_val}以下でなければなりません。")
                    continue

                return value
            except ValueError:
                self.show_warning("有効な整数を入力してください。")

    def prompt_bool(self, prompt: str, default: bool = False) -> bool:
        """真偽値入力を求める"""
        default_text = "(Y/n)" if default else "(y/N)"
        while True:
            result = input(f"{prompt} {default_text}: ").strip().lower()
            if not result:
                return default
            elif result in ['y', 'yes', 'true', '1']:
                return True
            elif result in ['n', 'no', 'false', '0']:
                return False
            else:
                self.show_warning("y/n または yes/no で回答してください。")

    def confirm_action(self, message: str) -> bool:
        """ユーザーの確認を求める"""
        return self.prompt_bool(f"{message} 実行しますか？")

    def show_error(self, error_message: str):
        """エラーメッセージを表示"""
        print(f"\n❌ エラー: {error_message}")
        logger.error(error_message)

    def show_warning(self, warning_message: str):
        """警告メッセージを表示"""
        print(f"\n⚠️  警告: {warning_message}")
        logger.warning(warning_message)

    def show_success(self, success_message: str):
        """成功メッセージを表示"""
        print(f"\n✅ 成功: {success_message}")
        logger.info(success_message)

    def show_info(self, info_message: str):
        """情報メッセージを表示"""
        print(f"\nℹ️  情報: {info_message}")
        logger.info(info_message)

    def show_progress(self, current: int, total: int, task: str = "処理中"):
        """プログレス情報を表示"""
        if total > 0:
            percentage = (current / total) * 100
            progress_message = f"{task}: {current}/{total} ({percentage:.1f}%)"
            logger.debug(progress_message)
            print(f"\r{progress_message}", end="", flush=True)

            if current >= total:
                print()  # 改行
                logger.info(f"Progress complete for {task}.")

    def create_table(self, headers: list, rows: list) -> str:
        """テーブル形式でデータを表示"""
        if not rows:
            return ""

        # 列幅の計算
        col_widths = [len(header) for header in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        # セパレーターの作成
        separator = "+" + "+".join("-" * (width + 2) for width in col_widths) + "+"

        # テーブルの作成
        table_lines = [separator]

        # ヘッダー行
        header_line = "|" + "|".join(f" {header:<{width}} " for header, width in zip(headers, col_widths)) + "|"
        table_lines.append(header_line)
        table_lines.append(separator)

        # データ行
        for row in rows:
            data_line = "|" + "|".join(f" {str(cell):<{width}} " for cell, width in zip(row, col_widths)) + "|"
            table_lines.append(data_line)

        table_lines.append(separator)

        return "\n".join(table_lines)

    def run_interactive_mode(self, collector):
        """対話型モードを実行"""
        self.show_welcome()

        while True:
            try:
                self.show_menu()
                choice = self.get_user_choice("選択してください (1-10)", ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

                if choice == 'quit':
                    self.show_info("終了します...")
                    break

                elif choice == '1':
                    self._handle_single_blog_collection(collector)
                elif choice == '2':
                    self._handle_multi_blog_search(collector)
                elif choice == '3':
                    self._handle_tag_search(collector)
                elif choice == '4':
                    self._handle_metadata_export(collector)
                elif choice == '5':
                    self._handle_stats_display(collector)
                elif choice == '6':
                    self._handle_language_settings()
                elif choice == '7':
                    self._handle_config_change(collector)
                elif choice == '8':
                    self._handle_translation_management()
                elif choice == '9':
                    self._show_help()
                elif choice == '10':
                    break

            except KeyboardInterrupt:
                self.show_warning("ユーザーによる中断を検出しました。")
                break
            except Exception as e:
                self.show_error(f"予期せぬエラー: {e}")
                logger.exception("Interactive mode error")

    def _handle_single_blog_collection(self, collector):
        """単一ブログ収集の処理（URL検証機能強化版）"""
        blog_name = self.prompt_blog_name()
        if not blog_name:
            return

        if self.confirm_action(f"ブログ '{blog_name}' から画像を収集しますか？"):
            try:
                self.progress.start_task(f"ブログ '{blog_name}' からの画像収集", 0)
                collector.run(blog_name)
                self.progress.complete_task(True)
                self.show_success(f"ブログ '{blog_name}' からの収集が完了しました。")
            except Exception as e:
                self.progress.complete_task(False)
                error_type = self.parse_error_type(e)
                context = {'blog_name': blog_name}
                self.show_enhanced_error(error_type, e, context)

    def _handle_multi_blog_search(self, collector):
        """複数ブログ検索の処理"""
        blogs_input = self.prompt_string("検索するブログ名をカンマ区切りで入力してください（例: blog1,blog2,blog3）")
        if not blogs_input:
            self.show_warning("ブログ名が入力されませんでした。")
            return

        blog_names = [blog.strip() for blog in blogs_input.split(",") if blog.strip()]

        if self.confirm_action(f"{len(blog_names)}個のブログを検索しますか？"):
            total_blogs = len(blog_names)
            successful_blogs = 0

            for i, blog_name in enumerate(blog_names):
                try:
                    self.progress.show_progress(i, total_blogs, f"ブログ検索中")
                    collector.run(blog_name)
                    successful_blogs += 1
                except Exception as e:
                    self.show_error(f"ブログ '{blog_name}' の処理中にエラーが発生しました: {e}")
                    error_type = self.parse_error_type(e)
                    context = {'blog_name': blog_name, 'index': i + 1, 'total': total_blogs}
                    self.show_enhanced_error(error_type, e, context)

            self.show_success(msg("blogs_searched", count=successful_blogs))

    def _handle_tag_search(self, collector):
        """タグ検索の処理"""
        tag = self.prompt_string("検索するタグを入力してください")
        if not tag:
            self.show_warning("タグが入力されませんでした。")
            return

        count = self.prompt_int("取得する画像数", 20, 1, 1000)

        if self.confirm_action(msg("images_collected", count=count)):
            try:
                self.progress.start_task(f"タグ '{tag}' の画像検索", count)
                # タグ検索の実装（実際のメソッドに合わせる）
                # collector.search_by_tag(tag, count)
                self.progress.complete_task(True)
                self.show_success("タグ検索が完了しました。")
            except Exception as e:
                self.progress.complete_task(False)
                self.show_error(f"検索中にエラーが発生しました: {e}")

    def _handle_metadata_export(self, collector):
        """メタデータエクスポートの処理"""
        format_choice = self.get_user_choice("エクスポート形式を選択してください (json/csv)", ['json', 'csv'])

        if self.confirm_action(f"メタデータを{format_choice.upper()}形式でエクスポートしますか？"):
            try:
                export_path = collector.export_metadata(format_choice)
                self.show_success(f"メタデータがエクスポートされました: {export_path}")
            except Exception as e:
                self.show_error(f"エクスポート中にエラーが発生しました: {e}")

    def _handle_stats_display(self, collector):
        """統計情報の表示処理"""
        try:
            collector.print_download_stats()
            self.progress.show_final_stats(collector._download_stats)
        except Exception as e:
            self.show_error(f"統計情報の取得中にエラーが発生しました: {e}")

    def _handle_config_change(self, collector):
        """設定変更の処理"""
        self.show_info("設定変更機能は開発中です。")
        # 設定変更の実装

    def _show_help(self):
        """ヘルプの表示"""
        help_text = msg("help_text")
    def _handle_translation_management(self):
        """翻訳管理の処理"""
        if not _LOCALIZATION_AVAILABLE:
            self.show_warning(msg("translation_management_unavailable"))
            return

        while True:
            try:
                self.show_info(msg("translation_management"))
                self.show_info("1. " + msg("translation_report_generated").replace("{path}", "").replace(":", "").replace("生成されました", "生成"))
                self.show_info("2. " + msg("translation_coverage_summary").replace(":", ""))
                self.show_info("3. " + msg("missing_translations_for").replace("{lang}", "").replace(":", "").replace("の不足翻訳", "不足翻訳検出"))
                self.show_info("4. " + msg("unused_translations_check"))
                self.show_info("5. " + msg("new_language_code_prompt").replace(" (例: fr, de, es)", ""))
                self.show_info("6. " + msg("back"))

                choice = self.get_user_choice(msg("translation_management_menu"), ['1', '2', '3', '4', '5', '6'])

                if choice == '1':
                    self._generate_translation_report()
                elif choice == '2':
                    self._show_translation_coverage()
                elif choice == '3':
                    self._show_missing_translations()
                elif choice == '4':
                    self._show_unused_translations()
                elif choice == '5':
                    self._create_new_language_template()
                elif choice == '6':
                    break

            except Exception as e:
                self.show_error(f"翻訳管理中にエラーが発生しました: {e}")

    def _generate_translation_report(self):
        """翻訳品質レポートを生成"""
        try:
            report_file = generate_translation_report("translation_quality_report.json")
            if report_file:
                self.show_success(f"翻訳品質レポートが生成されました: {report_file}")

                # レポートの内容を表示
                quality_report = validate_translation_quality()
                if quality_report and 'languages' in quality_report:
                    self.show_info("翻訳品質サマリー:")
                    for lang, data in quality_report['languages'].items():
                        coverage_pct = data['coverage'] * 100
                        quality_pct = data['quality_score'] * 100
                        self.show_info(f"  {lang}: カバレッジ {coverage_pct:.1f}%, 品質スコア {quality_pct:.1f}%")

                        if data['missing_keys']:
                            self.show_info(f"    未翻訳: {len(data['missing_keys'])} 項目")
            else:
                self.show_warning("レポートの生成に失敗しました。")
        except Exception as e:
            self.show_error(f"レポート生成エラー: {e}")

    def _show_translation_coverage(self):
        """翻訳カバレッジを表示"""
        try:
            from localization import get_localization_manager
            manager = get_localization_manager()
            available_langs = manager.get_available_languages()

            self.show_info(f"利用可能な言語数: {len(available_langs)}")
            self.show_info(f"利用可能な言語: {', '.join(available_langs)}")

            # 各言語のカバレッジを表示
            reference_keys = set(manager._messages.get('en', {}).keys())
            total_keys = len(reference_keys)

            for lang in available_langs:
                lang_messages = manager._messages.get(lang, {})
                lang_keys = set(lang_messages.keys())
                covered_keys = len(lang_keys.intersection(reference_keys))
                coverage = (covered_keys / total_keys * 100) if total_keys > 0 else 0

                empty_translations = [key for key, value in lang_messages.items()
                                   if not value or value.strip() == ""]

                self.show_info(f"  {lang}:")
                self.show_info(f"    カバレッジ: {coverage:.1f}% ({covered_keys}/{total_keys})")
                self.show_info(f"    空翻訳: {len(empty_translations)} 項目")
                self.show_info(f"    RTL言語: {'はい' if manager.is_rtl_language(lang) else 'いいえ'}")

        except Exception as e:
            self.show_error(f"カバレッジ表示エラー: {e}")

    def _show_missing_translations(self):
        """不足翻訳を表示"""
        try:
            available_langs = get_localization_manager().get_available_languages()

            self.show_info("不足翻訳の検出:")
            for lang in available_langs:
                if lang == 'en':  # 基準言語はスキップ
                    continue

                missing_keys = find_missing_translations(lang)
                if missing_keys:
                    self.show_info(f"  {lang}: {len(missing_keys)} 項目不足")
                    for key in missing_keys[:5]:  # 最初の5つを表示
                        self.show_info(f"    - {key}")
                    if len(missing_keys) > 5:
                        self.show_info(f"    ... ほか {len(missing_keys) - 5} 項目")
                else:
                    self.show_info(f"  {lang}: 完全翻訳済み ✓")

        except Exception as e:
            self.show_error(f"不足翻訳検出エラー: {e}")

    def _show_unused_translations(self):
        """未使用翻訳を表示"""
        try:
            # 簡易的な未使用翻訳チェック（完全な実装はtranslation_manager.pyで提供）
            self.show_info("未使用翻訳の検索:")
            self.show_info("  完全な未使用翻訳チェックはコマンドラインで実行してください:")
            self.show_info("  python translation_manager.py --unused")
        except Exception as e:
            self.show_error(f"未使用翻訳チェックエラー: {e}")

    def _create_new_language_template(self):
        """新しい言語テンプレートを作成"""
        try:
            lang_code = self.prompt_string("新しい言語のコードを入力してください (例: fr, de, es)")
            if not lang_code:
                return

            lang_name = self.prompt_string("言語名を入力してください (例: French, German, Spanish)")
            if not lang_name:
                return

            if self.confirm_action(f"言語 '{lang_name}' ({lang_code}) のテンプレートを作成しますか？"):
                try:
                    # translation_manager.pyの機能を使用
                    import subprocess
                    result = subprocess.run([
                        sys.executable, 'translation_manager.py',
                        '--create-template', lang_code, lang_name
                    ], capture_output=True, text=True)

                    if result.returncode == 0:
                        self.show_success(f"言語テンプレートが作成されました: {lang_code}")
                    else:
                        self.show_error(f"テンプレート作成に失敗しました: {result.stderr}")

                except Exception as e:
                    self.show_error(f"テンプレート作成エラー: {e}")

        except Exception as e:
            self.show_error(f"言語テンプレート作成エラー: {e}")

    def _handle_language_settings(self):
        """言語設定の処理"""
        if not _LOCALIZATION_AVAILABLE:
            self.show_warning(msg("language_functionality_unavailable"))
            return

        while True:
            try:
                current_lang = get_language()
                self.show_info(msg("current_language", lang=current_lang))
                self.show_info(msg("available_languages", langs=", ".join(get_localization_manager().get_available_languages())))

                choice = self.get_user_choice(msg("language_settings_menu"), ['1', '2', '3'])

                if choice == '1':
                    self._change_language()
                elif choice == '2':
                    self._show_language_info()
                elif choice == '3':
                    break

            except Exception as e:
                self.show_error(f"言語設定中にエラーが発生しました: {e}")

    def _change_language(self):
        """言語を変更"""
        if not _LOCALIZATION_AVAILABLE:
            return

        lang_code = input(msg("enter_language_code")).strip().lower()

        if set_language(lang_code):
            self.show_success(msg("language_changed", lang=lang_code))
        else:
            self.show_warning(msg("language_not_available", lang=lang_code))

    def _show_language_info(self):
        """言語情報を表示"""
        if not _LOCALIZATION_AVAILABLE:
            return

        manager = get_localization_manager()
        available_langs = manager.get_available_languages()

        self.show_info(msg("current_language", lang=get_language()))
        self.show_info(msg("available_languages", langs=", ".join(available_langs)))
        self.show_info(f"利用可能な言語数: {len(available_langs)}")

        # RTL言語の確認
        rtl_langs = [lang for lang in available_langs if manager.is_rtl_language(lang)]
        if rtl_langs:
            self.show_info(f"RTL言語: {', '.join(rtl_langs)}")

        # 翻訳品質レポートの生成と表示
        try:
            report_file = generate_translation_report("translation_quality_report.json")
            if report_file:
                self.show_info(f"翻訳品質レポートが生成されました: {report_file}")

                # 品質レポートの表示
                quality_report = validate_translation_quality()
                if quality_report and 'languages' in quality_report:
                    self.show_info("翻訳品質サマリー:")
                    for lang, data in quality_report['languages'].items():
                        coverage_pct = data['coverage'] * 100
                        quality_pct = data['quality_score'] * 100
                        self.show_info(f"  {lang}: カバレッジ {coverage_pct:.1f}%, 品質スコア {quality_pct:.1f}%")

                        if data['missing_keys']:
                            self.show_info(f"    未翻訳: {len(data['missing_keys'])} 項目")
        except Exception as e:
            self.show_warning(f"翻訳品質レポートの生成に失敗しました: {e}")

    def parse_error_type(self, error: Exception) -> str:
        """エラータイプを解析（実装が必要）"""
        # 実際の実装ではエラータイプを解析するロジックを追加
        return "generic_error"

    def show_enhanced_error(self, error_type: str, error: Exception, context: dict = None):
        """エラーメッセージを改善して表示（実装が必要）"""
        # 実際の実装ではエラータイプに応じた改善されたエラーメッセージを表示
        self.show_error(f"{error_type}: {str(error)}")

    def prompt_string(self, prompt: str) -> str:
        """文字列入力を求める（実装が必要）"""
        return input(f"{prompt}: ").strip()
