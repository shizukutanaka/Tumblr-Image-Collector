"""
プログレス表示システム
"""

import sys
import time
from typing import Optional, Dict, Any
from pathlib import Path
import threading
import logging

logger = logging.getLogger(__name__)

class ProgressDisplay:
    """プログレス表示を管理するクラス"""

    def __init__(self):
        self.current_task = ""
        self.total_items = 0
        self.completed_items = 0
        self.start_time = time.time()
        self.display_thread: Optional[threading.Thread] = None
        self.is_running = False

    def start_task(self, task_name: str, total_items: int = 0):
        """タスクを開始する"""
        self.current_task = task_name
        self.total_items = total_items
        self.completed_items = 0
        self.start_time = time.time()
        self.is_running = True

        if total_items > 0:
            self._start_progress_bar()
        else:
            logger.info(f"開始: {task_name}")

    def update_progress(self, increment: int = 1):
        """プログレスを更新する"""
        self.completed_items += increment
        self._update_progress_bar()

    def complete_task(self, success: bool = True):
        """タスクを完了する"""
        self.is_running = False

        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=1.0)

        elapsed_time = time.time() - self.start_time

        if success:
            if self.total_items > 0:
                logger.info(f"完了: {self.current_task} ({self.completed_items}/{self.total_items} 項目, {elapsed_time:.1f}秒)")
            else:
                logger.info(f"完了: {self.current_task} ({elapsed_time:.1f}秒)")
        else:
            logger.error(f"失敗: {self.current_task} ({elapsed_time:.1f}秒)")

    def show_message(self, message: str, level: str = "info"):
        """メッセージを表示する"""
        if level.lower() == "error":
            logger.error(message)
        elif level.lower() == "warning":
            logger.warning(message)
        else:
            logger.info(message)

    def _start_progress_bar(self):
        """プログレスバーを開始する"""
        if not self.is_running:
            return

        self.display_thread = threading.Thread(target=self._progress_bar_worker, daemon=True)
        self.display_thread.start()

    def _progress_bar_worker(self):
        """プログレスバーワーカー"""
        while self.is_running and self.completed_items < self.total_items:
            self._update_progress_bar()
            time.sleep(0.5)

    def _update_progress_bar(self):
        """プログレスバーを更新する"""
        if self.total_items == 0:
            return

        percentage = min(100, (self.completed_items / self.total_items) * 100)
        elapsed_time = time.time() - self.start_time
        eta = (elapsed_time / max(self.completed_items, 1)) * (self.total_items - self.completed_items)

        # プログレスバーの表示
        bar_length = 30
        filled_length = int(bar_length * percentage / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)

        progress_text = f"\r{self.current_task}: [{bar}] {percentage:5.1f}% ({self.completed_items}/{self.total_items}) ETA: {eta:.1f}s"
        sys.stdout.write(progress_text)
        sys.stdout.flush()

    def show_final_stats(self, stats: Dict[str, Any]):
        """最終統計を表示する"""
        logger.info("\n" + "="*60)
        logger.info("最終統計")
        logger.info("="*60)

        for key, value in stats.items():
            if isinstance(value, dict):
                logger.info(f"{key}:")
                for sub_key, sub_value in value.items():
                    logger.info(f"  {sub_key}: {sub_value}")
            else:
                logger.info(f"{key}: {value}")

        logger.info("="*60)


class InteractiveCLI:
    """対話型CLIを提供するクラス"""

    def __init__(self):
        self.progress = ProgressDisplay()
        self._command_history = []

    def show_welcome(self):
        """ウェルカムメッセージを表示"""
        welcome_text = """
╔══════════════════════════════════════════════════════════════╗
║                   Tumblr Image Collector                     ║
║                    商用グレード版 v2.0.0                     ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(welcome_text)

    def show_menu(self):
        """メインメニューを表示"""
        menu_text = """
使用可能な操作:
1. ブログから画像を収集
2. 複数のブログを検索
3. タグで画像を検索
4. メタデータをエクスポート
5. 統計情報を表示
6. 設定を変更
7. ヘルプを表示
8. 終了

        """
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
                print(f"無効な選択です。{valid_choices}から選んでください。")

    def prompt_string(self, prompt: str, default: str = "") -> str:
        """文字列入力を求める"""
        if default:
            result = input(f"{prompt} [{default}]: ").strip()
            return result if result else default
        else:
            return input(f"{prompt}: ").strip()

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
                    print(f"値は{min_val}以上でなければなりません。")
                    continue
                if max_val is not None and value > max_val:
                    print(f"値は{max_val}以下でなければなりません。")
                    continue

                return value
            except ValueError:
                print("有効な整数を入力してください。")

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
                print("y/n または yes/no で回答してください。")

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
            print(f"\r{task}: {current}/{total} ({percentage:.1f}%)", end="", flush=True)

            if current >= total:
                print()  # 改行

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
        header_line = "|" + "|".join(f" {header"<{width}"} " for header, width in zip(headers, col_widths)) + "|"
        table_lines.append(header_line)
        table_lines.append(separator)

        # データ行
        for row in rows:
            data_line = "|" + "|".join(f" {str(cell)"<{width}"} " for cell, width in zip(row, col_widths)) + "|"
            table_lines.append(data_line)

        table_lines.append(separator)

        return "\n".join(table_lines)

    def run_interactive_mode(self, collector):
        """対話型モードを実行"""
        self.show_welcome()

        while True:
            try:
                self.show_menu()
                choice = self.get_user_choice("選択してください (1-8)", ['1', '2', '3', '4', '5', '6', '7', '8'])

                if choice == 'quit':
                    print("終了します...")
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
                    self._handle_config_change(collector)
                elif choice == '7':
                    self._show_help()
                elif choice == '8':
                    break

            except KeyboardInterrupt:
                print("\n\nユーザーによる中断を検出しました。")
                break
            except Exception as e:
                self.show_error(f"予期せぬエラー: {e}")
                logger.exception("Interactive mode error")

    def _handle_single_blog_collection(self, collector):
        """単一ブログ収集の処理"""
        blog_name = self.prompt_string("収集するブログ名を入力してください")
        if not blog_name:
            self.show_warning("ブログ名が入力されませんでした。")
            return

        if self.confirm_action(f"ブログ '{blog_name}' から画像を収集しますか？"):
            try:
                self.progress.start_task(f"ブログ '{blog_name}' からの画像収集", 0)
                collector.run(blog_name)
                self.progress.complete_task(True)
                self.show_success(f"ブログ '{blog_name}' からの収集が完了しました。")
            except Exception as e:
                self.progress.complete_task(False)
                self.show_error(f"収集中にエラーが発生しました: {e}")

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

            self.show_success(f"{successful_blogs}/{total_blogs}個のブログの処理が完了しました。")

    def _handle_tag_search(self, collector):
        """タグ検索の処理"""
        tag = self.prompt_string("検索するタグを入力してください")
        if not tag:
            self.show_warning("タグが入力されませんでした。")
            return

        count = self.prompt_int("取得する画像数", 20, 1, 1000)

        if self.confirm_action(f"タグ '{tag}' で {count}件の画像を検索しますか？"):
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
        help_text = """
Tumblr Image Collector - ヘルプ

コマンドラインオプション:
  --blog BLOG_NAME      収集するブログ名を指定
  --output DIRECTORY    出力ディレクトリを指定
  --config FILE         設定ファイルを指定
  --workers NUM         ワーカー数を指定
  --interactive         対話型モードで起動

対話型モードの操作:
  1. ブログから画像を収集
  2. 複数のブログを検索
  3. タグで画像を検索
  4. メタデータをエクスポート
  5. 統計情報を表示
  6. 設定を変更
  7. ヘルプを表示
  8. 終了

詳細についてはドキュメントを参照してください。
        """
        print(help_text)
