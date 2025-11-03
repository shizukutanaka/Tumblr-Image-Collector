#!/usr/bin/env python3
"""
Tumblr Image Collector GUI Module
Tkinterベースのグラフィカルユーザーインターフェースを提供
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import logging
import os
from pathlib import Path
import json
from typing import Optional, Dict, Any
import sys
import time

# 外部および内部モジュールのインポート
try:
    from PIL import Image, ImageTk
    from core.design_tokens import DesignTokens
    from tumblr_image_collector import TumblrImageCollector
    from youtube_downloader import YouTubeDownloader
    from arxiv_collector import ArXivCollector
    from semantic_scholar_collector import SemanticScholarCollector
except ImportError as e:
    # パスを追加して再試行
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    try:
        from PIL import Image, ImageTk
        from core.design_tokens import DesignTokens
        from tumblr_image_collector import TumblrImageCollector
        from youtube_downloader import YouTubeDownloader
        from arxiv_collector import ArXivCollector
        from semantic_scholar_collector import SemanticScholarCollector
    except ImportError:
        # ユーザーに依存関係のインストールを促す
        print("エラー: 必要なライブラリが見つかりません (Pillowなど)。pip install -r requirements.txt を実行してください。", file=sys.stderr)
        sys.exit(1)

logger = logging.getLogger(__name__)

class TumblrCollectorGUI:
    """Tumblr Image CollectorのGUIクラス"""

    def __init__(self, root):
        self.root = root
        self.root.title("Tumblr Image Collector v2.1")
        self.root.geometry("850x750")
        self.root.configure(background=DesignTokens.COLOR_BACKGROUND)

        # バリデーション状態
        self.validation_errors = {}

        # コレクターインスタンス
        self.collector = None
        self.youtube_downloader = None
        self.arxiv_collector = None
        self.semantic_scholar_collector = None
        self.is_running = False
        self.message_queue = queue.Queue()

        # クリップボードモニター
        self.clipboard_monitor = None
        self.detected_urls = set()

        # プレビューシステム
        self.preview_system = None
        self.downloaded_images = []

        # ファイルリネームシステム
        self.file_renamer = None

        # 帯域制限システム
        self.bandwidth_limiter = None

        # 出力フォーマットシステム
        self.output_formatter = None

        # タグインデックスシステム
        self.tag_indexer = None

        # UIコンポーネントの初期化
        self._setup_styles()
        self._setup_ui()
        self._setup_menu()

        # メッセージ処理タイマー
        self.root.after(100, self._process_messages)

        # クリップボードモニターの初期化
        self._init_clipboard_monitor()

        # プレビューシステムの初期化
        self._init_preview_system()

        # ファイルリネームシステムの初期化
        self._init_file_renamer()

        # 帯域制限システムの初期化
        self._init_bandwidth_limiter()

        # 出力フォーマットシステムの初期化
        self._init_output_formatter()

        # タグインデックスシステムの初期化
        self._init_tag_indexer()

    def _setup_styles(self):
        """スタイルの設定（Atlassian Design System準拠）"""
        style = ttk.Style()
        style.theme_use('clam')

        style.configure("TFrame", background=DesignTokens.COLOR_BACKGROUND)
        style.configure("TLabel",
                       font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_MEDIUM),
                       background=DesignTokens.COLOR_BACKGROUND,
                       foreground=DesignTokens.COLOR_TEXT)
        style.configure("TButton",
                       font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_MEDIUM, DesignTokens.FONT_WEIGHT_NORMAL),
                       padding=(DesignTokens.SPACE_150, DesignTokens.SPACE_075),
                       relief="flat",
                       borderwidth=0,
                       focuscolor='none')
        style.map("TButton",
                 background=[("active", DesignTokens.COLOR_PRIMARY_HOVER), ("!disabled", "white")],
                 foreground=[("active", DesignTokens.COLOR_BACKGROUND)])

        # カスタムスタイル
        style.configure("Primary.TButton",
                       background=DesignTokens.COLOR_PRIMARY,
                       foreground=DesignTokens.COLOR_BACKGROUND)
        style.map("Primary.TButton",
                 background=[("active", DesignTokens.COLOR_PRIMARY_HOVER)])

        style.configure("Secondary.TButton",
                       background=DesignTokens.COLOR_SECONDARY,
                       foreground=DesignTokens.COLOR_BACKGROUND)
        style.map("Secondary.TButton",
                 background=[("active", "#5A6B87")])

        style.configure("Success.TButton",
                       background=DesignTokens.COLOR_SUCCESS,
                       foreground=DesignTokens.COLOR_BACKGROUND)
        style.map("Success.TButton",
                 background=[("active", "#00A368")])

        style.configure("Header.TLabel",
                       font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_LARGE, DesignTokens.FONT_WEIGHT_BOLD),
                       foreground=DesignTokens.COLOR_TEXT)
        style.configure("Card.TFrame",
                       background=DesignTokens.COLOR_SURFACE,
                       relief="solid",
                       borderwidth=1,
                       bordercolor=DesignTokens.COLOR_BORDER)
        style.configure("TCheckbutton",
                        background=DesignTokens.COLOR_BACKGROUND,
                        font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_MEDIUM))
        style.configure("TEntry", 
                        fieldbackground="white",
                        bordercolor=DesignTokens.COLOR_BORDER,
                        lightcolor=DesignTokens.COLOR_BORDER,
                        darkcolor=DesignTokens.COLOR_BORDER)

    def _setup_ui(self):
        """UIコンポーネントの設定（タブ付きインターフェース）"""
        main_frame = ttk.Frame(self.root, padding=DesignTokens.SPACE_300)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # タブウィジェットを作成
        self.tab_control = ttk.Notebook(main_frame)
        self.tab_control.grid(row=0, column=0, sticky="nsew")

        # Tumblrタブ
        self.tumblr_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tumblr_tab, text="Tumblr")

        # YouTubeタブ
        self.youtube_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.youtube_tab, text="YouTube")

        # 論文タブ
        self.paper_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.paper_tab, text="論文")

        # 各タブのUIを設定
        self._setup_tumblr_ui()
        self._setup_youtube_ui()
        self._setup_paper_ui()

    def _setup_tumblr_ui(self):
        """TumblrタブのUI設定"""
        # ヘッダー
        header_label = ttk.Label(self.tumblr_tab, text="Tumblr Image Collector", style="Header.TLabel")
        header_label.grid(row=0, column=0, sticky="ew", pady=(0, DesignTokens.SPACE_300))

        # --- 設定カード ---
        config_card = ttk.Frame(self.tumblr_tab, style="Card.TFrame", padding=DesignTokens.SPACE_200)
        config_card.grid(row=1, column=0, sticky="ew", pady=(0, DesignTokens.SPACE_200))
        config_card.columnconfigure(1, weight=1)

        config_header = ttk.Label(config_card, text="収集設定", style="Header.TLabel")
        config_header.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, DesignTokens.SPACE_200))

        # ブログ名
        ttk.Label(config_card, text="ブログ名").grid(row=1, column=0, sticky="w")
        self.blog_entry = ttk.Entry(config_card, font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_MEDIUM))
        self.blog_entry.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(DesignTokens.SPACE_100, 0))
        self.blog_error_label = ttk.Label(config_card, text="", foreground=DesignTokens.COLOR_DANGER, font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_SMALL))
        self.blog_error_label.grid(row=2, column=1, columnspan=2, sticky="w", padx=(DesignTokens.SPACE_100, 0))

        # 出力フォルダ
        ttk.Label(config_card, text="出力フォルダ").grid(row=3, column=0, sticky="w", pady=(DesignTokens.SPACE_100, 0))
        self.output_entry = ttk.Entry(config_card, font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_MEDIUM))
        self.output_entry.grid(row=3, column=1, sticky="ew", padx=(DesignTokens.SPACE_100, 0), pady=(DesignTokens.SPACE_100, 0))
        self.output_entry.insert(0, os.path.join(os.getcwd(), "downloads"))
        self.browse_button = ttk.Button(config_card, text="参照", style="Secondary.TButton", command=self._browse_output_dir)
        self.browse_button.grid(row=3, column=2, sticky="e", padx=(DesignTokens.SPACE_100, 0), pady=(DesignTokens.SPACE_100, 0))
        self.output_error_label = ttk.Label(config_card, text="", foreground=DesignTokens.COLOR_DANGER, font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_SMALL))
        self.output_error_label.grid(row=4, column=1, columnspan=2, sticky="w", padx=(DesignTokens.SPACE_100, 0))

        # タグ
        ttk.Label(config_card, text="タグ (カンマ区切り)").grid(row=5, column=0, sticky="w", pady=(DesignTokens.SPACE_100, 0))
        self.tags_entry = ttk.Entry(config_card, font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_MEDIUM))
        self.tags_entry.grid(row=5, column=1, columnspan=2, sticky="ew", padx=(DesignTokens.SPACE_100, 0), pady=(DesignTokens.SPACE_100, 0))

        # オプション
        options_frame = ttk.Frame(config_card)
        options_frame.grid(row=6, column=0, columnspan=3, sticky="w", pady=(DesignTokens.SPACE_200, 0))
        self.include_likes_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="いいねした投稿も含める", variable=self.include_likes_var).pack(side=tk.LEFT, padx=(0, DesignTokens.SPACE_200))
        self.interactive_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="対話モード", variable=self.interactive_var).pack(side=tk.LEFT)

        # 並列数
        workers_frame = ttk.Frame(config_card)
        workers_frame.grid(row=7, column=0, columnspan=3, sticky="w", pady=(DesignTokens.SPACE_100, 0))
        ttk.Label(workers_frame, text="並列数:").pack(side=tk.LEFT)
        self.workers_spinbox = tk.Spinbox(workers_frame, from_=1, to=20, width=5, font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_MEDIUM))
        self.workers_spinbox.pack(side=tk.LEFT, padx=(DesignTokens.SPACE_100, 0))
        self.workers_spinbox.delete(0, tk.END)
        self.workers_spinbox.insert(0, "5")
        self.workers_error_label = ttk.Label(workers_frame, text="", foreground=DesignTokens.COLOR_DANGER, font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_SMALL))
        self.workers_error_label.pack(side=tk.LEFT, padx=(DesignTokens.SPACE_100, 0))

        # --- クリップボードモニター設定 ---
        clipboard_frame = ttk.Frame(config_card, style="Card.TFrame", padding=DesignTokens.SPACE_100)
        clipboard_frame.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(DesignTokens.SPACE_200, 0))
        clipboard_frame.columnconfigure(1, weight=1)

        clipboard_header = ttk.Label(clipboard_frame, text="クリップボードモニター", style="Header.TLabel")
        clipboard_header.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, DesignTokens.SPACE_100))

        # クリップボードモニター有効/無効
        self.clipboard_monitor_var = tk.BooleanVar(value=True)
        self.clipboard_monitor_check = ttk.Checkbutton(
            clipboard_frame,
            text="クリップボードからTumblr URLを自動検出",
            variable=self.clipboard_monitor_var,
            command=self._toggle_clipboard_monitor
        )
        self.clipboard_monitor_check.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, DesignTokens.SPACE_100))

        # 通知設定
        notification_frame = ttk.Frame(clipboard_frame)
        notification_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, DesignTokens.SPACE_100))

        self.show_notifications_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(notification_frame, text="デスクトップ通知", variable=self.show_notifications_var).pack(side=tk.LEFT, padx=(0, DesignTokens.SPACE_200))

        self.notification_sound_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(notification_frame, text="通知音", variable=self.notification_sound_var).pack(side=tk.LEFT)

        # 検出されたURLリスト
        detected_frame = ttk.Frame(clipboard_frame)
        detected_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(DesignTokens.SPACE_100, 0))
        detected_frame.columnconfigure(0, weight=1)
        detected_frame.rowconfigure(1, weight=1)

        ttk.Label(detected_frame, text="検出されたブログ:").grid(row=0, column=0, sticky="w")

        # 検出されたURLのリストボックス
        listbox_frame = ttk.Frame(detected_frame, style="Card.TFrame")
        listbox_frame.grid(row=1, column=0, sticky="ew", pady=(DesignTokens.SPACE_50, 0))
        listbox_frame.columnconfigure(0, weight=1)
        listbox_frame.rowconfigure(0, weight=1)

        self.detected_urls_listbox = tk.Listbox(
            listbox_frame,
            height=3,
            font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_SMALL),
            selectmode=tk.EXTENDED
        )
        detected_scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.detected_urls_listbox.yview)
        self.detected_urls_listbox.configure(yscrollcommand=detected_scrollbar.set)

        self.detected_urls_listbox.grid(row=0, column=0, sticky="ew")
        detected_scrollbar.grid(row=0, column=1, sticky="ns")

        # リスト操作ボタン
        button_frame = ttk.Frame(detected_frame)
        button_frame.grid(row=2, column=0, sticky="ew", pady=(DesignTokens.SPACE_50, 0))

        ttk.Button(button_frame, text="クリア", style="Secondary.TButton", command=self._clear_detected_urls).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="ダウンロード開始", style="Success.TButton", command=self._start_from_detected).pack(side=tk.LEFT, padx=(DesignTokens.SPACE_100, 0))
        ttk.Button(button_frame, text="選択項目を削除", style="Secondary.TButton", command=self._remove_selected_detected).pack(side=tk.RIGHT)

        # --- プレビュー設定 ---
        preview_frame = ttk.Frame(config_card, style="Card.TFrame", padding=DesignTokens.SPACE_100)
        preview_frame.grid(row=10, column=0, columnspan=3, sticky="ew", pady=(DesignTokens.SPACE_200, 0))
        preview_frame.columnconfigure(1, weight=1)

        preview_header = ttk.Label(preview_frame, text="プレビュー機能", style="Header.TLabel")
        preview_header.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, DesignTokens.SPACE_100))

        # プレビュー有効/無効
        self.preview_enabled_var = tk.BooleanVar(value=True)
        self.preview_check = ttk.Checkbutton(
            preview_frame,
            text="ダウンロード前に画像をプレビュー",
            variable=self.preview_enabled_var
        )
        self.preview_check.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, DesignTokens.SPACE_100))

        # プレビュー設定
        preview_settings_frame = ttk.Frame(preview_frame)
        preview_settings_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, DesignTokens.SPACE_100))

        # サムネイルサイズ
        ttk.Label(preview_settings_frame, text="サムネイルサイズ:").pack(side=tk.LEFT)
        self.thumbnail_size_var = tk.StringVar(value="200x200")
        thumbnail_combo = ttk.Combobox(preview_settings_frame, textvariable=self.thumbnail_size_var,
                                     values=["150x150", "200x200", "300x300"], state="readonly", width=10)
        thumbnail_combo.pack(side=tk.LEFT, padx=(DesignTokens.SPACE_100, DesignTokens.SPACE_200))

        # スライドショー機能
        self.slideshow_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(preview_settings_frame, text="スライドショー有効", variable=self.slideshow_var).pack(side=tk.LEFT)

        # プレビュー操作ボタン
        preview_button_frame = ttk.Frame(preview_frame)
        preview_button_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(DesignTokens.SPACE_100, 0))

        ttk.Button(preview_button_frame, text="サムネイル生成", style="Secondary.TButton", command=self._generate_thumbnails).pack(side=tk.LEFT)
        ttk.Button(preview_button_frame, text="スライドショー開始", style="Secondary.TButton", command=self._start_slideshow).pack(side=tk.LEFT, padx=(DesignTokens.SPACE_100, 0))
        ttk.Button(preview_button_frame, text="フルプレビュー", style="Secondary.TButton", command=self._show_full_preview).pack(side=tk.LEFT, padx=(DesignTokens.SPACE_100, 0))
        ttk.Button(preview_button_frame, text="プレビューキャッシュクリア", style="Secondary.TButton", command=self._clear_preview_cache).pack(side=tk.RIGHT)

        # --- ファイルリネーム設定 ---
        rename_frame = ttk.Frame(config_card, style="Card.TFrame", padding=DesignTokens.SPACE_100)
        rename_frame.grid(row=11, column=0, columnspan=3, sticky="ew", pady=(DesignTokens.SPACE_200, 0))
        rename_frame.columnconfigure(1, weight=1)

        rename_header = ttk.Label(rename_frame, text="ファイル命名", style="Header.TLabel")
        rename_header.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, DesignTokens.SPACE_100))

        # 命名テンプレート
        ttk.Label(rename_frame, text="命名テンプレート:").grid(row=1, column=0, sticky="w")
        self.rename_template_var = tk.StringVar(value="{blog}_{timestamp}_{id}_{tags}")
        self.rename_template_entry = ttk.Entry(rename_frame, textvariable=self.rename_template_var, font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_SMALL))
        self.rename_template_entry.grid(row=1, column=1, sticky="ew", padx=(DesignTokens.SPACE_100, 0))

        # テンプレートヘルプボタン
        ttk.Button(rename_frame, text="?", width=2, command=self._show_template_help).grid(row=1, column=2, sticky="w", padx=(DesignTokens.SPACE_50, 0))

        # プレビュー
        ttk.Label(rename_frame, text="プレビュー:").grid(row=2, column=0, sticky="w", pady=(DesignTokens.SPACE_100, 0))

        self.rename_preview_var = tk.StringVar(value="")
        self.rename_preview_label = ttk.Label(rename_frame, textvariable=self.rename_preview_var, font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_SMALL), foreground=DesignTokens.COLOR_TEXT_SECONDARY)
        self.rename_preview_label.grid(row=2, column=1, sticky="w", padx=(DesignTokens.SPACE_100, 0))

        # テンプレート更新ボタン
        ttk.Button(rename_frame, text="プレビュー更新", style="Secondary.TButton", command=self._update_rename_preview).grid(row=2, column=2, sticky="w", padx=(DesignTokens.SPACE_50, 0))

        # 共通テンプレート選択
        template_frame = ttk.Frame(rename_frame)
        template_frame.grid(row=3, column=0, columnspan=3, sticky="w", pady=(DesignTokens.SPACE_100, 0))

        ttk.Label(template_frame, text="テンプレート:").pack(side=tk.LEFT)

        # テンプレート選択コンボボックス
        self.template_combo = ttk.Combobox(template_frame, state="readonly", width=20)
        self.template_combo.pack(side=tk.LEFT, padx=(DesignTokens.SPACE_100, 0))

        # ファイルリネームオプション
        rename_options_frame = ttk.Frame(rename_frame)
        rename_options_frame.grid(row=4, column=0, columnspan=3, sticky="w", pady=(DesignTokens.SPACE_100, 0))

        self.rename_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(rename_options_frame, text="ファイルリネームを有効化", variable=self.rename_enabled_var).pack(side=tk.LEFT, padx=(0, DesignTokens.SPACE_200))

        self.collision_resolve_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(rename_options_frame, text="名前の衝突を解決", variable=self.collision_resolve_var).pack(side=tk.LEFT)

        # --- 帯域制限設定 ---
        bandwidth_frame = ttk.Frame(config_card, style="Card.TFrame", padding=DesignTokens.SPACE_100)
        bandwidth_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(DesignTokens.SPACE_200, 0))
        bandwidth_frame.columnconfigure(1, weight=1)

        bandwidth_header = ttk.Label(bandwidth_frame, text="帯域制限", style="Header.TLabel")
        bandwidth_header.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, DesignTokens.SPACE_100))

        # 帯域制限有効/無効
        self.bandwidth_limit_var = tk.BooleanVar(value=False)
        self.bandwidth_check = ttk.Checkbutton(
            bandwidth_frame,
            text="ダウンロード速度を制限",
            variable=self.bandwidth_limit_var,
            command=self._toggle_bandwidth_limit
        )
        self.bandwidth_check.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, DesignTokens.SPACE_100))

        # 速度設定
        speed_frame = ttk.Frame(bandwidth_frame)
        speed_frame.grid(row=2, column=0, columnspan=3, sticky="w", pady=(0, DesignTokens.SPACE_100))

        ttk.Label(speed_frame, text="最大速度 (KB/s):").pack(side=tk.LEFT)

        self.bandwidth_rate_var = tk.StringVar(value="1024")
        self.bandwidth_rate_spinbox = tk.Spinbox(
            speed_frame,
            from_=1,
            to=10240,
            width=8,
            textvariable=self.bandwidth_rate_var,
            font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_MEDIUM)
        )
        self.bandwidth_rate_spinbox.pack(side=tk.LEFT, padx=(DesignTokens.SPACE_100, 0))

        # プリセット選択
        self.bandwidth_preset_var = tk.StringVar(value="normal")
        preset_combo = ttk.Combobox(speed_frame, textvariable=self.bandwidth_preset_var,
                                  values=["slow", "normal", "fast"], state="readonly", width=8)
        preset_combo.pack(side=tk.LEFT, padx=(DesignTokens.SPACE_100, 0))
        preset_combo.bind('<<ComboboxSelected>>', self._on_bandwidth_preset_selected)

        # 自動調整
        self.auto_adjust_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(speed_frame, text="ネットワーク状況に応じて自動調整", variable=self.auto_adjust_var).pack(side=tk.LEFT, padx=(DesignTokens.SPACE_200, 0))

        # 帯域情報表示
        info_frame = ttk.Frame(bandwidth_frame)
        info_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(DesignTokens.SPACE_100, 0))
        info_frame.columnconfigure(1, weight=1)

        ttk.Label(info_frame, text="現在の速度:").grid(row=0, column=0, sticky="w")
        self.current_rate_label = ttk.Label(info_frame, text="0 KB/s", font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_SMALL), foreground=DesignTokens.COLOR_TEXT_SECONDARY)
        self.current_rate_label.grid(row=0, column=1, sticky="w")

        ttk.Label(info_frame, text="ピーク速度:").grid(row=1, column=0, sticky="w")
        self.peak_rate_label = ttk.Label(info_frame, text="0 KB/s", font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_SMALL), foreground=DesignTokens.COLOR_TEXT_SECONDARY)
        self.peak_rate_label.grid(row=1, column=1, sticky="w")

        ttk.Label(info_frame, text="総ダウンロード:").grid(row=2, column=0, sticky="w")
        self.total_bytes_label = ttk.Label(info_frame, text="0 MB", font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_SMALL), foreground=DesignTokens.COLOR_TEXT_SECONDARY)
        self.total_bytes_label.grid(row=2, column=1, sticky="w")

        # 帯域制御ボタン
        control_frame = ttk.Frame(bandwidth_frame)
        control_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(DesignTokens.SPACE_100, 0))

        ttk.Button(control_frame, text="制限開始", style="Secondary.TButton", command=self._start_bandwidth_limit).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="制限停止", style="Secondary.TButton", command=self._stop_bandwidth_limit).pack(side=tk.LEFT, padx=(DesignTokens.SPACE_100, 0))
        ttk.Button(control_frame, text="統計リセット", style="Secondary.TButton", command=self._reset_bandwidth_stats).pack(side=tk.RIGHT)

        # --- 出力フォーマット設定 ---
        format_frame = ttk.Frame(config_card, style="Card.TFrame", padding=DesignTokens.SPACE_100)
        format_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(DesignTokens.SPACE_200, 0))
        format_frame.columnconfigure(1, weight=1)

        format_header = ttk.Label(format_frame, text="出力フォーマット", style="Header.TLabel")
        format_header.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, DesignTokens.SPACE_100))

        # フォーマット選択
        format_options_frame = ttk.Frame(format_frame)
        format_options_frame.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, DesignTokens.SPACE_100))

        self.output_formats = []
        self.format_vars = {}

        available_formats = [
            ('JSON', 'json', 'メタデータをJSON形式で保存'),
            ('Markdown', 'markdown', '投稿をMarkdown形式で保存'),
            ('HTML5', 'html5', 'HTML5アーカイブページを生成'),
            ('Blosxom', 'blosxom', 'Blosxom形式で保存 (互換性)'),
            ('Text', 'text', 'プレーンテキスト形式で保存')
        ]

        for i, (name, format_key, description) in enumerate(available_formats):
            var = tk.BooleanVar(value=format_key in ['json', 'html5'])
            self.format_vars[format_key] = var

            frame = ttk.Frame(format_options_frame)
            frame.pack(side=tk.LEFT, padx=(0, DesignTokens.SPACE_200))

            ttk.Checkbutton(frame, text=name, variable=var).pack(side=tk.TOP)
            ttk.Label(frame, text=description, font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_SMALL), foreground=DesignTokens.COLOR_TEXT_SECONDARY).pack(side=tk.TOP)

        # フォーマット設定
        settings_frame = ttk.Frame(format_frame)
        settings_frame.grid(row=2, column=0, columnspan=3, sticky="w", pady=(DesignTokens.SPACE_100, 0))

        # JSON設定
        json_frame = ttk.Frame(settings_frame)
        json_frame.pack(side=tk.LEFT, padx=(0, DesignTokens.SPACE_200))

        self.json_pretty_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(json_frame, text="JSON整形", variable=self.json_pretty_var).pack(side=tk.TOP)

        self.json_metadata_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(json_frame, text="メタデータ含む", variable=self.json_metadata_var).pack(side=tk.TOP)

        # HTML5設定
        html5_frame = ttk.Frame(settings_frame)
        html5_frame.pack(side=tk.LEFT, padx=(0, DesignTokens.SPACE_200))

        self.html5_responsive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(html5_frame, text="レスポンシブデザイン", variable=self.html5_responsive_var).pack(side=tk.TOP)

        # Markdown設定
        markdown_frame = ttk.Frame(settings_frame)
        markdown_frame.pack(side=tk.LEFT)

        self.markdown_images_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(markdown_frame, text="画像リンク含む", variable=self.markdown_images_var).pack(side=tk.TOP)

        # フォーマット制御ボタン
        control_frame = ttk.Frame(format_frame)
        control_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(DesignTokens.SPACE_100, 0))

        ttk.Button(control_frame, text="フォーマット実行", style="Secondary.TButton", command=self._run_format_output).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="プレビュー", style="Secondary.TButton", command=self._preview_formats).pack(side=tk.LEFT, padx=(DesignTokens.SPACE_100, 0))
        ttk.Button(control_frame, text="フォーマット設定リセット", style="Secondary.TButton", command=self._reset_format_settings).pack(side=tk.RIGHT)

        # --- タグインデックス設定 ---
        tag_index_frame = ttk.Frame(config_card, style="Card.TFrame", padding=DesignTokens.SPACE_100)
        tag_index_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(DesignTokens.SPACE_200, 0))
        tag_index_frame.columnconfigure(1, weight=1)

        tag_index_header = ttk.Label(tag_index_frame, text="タグインデックス", style="Header.TLabel")
        tag_index_header.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, DesignTokens.SPACE_100))

        # タグインデックス有効/無効
        self.tag_index_enabled_var = tk.BooleanVar(value=True)
        self.tag_index_check = ttk.Checkbutton(
            tag_index_frame,
            text="タグインデックスを自動生成",
            variable=self.tag_index_enabled_var
        )
        self.tag_index_check.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, DesignTokens.SPACE_100))

        # タグ処理設定
        tag_settings_frame = ttk.Frame(tag_index_frame)
        tag_settings_frame.grid(row=2, column=0, columnspan=3, sticky="w", pady=(0, DesignTokens.SPACE_100))

        # 最小頻度
        ttk.Label(tag_settings_frame, text="最小頻度:").pack(side=tk.LEFT)
        self.min_tag_freq_var = tk.StringVar(value="1")
        min_freq_spinbox = tk.Spinbox(tag_settings_frame, from_=1, to=100, width=5, textvariable=self.min_tag_freq_var)
        min_freq_spinbox.pack(side=tk.LEFT, padx=(DesignTokens.SPACE_100, 0))

        # 最大タグ数
        ttk.Label(tag_settings_frame, text="最大タグ数/投稿:").pack(side=tk.LEFT, padx=(DesignTokens.SPACE_200, 0))
        self.max_tags_var = tk.StringVar(value="50")
        max_tags_spinbox = tk.Spinbox(tag_settings_frame, from_=1, to=200, width=5, textvariable=self.max_tags_var)
        max_tags_spinbox.pack(side=tk.LEFT, padx=(DesignTokens.SPACE_100, 0))

        # タグ処理オプション
        tag_options_frame = ttk.Frame(tag_index_frame)
        tag_options_frame.grid(row=3, column=0, columnspan=3, sticky="w", pady=(0, DesignTokens.SPACE_100))

        self.ignore_case_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tag_options_frame, text="大文字小文字を無視", variable=self.ignore_case_var).pack(side=tk.LEFT, padx=(0, DesignTokens.SPACE_200))

        self.strip_chars_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tag_options_frame, text="特殊文字を除去", variable=self.strip_chars_var).pack(side=tk.LEFT)

        # インデックス出力形式
        format_selection_frame = ttk.Frame(tag_index_frame)
        format_selection_frame.grid(row=4, column=0, columnspan=3, sticky="w", pady=(0, DesignTokens.SPACE_100))

        ttk.Label(format_selection_frame, text="出力形式:").pack(side=tk.LEFT)

        self.tag_index_format_var = tk.StringVar(value="html")
        format_combo = ttk.Combobox(format_selection_frame, textvariable=self.tag_index_format_var,
                                  values=["html", "markdown", "json", "text"], state="readonly", width=10)
        format_combo.pack(side=tk.LEFT, padx=(DesignTokens.SPACE_100, 0))

        # タグクラウド
        self.tag_cloud_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(format_selection_frame, text="タグクラウド生成", variable=self.tag_cloud_var).pack(side=tk.LEFT, padx=(DesignTokens.SPACE_200, 0))

        # タグインデックス制御ボタン
        tag_control_frame = ttk.Frame(tag_index_frame)
        tag_control_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(DesignTokens.SPACE_100, 0))

        ttk.Button(tag_control_frame, text="インデックス生成", style="Secondary.TButton", command=self._generate_tag_index).pack(side=tk.LEFT)
        ttk.Button(tag_control_frame, text="タグクラウド生成", style="Secondary.TButton", command=self._generate_tag_cloud).pack(side=tk.LEFT, padx=(DesignTokens.SPACE_100, 0))
        ttk.Button(tag_control_frame, text="タグ検索", style="Secondary.TButton", command=self._search_by_tags_dialog).pack(side=tk.LEFT, padx=(DesignTokens.SPACE_100, 0))
        ttk.Button(tag_control_frame, text="インデックスリセット", style="Secondary.TButton", command=self._reset_tag_index).pack(side=tk.RIGHT)

        # タグ統計表示
        stats_frame = ttk.Frame(tag_index_frame)
        stats_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(DesignTokens.SPACE_100, 0))
        stats_frame.columnconfigure(1, weight=1)

        ttk.Label(stats_frame, text="タグ統計:").grid(row=0, column=0, sticky="w")
        self.tag_stats_label = ttk.Label(stats_frame, text="タグ数: 0 | 投稿数: 0", font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_SMALL), foreground=DesignTokens.COLOR_TEXT_SECONDARY)
        self.tag_stats_label.grid(row=0, column=1, sticky="w")

        ttk.Label(stats_frame, text="人気タグ:").grid(row=1, column=0, sticky="w")
        self.popular_tags_label = ttk.Label(stats_frame, text="なし", font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_SMALL), foreground=DesignTokens.COLOR_TEXT_SECONDARY)
        self.popular_tags_label.grid(row=1, column=1, sticky="w")

        # --- コントロールセクション ---
        control_frame = ttk.Frame(self.tumblr_tab)
        control_frame.grid(row=15, column=0, sticky="ew", pady=(0, DesignTokens.SPACE_200))
        self.start_button = ttk.Button(control_frame, text="開始", style="Success.TButton", command=self._start_collection)
        self.start_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(control_frame, text="停止", style="Secondary.TButton", command=self._stop_collection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(DesignTokens.SPACE_100, 0))
        ttk.Button(control_frame, text="設定保存", style="Secondary.TButton", command=self._save_config).pack(side=tk.LEFT, padx=(DesignTokens.SPACE_100, 0))
        ttk.Button(control_frame, text="設定読み込み", style="Secondary.TButton", command=self._load_config).pack(side=tk.LEFT, padx=(DesignTokens.SPACE_100, 0))

        # イベントバインディングでリアルタイムバリデーション
        self.blog_entry.bind('<KeyRelease>', lambda e: self._validate_form())
        self.output_entry.bind('<KeyRelease>', lambda e: self._validate_form())
        self.workers_spinbox.bind('<KeyRelease>', lambda e: self._validate_form())
        self.include_likes_var.trace('w', lambda *args: self._validate_form())

        # 初期バリデーション
        self._validate_form()

        # クリップボード設定の読み込み
        self._load_clipboard_settings()

        # プレビュー設定の読み込み
        self._load_preview_settings()

        # ファイルリネーム設定の読み込み
        self._load_rename_settings()

        # 帯域制限設定の読み込み
        self._load_bandwidth_settings()

        # 出力フォーマット設定の読み込み
        self._load_format_settings()

    def _setup_youtube_ui(self):
        """YouTubeタブのUI設定"""
        # ヘッダー
        header_label = ttk.Label(self.youtube_tab, text="YouTube Video Downloader", style="Header.TLabel")
        header_label.grid(row=0, column=0, sticky="ew", pady=(0, DesignTokens.SPACE_300))

        # --- 設定カード ---
        youtube_card = ttk.Frame(self.youtube_tab, style="Card.TFrame", padding=DesignTokens.SPACE_200)
        youtube_card.grid(row=1, column=0, sticky="ew", pady=(0, DesignTokens.SPACE_200))
        youtube_card.columnconfigure(1, weight=1)

        # YouTube URL
        ttk.Label(youtube_card, text="YouTube URL").grid(row=0, column=0, sticky="w")
        self.youtube_url_entry = ttk.Entry(youtube_card, font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_MEDIUM))
        self.youtube_url_entry.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(DesignTokens.SPACE_100, 0))

        # 出力フォルダ
        ttk.Label(youtube_card, text="出力フォルダ").grid(row=1, column=0, sticky="w", pady=(DesignTokens.SPACE_100, 0))
        self.youtube_output_entry = ttk.Entry(youtube_card, font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_MEDIUM))
        self.youtube_output_entry.grid(row=1, column=1, sticky="ew", padx=(DesignTokens.SPACE_100, 0), pady=(DesignTokens.SPACE_100, 0))
        self.youtube_output_entry.insert(0, os.path.join(os.getcwd(), "downloads", "youtube"))
        self.youtube_browse_button = ttk.Button(youtube_card, text="参照", style="Secondary.TButton", command=self._browse_youtube_output_dir)
        self.youtube_browse_button.grid(row=1, column=2, sticky="e", padx=(DesignTokens.SPACE_100, 0), pady=(DesignTokens.SPACE_100, 0))

        # 解像度選択
        ttk.Label(youtube_card, text="解像度").grid(row=2, column=0, sticky="w", pady=(DesignTokens.SPACE_100, 0))
        self.resolution_combo = ttk.Combobox(youtube_card, values=["144p", "240p", "360p", "480p", "720p", "1080p"], state="readonly")
        self.resolution_combo.set("720p")
        self.resolution_combo.grid(row=2, column=1, sticky="ew", padx=(DesignTokens.SPACE_100, 0), pady=(DesignTokens.SPACE_100, 0))

        # オプション
        self.audio_only_var = tk.BooleanVar()
        ttk.Checkbutton(youtube_card, text="音声のみ", variable=self.audio_only_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=(DesignTokens.SPACE_100, 0))

        # --- コントロールセクション ---
        youtube_control_frame = ttk.Frame(self.youtube_tab)
        youtube_control_frame.grid(row=2, column=0, sticky="ew", pady=(0, DesignTokens.SPACE_200))
        self.youtube_info_button = ttk.Button(youtube_control_frame, text="情報取得", style="Secondary.TButton", command=self._get_youtube_info)
        self.youtube_info_button.pack(side=tk.LEFT)
        self.youtube_download_button = ttk.Button(youtube_control_frame, text="ダウンロード", style="Success.TButton", command=self._download_youtube_video)
        self.youtube_download_button.pack(side=tk.LEFT, padx=(DesignTokens.SPACE_100, 0))

    def _setup_paper_ui(self):
        """論文タブのUI設定"""
        # ヘッダー
        header_label = ttk.Label(self.paper_tab, text="Academic Paper Collector", style="Header.TLabel")
        header_label.grid(row=0, column=0, sticky="ew", pady=(0, DesignTokens.SPACE_300))

        # --- 設定カード ---
        paper_card = ttk.Frame(self.paper_tab, style="Card.TFrame", padding=DesignTokens.SPACE_200)
        paper_card.grid(row=1, column=0, sticky="ew", pady=(0, DesignTokens.SPACE_200))
        paper_card.columnconfigure(1, weight=1)

        # 検索クエリ
        ttk.Label(paper_card, text="検索クエリ").grid(row=0, column=0, sticky="w")
        self.paper_query_entry = ttk.Entry(paper_card, font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_MEDIUM))
        self.paper_query_entry.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(DesignTokens.SPACE_100, 0))

        # 出力フォルダ
        ttk.Label(paper_card, text="出力フォルダ").grid(row=1, column=0, sticky="w", pady=(DesignTokens.SPACE_100, 0))
        self.paper_output_entry = ttk.Entry(paper_card, font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_MEDIUM))
        self.paper_output_entry.grid(row=1, column=1, sticky="ew", padx=(DesignTokens.SPACE_100, 0), pady=(DesignTokens.SPACE_100, 0))
        self.paper_output_entry.insert(0, os.path.join(os.getcwd(), "downloads", "papers"))
        self.paper_browse_button = ttk.Button(paper_card, text="参照", style="Secondary.TButton", command=self._browse_paper_output_dir)
        self.paper_browse_button.grid(row=1, column=2, sticky="e", padx=(DesignTokens.SPACE_100, 0), pady=(DesignTokens.SPACE_100, 0))

        # ソース選択
        source_frame = ttk.Frame(paper_card)
        source_frame.grid(row=2, column=0, columnspan=3, sticky="w", pady=(DesignTokens.SPACE_100, 0))
        self.paper_source_var = tk.StringVar(value="arxiv")
        ttk.Radiobutton(source_frame, text="arXiv", variable=self.paper_source_var, value="arxiv").pack(side=tk.LEFT, padx=(0, DesignTokens.SPACE_200))
        ttk.Radiobutton(source_frame, text="Semantic Scholar", variable=self.paper_source_var, value="semantic_scholar").pack(side=tk.LEFT)

        # 最大結果数
        ttk.Label(paper_card, text="最大結果数").grid(row=3, column=0, sticky="w", pady=(DesignTokens.SPACE_100, 0))
        self.paper_limit_spinbox = tk.Spinbox(paper_card, from_=1, to=100, width=5, font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_MEDIUM))
        self.paper_limit_spinbox.grid(row=3, column=1, sticky="w", padx=(DesignTokens.SPACE_100, 0), pady=(DesignTokens.SPACE_100, 0))
        self.paper_limit_spinbox.delete(0, tk.END)
        self.paper_limit_spinbox.insert(0, "10")

        # --- コントロールセクション ---
        paper_control_frame = ttk.Frame(self.paper_tab)
        paper_control_frame.grid(row=2, column=0, sticky="ew", pady=(0, DesignTokens.SPACE_200))
        self.paper_search_button = ttk.Button(paper_control_frame, text="検索", style="Secondary.TButton", command=self._search_papers)
        self.paper_search_button.pack(side=tk.LEFT)
        self.paper_download_button = ttk.Button(paper_control_frame, text="ダウンロード", style="Success.TButton", command=self._download_papers)
        self.paper_download_button.pack(side=tk.LEFT, padx=(DesignTokens.SPACE_100, 0))

        # 検索結果表示エリア
        result_frame = ttk.Frame(self.paper_tab, style="Card.TFrame", padding=DesignTokens.SPACE_200)
        result_frame.grid(row=3, column=0, sticky="nsew", pady=(0, DesignTokens.SPACE_200))
        self.paper_tab.rowconfigure(3, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(1, weight=1)

        result_header = ttk.Label(result_frame, text="検索結果", style="Header.TLabel")
        result_header.grid(row=0, column=0, sticky="w")

        self.paper_result_text = scrolledtext.ScrolledText(
            result_frame,
            height=10,
            wrap=tk.WORD,
            font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_SMALL),
            background=DesignTokens.COLOR_SURFACE,
            foreground=DesignTokens.COLOR_TEXT,
            relief="flat",
            borderwidth=0
        )
        self.paper_result_text.grid(row=1, column=0, sticky="nsew", padx=DesignTokens.SPACE_200, pady=(0, DesignTokens.SPACE_200))

    def _setup_common_ui(self, parent):
        """共通UI要素の設定"""
        # --- 進捗カード ---
        progress_card = ttk.Frame(parent, style="Card.TFrame", padding=DesignTokens.SPACE_200)
        progress_card.grid(row=1, column=0, sticky="ew", pady=(0, DesignTokens.SPACE_200))
        progress_card.columnconfigure(0, weight=1)

        # プログレスバー
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_card, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky="ew")

        # 詳細・ステータス
        status_frame = ttk.Frame(progress_card)
        status_frame.grid(row=1, column=0, sticky="ew", pady=(DesignTokens.SPACE_100, 0))
        self.progress_detail_label = ttk.Label(status_frame, text="0/0 項目", font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_SMALL), foreground=DesignTokens.COLOR_TEXT_SECONDARY)
        self.progress_detail_label.pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_frame, text="準備完了", foreground=DesignTokens.COLOR_SUCCESS, font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_MEDIUM, "bold"))
        self.status_label.pack(side=tk.LEFT, padx=DesignTokens.SPACE_100)
        self.time_label = ttk.Label(status_frame, text="", foreground=DesignTokens.COLOR_TEXT_SECONDARY, font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_SMALL))
        self.time_label.pack(side=tk.RIGHT)

        # 速度表示を追加
        speed_frame = ttk.Frame(progress_card)
        speed_frame.grid(row=2, column=0, sticky="w", pady=(DesignTokens.SPACE_50, 0))
        self.speed_label = ttk.Label(speed_frame, text="速度: 計算中...", foreground=DesignTokens.COLOR_TEXT_SECONDARY, font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_SMALL))
        self.speed_label.pack(side=tk.LEFT)

        # プログレス統計を初期化
        self.progress_stats = {'start_time': None, 'current': 0, 'total': 0, 'eta': 0}

        # --- ログカード ---
        log_card = ttk.Frame(parent, style="Card.TFrame")
        log_card.grid(row=2, column=0, sticky="nsew")
        parent.rowconfigure(2, weight=1)
        log_card.columnconfigure(0, weight=1)
        log_card.rowconfigure(1, weight=1)

        log_header = ttk.Label(log_card, text="ログ", style="Header.TLabel", padding=(DesignTokens.SPACE_200, DesignTokens.SPACE_100))
        log_header.grid(row=0, column=0, sticky="w")

        self.log_text = scrolledtext.ScrolledText(
            log_card,
            height=10,
            wrap=tk.WORD,
            font=(DesignTokens.FONT_FAMILY, DesignTokens.FONT_SIZE_SMALL),
            background=DesignTokens.COLOR_SURFACE,
            foreground=DesignTokens.COLOR_TEXT,
            relief="flat",
            borderwidth=0
        )
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=DesignTokens.SPACE_200, pady=(0, DesignTokens.SPACE_200))

        # イベントバインディングでリアルタイムバリデーション
        self.blog_entry.bind('<KeyRelease>', lambda e: self._validate_form())
        self.output_entry.bind('<KeyRelease>', lambda e: self._validate_form())
        self.workers_spinbox.bind('<KeyRelease>', lambda e: self._validate_form())
        self.include_likes_var.trace('w', lambda *args: self._validate_form())

        # 初期バリデーション
        self._validate_form()

    def _browse_youtube_output_dir(self):
        """YouTube出力ディレクトリを選択"""
        dir_path = filedialog.askdirectory(initialdir=self.youtube_output_entry.get())
        if dir_path:
            self.youtube_output_entry.delete(0, tk.END)
            self.youtube_output_entry.insert(0, dir_path)

    def _browse_paper_output_dir(self):
        """論文出力ディレクトリを選択"""
        dir_path = filedialog.askdirectory(initialdir=self.paper_output_entry.get())
        if dir_path:
            self.paper_output_entry.delete(0, tk.END)
            self.paper_output_entry.insert(0, dir_path)

    def _get_youtube_info(self):
        """YouTube動画情報を取得"""
        url = self.youtube_url_entry.get().strip()
        if not url:
            messagebox.showwarning("警告", "YouTube URLを入力してください。")
            return

        try:
            if not self.youtube_downloader:
                output_dir = self.youtube_output_entry.get().strip()
                self.youtube_downloader = YouTubeDownloader(output_dir)

            info = self.youtube_downloader.get_video_info(url)
            if info:
                info_text = f"""
タイトル: {info['title']}
著者: {info['author']}
長さ: {info['length']} 秒
視聴回数: {info['views']}
評価: {info['rating']}
利用可能な解像度: {', '.join([s['resolution'] for s in info['streams'][:5]])}
                """
                messagebox.showinfo("動画情報", info_text)
            else:
                messagebox.showerror("エラー", "動画情報の取得に失敗しました。")
        except Exception as e:
            messagebox.showerror("エラー", f"情報取得に失敗しました: {e}")

    def _download_youtube_video(self):
        """YouTube動画をダウンロード"""
        url = self.youtube_url_entry.get().strip()
        if not url:
            messagebox.showwarning("警告", "YouTube URLを入力してください。")
            return

        if self.is_running:
            messagebox.showwarning("警告", "他の処理が実行中です。")
            return

        try:
            if not self.youtube_downloader:
                output_dir = self.youtube_output_entry.get().strip()
                self.youtube_downloader = YouTubeDownloader(output_dir)

            resolution = self.resolution_combo.get()
            audio_only = self.audio_only_var.get()

            self.is_running = True
            self.status_label.config(text="ダウンロード中...", foreground=DesignTokens.COLOR_WARNING)

            if audio_only:
                download_thread = threading.Thread(target=self._download_youtube_audio, args=(url,), daemon=True)
            else:
                download_thread = threading.Thread(target=self._download_youtube_video_thread, args=(url, resolution), daemon=True)

            download_thread.start()

        except Exception as e:
            messagebox.showerror("エラー", f"ダウンロード開始に失敗しました: {e}")

    def _download_youtube_video_thread(self, url, resolution):
        """YouTube動画ダウンロードスレッド"""
        try:
            downloaded_path = self.youtube_downloader.download_video(url, resolution)
            if downloaded_path:
                self.message_queue.put(("status", "ダウンロード完了"))
                self.message_queue.put(("log", f"ダウンロード完了: {downloaded_path}"))
            else:
                self.message_queue.put(("error", "ダウンロードに失敗しました。"))
        except Exception as e:
            self.message_queue.put(("error", f"ダウンロードエラー: {e}"))
        finally:
            self.message_queue.put(("finished", None))

    def _download_youtube_audio(self, url):
        """YouTube音声ダウンロードスレッド"""
        try:
            downloaded_path = self.youtube_downloader.download_audio_only(url)
            if downloaded_path:
                self.message_queue.put(("status", "音声ダウンロード完了"))
                self.message_queue.put(("log", f"音声ダウンロード完了: {downloaded_path}"))
            else:
                self.message_queue.put(("error", "音声ダウンロードに失敗しました。"))
        except Exception as e:
            self.message_queue.put(("error", f"音声ダウンロードエラー: {e}"))
        finally:
            self.message_queue.put(("finished", None))

    def _search_papers(self):
        """論文を検索"""
        query = self.paper_query_entry.get().strip()
        if not query:
            messagebox.showwarning("警告", "検索クエリを入力してください。")
            return

        try:
            source = self.paper_source_var.get()
            limit = int(self.paper_limit_spinbox.get())

            if source == "arxiv":
                if not self.arxiv_collector:
                    output_dir = self.paper_output_entry.get().strip()
                    self.arxiv_collector = ArXivCollector(output_dir)

                papers = self.arxiv_collector.search_papers(query, max_results=limit)
            else:  # semantic_scholar
                if not self.semantic_scholar_collector:
                    output_dir = self.paper_output_entry.get().strip()
                    self.semantic_scholar_collector = SemanticScholarCollector(output_dir)

                papers = self.semantic_scholar_collector.search_papers(query, limit=limit)

            # 検索結果を表示
            self.paper_result_text.delete(1.0, tk.END)
            if papers:
                for i, paper in enumerate(papers, 1):
                    self.paper_result_text.insert(tk.END, f"{i}. {paper['title']}\n")
                    self.paper_result_text.insert(tk.END, f"   著者: {', '.join(paper['authors'])}\n")
                    if source == "arxiv":
                        self.paper_result_text.insert(tk.END, f"   arXiv ID: {paper['arxiv_id']}\n")
                    else:
                        self.paper_result_text.insert(tk.END, f"   年: {paper['year']}\n")
                    self.paper_result_text.insert(tk.END, f"   要約: {paper['abstract'][:200]}...\n\n")
            else:
                self.paper_result_text.insert(tk.END, "検索結果が見つかりませんでした。")

        except Exception as e:
            messagebox.showerror("エラー", f"検索に失敗しました: {e}")

    def _download_papers(self):
        """論文をダウンロード"""
        if self.is_running:
            messagebox.showwarning("警告", "他の処理が実行中です。")
            return

        try:
            source = self.paper_source_var.get()
            query = self.paper_query_entry.get().strip()
            if not query:
                messagebox.showwarning("警告", "検索クエリを入力してください。")
                return

            limit = int(self.paper_limit_spinbox.get())

            if source == "arxiv":
                if not self.arxiv_collector:
                    output_dir = self.paper_output_entry.get().strip()
                    self.arxiv_collector = ArXivCollector(output_dir)

                papers = self.arxiv_collector.search_papers(query, max_results=limit)
                download_func = self.arxiv_collector.download_paper
            else:  # semantic_scholar
                if not self.semantic_scholar_collector:
                    output_dir = self.paper_output_entry.get().strip()
                    self.semantic_scholar_collector = SemanticScholarCollector(output_dir)

                papers = self.semantic_scholar_collector.search_papers(query, limit=limit)
                # Semantic ScholarではPDFダウンロード機能がないので、arXivと同様に扱う
                download_func = None

            if not papers:
                messagebox.showwarning("警告", "ダウンロードする論文が見つかりません。まず検索を実行してください。")
                return

            self.is_running = True
            self.status_label.config(text="ダウンロード中...", foreground=DesignTokens.COLOR_WARNING)

            download_thread = threading.Thread(target=self._download_papers_thread, args=(papers, download_func), daemon=True)
            download_thread.start()

        except Exception as e:
            messagebox.showerror("エラー", f"ダウンロード開始に失敗しました: {e}")

    def _download_papers_thread(self, papers, download_func):
        """論文ダウンロードスレッド"""
        try:
            downloaded = 0
            for paper in papers:
                if not self.is_running:
                    break

                if download_func and 'arxiv_id' in paper:
                    downloaded_path = download_func(paper['arxiv_id'])
                    if downloaded_path:
                        self.message_queue.put(("log", f"ダウンロード完了: {downloaded_path}"))
                        downloaded += 1
                    else:
                        self.message_queue.put(("log", f"ダウンロード失敗: {paper['title']}"))

                # 進捗更新
                self.message_queue.put(("progress", (downloaded / len(papers)) * 100))

            self.message_queue.put(("status", f"{downloaded}/{len(papers)} 論文ダウンロード完了"))

        except Exception as e:
            self.message_queue.put(("error", f"ダウンロードエラー: {e}"))
        finally:
            self.message_queue.put(("finished", None))

    def _validate_form(self):
        """フォームのバリデーションを実行"""
        errors = {}

        # ブログ名のバリデーション（URL検証を含む）
        blog_name = self.blog_entry.get().strip()
        if not blog_name and not self.include_likes_var.get():
            errors['blog_name'] = "ブログ名を入力するか、「いいねした投稿も含める」をチェックしてください。"
        else:
            # URL検証を追加
            if blog_name:
                from production_url_manager import get_url_manager
                url_manager = get_url_manager()
                is_valid, reason = url_manager.validate_url_security(f"https://{blog_name}.tumblr.com")
                if not is_valid:
                    errors['blog_name'] = f"ブログ名が無効です: {reason}"

        # 出力ディレクトリのバリデーション
        output_dir = self.output_entry.get().strip()
        if not output_dir:
            errors['output_dir'] = "出力フォルダを指定してください。"
        elif not os.path.isdir(output_dir):
            errors['output_dir'] = "指定されたフォルダが存在しません。"

        # ワーカー数のバリデーション
        try:
            workers = int(self.workers_spinbox.get())
            if workers < 1 or workers > 20:
                errors['workers'] = "並列数は1-20の範囲で指定してください。"
        except ValueError:
            errors['workers'] = "並列数は有効な数値で指定してください。"

        self.validation_errors = errors

        # UIの更新
        self._update_validation_ui()

        # 開始ボタンの状態更新
        self.start_button.config(state=tk.NORMAL if not errors else tk.DISABLED)

    def _update_validation_ui(self):
        """バリデーションUIを更新"""
        # エラーメッセージを表示
        self.blog_error_label.config(text=self.validation_errors.get('blog_name', ''))
        self.output_error_label.config(text=self.validation_errors.get('output_dir', ''))
        self.workers_error_label.config(text=self.validation_errors.get('workers', ''))

    def _setup_menu(self):
        """メニューバーの設定"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # ファイルメニュー
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ファイル", menu=file_menu)
        file_menu.add_command(label="設定保存", command=self._save_config)
        file_menu.add_command(label="設定読み込み", command=self._load_config)
        file_menu.add_separator()
        file_menu.add_command(label="終了", command=self._on_closing)

        # ツールメニュー
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ツール", menu=tools_menu)
        tools_menu.add_command(label="設定ウィザード", command=self._run_config_wizard)
        tools_menu.add_command(label="統計表示", command=self._show_statistics)
        tools_menu.add_command(label="ログクリア", command=self._clear_log)

        # ヘルプメニュー
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ヘルプ", menu=help_menu)
        help_menu.add_command(label="使い方", command=self._show_help)
        help_menu.add_command(label="バージョン情報", command=self._show_about)

    def _setup_logging(self):
        """ログ表示の設定"""
        class GUITextHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
                self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

            def emit(self, record):
                msg = self.format(record)
                self.text_widget.after(0, lambda: self._append_text(msg))

            def _append_text(self, msg):
                self.text_widget.insert(tk.END, msg + '\n')
                self.text_widget.see(tk.END)
                # ログが多すぎる場合は古いものを削除
                if self.text_widget.index('end-1c').split('.')[0] > '1000':
                    self.text_widget.delete('1.0', '100.0')

        # GUIログハンドラーを追加
        gui_handler = GUITextHandler(self.log_text)
        gui_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(gui_handler)

    def _browse_output_dir(self):
        """出力ディレクトリを選択"""
        dir_path = filedialog.askdirectory(initialdir=self.output_entry.get())
        if dir_path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, dir_path)

    def _start_collection(self):
        """コレクションを開始"""
        if self.is_running or self.validation_errors:
            return

        # UI状態の更新
        self.start_button.config(state=tk.DISABLED, text="実行中...")
        self.stop_button.config(state=tk.NORMAL)
        self.is_running = True
        self.progress_var.set(0)
        self.status_label.config(text="実行中...", foreground=DesignTokens.COLOR_WARNING)

        # プログレス統計を初期化
        self.progress_stats = {
            'start_time': time.time(),
            'current': 0,
            'total': 0,
            'eta': 0
        }
        self._update_progress_display()

        # コレクターの初期化
        try:
            config_file = "config.json"
            output_dir = self.output_entry.get().strip()
            workers = int(self.workers_spinbox.get())

            self.collector = TumblrImageCollector(
                config_file=config_file,
                output_dir_override=output_dir,
                workers_override=workers
            )

            # コレクションを別スレッドで実行
            collection_thread = threading.Thread(target=self._run_collection, daemon=True)
            collection_thread.start()

        except Exception as e:
            messagebox.showerror("エラー", f"コレクターの初期化に失敗しました: {e}")
            logger.error(f"コレクター初期化エラー: {e}")
            self._reset_ui_state()

    def _run_collection(self):
        """コレクションを実行（別スレッド）"""
        try:
            blog_name = self.blog_entry.get().strip()
            tags = [tag.strip() for tag in self.tags_entry.get().split(',') if tag.strip()]
            include_likes = self.include_likes_var.get()

            # 進捗更新のためのコールバック設定
            self.collector.progress_callback = self._update_progress

            # 日付範囲の設定（未実装）
            date_range = None

            # コレクション実行
            self.collector.run(
                blog_name=blog_name,
                tags=tags,
                date_range=date_range,
                include_likes=include_likes
            )

            self.message_queue.put(("status", "完了"))
            self.message_queue.put(("progress", 100))

        except Exception as e:
            logger.error(f"コレクション実行エラー: {e}")
            self.message_queue.put(("error", str(e)))
        finally:
            self.message_queue.put(("finished", None))

    def _reset_ui_state(self):
        """UI状態をリセット"""
        self.start_button.config(state=tk.NORMAL, text="開始")
        self.stop_button.config(state=tk.DISABLED)
        self.is_running = False
        self.status_label.config(text="準備完了", foreground=DesignTokens.COLOR_SUCCESS)
        self.time_label.config(text="")
        self.progress_detail_label.config(text="0/0 項目")
        self._validate_form()  # バリデーションを再実行

    def _stop_collection(self):
        """コレクションを停止"""
        if self.collector and self.is_running:
            # コレクターの停止処理（未実装）
            logger.info("コレクション停止リクエスト")
            self.is_running = False
            self._reset_ui_state()

    def _update_progress_display(self):
        """プログレス表示を更新"""
        if not self.progress_stats['start_time']:
            return

        current = self.progress_stats['current']
        total = self.progress_stats['total']
        elapsed = time.time() - self.progress_stats['start_time']

        # 詳細情報を更新
        if total > 0:
            self.progress_detail_label.config(text=f"{current}/{total} 項目")
        else:
            self.progress_detail_label.config(text=f"{current} 項目")

        # 時間情報を更新
        elapsed_str = f"経過: {elapsed:.1f}s"
        if total > 0 and current > 0:
            eta = (elapsed / current) * (total - current)
            eta_str = f" | 残り: {eta:.1f}s"
            self.time_label.config(text=elapsed_str + eta_str)
        else:
            self.time_label.config(text=elapsed_str)

        # ダウンロード速度と失敗数の統計を追加
        if current > 0:
            speed = current / elapsed if elapsed > 0 else 0
            self.speed_label.config(text=f"速度: {speed:.2f} 項目/秒")
        else:
            self.speed_label.config(text="速度: 計算中...")

    def _update_progress(self, current, total, status="実行中"):
        """進捗を更新"""
        self.progress_stats['current'] = current
        self.progress_stats['total'] = total

        if total > 0:
            progress = (current / total) * 100
            self.message_queue.put(("progress", progress))

        self.message_queue.put(("status", status))
        self.message_queue.put(("progress_update", None))

    def _process_messages(self):
        """メッセージキューを処理"""
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()

                if msg_type == "progress":
                    self.progress_var.set(data)
                elif msg_type == "status":
                    self.status_label.config(text=data)
                elif msg_type == "progress_update":
                    self._update_progress_display()
                elif msg_type == "error":
                    messagebox.showerror("エラー", data)
                    self._reset_ui_state()
                elif msg_type == "finished":
                    self._reset_ui_state()
                    self.is_running = False
                elif msg_type == "log":
                    self.log_text.insert(tk.END, data + '\n')
                    self.log_text.see(tk.END)

        except queue.Empty:
            pass

        # 次の処理をスケジュール
        self.root.after(100, self._process_messages)

    def _save_config(self):
        """現在の設定を保存"""
        config = {
            "blog_name": self.blog_entry.get(),
            "output_dir": self.output_entry.get(),
            "tags": self.tags_entry.get(),
            "include_likes": self.include_likes_var.get(),
            "workers": int(self.workers_spinbox.get()),
            "interactive": self.interactive_var.get()
        }

        # クリップボード設定を追加
        config["clipboard_monitor"] = {
            "enabled": self.clipboard_monitor_var.get(),
            "show_notifications": self.show_notifications_var.get(),
            "notification_sound": self.notification_sound_var.get(),
            "detected_urls": list(self.detected_urls)
        }

        # プレビュー設定を追加
        config["preview"] = {
            "enabled": self.preview_enabled_var.get(),
            "thumbnail_size": self.thumbnail_size_var.get(),
            "slideshow_enabled": self.slideshow_var.get()
        }

        # ファイルリネーム設定を追加
        config["file_rename"] = {
            "enabled": self.rename_enabled_var.get(),
            "template": self.rename_template_var.get(),
            "collision_resolve": self.collision_resolve_var.get()
        }

        # 帯域制限設定を追加
        config["bandwidth_limit"] = {
            "enabled": self.bandwidth_limit_var.get(),
            "max_rate_kbps": self.bandwidth_rate_var.get(),
            "preset": self.bandwidth_preset_var.get(),
            "auto_adjust": self.auto_adjust_var.get()
        }

        # 出力フォーマット設定を追加
        selected_formats = []
        for format_key, var in self.format_vars.items():
            if var.get():
                selected_formats.append(format_key)

        config["output_format"] = {
            "formats": selected_formats,
            "json_pretty": self.json_pretty_var.get(),
            "json_metadata": self.json_metadata_var.get(),
            "html5_responsive": self.html5_responsive_var.get(),
            "markdown_images": self.markdown_images_var.get()
        }

        try:
            config_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if config_path:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                messagebox.showinfo("成功", f"設定を保存しました: {config_path}")
        except Exception as e:
            messagebox.showerror("エラー", f"設定保存に失敗しました: {e}")

    def _load_config(self):
        """設定を読み込み"""
        try:
            config_path = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if config_path:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                self.blog_entry.delete(0, tk.END)
                self.blog_entry.insert(0, config.get("blog_name", ""))

                self.output_entry.delete(0, tk.END)
                self.output_entry.insert(0, config.get("output_dir", ""))

                self.tags_entry.delete(0, tk.END)
                self.tags_entry.insert(0, config.get("tags", ""))

                self.include_likes_var.set(config.get("include_likes", False))
                self.interactive_var.set(config.get("interactive", False))

                workers = config.get("workers", 5)
                self.workers_spinbox.delete(0, tk.END)
                self.workers_spinbox.insert(0, str(workers))

                # クリップボード設定を読み込み
                clipboard_config = config.get("clipboard_monitor", {})
                self.clipboard_monitor_var.set(clipboard_config.get("enabled", True))
                self.show_notifications_var.set(clipboard_config.get("show_notifications", True))
                self.notification_sound_var.set(clipboard_config.get("notification_sound", True))

                # 検出されたURLを復元
                detected_urls = clipboard_config.get("detected_urls", [])
                self.detected_urls = set(detected_urls)
                self._update_detected_urls_listbox()

                # プレビュー設定を読み込み
                preview_config = config.get("preview", {})
                self.preview_enabled_var.set(preview_config.get("enabled", True))
                self.thumbnail_size_var.set(preview_config.get("thumbnail_size", "200x200"))
                self.slideshow_var.set(preview_config.get("slideshow_enabled", False))

                # ファイルリネーム設定を読み込み
                rename_config = config.get("file_rename", {})
                self.rename_enabled_var.set(rename_config.get("enabled", True))
                self.rename_template_var.set(rename_config.get("template", "{blog}_{timestamp}_{id}_{tags}"))
                self.collision_resolve_var.set(rename_config.get("collision_resolve", True))

                # プレビューを更新
                self._update_rename_preview()

                # 帯域制限設定を読み込み
                bandwidth_config = config.get("bandwidth_limit", {})
                self.bandwidth_limit_var.set(bandwidth_config.get("enabled", False))
                self.bandwidth_rate_var.set(bandwidth_config.get("max_rate_kbps", "1024"))
                self.bandwidth_preset_var.set(bandwidth_config.get("preset", "normal"))
                self.auto_adjust_var.set(bandwidth_config.get("auto_adjust", True))

                # 出力フォーマット設定を読み込み
                format_config = config.get("output_format", {})

                # フォーマット選択を復元
                saved_formats = format_config.get("formats", ["json", "html5"])
                for format_key, var in self.format_vars.items():
                    var.set(format_key in saved_formats)

                # 設定を復元
                self.json_pretty_var.set(format_config.get("json_pretty", True))
                self.json_metadata_var.set(format_config.get("json_metadata", True))
                self.html5_responsive_var.set(format_config.get("html5_responsive", True))
                self.markdown_images_var.set(format_config.get("markdown_images", True))

                messagebox.showinfo("成功", f"設定を読み込みました: {config_path}")
        except Exception as e:
            messagebox.showerror("エラー", f"設定読み込みに失敗しました: {e}")

    def _run_config_wizard(self):
        """設定ウィザードを実行"""
        try:
            from config import ConfigWizard
            wizard = ConfigWizard()
            wizard.run_wizard()
            messagebox.showinfo("完了", "設定ウィザードが完了しました。")
        except Exception as e:
            messagebox.showerror("エラー", f"設定ウィザードの実行に失敗しました: {e}")

    def _show_statistics(self):
        """統計情報を表示"""
        if self.collector:
            try:
                stats = self.collector.print_download_stats()
                # 統計表示ダイアログ（未実装）
                messagebox.showinfo("統計", "統計表示機能は開発中です。")
            except Exception as e:
                messagebox.showerror("エラー", f"統計取得に失敗しました: {e}")
        else:
            messagebox.showwarning("警告", "コレクターが初期化されていません。")

    def _clear_log(self):
        """ログをクリア"""
        self.log_text.delete(1.0, tk.END)

    def _show_help(self):
        """ヘルプを表示"""
        help_text = """
Tumblr Image Collector GUI ヘルプ

基本操作:
1. ブログ名を入力するか、「いいねした投稿も含める」をチェック
2. 必要に応じてタグ、出力フォルダを設定
3. 「開始」ボタンをクリックしてダウンロードを開始
4. 進捗はプログレスバーとログで確認できます

設定オプション:
- ブログ名: 収集対象のTumblrブログ名
- 出力フォルダ: ダウンロードしたファイルの保存先
- タグ: カンマ区切りで指定したタグの投稿を収集
- いいねした投稿も含める: 認証ユーザーのいいね投稿を収集
- 並列数: 同時にダウンロードするワーカー数

注意事項:
- Tumblr APIのレート制限に注意してください
- 大量のダウンロードには時間がかかります
- 設定はJSON形式で保存/読み込み可能です
        """
        messagebox.showinfo("ヘルプ", help_text)

    def _show_about(self):
        """バージョン情報を表示"""
        about_text = """
Tumblr Image Collector v2.1

機能:
- Tumblrブログからの画像/動画収集
- タグベースの検索
- いいね投稿の収集
- 並列ダウンロード
- 重複検出
- 進捗表示

ライセンス: MIT License
        """
        messagebox.showinfo("バージョン情報", about_text)

    def _on_closing(self):
        """ウィンドウ閉じイベント"""
        # クリップボードモニターを停止
        if self.clipboard_monitor:
            try:
                self.clipboard_monitor.stop_monitoring()
                logger.info("クリップボードモニターを停止")
            except Exception as e:
                logger.error(f"クリップボードモニター停止エラー: {e}")

        # 帯域制限システムを停止
        if self.bandwidth_limiter:
            try:
                self.bandwidth_limiter.stop_monitoring()
                logger.info("帯域制限システムを停止")
            except Exception as e:
                logger.error(f"帯域制限システム停止エラー: {e}")

        if self.is_running:
            if messagebox.askyesno("確認", "コレクションを実行中です。終了しますか？"):
                self._stop_collection()
                self.root.destroy()
        else:
            self.root.destroy()

    def _init_clipboard_monitor(self):
        """クリップボードモニターを初期化"""
        try:
            from clipboard_monitor import create_clipboard_monitor

            def on_tumblr_detected(url: str, blog_name: Optional[str]):
                """Tumblr URL検出時のコールバック"""
                self.detected_urls.add(url)
                self._update_detected_urls_listbox()

                # 設定に基づいて通知を表示
                if self.show_notifications_var.get():
                    if blog_name:
                        messagebox.showinfo("Tumblr URL検出", f"ブログ '{blog_name}' を検出しました。")
                    else:
                        messagebox.showinfo("Tumblr URL検出", "Tumblr URLを検出しました。")

            self.clipboard_monitor = create_clipboard_monitor(callback=on_tumblr_detected)

            # 設定を適用
            self.clipboard_monitor.update_config(
                show_notifications=self.show_notifications_var.get(),
                notification_sound=self.notification_sound_var.get()
            )

            # クリップボードモニタリングを開始
            if self.clipboard_monitor_var.get():
                self.clipboard_monitor.start_monitoring()

        except ImportError as e:
            logger.warning(f"クリップボードモニターの初期化に失敗: {e}")
            self.clipboard_monitor_var.set(False)
        except Exception as e:
            logger.error(f"クリップボードモニターエラー: {e}")

    def _toggle_clipboard_monitor(self):
        """クリップボードモニターの有効/無効を切り替え"""
        if self.clipboard_monitor_var.get():
            # モニタリングを開始
            if self.clipboard_monitor:
                try:
                    self.clipboard_monitor.start_monitoring()
                    logger.info("クリップボードモニタリングを開始")
                except Exception as e:
                    logger.error(f"クリップボードモニタリング開始エラー: {e}")
                    self.clipboard_monitor_var.set(False)
        else:
            # モニタリングを停止
            if self.clipboard_monitor:
                try:
                    self.clipboard_monitor.stop_monitoring()
                    logger.info("クリップボードモニタリングを停止")
                except Exception as e:
                    logger.error(f"クリップボードモニタリング停止エラー: {e}")

        # 設定を保存
        self._save_clipboard_settings()

    def _update_detected_urls_listbox(self):
        """検出されたURLリストボックスを更新"""
        if hasattr(self, 'detected_urls_listbox'):
            self.detected_urls_listbox.delete(0, tk.END)
            for url in self.detected_urls:
                blog_name = self._extract_blog_name_from_url(url)
                display_text = f"{blog_name} ({url})" if blog_name else url
                self.detected_urls_listbox.insert(tk.END, display_text)

    def _extract_blog_name_from_url(self, url: str) -> Optional[str]:
        """URLからブログ名を抽出"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.endswith('.tumblr.com'):
                return domain.replace('.tumblr.com', '')
        except Exception:
            pass
        return None

    def _clear_detected_urls(self):
        """検出されたURLをクリア"""
        self.detected_urls.clear()
        if self.clipboard_monitor:
            self.clipboard_monitor.clear_detected_urls()
        self._update_detected_urls_listbox()

    def _remove_selected_detected(self):
        """選択された検出URLを削除"""
        selected_indices = self.detected_urls_listbox.curselection()
        for index in reversed(selected_indices):
            url_to_remove = list(self.detected_urls)[index]
            self.detected_urls.remove(url_to_remove)
        self._update_detected_urls_listbox()

    def _start_from_detected(self):
        """検出されたURLからダウンロードを開始"""
        if not self.detected_urls:
            messagebox.showwarning("警告", "検出されたブログがありません。")
            return

        # 検出されたURLをブログ名として設定
        selected_indices = self.detected_urls_listbox.curselection()
        if selected_indices:
            # 選択された項目のみ使用
            selected_urls = [list(self.detected_urls)[i] for i in selected_indices]
            blog_names = [self._extract_blog_name_from_url(url) or url for url in selected_urls]
        else:
            # すべての検出されたURLを使用
            blog_names = [self._extract_blog_name_from_url(url) or url for url in self.detected_urls]

        # 最初のブログ名を入力フィールドに設定
        if blog_names:
            self.blog_entry.delete(0, tk.END)
            self.blog_entry.insert(0, blog_names[0])

        # ダウンロードを開始
        self._start_collection()

    def _save_clipboard_settings(self):
        """クリップボード設定を保存"""
        try:
            config = {}
            if os.path.exists('config.json'):
                with open('config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)

            if 'clipboard_monitor' not in config:
                config['clipboard_monitor'] = {}

            config['clipboard_monitor'].update({
                'enabled': self.clipboard_monitor_var.get(),
                'show_notifications': self.show_notifications_var.get(),
                'notification_sound': self.notification_sound_var.get(),
                'detected_urls': list(self.detected_urls)
            })

            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"クリップボード設定保存エラー: {e}")

    def _load_clipboard_settings(self):
        """クリップボード設定を読み込み"""
        try:
            if os.path.exists('config.json'):
                with open('config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)

                clipboard_config = config.get('clipboard_monitor', {})
                self.clipboard_monitor_var.set(clipboard_config.get('enabled', True))
                self.show_notifications_var.set(clipboard_config.get('show_notifications', True))
                self.notification_sound_var.set(clipboard_config.get('notification_sound', True))

                # 検出されたURLを復元
                detected_urls = clipboard_config.get('detected_urls', [])
                self.detected_urls = set(detected_urls)
                self._update_detected_urls_listbox()

        except Exception as e:
            logger.error(f"クリップボードモニターエラー: {e}")

    def _init_preview_system(self):
        """プレビューシステムを初期化"""
        try:
            from image_preview import create_preview_system
            self.preview_system = create_preview_system("preview_cache")
            logger.info("プレビューシステムを初期化しました")
        except ImportError as e:
            logger.warning(f"プレビューシステムの初期化に失敗: {e}")
            self.preview_enabled_var.set(False)
        except Exception as e:
            logger.error(f"プレビューシステムエラー: {e}")

    def _generate_thumbnails(self):
        """サムネイルを生成"""
        if not self.preview_system:
            messagebox.showwarning("警告", "プレビューシステムが初期化されていません。")
            return

        # ダウンロードされた画像のパスを取得
        output_dir = self.output_entry.get().strip()
        if not os.path.exists(output_dir):
            messagebox.showwarning("警告", "出力フォルダが存在しません。")
            return

        # 画像ファイルを検索
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

        image_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.lower().endswith(image_extensions + video_extensions):
                    image_files.append(os.path.join(root, file))

        if not image_files:
            messagebox.showinfo("情報", "生成する画像が見つかりません。")
            return

        try:
            # サムネイル生成の進行状況を表示
            self.status_label.config(text="サムネイル生成中...", foreground=DesignTokens.COLOR_WARNING)
            self.progress_var.set(0)

            # バッチ処理でサムネイルを生成
            thumbnails = self.preview_system.batch_generate_thumbnails(
                image_files,
                progress_callback=self._update_thumbnail_progress
            )

            self.status_label.config(text=f"サムネイル生成完了 ({len(thumbnails)}個)", foreground=DesignTokens.COLOR_SUCCESS)
            messagebox.showinfo("完了", f"{len(thumbnails)}個のサムネイルを生成しました。")

        except Exception as e:
            logger.error(f"サムネイル生成エラー: {e}")
            messagebox.showerror("エラー", f"サムネイル生成に失敗しました: {e}")
        finally:
            self._reset_ui_state()

    def _update_thumbnail_progress(self, progress: float, total: int, current: int):
        """サムネイル生成の進捗を更新"""
        self.progress_var.set(progress * 100)
        self.status_label.config(text=f"サムネイル生成中... {current}/{total}")

    def _start_slideshow(self):
        """スライドショーを開始"""
        if not self.preview_system:
            messagebox.showwarning("警告", "プレビューシステムが初期化されていません。")
            return

        # ダウンロードされた画像のパスを取得
        output_dir = self.output_entry.get().strip()
        if not os.path.exists(output_dir):
            messagebox.showwarning("警告", "出力フォルダが存在しません。")
            return

        # 画像ファイルを検索
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

        image_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.lower().endswith(image_extensions + video_extensions):
                    image_files.append(os.path.join(root, file))

        if not image_files:
            messagebox.showinfo("情報", "スライドショー用の画像が見つかりません。")
            return

        try:
            # サムネイルサイズを解析
            size_str = self.thumbnail_size_var.get()
            if 'x' in size_str:
                width, height = map(int, size_str.split('x'))
                self.preview_system.thumbnail_size = (width, height)

            # スライドショーを開始
            success = self.preview_system.start_slideshow(
                image_files,
                delay=3.0,
                window_title="Tumblr Image Slideshow"
            )

            if success:
                messagebox.showinfo("スライドショー", f"{len(image_files)}個の画像でスライドショーを開始しました。")
            else:
                messagebox.showerror("エラー", "スライドショーの開始に失敗しました。")

        except Exception as e:
            logger.error(f"スライドショー開始エラー: {e}")
            messagebox.showerror("エラー", f"スライドショーの開始に失敗しました: {e}")

    def _show_full_preview(self):
        """フルサイズプレビューを表示"""
        if not self.preview_system:
            messagebox.showwarning("警告", "プレビューシステムが初期化されていません。")
            return

        # プレビューする画像を選択
        output_dir = self.output_entry.get().strip()
        if not os.path.exists(output_dir):
            messagebox.showwarning("警告", "出力フォルダが存在しません。")
            return

        # 画像ファイルを検索
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

        image_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.lower().endswith(image_extensions + video_extensions):
                    image_files.append(os.path.join(root, file))

        if not image_files:
            messagebox.showinfo("情報", "プレビューする画像が見つかりません。")
            return

        # ファイル選択ダイアログ
        preview_window = tk.Toplevel(self.root)
        preview_window.title("画像選択")
        preview_window.geometry("600x400")

        # ファイルリスト
        listbox = tk.Listbox(preview_window, selectmode=tk.SINGLE)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # スクロールバー
        scrollbar = ttk.Scrollbar(preview_window, orient=tk.VERTICAL, command=listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.configure(yscrollcommand=scrollbar.set)

        # ファイルをリストに追加
        for file_path in image_files:
            filename = os.path.basename(file_path)
            listbox.insert(tk.END, filename)

        def on_preview():
            selection = listbox.curselection()
            if selection:
                index = selection[0]
                selected_file = image_files[index]

                # フルプレビューを表示
                success = self.preview_system.show_full_preview(
                    selected_file,
                    title=f"Preview - {os.path.basename(selected_file)}"
                )

                if not success:
                    messagebox.showerror("エラー", "プレビューの表示に失敗しました。")

        def on_info():
            selection = listbox.curselection()
            if selection:
                index = selection[0]
                selected_file = image_files[index]

                # 画像情報を取得
                info = self.preview_system.get_image_info(selected_file)
                if info:
                    info_text = f"""
ファイル名: {info['filename']}
サイズ: {info['size'][0]}x{info['size'][1]}
モード: {info['mode']}
形式: {info['format']}
ファイルサイズ: {info['file_size']} bytes
透明度: {'あり' if info['has_transparency'] else 'なし'}
                    """
                    messagebox.showinfo("画像情報", info_text.strip())
                else:
                    messagebox.showerror("エラー", "画像情報の取得に失敗しました。")

        # ボタンフレーム
        button_frame = ttk.Frame(preview_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="プレビュー", command=on_preview).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="情報", command=on_info).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(button_frame, text="閉じる", command=preview_window.destroy).pack(side=tk.RIGHT)

    def _clear_preview_cache(self):
        """プレビューキャッシュをクリア"""
        if not self.preview_system:
            messagebox.showwarning("警告", "プレビューシステムが初期化されていません。")
            return

        try:
            # キャッシュディレクトリの内容を削除
            cache_dir = Path("preview_cache")
            if cache_dir.exists():
                for file_path in cache_dir.glob("*"):
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                    except Exception as e:
                        logger.error(f"キャッシュファイル削除エラー: {e}")

                # キャッシュをクリア
                self.preview_system.cache.clear()

                messagebox.showinfo("完了", "プレビューキャッシュをクリアしました。")
            else:
                messagebox.showinfo("情報", "キャッシュディレクトリが存在しません。")

        except Exception as e:
            logger.error(f"キャッシュクリアエラー: {e}")
            messagebox.showerror("エラー", f"キャッシュのクリアに失敗しました: {e}")

    def _save_preview_settings(self):
        """プレビュー設定を保存"""
        try:
            config = {}
            if os.path.exists('config.json'):
                with open('config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)

            if 'preview' not in config:
                config['preview'] = {}

            config['preview'].update({
                'enabled': self.preview_enabled_var.get(),
                'thumbnail_size': self.thumbnail_size_var.get(),
                'slideshow_enabled': self.slideshow_var.get()
            })

            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"プレビュー設定保存エラー: {e}")

    def _load_preview_settings(self):
        """プレビュー設定を読み込み"""
        try:
            if os.path.exists('config.json'):
                with open('config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)

                preview_config = config.get('preview', {})
                self.preview_enabled_var.set(preview_config.get('enabled', True))
                self.thumbnail_size_var.set(preview_config.get('thumbnail_size', '200x200'))
                self.slideshow_var.set(preview_config.get('slideshow_enabled', False))

        except Exception as e:
            logger.error(f"プレビューシステムエラー: {e}")

    def _init_file_renamer(self):
        """ファイルリネームシステムを初期化"""
        try:
            from file_renamer import create_file_renamer, TEMPLATES

            # デフォルトテンプレートで初期化
            self.file_renamer = create_file_renamer("{blog}_{timestamp}_{id}_{tags}")

            # テンプレートコンボボックスの設定
            if hasattr(self, 'template_combo'):
                self.template_combo['values'] = list(TEMPLATES.keys())
                self.template_combo.bind('<<ComboboxSelected>>', self._on_template_selected)

            logger.info("ファイルリネームシステムを初期化しました")
        except ImportError as e:
            logger.warning(f"ファイルリネームシステムの初期化に失敗: {e}")
            self.rename_enabled_var.set(False)
        except Exception as e:
            logger.error(f"ファイルリネームシステムエラー: {e}")

    def _on_template_selected(self, event=None):
        """テンプレートが選択された時の処理"""
        if not self.file_renamer:
            return

        try:
            from file_renamer import TEMPLATES

            selected_template = self.template_combo.get()
            if selected_template in TEMPLATES:
                template = TEMPLATES[selected_template]
                self.rename_template_var.set(template)
                self._update_rename_preview()

        except Exception as e:
            logger.error(f"テンプレート選択エラー: {e}")

    def _update_rename_preview(self):
        """リネームプレビューを更新"""
        if not self.file_renamer:
            return

        try:
            # サンプルメタデータでプレビューを生成
            sample_metadata = {
                'blog_name': self.blog_entry.get().strip() or 'sample_blog',
                'post_id': 12345,
                'timestamp': datetime.now().timestamp(),
                'tags': [tag.strip() for tag in self.tags_entry.get().split(',') if tag.strip()] or ['sample', 'tag'],
                'media_type': 'image'
            }

            template = self.rename_template_var.get().strip()
            if template:
                self.file_renamer.set_template(template)

            preview_name = self.file_renamer.preview_rename(sample_metadata, "downloads", "jpg")
            self.rename_preview_var.set(preview_name)

        except Exception as e:
            logger.error(f"プレビュー更新エラー: {e}")
            self.rename_preview_var.set("プレビュー生成エラー")

    def _show_template_help(self):
        """テンプレートヘルプを表示"""
        help_text = """
使用可能なプレースホルダー:

{blog} - ブログ名
{id} - 投稿ID
{timestamp} - タイムスタンプ (YYYYMMDD_HHMMSS)
{date} - 日付 (YYYYMMDD)
{time} - 時刻 (HHMMSS)
{tags} - タグ (アンダースコア区切り)
{type} - メディアタイプ
{index} - インデックス番号 (001, 002, ...)
{hash} - ファイルハッシュ
{original} - 元のファイル名
{sequential} - 連番 (001, 002, ...)

例:
{blog}_{timestamp}_{id}_{tags}
{blog}_{date}_{sequential:03d}
{blog}_{type}_{id}

特殊文字は自動的にサニタイズされます。
        """

        help_window = tk.Toplevel(self.root)
        help_window.title("命名テンプレート ヘルプ")
        help_window.geometry("500x400")

        text_widget = tk.Text(help_window, wrap=tk.WORD, font=("Consolas", 10))
        text_widget.insert(tk.END, help_text.strip())
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Button(help_window, text="閉じる", command=help_window.destroy).pack(pady=10)

    def _save_rename_settings(self):
        """ファイルリネーム設定を保存"""
        try:
            config = {}
            if os.path.exists('config.json'):
                with open('config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)

            if 'file_rename' not in config:
                config['file_rename'] = {}

            config['file_rename'].update({
                'enabled': self.rename_enabled_var.get(),
                'template': self.rename_template_var.get(),
                'collision_resolve': self.collision_resolve_var.get()
            })

            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"ファイルリネーム設定保存エラー: {e}")

    def _load_rename_settings(self):
        """ファイルリネーム設定を読み込み"""
        try:
            if os.path.exists('config.json'):
                with open('config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)

                rename_config = config.get('file_rename', {})
                self.rename_enabled_var.set(rename_config.get('enabled', True))
                self.rename_template_var.set(rename_config.get('template', '{blog}_{timestamp}_{id}_{tags}'))
                self.collision_resolve_var.set(rename_config.get('collision_resolve', True))

                # プレビューを更新
                self._update_rename_preview()

        except Exception as e:
            logger.error(f"ファイルリネーム設定読み込みエラー: {e}")

    def _init_bandwidth_limiter(self):
        """帯域制限システムを初期化"""
        try:
            from bandwidth_limiter import create_bandwidth_limiter

            # デフォルト設定で初期化
            self.bandwidth_limiter = create_bandwidth_limiter(max_rate_kbps=1024, auto_adjust=True)
            self.bandwidth_limiter.rate_change_callback = self._on_rate_change

            # 定期的に帯域情報を更新
            self._update_bandwidth_info()

            logger.info("帯域制限システムを初期化しました")
        except ImportError as e:
            logger.warning(f"帯域制限システムの初期化に失敗: {e}")
            self.bandwidth_limit_var.set(False)
        except Exception as e:
            logger.error(f"帯域制限システムエラー: {e}")

    def _on_bandwidth_preset_selected(self, event=None):
        """帯域プリセットが選択された時の処理"""
        if not self.bandwidth_limiter:
            return

        try:
            from bandwidth_limiter import RATE_PRESETS

            preset = self.bandwidth_preset_var.get()
            if preset in RATE_PRESETS:
                rate_bps = RATE_PRESETS[preset]
                if rate_bps > 0:
                    self.bandwidth_limiter.set_max_rate(rate_bps)
                    self.bandwidth_rate_var.set(str(rate_bps // 1024))  # Convert to KB/s
                else:
                    # Unlimited
                    self.bandwidth_rate_var.set("0")

        except Exception as e:
            logger.error(f"帯域プリセット選択エラー: {e}")

    def _toggle_bandwidth_limit(self):
        """帯域制限の有効/無効を切り替え"""
        if self.bandwidth_limit_var.get():
            # 帯域制限を有効化
            if self.bandwidth_limiter:
                try:
                    # 現在の設定を適用
                    rate_kbps = int(self.bandwidth_rate_var.get())
                    if rate_kbps > 0:
                        self.bandwidth_limiter.set_max_rate(rate_kbps * 1024)
                        self.bandwidth_limiter.auto_adjust = self.auto_adjust_var.get()
                        logger.info(f"帯域制限を有効化: {rate_kbps}KB/s")
                    else:
                        # 無制限
                        logger.info("帯域制限を無制限に設定")
                except Exception as e:
                    logger.error(f"帯域制限有効化エラー: {e}")
                    self.bandwidth_limit_var.set(False)
        else:
            # 帯域制限を無効化
            if self.bandwidth_limiter:
                try:
                    logger.info("帯域制限を無効化")
                except Exception as e:
                    logger.error(f"帯域制限無効化エラー: {e}")

    def _start_bandwidth_limit(self):
        """帯域制限を開始"""
        if not self.bandwidth_limiter:
            messagebox.showwarning("警告", "帯域制限システムが初期化されていません。")
            return

        try:
            rate_kbps = int(self.bandwidth_rate_var.get())
            if rate_kbps > 0:
                self.bandwidth_limiter.set_max_rate(rate_kbps * 1024)
                self.bandwidth_limiter.start_monitoring()
                self.bandwidth_limit_var.set(True)
                messagebox.showinfo("帯域制限", f"帯域制限を開始しました: {rate_kbps}KB/s")
                logger.info(f"帯域制限開始: {rate_kbps}KB/s")
            else:
                messagebox.showwarning("警告", "有効な速度制限値を設定してください。")

        except ValueError:
            messagebox.showerror("エラー", "無効な速度制限値です。")
        except Exception as e:
            logger.error(f"帯域制限開始エラー: {e}")
            messagebox.showerror("エラー", f"帯域制限の開始に失敗しました: {e}")

    def _stop_bandwidth_limit(self):
        """帯域制限を停止"""
        if self.bandwidth_limiter:
            try:
                self.bandwidth_limiter.stop_monitoring()
                self.bandwidth_limit_var.set(False)
                messagebox.showinfo("帯域制限", "帯域制限を停止しました。")
                logger.info("帯域制限停止")
            except Exception as e:
                logger.error(f"帯域制限停止エラー: {e}")

    def _reset_bandwidth_stats(self):
        """帯域統計をリセット"""
        if self.bandwidth_limiter:
            try:
                self.bandwidth_limiter.reset_statistics()
                self._update_bandwidth_info()
                messagebox.showinfo("統計リセット", "帯域統計をリセットしました。")
                logger.info("帯域統計リセット")
            except Exception as e:
                logger.error(f"帯域統計リセットエラー: {e}")

    def _update_bandwidth_info(self):
        """帯域情報を更新"""
        if not self.bandwidth_limiter:
            return

        try:
            stats = self.bandwidth_limiter.get_statistics()

            # 現在の速度をKB/sで表示
            current_rate_kbps = stats['current_rate_bps'] / 1024
            self.current_rate_label.config(text=f"{current_rate_kbps:.1f} KB/s")

            # ピーク速度をKB/sで表示
            peak_rate_kbps = stats['peak_rate_bps'] / 1024
            self.peak_rate_label.config(text=f"{peak_rate_kbps:.1f} KB/s")

            # 総ダウンロード量をMBで表示
            total_mb = stats['total_bytes'] / (1024 * 1024)
            self.total_bytes_label.config(text=f"{total_mb:.2f} MB")

        except Exception as e:
            logger.error(f"帯域情報更新エラー: {e}")

        # 1秒後に再更新
        if hasattr(self, 'root'):
            self.root.after(1000, self._update_bandwidth_info)

    def _on_rate_change(self, new_rate: float):
        """帯域制限のレートが変更された時のコールバック"""
        logger.debug(f"帯域レート変更: {new_rate / 1024:.1f}KB/s")

    def _save_bandwidth_settings(self):
        """帯域制限設定を保存"""
        try:
            config = {}
            if os.path.exists('config.json'):
                with open('config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)

            if 'bandwidth_limit' not in config:
                config['bandwidth_limit'] = {}

            config['bandwidth_limit'].update({
                'enabled': self.bandwidth_limit_var.get(),
                'max_rate_kbps': self.bandwidth_rate_var.get(),
                'preset': self.bandwidth_preset_var.get(),
                'auto_adjust': self.auto_adjust_var.get()
            })

            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"帯域制限設定保存エラー: {e}")

    def _load_bandwidth_settings(self):
        """帯域制限設定を読み込み"""
        try:
            if os.path.exists('config.json'):
                with open('config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)

                bandwidth_config = config.get('bandwidth_limit', {})
                self.bandwidth_limit_var.set(bandwidth_config.get('enabled', False))
                self.bandwidth_rate_var.set(bandwidth_config.get('max_rate_kbps', '1024'))
                self.bandwidth_preset_var.set(bandwidth_config.get('preset', 'normal'))
                self.auto_adjust_var.set(bandwidth_config.get('auto_adjust', True))

        except Exception as e:
            logger.error(f"帯域制限設定読み込みエラー: {e}")

    def _init_output_formatter(self):
        """出力フォーマットシステムを初期化"""
        try:
            from output_formatter import create_output_formatter

            # デフォルト出力ディレクトリで初期化
            output_dir = os.path.join(os.getcwd(), "formatted_output")
            self.output_formatter = create_output_formatter(output_dir)

            logger.info("出力フォーマットシステムを初期化しました")
        except ImportError as e:
            logger.warning(f"出力フォーマットシステムの初期化に失敗: {e}")
        except Exception as e:
            logger.error(f"出力フォーマットシステムエラー: {e}")

    def _run_format_output(self):
        """出力フォーマットを実行"""
        if not self.output_formatter:
            messagebox.showwarning("警告", "出力フォーマットシステムが初期化されていません。")
            return

        # 選択されたフォーマットを収集
        selected_formats = []
        for format_key, var in self.format_vars.items():
            if var.get():
                selected_formats.append(format_key)

        if not selected_formats:
            messagebox.showwarning("警告", "少なくとも1つの出力フォーマットを選択してください。")
            return

        # 出力ディレクトリを取得
        output_dir = self.output_entry.get().strip()
        if not os.path.exists(output_dir):
            messagebox.showwarning("警告", "出力フォルダが存在しません。")
            return

        # ブログ情報を収集（サンプルデータ）
        blog_info = {
            'name': self.blog_entry.get().strip() or 'Unknown Blog',
            'description': 'Tumblr blog archive generated by Tumblr Image Collector',
            'url': f"https://{self.blog_entry.get().strip() or 'example'}.tumblr.com"
        }

        # 投稿データを収集（実際にはダウンロードしたデータを基に）
        posts = self._get_sample_posts_data()

        if not posts:
            messagebox.showinfo("情報", "フォーマットする投稿データが見つかりません。まずダウンロードを実行してください。")
            return

        try:
            # フォーマットオプションを準備
            format_options = {
                'json': {
                    'pretty': self.json_pretty_var.get(),
                    'include_metadata': self.json_metadata_var.get()
                },
                'markdown': {
                    'include_images': self.markdown_images_var.get()
                },
                'html5': {
                    'responsive': self.html5_responsive_var.get()
                }
            }

            # 各フォーマットを実行
            self.status_label.config(text="フォーマット実行中...", foreground=DesignTokens.COLOR_WARNING)
            self.progress_var.set(0)

            results = {}
            for i, format_type in enumerate(selected_formats):
                try:
                    if format_type == 'json':
                        # ブログ全体をJSONで出力
                        result = self.output_formatter._format_blog_json(posts, blog_info, **format_options.get(format_type, {}))

                        # ファイルを保存
                        json_path = os.path.join(output_dir, f"{blog_info['name']}_archive.json")
                        with open(json_path, 'w', encoding='utf-8') as f:
                            f.write(result)
                        results[format_type] = json_path

                    elif format_type == 'html5':
                        # ブログ全体をHTML5で出力
                        result = self.output_formatter._format_blog_html5(posts, blog_info, **format_options.get(format_type, {}))

                        # ファイルを保存
                        html_path = os.path.join(output_dir, f"{blog_info['name']}_archive.html")
                        with open(html_path, 'w', encoding='utf-8') as f:
                            f.write(result)
                        results[format_type] = html_path

                    else:
                        # 個別ファイルで出力
                        result = self.output_formatter._format_blog_individual(posts, blog_info, format_type, **format_options.get(format_type, {}))
                        results[format_type] = result

                    # 進捗更新
                    progress = (i + 1) / len(selected_formats)
                    self.progress_var.set(progress * 100)

                except Exception as e:
                    logger.error(f"フォーマット実行エラー ({format_type}): {e}")
                    results[format_type] = f"Error: {e}"

            self.status_label.config(text="フォーマット完了", foreground=DesignTokens.COLOR_SUCCESS)
            self.progress_var.set(100)

            # 結果を表示
            result_text = "フォーマット結果:\n"
            for format_type, path in results.items():
                if isinstance(path, str) and os.path.exists(path):
                    result_text += f"✓ {format_type.upper()}: {path}\n"
                else:
                    result_text += f"✗ {format_type.upper()}: {path}\n"

            messagebox.showinfo("フォーマット完了", result_text)

        except Exception as e:
            logger.error(f"フォーマット実行エラー: {e}")
            messagebox.showerror("エラー", f"フォーマットの実行に失敗しました: {e}")
        finally:
            self._reset_ui_state()

    def _get_sample_posts_data(self):
        """サンプル投稿データを取得（実際にはダウンロードしたデータを使用）"""
        # 実際の実装では、ダウンロードした投稿データをデータベースやファイルから読み込む
        # ここではサンプルデータを使用

        output_dir = self.output_entry.get().strip()
        if not os.path.exists(output_dir):
            return []

        # 画像ファイルを検索してサンプル投稿データを作成
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
        posts = []

        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.lower().endswith(image_extensions):
                    file_path = os.path.join(root, file)

                    # ファイルのメタデータから投稿データを作成
                    post_data = {
                        'id': hash(file_path) % 1000000,
                        'blog_name': self.blog_entry.get().strip() or 'sample_blog',
                        'type': 'photo',
                        'timestamp': os.path.getctime(file_path),
                        'content': f'<img src="{file_path}" alt="{file}" />',
                        'media': [{
                            'url': file_path,
                            'caption': f'Sample image: {file}'
                        }],
                        'tags': ['sample', 'generated']
                    }

                    posts.append(post_data)

        return posts[:10]  # 最初の10件のみ使用

    def _preview_formats(self):
        """フォーマットプレビューを表示"""
        if not self.output_formatter:
            messagebox.showwarning("警告", "出力フォーマットシステムが初期化されていません。")
            return

        # 選択されたフォーマットを収集
        selected_formats = []
        for format_key, var in self.format_vars.items():
            if var.get():
                selected_formats.append(format_key)

        if not selected_formats:
            messagebox.showwarning("警告", "少なくとも1つの出力フォーマットを選択してください。")
            return

        # サンプル投稿データ
        sample_post = {
            'id': 12345,
            'blog_name': self.blog_entry.get().strip() or 'sample_blog',
            'type': 'photo',
            'timestamp': datetime.now().timestamp(),
            'content': '<p>これはサンプル投稿です。<img src="sample.jpg" alt="Sample image" /></p>',
            'media': [{
                'url': 'https://example.com/sample.jpg',
                'caption': 'Sample image caption'
            }],
            'tags': ['sample', 'preview', 'test']
        }

        # プレビューウィンドウを作成
        preview_window = tk.Toplevel(self.root)
        preview_window.title("フォーマットプレビュー")
        preview_window.geometry("800x600")

        # ノートブックでタブ表示
        tab_control = ttk.Notebook(preview_window)
        tab_control.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 各フォーマットのプレビューを表示
        for format_type in selected_formats:
            try:
                formatted_content = self.output_formatter.format_post(sample_post, format_type)

                # テキストエリアで表示
                frame = ttk.Frame(tab_control)
                tab_control.add(frame, text=format_type.upper())

                text_widget = tk.Text(frame, wrap=tk.WORD, font=("Consolas", 9))
                text_widget.insert(tk.END, formatted_content)
                text_widget.config(state=tk.DISABLED)

                # スクロールバー
                scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text_widget.yview)
                text_widget.configure(yscrollcommand=scrollbar.set)

                text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            except Exception as e:
                logger.error(f"フォーマットプレビューエラー ({format_type}): {e}")

                frame = ttk.Frame(tab_control)
                tab_control.add(frame, text=format_type.upper())

                error_label = ttk.Label(frame, text=f"プレビュー生成エラー: {e}", foreground="red")
                error_label.pack(pady=20)

        # 閉じるボタン
        ttk.Button(preview_window, text="閉じる", command=preview_window.destroy).pack(pady=10)

    def _reset_format_settings(self):
        """フォーマット設定をリセット"""
        # デフォルト値に戻す
        for format_key, var in self.format_vars.items():
            var.set(format_key in ['json', 'html5'])

        self.json_pretty_var.set(True)
        self.json_metadata_var.set(True)
        self.html5_responsive_var.set(True)
        self.markdown_images_var.set(True)

        messagebox.showinfo("リセット", "フォーマット設定をデフォルト値にリセットしました。")

    def _save_format_settings(self):
        """出力フォーマット設定を保存"""
        try:
            config = {}
            if os.path.exists('config.json'):
                with open('config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)

            if 'output_format' not in config:
                config['output_format'] = {}

            # 選択されたフォーマットを保存
            selected_formats = []
            for format_key, var in self.format_vars.items():
                if var.get():
                    selected_formats.append(format_key)

            config['output_format'].update({
                'formats': selected_formats,
                'json_pretty': self.json_pretty_var.get(),
                'json_metadata': self.json_metadata_var.get(),
                'html5_responsive': self.html5_responsive_var.get(),
                'markdown_images': self.markdown_images_var.get()
            })

            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"出力フォーマット設定保存エラー: {e}")

    def _load_format_settings(self):
        """出力フォーマット設定を読み込み"""
        try:
            if os.path.exists('config.json'):
                with open('config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)

                format_config = config.get('output_format', {})

                # フォーマット選択を復元
                saved_formats = format_config.get('formats', ['json', 'html5'])
                for format_key, var in self.format_vars.items():
                    var.set(format_key in saved_formats)

                # 設定を復元
                self.json_pretty_var.set(format_config.get('json_pretty', True))
                self.json_metadata_var.set(format_config.get('json_metadata', True))
                self.html5_responsive_var.set(format_config.get('html5_responsive', True))
                self.markdown_images_var.set(format_config.get('markdown_images', True))

        except Exception as e:
            logger.error(f"出力フォーマット設定読み込みエラー: {e}")

    def run(self):
        """GUIを実行"""
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.mainloop()


def main():
    """GUIアプリケーションのエントリーポイント"""
    root = tk.Tk()
    app = TumblrCollectorGUI(root)
    app.run()


if __name__ == "__main__":
    main()
