"""
Tumblr Image Collector - テストスイート
エンドツーエンドテスト、ユニットテスト、統合テストを含む
"""

import unittest
import pytest
import tempfile
import shutil
import json
import csv
import os
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None
    _CV2_AVAILABLE = False

# テスト対象のモジュールをインポート
from tumblr_image_collector import TumblrImageCollector, DownloadError, DEFAULT_PAGE_LIMIT
from tumblr_image_collector_enhanced import EnhancedTumblrImageCollector
from image_classifier import ImageClassifier
from config import ConfigWizard
from i18n import _, set_locale, get_current_locale


class TestTumblrImageCollector(unittest.TestCase):
    """TumblrImageCollectorのユニットテスト"""

    def setUp(self):
        """テスト前のセットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.json"
        self.output_dir = Path(self.temp_dir) / "output"

        # テスト用の設定
        self.test_config = {
            "consumer_key": "test_key",
            "consumer_secret": "test_secret",
            "token": "test_token",
            "token_secret": "test_token_secret",
            "output_folder_name": "test_output",
            "max_download_workers": 2,
            "enable_deep_model": False,
            "filters": {
                "max_file_size_mb": 10,
                "nsfw_threshold": 0.4
            },
            "network": {
                "download_timeout_seconds": 10,
                "max_retries": 1,
                "backoff_factor": 0.1,
                "max_backoff_seconds": 5
            },
            "logging": {
                "level": "INFO",
                "max_bytes": 1048576,
                "backup_count": 3
            }
        }

        # 設定ファイルを保存
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)

    def tearDown(self):
        """テスト後のクリーンアップ"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('tumblr_image_collector.pytumblr.TumblrRestClient')
    def test_initialization(self, mock_client_class):
        """初期化のテスト"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        collector = TumblrImageCollector(
            config_file=str(self.config_file),
            output_dir_override=str(self.output_dir)
        )

        self.assertIsNotNone(collector)
        self.assertEqual(collector.output_folder, self.output_dir)
        self.assertEqual(collector.max_workers, 2)

    def test_input_validation(self):
        """入力検証のテスト"""
        collector = TumblrImageCollector(config_file=str(self.config_file))

        # 有効なURL
        self.assertTrue(collector._validate_input("https://example.com", "url"))

        # 無効なURL
        self.assertFalse(collector._validate_input("not_a_url", "url"))

        # 危険なファイル名
        self.assertFalse(collector._validate_input("../../etc/passwd", "filename"))

        # 有効なファイル名
        self.assertTrue(collector._validate_input("image.jpg", "filename"))

    def test_filename_sanitization(self):
        """ファイル名サニタイズのテスト"""
        collector = TumblrImageCollector(config_file=str(self.config_file))

        # 危険な文字を含むファイル名
        dangerous_name = 'image_../../../etc/passwd.jpg'
        sanitized = collector._sanitize_filename(dangerous_name)

        self.assertNotIn('/', sanitized)
        self.assertNotIn('\\', sanitized)
        self.assertNotIn(':', sanitized)

    def test_rate_limiting(self):
        """レート制限のテスト"""
        collector = TumblrImageCollector(config_file=str(self.config_file))

        # レート制限の初期化
        rate_limiter = collector._create_rate_limiter()
        self.assertIsInstance(rate_limiter, dict)
        self.assertIn('requests_per_minute', rate_limiter)

        # レート制限チェック
        can_request = collector._check_rate_limit()
        self.assertTrue(can_request)

    def test_image_processing_efficiency(self):
        """画像処理の効率性テスト"""
        collector = TumblrImageCollector(config_file=str(self.config_file))

        # テスト画像の作成
        test_image_path = Path(self.temp_dir) / "test_image.jpg"
        test_image = Image.new('RGB', (100, 100), color='red')
        test_image.save(test_image_path)

        # 効率的な画像処理のテスト
        features = collector._process_image_efficiently(str(test_image_path))

        self.assertIsInstance(features, dict)
        self.assertIn('dimensions', features)
        self.assertIn('aspect_ratio', features)

    def test_memory_cleanup(self):
        """メモリクリーンアップのテスト"""
        collector = TumblrImageCollector(config_file=str(self.config_file))

        # リソースのクリーンアップ
        collector._cleanup_resources()

        # クリーンアップ後の状態を確認
        self.assertIsNone(collector.executor)
        self.assertIsNone(collector.session)

    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        collector = TumblrImageCollector(config_file=str(self.config_file))

        # 無効な画像パスでのエラーハンドリング
        metadata = collector._extract_image_metadata("nonexistent_image.jpg")
        self.assertIsNone(metadata)

    def test_generate_output_filename(self):
        """出力ファイル名生成のテスト"""
        collector = TumblrImageCollector(config_file=str(self.config_file))

        temp_image_path = Path(self.temp_dir) / "temp_image.jpg"
        Image.new('RGB', (200, 100), color='white').save(temp_image_path)

        metadata = {
            'width': 200,
            'height': 100
        }

        filename = collector._generate_output_filename(
            str(temp_image_path),
            metadata,
            image_url="https://example-blog.tumblr.com/post/123456/image.jpg",
            post_data={'blog_name': 'example-blog', 'timestamp': 1737800400}
        )

        self.assertTrue(filename.endswith('.jpg'))
        self.assertIn('example-blog', filename)

    def test_nsfw_threshold_configuration(self):
        """NSFW閾値が設定値を反映することを確認"""
        collector = TumblrImageCollector(config_file=str(self.config_file))
        self.assertAlmostEqual(collector.nsfw_threshold, 0.4)

    @patch('tumblr_image_collector.pytumblr.TumblrRestClient')
    def test_environment_credentials_override(self, mock_client_class):
        """環境変数の資格情報が設定ファイルより優先されることを確認"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        env_values = {
            "TUMBLR_CONSUMER_KEY": "env_consumer_key",
            "TUMBLR_CONSUMER_SECRET": "env_consumer_secret",
            "TUMBLR_OAUTH_TOKEN": "env_token",
            "TUMBLR_OAUTH_TOKEN_SECRET": "env_token_secret"
        }

        with patch.dict(os.environ, env_values, clear=False):
            collector = TumblrImageCollector(config_file=str(self.config_file))

        self.assertEqual(collector.consumer_key, "env_consumer_key")
        self.assertEqual(collector.consumer_secret, "env_consumer_secret")
        self.assertEqual(collector.token, "env_token")
        self.assertEqual(collector.token_secret, "env_token_secret")
        # 元の設定ファイルは変更されないことを確認
        self.assertEqual(collector.config.get("consumer_key"), "test_key")
        self.assertEqual(collector.config.get("consumer_secret"), "test_secret")

    @patch('tumblr_image_collector.pytumblr.TumblrRestClient')
    def test_cli_post_filters_apply_tags_and_dates(self, mock_client_class):
        """CLIフィルタがタグと日時で投稿を絞り込むことを検証"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        collector = TumblrImageCollector(config_file=str(self.config_file))
        collector._cli_tags = ['art']
        collector._cli_start_date = datetime(2024, 1, 1)
        collector._cli_end_date = datetime(2024, 12, 31)

        in_range_post = {
            'tags': ['Art', 'photo'],
            'timestamp': int(datetime(2024, 5, 1).timestamp()),
            'type': 'photo',
            'photos': [{'original_size': {'url': 'https://example.com/a.jpg'}}]
        }

        out_of_range_post = {
            'tags': ['art'],
            'timestamp': int(datetime(2023, 5, 1).timestamp()),
            'type': 'photo',
            'photos': [{'original_size': {'url': 'https://example.com/b.jpg'}}]
        }

        missing_tag_post = {
            'tags': ['travel'],
            'timestamp': int(datetime(2024, 6, 1).timestamp()),
            'type': 'photo',
            'photos': [{'original_size': {'url': 'https://example.com/c.jpg'}}]
        }

        filtered = collector._apply_cli_post_filters([
            in_range_post,
            out_of_range_post,
            missing_tag_post
        ])

        self.assertEqual(len(filtered), 1)
        self.assertIs(filtered[0], in_range_post)

    @patch('tumblr_image_collector.pytumblr.TumblrRestClient')
    def test_collect_liked_posts_respects_filters(self, mock_client_class):
        """Like収集がフィルタを適用してから処理することを検証"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        collector = TumblrImageCollector(config_file=str(self.config_file))
        collector._cli_tags = ['art']
        collector._cli_start_date = datetime(2024, 1, 1)
        collector._cli_end_date = datetime(2024, 12, 31)

        liked_posts_batch = {
            'liked_posts': [
                {
                    'tags': ['art'],
                    'timestamp': int(datetime(2024, 3, 15).timestamp()),
                    'type': 'photo',
                    'photos': [{'original_size': {'url': 'https://example.com/like1.jpg'}}]
                },
                {
                    'tags': ['travel'],
                    'timestamp': int(datetime(2024, 3, 15).timestamp()),
                    'type': 'photo',
                    'photos': [{'original_size': {'url': 'https://example.com/like2.jpg'}}]
                }
            ]
        }

        mock_client.likes.side_effect = [liked_posts_batch, {'liked_posts': []}]

        with patch.object(collector, 'process_posts', return_value=True) as mock_process, \
             patch.object(collector, 'wait_and_resume') as mock_wait:
            collector._collect_liked_posts(batch_size=10)

        mock_process.assert_called_once()
        processed_posts = mock_process.call_args[0][0]
        self.assertEqual(len(processed_posts), 1)
        self.assertEqual(processed_posts[0]['tags'][0], 'art')
        mock_wait.assert_not_called()

    @patch('tumblr_image_collector.pytumblr.TumblrRestClient')
    def test_persist_runtime_state_serializes_cli_filters(self, mock_client_class):
        """ランタイム状態の保存がCLIフィルタとオフセットを記録することを確認"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        collector = TumblrImageCollector(config_file=str(self.config_file))
        collector._cli_tags = ['art', 'photography']
        collector._cli_start_date = datetime(2024, 1, 1)
        collector._cli_end_date = datetime(2024, 6, 30)
        collector._include_likes = True
        collector.downloaded_files.update({'file1.jpg', 'file2.png'})

        captured_payload = {}
        with patch.object(collector, '_save_collection_state') as mock_save:
            collector._persist_runtime_state(blog_name='blog1', offset=120)
            self.assertTrue(mock_save.called)
            captured_payload = mock_save.call_args[0][0]

        self.assertEqual(sorted(captured_payload['downloaded_images']), ['file1.jpg', 'file2.png'])
        self.assertEqual(captured_payload['offsets'], {'blog1': 120})
        cli_filters = captured_payload['cli_filters']
        self.assertEqual(cli_filters['tags'], ['art', 'photography'])
        self.assertEqual(cli_filters['include_likes'], True)
        self.assertEqual(cli_filters['start_date'], '2024-01-01T00:00:00')
        self.assertEqual(cli_filters['end_date'], '2024-06-30T00:00:00')

    @patch('tumblr_image_collector.pytumblr.TumblrRestClient')
    def test_resume_image_collection_restores_cli_filters(self, mock_client_class):
        """resume_image_collection が状態ファイルからCLIフィルタを復元することを確認"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        collector = TumblrImageCollector(config_file=str(self.config_file))

        previous_state = {
            'total_found': 0,
            'downloaded_images': [],
            'skipped_images': [],
            'errors': [],
            'cli_filters': {
                'tags': ['landscape'],
                'start_date': '2024-02-01T00:00:00',
                'end_date': '2024-03-01T00:00:00',
                'include_likes': True
            },
            'collection_params': {
                'tags': ['landscape'],
                'date_range': {
                    'start': '2024-02-01T00:00:00',
                    'end': '2024-03-01T00:00:00'
                }
            }
        }

        new_results = {
            'total_found': 1,
            'downloaded_images': ['new_file.jpg'],
            'skipped_images': [],
            'errors': [],
            'collection_params': {'tags': ['landscape']},
            'cli_filters': collector._serialize_cli_filters(),
            'offsets': {}
        }

        with patch.object(collector, '_save_collection_state') as mock_save, \
             patch.object(collector, 'auto_image_collection', return_value=new_results) as mock_auto:
            merged = collector.resume_image_collection(previous_collection_results=previous_state, additional_params={'extend_date_range': False})

        self.assertTrue(mock_save.called)
        self.assertTrue(mock_auto.called)

        resumed_params = mock_auto.call_args[0][0]
        self.assertEqual(resumed_params['tags'], ['landscape'])
        self.assertTrue(resumed_params['include_likes'])
        self.assertIsInstance(resumed_params['date_range']['start'], datetime)
        self.assertIsInstance(resumed_params['date_range']['end'], datetime)

        self.assertIn('cli_filters', merged)
        self.assertEqual(collector._cli_tags, ['landscape'])
        self.assertEqual(collector._include_likes, True)

    @patch('tumblr_image_collector.pytumblr.TumblrRestClient')
    def test_multi_blog_search_filters_results(self, mock_client_class):
        """multi_blog_search が条件に従って投稿をフィルタリングすることを確認"""
        collector = TumblrImageCollector(config_file=str(self.config_file))

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        now = datetime.now()
        valid_timestamp = int(now.timestamp())
        old_timestamp = int((now - timedelta(days=DEFAULT_DAYS_BACK + 5)).timestamp())

        mock_client.posts.return_value = {
            'posts': [
                {
                    'timestamp': valid_timestamp,
                    'note_count': 10,
                    'likes': 8,
                    'is_nsfw': False,
                    'type': 'photo',
                    'photos': [{'original_size': {'url': 'https://example.com/valid.jpg'}}]
                },
                {
                    'timestamp': valid_timestamp,
                    'note_count': 10,
                    'likes': 3,
                    'is_nsfw': False,
                    'type': 'photo',
                    'photos': [{'original_size': {'url': 'https://example.com/low_likes.jpg'}}]
                },
                {
                    'timestamp': valid_timestamp,
                    'note_count': 1,
                    'likes': 0,
                    'is_nsfw': False,
                    'type': 'photo',
                    'photos': [{'original_size': {'url': 'https://example.com/low_notes.jpg'}}]
                },
                {
                    'timestamp': valid_timestamp,
                    'note_count': 10,
                    'likes': 8,
                    'is_nsfw': True,
                    'type': 'photo',
                    'photos': [{'original_size': {'url': 'https://example.com/nsfw.jpg'}}]
                },
                {
                    'timestamp': old_timestamp,
                    'note_count': 10,
                    'likes': 8,
                    'is_nsfw': False,
                    'type': 'photo',
                    'photos': [{'original_size': {'url': 'https://example.com/old.jpg'}}]
                },
                {
                    'timestamp': valid_timestamp,
                    'note_count': 15,
                    'likes': 12,
                    'is_nsfw': False,
                    'type': 'video',
                    'photos': [{'original_size': {'url': 'https://example.com/video.jpg'}}]
                }
            ]
        }

        search_params = {
            'min_likes': 5,
            'min_notes': 5,
            'date_range': {
                'start': now - timedelta(days=7),
                'end': now + timedelta(minutes=1)
            },
            'max_pages': 1,
            'limit': 5,
            'content_type': ['photo']
        }

        results = collector.multi_blog_search(blogs=['blog1'], search_params=search_params)

        self.assertEqual(results, ['https://example.com/valid.jpg'])
        mock_client.posts.assert_called_with(
            blogname='blog1',
            type='photo',
            limit=5,
            offset=0
        )

    @patch('tumblr_image_collector.pytumblr.TumblrRestClient')
    def test_multi_blog_search_uses_tag_endpoint(self, mock_client_class):
        """タグ指定時に tagged API を使用し、タイムスタンプベースでページングする"""
        collector = TumblrImageCollector(config_file=str(self.config_file))

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        now = datetime.now()
        timestamps = [int(now.timestamp()), int((now - timedelta(minutes=5)).timestamp())]

        mock_client.tagged.side_effect = [
            [
                {
                    'timestamp': timestamps[0],
                    'note_count': 5,
                    'likes': 5,
                    'is_nsfw': False,
                    'type': 'photo',
                    'photos': [{'original_size': {'url': 'https://example.com/tagged1.jpg'}}]
                },
                {
                    'timestamp': timestamps[1],
                    'note_count': 5,
                    'likes': 5,
                    'is_nsfw': False,
                    'type': 'photo',
                    'photos': [{'original_size': {'url': 'https://example.com/tagged2.jpg'}}]
                }
            ],
            []
        ]

        search_params = {
            'date_range': {
                'start': now - timedelta(days=1),
                'end': now
            },
            'max_pages': 2,
            'limit': 3,
            'content_type': ['photo']
        }

        results = collector.multi_blog_search(tags=['art', 'design'], search_params=search_params)

        self.assertEqual(
            results,
            ['https://example.com/tagged1.jpg', 'https://example.com/tagged2.jpg']
        )

        mock_client.tagged.assert_any_call(
            tag='art',
            before=int(search_params['date_range']['end'].timestamp()),
            limit=3
        )

    @patch('tumblr_image_collector.pytumblr.TumblrRestClient')
    def test_multi_blog_search_handles_multiple_tags(self, mock_client_class):
        """複数タグ指定で重複URLが排除されることを確認"""
        collector = TumblrImageCollector(config_file=str(self.config_file))

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        now = datetime.now()
        ts = int(now.timestamp())

        tag_posts = [
            [
                {
                    'timestamp': ts,
                    'note_count': 10,
                    'likes': 6,
                    'is_nsfw': False,
                    'photos': [{'original_size': {'url': 'https://example.com/shared.jpg'}}]
                }
            ],
            [
                {
                    'timestamp': ts - 60,
                    'note_count': 10,
                    'likes': 6,
                    'is_nsfw': False,
                    'photos': [{'original_size': {'url': 'https://example.com/shared.jpg'}}]
                },
                {
                    'timestamp': ts - 120,
                    'note_count': 10,
                    'likes': 6,
                    'is_nsfw': False,
                    'photos': [{'original_size': {'url': 'https://example.com/tag2.jpg'}}]
                }
            ]
        ]

        mock_client.tagged.side_effect = tag_posts + [[], []]

        search_params = {
            'date_range': {
                'start': now - timedelta(hours=1),
                'end': now
            },
            'max_pages': 2
        }

        results = collector.multi_blog_search(tags=['tag1', 'tag2'], search_params=search_params)

        self.assertEqual(
            results,
            ['https://example.com/shared.jpg', 'https://example.com/tag2.jpg']
        )

    @patch.object(TumblrImageCollector, 'multi_blog_search', return_value=[])
    def test_auto_image_collection_returns_structure(self, mock_multi_blog):
        """自動画像収集の戻り値構造を検証"""
        collector = TumblrImageCollector(config_file=str(self.config_file))

        params = {
            'download_options': {
                'output_directory': str(self.output_dir / 'auto'),
                'overwrite': True
            }
        }

        results = collector.auto_image_collection(params)

        self.assertIsInstance(results, dict)
        self.assertIn('total_found', results)
        self.assertIn('downloaded_images', results)
        self.assertIn('skipped_images', results)
        self.assertIn('errors', results)

    @patch.object(TumblrImageCollector, '_save_collection_state')
    @patch.object(TumblrImageCollector, 'auto_image_collection')
    def test_resume_image_collection_merges_results(self, mock_auto_collection, mock_save_state):
        """収集再開で既存結果にマージされることを検証"""
        collector = TumblrImageCollector(config_file=str(self.config_file))

        previous_results = {
            'total_found': 10,
            'downloaded_images': ['a.jpg'],
            'skipped_images': ['b.jpg'],
            'errors': [],
            'collection_params': {'blogs': ['blog1']}
        }

        new_results = {
            'total_found': 5,
            'downloaded_images': ['c.jpg'],
            'skipped_images': ['d.jpg'],
            'errors': [{'error': 'network'}],
            'collection_params': {'blogs': ['blog1']}
        }

        mock_auto_collection.return_value = new_results

        merged = collector.resume_image_collection(previous_results)

        self.assertEqual(merged['total_found'], 15)
        self.assertIn('a.jpg', merged['downloaded_images'])
        self.assertIn('c.jpg', merged['downloaded_images'])
        self.assertIn({'error': 'network'}, merged['errors'])
        mock_save_state.assert_called_once()

    @patch('tumblr_image_collector.requests.get')
    def test_analyze_image_details_extracts_metadata(self, mock_get):
        """画像詳細分析が基本メタデータを抽出することを確認"""
        collector = TumblrImageCollector(config_file=str(self.config_file))

        buffer = BytesIO()
        Image.new('RGB', (64, 32), color='white').save(buffer, format='PNG')
        buffer.seek(0)

        mock_get.return_value = MagicMock(content=buffer.read())

        details = collector._analyze_image_details('https://example.com/test.png')

        self.assertIsNotNone(details)
        self.assertEqual(details['width'], 64)
        self.assertEqual(details['height'], 32)
        self.assertIn('local_path', details)

    def test_apply_advanced_filters_passes_when_within_constraints(self):
        """高度フィルタが条件を満たす場合にTrueを返す"""
        collector = TumblrImageCollector(config_file=str(self.config_file))

        image_info = {
            'width': 800,
            'height': 600,
            'aspect_ratio': 800 / 600,
            'local_path': str(self.output_dir / 'dummy.jpg')
        }

        # ダミー画像生成
        dummy_image = Image.new('RGB', (800, 600), color='gray')
        dummy_image.save(image_info['local_path'])

        params = {
            'min_resolution': (500, 500),
            'advanced_filters': {
                'aspect_ratio_range': (1.0, 1.5),
                'entropy_threshold': None,
                'color_palette': None
            }
        }

        self.assertTrue(collector._apply_advanced_filters(image_info, params))

    def test_save_and_load_collection_state_roundtrip(self):
        """収集状態の保存と読み込みの往復を確認"""
        collector = TumblrImageCollector(config_file=str(self.config_file))

        test_results = {
            'total_found': 3,
            'downloaded_images': ['x.jpg'],
            'skipped_images': ['y.jpg'],
            'errors': []
        }

        with pytest.monkeypatch.context() as m:
            m.setattr(os, 'getcwd', lambda: str(self.temp_dir))
            collector._save_collection_state(test_results, 'state.json')
            loaded = collector._load_last_collection_state('state.json')

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded['total_found'], 3)
        self.assertIn('saved_timestamp', loaded)

    def test_merge_collection_results_combines_entries(self):
        """収集結果のマージ処理を検証"""
        collector = TumblrImageCollector(config_file=str(self.config_file))

        previous = {
            'total_found': 2,
            'downloaded_images': ['a.jpg'],
            'skipped_images': ['b.jpg'],
            'errors': [],
            'collection_params': {'blogs': ['blog1']}
        }

        new = {
            'total_found': 1,
            'downloaded_images': ['c.jpg'],
            'skipped_images': ['b.jpg'],
            'errors': [{'error': 'timeout'}],
            'collection_params': {'blogs': ['blog1']}
        }

        merged = collector._merge_collection_results(previous, new)

        self.assertEqual(merged['total_found'], 3)
        self.assertEqual(sorted(merged['downloaded_images']), ['a.jpg', 'c.jpg'])
        self.assertEqual(sorted(merged['skipped_images']), ['b.jpg'])
        self.assertIn({'error': 'timeout'}, merged['errors'])

    @pytest.mark.skipif(not _CV2_AVAILABLE, reason="OpenCVが利用できない環境ではスキップ")
    def test_nsfw_scoring(self):
        """NSFWヒューリスティックが肌色画像と非肌色画像を識別できることを検証"""
        collector = TumblrImageCollector(config_file=str(self.config_file))

        skin_bgr = np.full((200, 200, 3), (45, 160, 190), dtype=np.uint8)
        low_skin_bgr = np.full((200, 200, 3), (30, 60, 200), dtype=np.uint8)

        skin_score = collector._estimate_nsfw_content(skin_bgr)
        low_skin_score = collector._estimate_nsfw_content(low_skin_bgr)

        self.assertGreater(skin_score, low_skin_score)
        self.assertGreaterEqual(skin_score, 0.0)
        self.assertLessEqual(skin_score, 1.0)

    @pytest.mark.skipif(not (_CV2_AVAILABLE and _NUMPY_AVAILABLE), reason="OpenCV/Numpyが必要")
    def test_metadata_contains_nsfw_score(self):
        """メタデータにnsfw_scoreが含まれることを確認"""
        collector = TumblrImageCollector(config_file=str(self.config_file))

        temp_image_path = Path(self.temp_dir) / "temp_metadata.jpg"
        Image.new('RGB', (128, 128), color=(240, 200, 180)).save(temp_image_path)

        metadata = collector._extract_image_metadata(str(temp_image_path))
        self.assertIsNotNone(metadata)
        metrics = metadata['ai_classification'].get('metrics', {})
        self.assertIn('nsfw_score', metrics)
        self.assertGreaterEqual(metrics['nsfw_score'], 0.0)
        self.assertLessEqual(metrics['nsfw_score'], 1.0)

    def test_record_classification_stats_metrics_summary(self):
        """metrics_summary に数値メトリクスが集計されることを検証"""
        collector = TumblrImageCollector(config_file=str(self.config_file))

        classification_result = {
            'is_valid': True,
            'is_high_resolution': False,
            'is_potentially_nsfw': True,
            'top_predictions': [
                {'label': 'portrait', 'confidence': 0.82}
            ],
            'metrics': {
                'nsfw_score': 0.67,
                'quality_score': 0.91,
                'notes': 'non-numeric should be ignored'
            }
        }

        collector._record_classification_stats(classification_result)

        stats = collector._download_stats['ai_classification_stats']
        metrics_summary = stats.get('metrics_summary', {})

        self.assertIn('nsfw_score', metrics_summary)
        nsfw_entry = metrics_summary['nsfw_score']
        self.assertEqual(nsfw_entry['count'], 1)
        self.assertAlmostEqual(nsfw_entry['sum'], 0.67)
        self.assertAlmostEqual(nsfw_entry['min'], 0.67)
        self.assertAlmostEqual(nsfw_entry['max'], 0.67)

        self.assertIn('quality_score', metrics_summary)
        quality_entry = metrics_summary['quality_score']
        self.assertEqual(quality_entry['count'], 1)
        self.assertAlmostEqual(quality_entry['sum'], 0.91)
        self.assertAlmostEqual(quality_entry['min'], 0.91)
        self.assertAlmostEqual(quality_entry['max'], 0.91)

        self.assertNotIn('notes', metrics_summary)

    def test_save_statistics_exports_json_and_csv(self):
        """統計保存がJSON/CSV双方にメトリクス情報を出力することを確認"""
        collector = TumblrImageCollector(config_file=str(self.config_file))

        collector.script_dir = Path(self.temp_dir)
        collector._download_stats = {
            'total_attempts': 3,
            'successful_downloads': 2,
            'failed_downloads': 1,
            'skipped_duplicates': 0,
            'cache_hits': 0,
            'ai_classification_stats': {
                'valid_images': 2,
                'invalid_images': 1,
                'high_resolution_images': 1,
                'low_resolution_images': 1,
                'potentially_nsfw_images': 1,
                'image_type_distribution': {},
                'metrics_summary': {
                    'nsfw_score': {
                        'count': 2,
                        'sum': 0.9,
                        'min': 0.4,
                        'max': 0.5
                    }
                }
            }
        }

        collector._save_statistics()

        json_path = collector.script_dir / 'download_statistics.json'
        csv_path = collector.script_dir / 'download_statistics.csv'
        self.assertTrue(json_path.exists())
        self.assertTrue(csv_path.exists())

        with open(json_path, 'r', encoding='utf-8') as fh:
            saved_json = json.load(fh)
        self.assertEqual(saved_json['total_attempts'], 3)
        self.assertEqual(
            saved_json['ai_classification_stats']['metrics_summary']['nsfw_score']['count'],
            2
        )

        with open(csv_path, 'r', encoding='utf-8') as fh:
            rows = [row for row in csv.reader(fh) if row]

        self.assertIn(['metric', 'value'], rows)
        self.assertIn(['total_attempts', '3'], rows)
        self.assertIn(['Metric', 'count', 'mean', 'min', 'max'], rows)
        self.assertIn(['nsfw_score', '2', '0.4500', '0.4000', '0.5000'], rows)

    def test_validate_tumblr_credentials(self):
        """認証情報の形式検証をテスト"""
        wizard = ConfigWizard()

        valid_key = "abcdEFGHijklMNOPqrst"
        valid_secret = "qrstUVWXyzabCDEFghij"
        self.assertTrue(wizard._validate_tumblr_credentials(valid_key, valid_secret))

        too_short = "short"
        self.assertFalse(wizard._validate_tumblr_credentials(too_short, valid_secret))

        invalid_chars = "inv@lid_key_1234567890"
        self.assertFalse(wizard._validate_tumblr_credentials(invalid_chars, valid_secret))

        identical = "sameVALUEfortesting123"
        self.assertFalse(wizard._validate_tumblr_credentials(identical, identical))


class TestImageClassifier(unittest.TestCase):
    """ImageClassifierのユニットテスト"""

    def setUp(self):
        """テスト前のセットアップ"""
        self.classifier = ImageClassifier(enable_deep_model=False)

    def test_initialization(self):
        """初期化のテスト"""
        self.assertIsNotNone(self.classifier)
        self.assertFalse(self.classifier.enable_deep_model)

    def test_image_analysis(self):
        """画像分析のテスト"""
        # テスト画像の作成
        test_image = Image.new('RGB', (100, 100), color='blue')

        # 分析結果のテスト
        result = self.classifier.analyze_image(test_image)

        self.assertIsInstance(result, dict)
        self.assertIn('is_valid', result)

    def test_valid_image_check(self):
        """有効画像チェックのテスト"""
        # テスト画像の作成
        test_image = Image.new('RGB', (100, 100), color='green')

        # 有効性チェック
        is_valid = self.classifier.is_valid_image(test_image)

        # 結果の検証（実際の画像サイズによる）
        self.assertIsInstance(is_valid, bool)


class TestConfigWizard(unittest.TestCase):
    """ConfigWizardのユニットテスト"""

    def setUp(self):
        """テスト前のセットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.wizard = ConfigWizard()

    def tearDown(self):
        """テスト後のクリーンアップ"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_loading(self):
        """設定読み込みのテスト"""
        config_file = Path(self.temp_dir) / "test_config.json"
        test_config = {"test_key": "test_value"}

        with open(config_file, 'w') as f:
            json.dump(test_config, f)

        loaded_config = self.wizard.load_config(str(config_file))

        self.assertEqual(loaded_config, test_config)

    def test_config_saving(self):
        """設定保存のテスト"""
        config_file = Path(self.temp_dir) / "test_config.json"
        test_config = {"test_key": "test_value"}

        self.wizard.save_config(test_config, str(config_file))

        # ファイルが作成されたことを確認
        self.assertTrue(config_file.exists())

        # 内容を確認
        with open(config_file, 'r') as f:
            saved_config = json.load(f)

        self.assertEqual(saved_config, test_config)

    def test_prompt_validation(self):
        """プロンプト検証のテスト"""
        # テスト用の入力をシミュレート
        with patch('builtins.input', return_value='test_value'):
            result = self.wizard.prompt_string("Enter value", "default")

        self.assertEqual(result, 'test_value')

    def test_filter_defaults_include_nsfw_threshold(self):
        """フィルタ設定にNSFW閾値が含まれることを確認"""
        self.wizard.config = {'filters': {'max_file_size_mb': 8}}
        self.wizard._apply_defaults()
        filters = self.wizard.config.get('filters', {})

        self.assertEqual(filters.get('max_file_size_mb'), 8)
        self.assertIn('nsfw_threshold', filters)


class TestInternationalization(unittest.TestCase):
    """国際化システムのテスト"""

    def setUp(self):
        """テスト前のセットアップ"""
        self.temp_dir = tempfile.mkdtemp()

        # テスト用のロケールディレクトリ
        self.locale_dir = Path(self.temp_dir) / "locales"
        self.locale_dir.mkdir()

        # テスト用の英語ロケールファイル
        en_locale = {
            "test_message": "This is a test message",
            "greeting": "Hello, {name}!"
        }

        with open(self.locale_dir / "en.json", 'w', encoding='utf-8') as f:
            json.dump(en_locale, f, ensure_ascii=False, indent=2)

        # テスト用の日本語ロケールファイル
        ja_locale = {
            "test_message": "これはテストメッセージです",
            "greeting": "こんにちは、{name}！"
        }

        with open(self.locale_dir / "ja.json", 'w', encoding='utf-8') as f:
            json.dump(ja_locale, f, ensure_ascii=False, indent=2)

    def tearDown(self):
        """テスト後のクリーンアップ"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('i18n.Internationalization.__init__', return_value=None)
    @patch('i18n.Internationalization.load_locale')
    def test_locale_setting(self, mock_load_locale, mock_init):
        """ロケール設定のテスト"""
        mock_load_locale.return_value = True

        # 英語に設定
        success = set_locale('en')
        self.assertTrue(success)

        # 日本語に設定
        success = set_locale('ja')
        self.assertTrue(success)

    def test_translation_function(self):
        """翻訳関数のテスト"""
        # デフォルトの英語ロケールでテスト
        message = _("test_message", "Default message")
        self.assertEqual(message, "Default message")  # デフォルト値が返されるはず

    def test_current_locale(self):
        """現在のロケールのテスト"""
        locale = get_current_locale()
        self.assertIsInstance(locale, str)


class TestIntegration(unittest.TestCase):
    """統合テスト"""

    def setUp(self):
        """テスト前のセットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "config.json"
        self.output_dir = Path(self.temp_dir) / "output"

        # 統合テスト用の設定
        self.test_config = {
            "consumer_key": "test_key",
            "consumer_secret": "test_secret",
            "token": "test_token",
            "token_secret": "test_token_secret",
            "output_folder_name": "test_output",
            "max_download_workers": 1,
            "enable_deep_model": False,
            "network": {
                "download_timeout_seconds": 5,
                "max_retries": 1,
                "backoff_factor": 0.1,
                "max_backoff_seconds": 1
            }
        }

        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)

    def tearDown(self):
        """テスト後のクリーンアップ"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('tumblr_image_collector.pytumblr.TumblrRestClient')
    def test_end_to_end_workflow(self, mock_client_class):
        """エンドツーエンドのワークフローテスト"""
        # モッククライアントの設定
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # モック投稿データの設定
        mock_posts = {
            'posts': [
                {
                    'type': 'photo',
                    'photos': [
                        {
                            'original_size': {
                                'url': 'https://example.com/test_image.jpg'
                            }
                        }
                    ],
                    'tags': ['test', 'image']
                }
            ]
        }
        mock_client.posts.return_value = mock_posts

        # コレクターの初期化
        collector = TumblrImageCollector(
            config_file=str(self.config_file),
            output_dir_override=str(self.output_dir)
        )

        # ブログ投稿の取得テスト
        posts = collector.get_blog_posts("test_blog", limit=1, offset=0)
        self.assertIsNotNone(posts)
        self.assertEqual(len(posts), 1)

        # レート制限のテスト
        can_request = collector._check_rate_limit()
        self.assertTrue(can_request)

    def test_configuration_workflow(self):
        """設定ワークフローのテスト"""
        wizard = ConfigWizard()

        # 設定の保存と読み込み
        test_config = {"test_setting": "test_value"}
        config_file = Path(self.temp_dir) / "workflow_config.json"

        wizard.save_config(test_config, str(config_file))
        loaded_config = wizard.load_config(str(config_file))

        self.assertEqual(test_config, loaded_config)


class TestPerformance(unittest.TestCase):
    """パフォーマンステスト"""

    def setUp(self):
        """テスト前のセットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.collector = TumblrImageCollector(
            config_file=Path(self.temp_dir) / "config.json"
        )

    def tearDown(self):
        """テスト後のクリーンアップ"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_memory_efficiency(self):
        """メモリ効率のテスト"""
        import psutil
        import os

        # 初期メモリ使用量
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 画像処理の実行
        test_image_path = Path(self.temp_dir) / "large_test_image.jpg"
        large_image = Image.new('RGB', (2000, 2000), color='purple')
        large_image.save(test_image_path)

        # 効率的な処理
        features = self.collector._process_image_efficiently(str(test_image_path))

        # メモリ使用量の確認
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # メモリ増加が許容範囲内であることを確認
        self.assertLess(memory_increase, 100)  # 100MB未満の増加

    def test_processing_speed(self):
        """処理速度のテスト"""
        import time

        # 複数のテスト画像を作成
        test_images = []
        for i in range(5):
            image_path = Path(self.temp_dir) / f"test_image_{i}.jpg"
            image = Image.new('RGB', (500, 500), color=f'rgb({i*50}, {i*50}, {i*50})')
            image.save(image_path)
            test_images.append(str(image_path))

        # 処理時間の測定
        start_time = time.time()

        for image_path in test_images:
            features = self.collector._process_image_efficiently(image_path)

        end_time = time.time()
        processing_time = end_time - start_time

        # 処理時間が許容範囲内であることを確認
        self.assertLess(processing_time, 10)  # 10秒未満


class TestSecurity(unittest.TestCase):
    """セキュリティテスト"""

    def setUp(self):
        """テスト前のセットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.collector = TumblrImageCollector(
            config_file=Path(self.temp_dir) / "config.json"
        )

    def tearDown(self):
        """テスト後のクリーンアップ"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_path_traversal_prevention(self):
        """パストラバーサル攻撃防止のテスト"""
        # 危険なパス
        dangerous_path = "../../etc/passwd"

        is_valid = self.collector._validate_input(dangerous_path, "path")
        self.assertFalse(is_valid)

        # 危険なファイル名
        dangerous_filename = "image_../../../etc/passwd.jpg"
        sanitized = self.collector._sanitize_filename(dangerous_filename)

        self.assertNotIn('/', sanitized)
        self.assertNotIn('\\', sanitized)

    def test_url_validation(self):
        """URL検証のテスト"""
        # 有効なURL
        valid_urls = [
            "https://example.com/image.jpg",
            "http://test.com/photo.png",
            "https://subdomain.example.org/image.gif"
        ]

        for url in valid_urls:
            self.assertTrue(self.collector._validate_input(url, "url"))

        # 無効なURL
        invalid_urls = [
            "not_a_url",
            "ftp://example.com/file.txt",
            "https://",
            "javascript:alert('xss')"
        ]

        for url in invalid_urls:
            self.assertFalse(self.collector._validate_input(url, "url"))

    def test_input_length_limits(self):
        """入力長制限のテスト"""
        # 長すぎる入力
        long_input = "a" * 1001  # 1000文字を超える

        is_valid = self.collector._validate_input(long_input, "text", max_length=1000)
        self.assertFalse(is_valid)

        # 適切な長さの入力
        normal_input = "a" * 100

        is_valid = self.collector._validate_input(normal_input, "text", max_length=1000)
        self.assertTrue(is_valid)


# pytest用のテスト
def test_basic_import():
    """基本的なインポートのテスト"""
    try:
        from tumblr_image_collector import TumblrImageCollector
        from image_classifier import ImageClassifier
        from config import ConfigWizard
        from i18n import _, set_locale
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_image_processing():
    """画像処理のテスト"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # テスト画像の作成
        test_image_path = Path(temp_dir) / "test.jpg"
        test_image = Image.new('RGB', (100, 100), color='red')
        test_image.save(test_image_path)

        # 画像処理のテスト
        collector = TumblrImageCollector(config_file=Path(temp_dir) / "config.json")

        features = collector._process_image_efficiently(str(test_image_path))

        assert isinstance(features, dict)
        assert 'dimensions' in features
        assert 'aspect_ratio' in features


if __name__ == '__main__':
    # unittestの実行
    unittest.main()
