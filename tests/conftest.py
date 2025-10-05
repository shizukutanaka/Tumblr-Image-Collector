"""
pytest設定ファイル
"""

import pytest
import sys
import os
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# テスト設定
def pytest_configure(config):
    """pytestの設定"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "security: marks tests as security tests")

# テスト用のフィクスチャ
@pytest.fixture
def temp_directory():
    """一時ディレクトリのフィクスチャ"""
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def sample_image(temp_directory):
    """サンプル画像のフィクスチャ"""
    from PIL import Image

    image_path = temp_directory / "sample.jpg"
    image = Image.new('RGB', (800, 600), color='blue')
    image.save(image_path)

    return image_path

@pytest.fixture
def test_config(temp_directory):
    """テスト設定のフィクスチャ"""
    import json

    config = {
        "consumer_key": "test_key",
        "consumer_secret": "test_secret",
        "token": "test_token",
        "token_secret": "test_token_secret",
        "output_folder_name": "test_output",
        "max_download_workers": 2,
        "enable_deep_model": False,
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

    config_file = temp_directory / "test_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f)

    return config_file

@pytest.fixture
def mock_tumblr_client():
    """モックTumblrクライアントのフィクスチャ"""
    client = Mock()

    # モック投稿データ
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
    client.posts.return_value = mock_posts

    return client

# カバレッジ設定
@pytest.fixture(scope="session", autouse=True)
def configure_coverage():
    """カバレッジの設定"""
    try:
        import pytest_cov
        # カバレッジレポートの設定
        os.environ['COVERAGE_FILE'] = '.coverage'
    except ImportError:
        pass  # pytest-covがインストールされていない場合は何もしない

# ログ設定
@pytest.fixture(autouse=True)
def configure_logging():
    """テスト用のログ設定"""
    import logging

    # ログレベルをWARNING以上に設定してテスト出力をクリーンに保つ
    logging.basicConfig(level=logging.WARNING)

    yield

    # テスト後のクリーンアップ
    logging.shutdown()

# パフォーマンステスト用の設定
def pytest_collection_modifyitems(config, items):
    """テストアイテムの修正"""
    for item in items:
        # 遅いテストにマークを付ける
        if "performance" in item.nodeid or "integration" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # セキュリティテストにマークを付ける
        if "security" in item.nodeid:
            item.add_marker(pytest.mark.security)

# テスト失敗時のスクリーンショット（オプション）
try:
    import pytest_html
    # HTMLレポートの設定
    def pytest_html_report_title(report):
        report.title = "Tumblr Image Collector - Test Report"

    def pytest_html_results_summary_prefix(prefix, summary, postfix):
        prefix.extend(["<p>商用グレードのTumblr画像収集ツールのテスト結果</p>"])

except ImportError:
    pass  # pytest-htmlがインストールされていない場合は何もしない
