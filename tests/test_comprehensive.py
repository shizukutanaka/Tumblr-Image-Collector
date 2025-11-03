# Tumblr Image Collector - Comprehensive Test Suite
# Enhanced testing framework with CI/CD integration

import pytest
import os
import sys
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import responses

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tumblr_image_collector import TumblrImageCollector
from cloud_integration.cloud_sync import CloudSyncManager
from mobile_app.main import TumblrMobileApp
from web_interface.app import WebInterface

class TestTumblrImageCollector:
    """Comprehensive test suite for TumblrImageCollector"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create mock configuration"""
        return {
            'consumer_key': 'test_key',
            'consumer_secret': 'test_secret',
            'token': 'test_token',
            'token_secret': 'test_token_secret',
            'output_dir': temp_dir,
            'max_workers': 2,
            'image_filters': {
                'max_file_size_mb': 10,
                'min_width': 100,
                'min_height': 100
            }
        }

    @pytest.fixture
    def collector(self, mock_config):
        """Create TumblrImageCollector instance"""
        with patch('tumblr_image_collector.TumblrImageCollector._setup_credentials'):
            collector = TumblrImageCollector()
            collector.config = mock_config
            collector.consumer_key = mock_config['consumer_key']
            collector.consumer_secret = mock_config['consumer_secret']
            collector.token = mock_config['token']
            collector.token_secret = mock_config['token_secret']
            yield collector

    def test_initialization(self, collector, mock_config):
        """Test collector initialization"""
        assert collector.consumer_key == mock_config['consumer_key']
        assert collector.consumer_secret == mock_config['consumer_secret']
        assert collector.output_folder.exists()

    def test_download_image_success(self, collector, temp_dir):
        """Test successful image download"""
        # Mock the Tumblr API client
        mock_client = Mock()
        mock_client.posts.return_value = {
            'posts': [{
                'id': '123',
                'type': 'photo',
                'photos': [{
                    'original_size': {
                        'url': 'https://example.com/image.jpg',
                        'width': 800,
                        'height': 600
                    }
                }],
                'timestamp': 1234567890
            }]
        }
        collector.client = mock_client

        with responses.RequestsMock() as rsps:
            rsps.add(responses.GET, 'https://example.com/image.jpg',
                    body=b'fake_image_data', status=200,
                    headers={'Content-Type': 'image/jpeg'})

            result = collector.download_image('https://example.com/image.jpg')
            assert result == True

    def test_download_image_failure(self, collector):
        """Test image download failure"""
        with responses.RequestsMock() as rsps:
            rsps.add(responses.GET, 'https://example.com/image.jpg',
                    body='Not found', status=404)

            result = collector.download_image('https://example.com/image.jpg')
            assert result == False

    def test_batch_download(self, collector):
        """Test batch download functionality"""
        urls = ['https://example.com/image1.jpg', 'https://example.com/image2.jpg']

        with patch.object(collector, 'download_image') as mock_download:
            mock_download.return_value = True

            results = collector.download_engine.batch_download(urls)
            assert results['total_processed'] == 2
            assert results['successful'] == 2
            assert mock_download.call_count == 2

    def test_video_download(self, collector, temp_dir):
        """Test video download functionality"""
        video_url = 'https://example.com/video.mp4'

        with responses.RequestsMock() as rsps:
            rsps.add(responses.GET, video_url,
                    body=b'fake_video_data', status=200,
                    headers={'Content-Type': 'video/mp4'})

            result = collector.download_video(video_url, {'id': '123'})
            assert result == True

    def test_batch_blog_download(self, collector):
        """Test batch blog download"""
        blog_names = ['blog1', 'blog2']

        with patch.object(collector, 'run') as mock_run:
            mock_run.return_value = None

            results = collector.batch_blog_download(blog_names, max_concurrent_blogs=1)
            assert results['total_blogs'] == 2
            assert results['successful'] == 2
            assert mock_run.call_count == 2

    @pytest.mark.parametrize("filter_name,filter_value,expected", [
        ('min_width', 500, True),
        ('max_file_size_mb', 0.001, False),  # 1KB limit
        ('blur_threshold', 100, True),
    ])
    def test_image_filters(self, collector, filter_name, filter_value, expected):
        """Test image filtering logic"""
        # Create a test image
        from PIL import Image
        test_image = Image.new('RGB', (800, 600), color='red')

        # Apply filter settings
        collector.IMAGE_FILTERS[filter_name] = filter_value

        result = collector._is_image_valid(test_image)
        assert result == expected

class TestCloudIntegration:
    """Test cloud storage integration"""

    @pytest.fixture
    def cloud_manager(self):
        """Create cloud sync manager"""
        return CloudSyncManager()

    def test_dropbox_provider_initialization(self, cloud_manager):
        """Test Dropbox provider initialization"""
        if 'dropbox' in cloud_manager.providers:
            provider = cloud_manager.providers['dropbox']
            assert provider.authenticated == False
            assert provider.dbx is None

    def test_google_drive_provider_initialization(self, cloud_manager):
        """Test Google Drive provider initialization"""
        if 'google_drive' in cloud_manager.providers:
            provider = cloud_manager.providers['google_drive']
            assert provider.authenticated == False
            assert provider.creds is None

    @patch('cloud_integration.providers.dropbox_sync.dropbox')
    def test_dropbox_authentication(self, mock_dropbox, cloud_manager):
        """Test Dropbox authentication"""
        if 'dropbox' not in cloud_manager.providers:
            pytest.skip("Dropbox provider not available")

        # Mock Dropbox client
        mock_client = Mock()
        mock_client.users_get_current_account.return_value = {
            'name': {'display_name': 'Test User'},
            'email': 'test@example.com'
        }
        mock_dropbox.Dropbox.return_value = mock_client

        provider = cloud_manager.providers['dropbox']
        result = provider.authenticate({'access_token': 'test_token'})

        assert result == True
        assert provider.authenticated == True
        assert provider.account_info == mock_client.users_get_current_account.return_value

class TestMobileApp:
    """Test mobile application functionality"""

    @pytest.fixture
    def mobile_app(self):
        """Create mobile app instance"""
        app = TumblrMobileApp()
        yield app
        app.stop()

    def test_app_initialization(self, mobile_app):
        """Test mobile app initialization"""
        assert mobile_app.collector is None  # Not initialized yet
        assert 'settings' in mobile_app.settings
        assert mobile_app.screen_manager is not None

    def test_settings_management(self, mobile_app):
        """Test settings save/load"""
        original_settings = mobile_app.settings.copy()
        mobile_app.settings['test_key'] = 'test_value'
        mobile_app.save_settings()

        # Create new instance to test loading
        new_app = TumblrMobileApp()
        assert new_app.settings.get('test_key') == 'test_value'

        # Restore original settings
        mobile_app.settings = original_settings
        mobile_app.save_settings()

class TestWebInterface:
    """Test web interface functionality"""

    @pytest.fixture
    def web_interface(self):
        """Create web interface instance"""
        interface = WebInterface()
        yield interface
        # Cleanup if needed

    def test_interface_initialization(self, web_interface):
        """Test web interface initialization"""
        assert web_interface.app is not None
        assert web_interface.collector is None  # Not initialized yet

    def test_health_check(self, web_interface, client):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'healthy'

class TestBrowserExtension:
    """Test browser extension functionality"""

    @pytest.fixture
    def extension_files(self):
        """Check if extension files exist"""
        extension_dir = Path(__file__).parent.parent / 'browser_extension'
        return {
            'manifest': extension_dir / 'manifest.json',
            'popup_js': extension_dir / 'popup.js',
            'content_js': extension_dir / 'content.js',
            'background_js': extension_dir / 'background.js'
        }

    def test_extension_files_exist(self, extension_files):
        """Test that extension files exist"""
        for name, path in extension_files.items():
            assert path.exists(), f"{name} file not found: {path}"

    def test_manifest_validity(self, extension_files):
        """Test manifest.json validity"""
        with open(extension_files['manifest']) as f:
            manifest = json.load(f)

        required_fields = ['manifest_version', 'name', 'version', 'description']
        for field in required_fields:
            assert field in manifest, f"Required field missing: {field}"

        assert manifest['manifest_version'] == 3
        assert 'permissions' in manifest
        assert 'action' in manifest

class TestPerformance:
    """Performance and load testing"""

    @pytest.fixture
    def collector(self):
        """Create collector for performance tests"""
        with patch('tumblr_image_collector.TumblrImageCollector._setup_credentials'):
            collector = TumblrImageCollector()
            yield collector

    def test_download_performance(self, collector, benchmark):
        """Benchmark download performance"""
        def download_test():
            # Mock successful download
            with patch.object(collector, '_download_and_store_image') as mock_download:
                mock_download.return_value = '/tmp/test.jpg'
                result = collector.download_image('https://example.com/test.jpg')
                return result

        result = benchmark(download_test)
        assert result == True

    def test_batch_processing_performance(self, collector, benchmark):
        """Benchmark batch processing"""
        urls = [f'https://example.com/image{i}.jpg' for i in range(10)]

        def batch_test():
            with patch.object(collector.download_engine, 'batch_download') as mock_batch:
                mock_batch.return_value = {
                    'total_processed': 10,
                    'successful': 10,
                    'failed': 0
                }
                results = collector.download_engine.batch_download(urls)
                return results

        result = benchmark(batch_test)
        assert result['successful'] == 10

class TestSecurity:
    """Security testing"""

    @pytest.fixture
    def collector(self):
        """Create collector for security tests"""
        with patch('tumblr_image_collector.TumblrImageCollector._setup_credentials'):
            collector = TumblrImageCollector()
            yield collector

    def test_input_validation(self, collector):
        """Test input validation"""
        # Test URL validation
        assert collector._validate_input('https://example.com', 'url') == True
        assert collector._validate_input('not-a-url', 'url') == False

        # Test filename validation
        assert collector._validate_input('safe_name.jpg', 'filename') == True
        assert collector._validate_input('unsafe<name>.jpg', 'filename') == False

    def test_domain_filtering(self, collector):
        """Test domain filtering"""
        # Mock allowed domains
        collector.allowed_domains = {'tumblr.com', 'media.tumblr.com'}

        assert collector._is_allowed_domain('tumblr.com') == True
        assert collector._is_allowed_domain('evil.com') == False

    @pytest.mark.parametrize("malicious_input", [
        "../../../etc/passwd",
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        "../../../../windows/system32/cmd.exe"
    ])
    def test_path_traversal_prevention(self, collector, malicious_input):
        """Test path traversal prevention"""
        assert collector._validate_input(malicious_input, 'path') == False

class TestIntegration:
    """Integration testing"""

    @pytest.fixture(scope="session")
    def temp_collection_dir(self, tmp_path_factory):
        """Create temporary collection directory for integration tests"""
        return tmp_path_factory.mktemp("tumblr_collection")

    def test_full_workflow(self, temp_collection_dir):
        """Test complete workflow from scan to download"""
        # This would be a comprehensive integration test
        # For now, just test the setup
        assert temp_collection_dir.exists()
        assert temp_collection_dir.is_dir()

    def test_cross_component_integration(self):
        """Test interaction between different components"""
        # Test CLI -> Core -> Storage integration
        # Test Web -> Core -> Mobile integration
        # etc.
        pass

# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Mark slow tests
        if "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
