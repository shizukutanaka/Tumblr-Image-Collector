import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from PIL import Image

from tumblr_collector_simple import SimpleTumblrCollector


class TestSimpleTumblrCollectorState(TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(self._temp_dir.name)
        self.output_dir = temp_path / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = temp_path / "config.json"
        self.state_file_name = "collector_state.json"
        config = {
            "api_key": "test_key",
            "api_secret": "test_secret",
            "output_dir": str(self.output_dir),
            "state_file": self.state_file_name,
            "log_dir": str(temp_path / "logs")
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    def _create_collector(self) -> SimpleTumblrCollector:
        with patch("tumblr_collector_simple.pytumblr.TumblrRestClient"):
            return SimpleTumblrCollector(config_file=str(self.config_path))

    def test_load_state_from_existing_file(self) -> None:
        state_payload = {
            "downloaded_files": ["example.jpg"],
            "downloaded_hashes": ["deadbeef"],
            "downloaded_urls": ["https://example.com/example.jpg"],
            "output_dir": str(self.output_dir)
        }
        with open(self.output_dir / self.state_file_name, "w", encoding="utf-8") as f:
            json.dump(state_payload, f)

        collector = self._create_collector()

        self.assertIn("example.jpg", collector.downloaded_files)
        self.assertIn("deadbeef", collector.downloaded_hashes)
        self.assertIn("https://example.com/example.jpg", collector.downloaded_urls)

    def test_populate_state_from_existing_images(self) -> None:
        image_path = self.output_dir / "existing.png"
        Image.new("RGB", (2, 2), color=(255, 0, 0)).save(image_path)

        collector = self._create_collector()

        self.assertIn("existing.png", collector.downloaded_files)
        self.assertGreaterEqual(len(collector.downloaded_hashes), 1)
        self.assertTrue((self.output_dir / self.state_file_name).exists())
        self.assertEqual(len(collector.downloaded_urls), 0)

    def test_download_skips_duplicate_url(self) -> None:
        collector = self._create_collector()

        class DummyResponse:
            def __init__(self, payload: bytes) -> None:
                self._payload = payload
                self.headers = {"Content-Type": "image/jpeg"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def iter_content(self, chunk_size: int):
                yield self._payload

            def raise_for_status(self) -> None:
                return None

        image_bytes = BytesIO()
        Image.new("RGB", (2, 2), color=(0, 255, 0)).save(image_bytes, format="JPEG")
        payload = image_bytes.getvalue()

        dummy_response = DummyResponse(payload)

        with patch.object(collector.session, "get", return_value=dummy_response), \
                patch.object(collector, "_is_valid_image", return_value=True), \
                patch.object(collector, "_calculate_image_hash_from_bytes", return_value="hash123"):
            first = collector._download_image(
                "https://example.com/image.jpg",
                "sampleblog",
                {"id": "post1"}
            )
            second = collector._download_image(
                "https://example.com/image.jpg",
                "sampleblog",
                {"id": "post2"}
            )

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertIn("https://example.com/image.jpg", collector.downloaded_urls)
