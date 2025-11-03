#!/usr/bin/env python3
"""
Enhanced Collectors Test Suite
YouTube、論文、Webスクレイピングの拡張機能をテスト
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from youtube_downloader import EnhancedYouTubeDownloader
from arxiv_collector import EnhancedArXivCollector
from semantic_scholar_collector import SemanticScholarCollector
from adaptive_scraper import AdaptiveScraper


class TestEnhancedYouTubeDownloader:
    """Enhanced YouTube Downloader テスト"""

    @pytest.fixture
    def downloader(self):
        """テスト用ダウンローダーインスタンス"""
        return EnhancedYouTubeDownloader(download_path="tests/downloads/youtube")

    def test_initialization(self, downloader):
        """初期化テスト"""
        assert downloader.download_path == "tests/downloads/youtube"
        assert len(downloader.supported_resolutions) > 0
        assert len(downloader.subtitle_languages) > 0

    def test_url_validation(self, downloader):
        """URL検証テスト"""
        # 有効なURL
        assert downloader._is_valid_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert downloader._is_valid_youtube_url("https://youtu.be/dQw4w9WgXcQ")

        # 無効なURL
        assert not downloader._is_valid_youtube_url("https://example.com")
        assert not downloader._is_valid_youtube_url("invalid")

    def test_filename_sanitization(self, downloader):
        """ファイル名サニタイゼーションテスト"""
        dangerous_name = "Test<>:\"/\\|?*Video"
        sanitized = downloader._sanitize_filename(dangerous_name)

        assert '<' not in sanitized
        assert '>' not in sanitized
        assert ':' not in sanitized
        assert '"' not in sanitized

    def test_quality_to_height_conversion(self, downloader):
        """品質→高さ変換テスト"""
        assert downloader._quality_to_height('1080p') == 1080
        assert downloader._quality_to_height('4k') == 2160
        assert downloader._quality_to_height('8k') == 4320
        assert downloader._quality_to_height('hd') == 720

    @pytest.mark.asyncio
    async def test_playlist_download_advanced(self, downloader):
        """高度なプレイリストダウンロードテスト（モック）"""
        # 実際のダウンロードはスキップし、メソッドの存在を確認
        assert hasattr(downloader, 'download_playlist_advanced')

        # モックURLでテスト
        results = downloader.download_playlist_advanced(
            "https://www.youtube.com/playlist?list=PLtest",
            quality='720p',
            max_videos=1
        )

        assert isinstance(results, list)


class TestEnhancedArXivCollector:
    """Enhanced arXiv Collector テスト"""

    @pytest.fixture
    def collector(self):
        """テスト用コレクターインスタンス"""
        return EnhancedArXivCollector(download_path="tests/downloads/papers")

    def test_initialization(self, collector):
        """初期化テスト"""
        assert collector.download_path == "tests/downloads/papers"
        assert collector.base_url == "http://export.arxiv.org/api/query"

    def test_keyword_extraction(self, collector):
        """キーワード抽出テスト"""
        abstract = """This paper presents a novel approach to machine learning using deep neural networks
        for image classification. We demonstrate superior performance on benchmark datasets."""

        keywords = collector._extract_keywords(abstract)

        assert isinstance(keywords, list)
        assert 'machine' in keywords or 'learning' in keywords
        assert len(keywords) <= 20

    def test_readability_score_calculation(self, collector):
        """読みやすさスコア計算テスト"""
        simple_text = "This is a simple text. It is easy to read. Very short words."
        complex_text = "Notwithstanding aforementioned considerations, implementation necessitates comprehensive understanding."

        simple_score = collector._calculate_readability_score(simple_text)
        complex_score = collector._calculate_readability_score(complex_text)

        assert isinstance(simple_score, float)
        assert isinstance(complex_score, float)
        assert 0 <= simple_score <= 100
        assert 0 <= complex_score <= 100

    def test_impact_score_calculation(self, collector):
        """影響度スコア計算テスト"""
        categories = [{'term': 'cs.AI'}, {'term': 'cs.LG'}]
        authors = ['Author 1', 'Author 2', 'Author 3']
        abstract = "A comprehensive study of AI with significant implications."

        score = collector._calculate_impact_score(categories, authors, abstract)

        assert isinstance(score, float)
        assert score <= 10.0
        assert score > 0

    def test_pdf_text_extraction(self, collector):
        """PDF テキスト抽出テスト（モック）"""
        # メソッドの存在を確認
        assert hasattr(collector, 'extract_pdf_text')

    @pytest.mark.asyncio
    async def test_paper_content_analysis(self, collector):
        """論文コンテンツ分析テスト（モック）"""
        assert hasattr(collector, 'analyze_paper_content')

    def test_section_extraction(self, collector):
        """セクション抽出テスト"""
        text = """
        1. Introduction
        This is the introduction.

        2. Methods
        This section describes the methods.

        CONCLUSION
        This is the conclusion.
        """

        sections = collector._extract_sections(text)

        assert isinstance(sections, list)
        assert len(sections) > 0


class TestSemanticScholarCollector:
    """Semantic Scholar Collector テスト"""

    @pytest.fixture
    def collector(self):
        """テスト用コレクターインスタンス"""
        return SemanticScholarCollector(download_path="tests/downloads/papers")

    def test_initialization(self, collector):
        """初期化テスト"""
        assert collector.download_path == "tests/downloads/papers"
        assert collector.base_url == "https://api.semanticscholar.org/graph/v1"

    def test_citation_count_search(self, collector):
        """引用数検索テスト（モック）"""
        assert hasattr(collector, 'search_by_citation_count')

    def test_author_details(self, collector):
        """著者詳細取得テスト（モック）"""
        assert hasattr(collector, 'get_author_details')

    @pytest.mark.asyncio
    async def test_pdf_download(self, collector):
        """PDF ダウンロードテスト（モック）"""
        assert hasattr(collector, 'download_paper_pdf')

    def test_trending_papers(self, collector):
        """トレンディング論文取得テスト（モック）"""
        assert hasattr(collector, 'get_trending_papers')


class TestAdaptiveScraper:
    """Adaptive Scraper テスト"""

    @pytest.fixture
    def scraper(self):
        """テスト用スクレイパーインスタンス"""
        return AdaptiveScraper()

    def test_initialization(self, scraper):
        """初期化テスト"""
        assert scraper.scraper_config['enabled'] == True
        assert scraper.scraper_config['framework'] == 'CCCD'

    def test_user_agent_rotation(self, scraper):
        """ユーザーエージェントローテーションテスト"""
        ua1 = scraper._get_rotating_user_agent()
        ua2 = scraper._get_rotating_user_agent()

        assert isinstance(ua1, str)
        assert isinstance(ua2, str)
        assert len(ua1) > 0
        assert len(ua2) > 0

    def test_data_validation(self, scraper):
        """データ検証テスト"""
        # 有効なデータ
        valid_data = {
            'url': 'https://example.com',
            'title': 'Test Page',
            'metadata': {'content_length': 1000}
        }

        is_valid, message = scraper._validate_data_item(valid_data)
        assert is_valid

        # 無効なデータ
        invalid_data = {
            'title': 'Test Page'
        }

        is_valid, message = scraper._validate_data_item(invalid_data)
        assert not is_valid

    def test_adaptive_rate_limiting(self, scraper):
        """適応型レート制限テスト"""
        # モニタリングデータを設定
        scraper.monitoring_data['request_count'] = 100
        scraper.monitoring_data['success_count'] = 40

        initial_delay = scraper.adaptive_params['current_delay']
        scraper._apply_adaptive_rate_limiting()

        # 成功率が低いため、遅延が増加するはず
        assert scraper.adaptive_params['current_delay'] >= initial_delay

    @pytest.mark.asyncio
    async def test_scrape_with_playwright(self, scraper):
        """Playwright スクレイピングテスト（モック）"""
        assert hasattr(scraper, 'scrape_with_playwright')

    @pytest.mark.asyncio
    async def test_scrape_with_selenium(self, scraper):
        """Selenium スクレイピングテスト（モック）"""
        assert hasattr(scraper, 'scrape_with_selenium')

    def test_scrape_with_beautifulsoup(self, scraper):
        """BeautifulSoup スクレイピングテスト（モック）"""
        assert hasattr(scraper, 'scrape_with_beautifulsoup')

    @pytest.mark.asyncio
    async def test_scrape_with_httpx(self, scraper):
        """httpx スクレイピングテスト（モック）"""
        assert hasattr(scraper, 'scrape_with_httpx')

    def test_cccd_framework_execution(self, scraper):
        """CCCD フレームワーク実行テスト（モック）"""
        result = scraper.execute_cccd_framework(
            target_url="https://example.com",
            config={}
        )

        assert isinstance(result, dict)
        assert 'framework' in result
        assert 'phases' in result

    def test_scraper_stats(self, scraper):
        """スクレイパー統計テスト"""
        stats = scraper.get_scraper_stats()

        assert isinstance(stats, dict)
        assert 'total_operations' in stats
        assert 'successful_operations' in stats
        assert 'cccd_state' in stats


class TestIntegration:
    """統合テスト"""

    def test_all_collectors_instantiation(self):
        """全コレクターの同時インスタンス化テスト"""
        youtube = EnhancedYouTubeDownloader()
        arxiv = EnhancedArXivCollector()
        semantic = SemanticScholarCollector()
        scraper = AdaptiveScraper()

        assert youtube is not None
        assert arxiv is not None
        assert semantic is not None
        assert scraper is not None

    def test_directory_creation(self):
        """ディレクトリ作成テスト"""
        test_path = "tests/downloads/integration_test"

        youtube = EnhancedYouTubeDownloader(download_path=test_path)

        assert os.path.exists(test_path)

        # クリーンアップ
        if os.path.exists(test_path):
            os.rmdir(test_path)


# テスト実行用メインブロック
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
