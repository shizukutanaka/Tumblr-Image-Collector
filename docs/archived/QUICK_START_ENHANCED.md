# Quick Start Guide - Enhanced Collectors

## YouTube、論文、Webスクレイピングの拡張機能クイックスタート

---

## 🚀 セットアップ

### 1. 依存パッケージのインストール

```bash
# 基本パッケージ
pip install -r requirements.txt

# Playwrightセットアップ（Webスクレイピング用）
pip install playwright
playwright install chromium

# オプション: GPU加速
pip install tensorflow[and-cuda]
```

### 2. 環境変数設定（オプション）

```bash
# .envファイルを作成
cat > .env <<EOF
# YouTube設定
YOUTUBE_QUALITY=1080p
YOUTUBE_DOWNLOAD_PATH=downloads/youtube

# 論文設定
PAPERS_DOWNLOAD_PATH=downloads/papers
OPENAI_API_KEY=your_api_key_here  # AI要約生成用

# Webスクレイピング設定
SCRAPER_USER_AGENTS=Mozilla/5.0...
SCRAPER_PROXY_URL=http://proxy:port
EOF
```

---

## 📺 YouTube ダウンローダー

### 基本的な使い方

```python
from youtube_downloader import EnhancedYouTubeDownloader

# インスタンス作成
downloader = EnhancedYouTubeDownloader()

# 動画情報を取得
info = downloader.get_video_info_enhanced("https://www.youtube.com/watch?v=...")
print(f"Title: {info['title']}")
print(f"Duration: {info['duration']}s")
print(f"Views: {info['view_count']}")

# 動画をダウンロード
import asyncio
result = asyncio.run(downloader.download_video_enhanced(
    url="https://www.youtube.com/watch?v=...",
    quality='1080p',
    download_subtitles=True,
    subtitle_langs=['en', 'ja']
))

print(f"Downloaded: {result['video_path']}")
```

### 高度な機能

```python
# プレイリスト全体をダウンロード
results = downloader.download_playlist_advanced(
    playlist_url="https://www.youtube.com/playlist?list=...",
    quality='1080p',
    max_videos=50,
    download_subtitles=True
)

# ライブストリーミングを録画
live_path = downloader.download_live_stream(
    url="https://www.youtube.com/watch?v=...",
    duration=3600,  # 1時間
    quality='1080p'
)

# チャンネルの動画を一括ダウンロード
channel_results = downloader.download_channel_videos(
    channel_url="https://www.youtube.com/@channel",
    limit=50,
    quality='720p'
)
```

---

## 📚 学術論文コレクター

### arXiv Collector

```python
from arxiv_collector import EnhancedArXivCollector

# インスタンス作成
collector = EnhancedArXivCollector()

# 高度な検索
papers = collector.search_papers_enhanced(
    query="quantum computing",
    max_results=20,
    filters={
        'categories': ['quant-ph', 'cs.QI'],
        'date_from': '2024-01-01',
        'title_only': False
    }
)

# 論文を高度な機能付きでダウンロード
import asyncio
result = asyncio.run(collector.download_paper_enhanced(
    arxiv_id="2101.12345",
    extract_citations=True,    # 引用情報を抽出
    generate_summary=True       # AI要約を生成
))

print(f"PDF: {result['pdf_path']}")
print(f"Keywords: {result['keywords']}")
print(f"Summary: {result['summary']}")
print(f"Related Papers: {len(result['related_papers'])}")

# 論文ライブラリをエクスポート
collector.export_paper_library(
    output_path='my_papers.bib',
    format='bib'  # または 'json', 'csv'
)
```

### Semantic Scholar Collector

```python
from semantic_scholar_collector import SemanticScholarCollector

# インスタンス作成
collector = SemanticScholarCollector()

# 論文検索
papers = collector.search_papers(
    query="machine learning",
    limit=20
)

# 高被引用論文を検索
popular_papers = collector.search_by_citation_count(
    query="deep learning",
    min_citations=100,
    limit=10
)

# 著者情報を取得
author = collector.get_author_details("author_id")
print(f"Name: {author['name']}")
print(f"h-index: {author['h_index']}")
print(f"Papers: {author['paper_count']}")

# トレンディング論文を取得
trending = collector.get_trending_papers(
    field='computer-science',
    limit=10
)

# PDFをダウンロード
import asyncio
pdf_path = asyncio.run(collector.download_paper_pdf("paper_id"))
```

---

## 🌐 Web スクレイピング

### Adaptive Scraper

```python
from adaptive_scraper import AdaptiveScraper, get_adaptive_scraper

# インスタンス作成
scraper = get_adaptive_scraper()

# CCCDフレームワークで包括的スクレイピング
result = scraper.execute_cccd_framework(
    target_url="https://example.com",
    config={
        'max_depth': 3,
        'follow_external': False
    }
)

print(f"URLs found: {result['phases']['crawling']['urls_found']}")
print(f"Data collected: {result['phases']['collection']['collected_count']}")
```

### 各種スクレイピングエンジン

```python
import asyncio

# Playwright（推奨: 最も強力）
result = asyncio.run(scraper.scrape_with_playwright(
    url="https://example.com",
    wait_for=".content-loaded"
))

# Selenium（従来型）
result = asyncio.run(scraper.scrape_with_selenium(
    url="https://example.com",
    wait_time=5
))

# BeautifulSoup（軽量・高速）
result = scraper.scrape_with_beautifulsoup(
    url="https://example.com",
    parser='lxml'
)

# httpx（非同期HTTP）
result = asyncio.run(scraper.scrape_with_httpx(
    url="https://example.com",
    follow_redirects=True
))
```

### 統計とモニタリング

```python
# スクレイパー統計を取得
stats = scraper.get_scraper_stats()
print(f"Total operations: {stats['total_operations']}")
print(f"Success rate: {stats['success_rate']:.2%}")

# CCCD状態を確認
status = scraper.get_cccd_status()
print(f"Crawling: {status['state']['crawling']}")
print(f"Collection: {status['state']['collection']}")
```

---

## 🧪 テスト実行

### すべてのテストを実行

```bash
pytest tests/test_enhanced_collectors.py -v
```

### 特定のコレクターのみテスト

```bash
# YouTube
pytest tests/test_enhanced_collectors.py::TestEnhancedYouTubeDownloader -v

# arXiv
pytest tests/test_enhanced_collectors.py::TestEnhancedArXivCollector -v

# Semantic Scholar
pytest tests/test_enhanced_collectors.py::TestSemanticScholarCollector -v

# Webスクレイピング
pytest tests/test_enhanced_collectors.py::TestAdaptiveScraper -v
```

### カバレッジレポート生成

```bash
pytest tests/test_enhanced_collectors.py --cov=. --cov-report=html
```

---

## 💡 ベストプラクティス

### YouTube ダウンロード

1. **品質選択**
   - 一般用途: 720p または 1080p
   - 高品質保存: 1440p または 2160p (4K)
   - アーカイブ: 4320p (8K)

2. **字幕**
   - 常に複数言語をダウンロード推奨
   - 自動生成字幕も有用

3. **バッチダウンロード**
   - プレイリストは `download_playlist_advanced` を使用
   - API制限を避けるため適切な遅延を設定

### 学術論文収集

1. **検索最適化**
   - 具体的なカテゴリを指定
   - 日付範囲で絞り込み
   - 著者名で精度向上

2. **メタデータ管理**
   - BibTeX形式でエクスポート
   - 定期的にライブラリをバックアップ

3. **AI要約**
   - OpenAI APIキーを設定
   - バッチ処理でコスト削減

### Webスクレイピング

1. **エンジン選択**
   - 静的サイト → BeautifulSoup
   - 動的サイト → Playwright
   - Cloudflare保護 → Selenium + undetected-chromedriver

2. **レート制限**
   - 適応型レート制限を有効化
   - プロキシローテーションを使用
   - ユーザーエージェントを定期的に変更

3. **エラーハンドリング**
   - 常にtry-exceptブロックを使用
   - リトライロジックを実装
   - ログを詳細に記録

---

## ⚠️ 注意事項

### 法的遵守

- YouTube: 利用規約を遵守
- 学術論文: 著作権法を確認
- Webスクレイピング: robots.txtを尊重

### パフォーマンス

- 大量ダウンロード時はディスク容量を確認
- メモリ使用量に注意（特に8K動画）
- ネットワーク帯域を考慮

### セキュリティ

- APIキーは環境変数で管理
- ダウンロードファイルのウイルススキャン
- プロキシは信頼できるもののみ使用

---

## 🔧 トラブルシューティング

### YouTube ダウンロードが失敗する

```python
# yt-dlpの更新
pip install --upgrade yt-dlp

# Pythonバージョン確認（3.8以上推奨）
python --version

# 詳細ログ有効化
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 論文PDFが見つからない

```python
# arXiv IDの形式を確認
# 正: "2101.12345" または "arXiv:2101.12345"
# 誤: "2101.12345v1"

# DOIからarXiv IDを取得
from crossref_commons.iteration import iterate_publications_as_json
```

### Webスクレイピングがブロックされる

```python
# プロキシを追加
scraper.add_proxy("http://proxy:port", "username", "password")

# ユーザーエージェントをカスタマイズ
custom_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
    # ... more user agents
]
scraper.set_user_agents(custom_agents)

# 遅延を増やす
scraper.adaptive_params['current_delay'] = 5.0  # 5秒
```

---

## 📖 詳細ドキュメント

- [完全な改善リスト](ENHANCED_COLLECTORS_IMPROVEMENTS.md)
- [APIリファレンス](docs/API_REFERENCE.md)
- [アーキテクチャ設計](docs/ARCHITECTURE.md)

---

**最終更新**: 2025-10-30
**バージョン**: 2.0.0
