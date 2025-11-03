# Tumblr Image Collector - Ultimate Enhancement Edition

## 🚀 最強の多目的コンテンツコレクター

YouTube、学術論文、Webコンテンツを統合管理する究極のコレクションシステム

---

## ✨ 主要機能

### 📺 YouTube収集 (8K対応)
- **8K (4320p) 動画**ダウンロード
- **プレイリスト**一括処理
- **ライブストリーム**録画
- **30+言語字幕**自動取得
- **チャプター・メタデータ**完全保存

### 📚 学術論文収集
- **arXiv** 完全統合
- **Semantic Scholar** API統合
- **CrossRef/PubMed** クロスリファレンス
- **引用ネットワーク**自動構築
- **AI要約**生成 (OpenAI)
- **BibTeX/APA/MLA**自動生成

### 🌐 Webスクレイピング
- **Playwright** (最先端ブラウザ自動化)
- **Selenium** (従来型自動化)
- **BeautifulSoup** (軽量・高速)
- **httpx** (非同期HTTP)
- **ステルスモード**対応

### 🎬 動画処理
- **FFmpeg統合**による高度な処理
- **フォーマット変換** (H.264/H.265)
- **クリップ抽出**
- **音声抽出** (MP3/FLAC)
- **GIFアニメーション**生成
- **サムネイル**自動生成

### 🔗 統合管理
- **マルチソース**一元管理
- **並列ダウンロード** (最大15並列)
- **自動重複検出**
- **進捗追跡**
- **レポート生成**

---

## 📦 インストール

### 1. 基本パッケージ

```bash
pip install -r requirements.txt
```

### 2. Playwright セットアップ

```bash
pip install playwright
playwright install chromium
```

### 3. FFmpeg インストール

**Windows:**
```bash
choco install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

---

## 🎯 クイックスタート

### 例1: YouTube 8K動画ダウンロード

```python
from youtube_downloader import EnhancedYouTubeDownloader
import asyncio

async def download_8k_video():
    downloader = EnhancedYouTubeDownloader()

    result = await downloader.download_video_enhanced(
        url="https://www.youtube.com/watch?v=...",
        quality='4320p',  # 8K
        download_subtitles=True,
        subtitle_langs=['en', 'ja', 'es']
    )

    print(f"Downloaded: {result['video_path']}")

asyncio.run(download_8k_video())
```

### 例2: 学術論文の完全収集

```python
from arxiv_collector import EnhancedArXivCollector
from academic_cross_reference import get_cross_ref_resolver
import asyncio

async def collect_paper():
    # arXiv論文ダウンロード
    collector = EnhancedArXivCollector()
    paper = await collector.download_paper_enhanced(
        arxiv_id="2101.12345",
        extract_citations=True,
        generate_summary=True
    )

    # クロスリファレンス解決
    resolver = get_cross_ref_resolver()
    cross_ref = resolver.resolve_arxiv("2101.12345")

    # 関連論文検索
    related = resolver.find_related_papers(cross_ref, max_results=10)

    # 書誌情報エクスポート
    resolver.export_bibliography(
        cross_refs=[cross_ref] + related,
        output_path='bibliography.bib',
        format='bibtex'
    )

asyncio.run(collect_paper())
```

### 例3: 統合コレクター

```python
from unified_collector import get_unified_collector
import asyncio

async def unified_collection():
    collector = get_unified_collector()

    # YouTube動画追加
    await collector.add_youtube_video(
        url="https://www.youtube.com/watch?v=...",
        quality='1080p'
    )

    # arXiv論文追加
    await collector.add_arxiv_paper(
        arxiv_id="2101.12345",
        extract_citations=True
    )

    # Webコンテンツ追加
    await collector.add_web_content(
        url="https://example.com",
        scraper_type='playwright'
    )

    # 並列処理開始
    await collector.start_workers(num_workers=3)

    # レポート生成
    collector.export_collection_report('report.json')

asyncio.run(unified_collection())
```

### 例4: 動画処理

```python
from advanced_video_processor import get_video_processor

processor = get_video_processor()

# Web用に変換
processor.convert_format(
    input_path="video.mp4",
    output_path="web_version.mp4",
    preset='web_hd'
)

# GIFアニメーション作成
processor.create_gif(
    input_path="video.mp4",
    output_path="preview.gif",
    duration=5.0
)

# サムネイル生成
processor.generate_thumbnail(
    input_path="video.mp4",
    output_path="thumbnail.jpg"
)
```

---

## 📚 完全なワークフロー例

```bash
# 例を実行
python examples/complete_workflow.py youtube      # YouTube処理
python examples/complete_workflow.py paper        # 論文収集
python examples/complete_workflow.py unified      # 統合収集
python examples/complete_workflow.py web2video    # Web→動画
python examples/complete_workflow.py all          # すべて実行
```

---

## 🏗️ アーキテクチャ

```
┌─────────────────────────────────────────┐
│      Unified Collector (統合管理)        │
│  - タスク管理                             │
│  - 並列処理                               │
│  - 重複検出                               │
└─────────────┬───────────────────────────┘
              │
    ┌─────────┼─────────┬─────────┐
    │         │         │         │
┌───▼───┐ ┌──▼──┐ ┌───▼───┐ ┌──▼──┐
│YouTube│ │arXiv│ │Semantic│ │ Web │
│  8K   │ │ PDF │ │Scholar │ │Scrape│
└───┬───┘ └──┬──┘ └───┬───┘ └──┬──┘
    │        │        │        │
    │   ┌────▼────────▼────┐   │
    │   │Cross Reference  │   │
    │   │   Resolver      │   │
    │   └─────────────────┘   │
    │                         │
┌───▼─────────────────────────▼───┐
│    Advanced Video Processor     │
│  - 変換 - GIF - サムネイル      │
└─────────────────────────────────┘
```

---

## 📊 パフォーマンス

| 機能 | スペック |
|------|---------|
| YouTube最大解像度 | 8K (4320p) |
| 並列ダウンロード | 最大15並列 |
| 対応論文データベース | 4+ (arXiv, CrossRef, PubMed, Semantic Scholar) |
| Webスクレイピングエンジン | 4+ (Playwright, Selenium, BeautifulSoup, httpx) |
| 動画処理速度 | FFmpeg最適化 |
| 重複検出 | SHA-256ハッシュ |

---

## 📁 プロジェクト構造

```
tumblr-image-collector/
├── youtube_downloader.py           # YouTube収集
├── arxiv_collector.py              # arXiv収集
├── semantic_scholar_collector.py   # Semantic Scholar
├── adaptive_scraper.py             # Webスクレイピング
├── unified_collector.py            # 統合管理 ★
├── advanced_video_processor.py     # 動画処理 ★
├── academic_cross_reference.py     # クロスリファレンス ★
├── requirements.txt                # 依存パッケージ
├── examples/
│   └── complete_workflow.py        # 完全ワークフロー ★
├── tests/
│   └── test_enhanced_collectors.py # テストスイート
└── docs/
    ├── ENHANCED_COLLECTORS_IMPROVEMENTS.md
    ├── QUICK_START_ENHANCED.md
    ├── IMPROVEMENTS_SUMMARY.md
    └── FINAL_IMPROVEMENTS_COMPLETE.md
```

---

## 🧪 テスト

```bash
# すべてのテストを実行
pytest tests/test_enhanced_collectors.py -v

# カバレッジレポート
pytest tests/test_enhanced_collectors.py --cov=. --cov-report=html

# 特定のテスト
pytest tests/test_enhanced_collectors.py::TestEnhancedYouTubeDownloader -v
```

---

## 📖 ドキュメント

- [📘 詳細改善リスト](ENHANCED_COLLECTORS_IMPROVEMENTS.md)
- [🚀 クイックスタート](QUICK_START_ENHANCED.md)
- [📊 改善サマリー](IMPROVEMENTS_SUMMARY.md)
- [✅ 完了レポート](FINAL_IMPROVEMENTS_COMPLETE.md)

---

## 🎓 技術スタック

### コア
- Python 3.8+
- asyncio (非同期処理)
- dataclasses (データ構造)

### YouTube
- yt-dlp 2024.10.7
- pytube 15.0.0

### 学術論文
- PyPDF2 3.0.1
- pdfplumber 0.11.0
- bibtexparser 1.4.1
- semanticscholar 0.8.2
- crossref-commons 0.0.7

### Webスクレイピング
- playwright 1.47.0
- selenium 4.25.0
- beautifulsoup4 4.12.3
- httpx 0.27.2
- aiohttp 3.10.5

### 動画処理
- FFmpeg (外部依存)

---

## 💡 ユースケース

### 1. 研究者向け
- 学術論文の系統的レビュー
- 引用ネットワーク分析
- 文献管理システム統合

### 2. 教育者向け
- 教材動画の収集・編集
- マルチメディア教材作成
- オンライン講義アーカイブ

### 3. コンテンツクリエイター向け
- ソーシャルメディア用動画変換
- プレビューGIF自動生成
- サムネイル一括作成

### 4. データサイエンティスト向け
- 大規模動画データ収集
- Webコンテンツマイニング
- マルチモーダルデータセット構築

---

## 🔧 高度な設定

### 環境変数

```bash
# YouTube設定
export YOUTUBE_QUALITY=1080p
export YOUTUBE_DOWNLOAD_PATH=/path/to/downloads

# OpenAI (AI要約用)
export OPENAI_API_KEY=your_api_key

# プロキシ
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=https://proxy:port
```

### 設定ファイル

```python
# config.py
CONFIG = {
    'youtube': {
        'quality': '1080p',
        'subtitles': True,
        'languages': ['en', 'ja']
    },
    'papers': {
        'extract_citations': True,
        'generate_summary': True
    },
    'unified': {
        'num_workers': 5,
        'skip_duplicates': True,
        'max_retries': 3
    }
}
```

---

## 🤝 貢献

プルリクエストを歓迎します！

1. Fork
2. Feature branch作成 (`git checkout -b feature/amazing`)
3. Commit (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing`)
5. Pull Request作成

---

## 📄 ライセンス

元のTumblr Image Collectorのライセンスに従います。

---

## 🙏 謝辞

- yt-dlp チーム
- arXiv
- Semantic Scholar
- Playwright チーム
- FFmpeg プロジェクト

---

## 📞 サポート

- Issues: [GitHub Issues](https://github.com/your-repo/issues)
- Discussions: [GitHub Discussions](https://github.com/your-repo/discussions)

---

**バージョン**: 3.0.0 Ultimate Edition
**最終更新**: 2025-10-30
**ステータス**: ✅ Production Ready

---

Made with ❤️ by Claude Code Enhancement Team
