#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Workflow Examples
YouTube、論文、Web収集の完全統合ワークフロー例
"""

import asyncio
import os
from pathlib import Path


# ============================================================================
# 例1: YouTube 研究動画の完全処理ワークフロー
# ============================================================================

async def youtube_research_workflow():
    """
    YouTubeの研究動画を完全処理するワークフロー
    - 8K動画ダウンロード
    - Web用変換
    - サムネイル・プレビュー生成
    - メタデータ保存
    """
    from youtube_downloader import EnhancedYouTubeDownloader
    from advanced_video_processor import get_video_processor

    print("=== YouTube Research Video Workflow ===")

    # 1. 動画ダウンロード
    downloader = EnhancedYouTubeDownloader()

    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # 例

    print("1. Downloading video...")
    result = await downloader.download_video_enhanced(
        url=video_url,
        quality='1080p',  # または '4320p' for 8K
        download_subtitles=True,
        subtitle_langs=['en', 'ja', 'es']
    )

    if not result or 'error' in result:
        print(f"Download failed: {result.get('error')}")
        return

    video_path = result['video_path']
    print(f"   ✓ Downloaded: {video_path}")

    # 2. 動画処理
    processor = get_video_processor()

    # Web用に最適化
    print("2. Converting for web...")
    web_path = video_path.replace('.mp4', '_web.mp4')
    processor.convert_format(
        input_path=video_path,
        output_path=web_path,
        preset='web_hd'
    )
    print(f"   ✓ Web version: {web_path}")

    # モバイル用変換
    print("3. Converting for mobile...")
    mobile_path = video_path.replace('.mp4', '_mobile.mp4')
    processor.convert_format(
        input_path=video_path,
        output_path=mobile_path,
        preset='mobile'
    )
    print(f"   ✓ Mobile version: {mobile_path}")

    # 4. サムネイル生成
    print("4. Generating thumbnail...")
    thumb_path = video_path.replace('.mp4', '_thumb.jpg')
    processor.generate_thumbnail(
        input_path=video_path,
        output_path=thumb_path,
        timestamp=5.0
    )
    print(f"   ✓ Thumbnail: {thumb_path}")

    # 5. プレビューGIF作成
    print("5. Creating preview GIF...")
    gif_path = video_path.replace('.mp4', '_preview.gif')
    processor.create_gif(
        input_path=video_path,
        output_path=gif_path,
        start_time=10.0,
        duration=3.0,
        fps=10,
        width=480
    )
    print(f"   ✓ Preview GIF: {gif_path}")

    # 6. 音声抽出
    print("6. Extracting audio...")
    audio_path = video_path.replace('.mp4', '_audio.mp3')
    processor.extract_audio(
        input_path=video_path,
        output_path=audio_path,
        format='mp3',
        bitrate='192k'
    )
    print(f"   ✓ Audio: {audio_path}")

    print("\n✅ Complete! Generated files:")
    print(f"   - Original: {video_path}")
    print(f"   - Web: {web_path}")
    print(f"   - Mobile: {mobile_path}")
    print(f"   - Thumbnail: {thumb_path}")
    print(f"   - Preview: {gif_path}")
    print(f"   - Audio: {audio_path}")
    print(f"   - Subtitles: {len(result.get('subtitle_paths', []))} files")


# ============================================================================
# 例2: 学術論文の完全収集・分析ワークフロー
# ============================================================================

async def academic_paper_workflow():
    """
    学術論文の完全収集・分析ワークフロー
    - arXivダウンロード
    - クロスリファレンス解決
    - 関連論文検索
    - 引用ネットワーク構築
    - 書誌情報エクスポート
    """
    from arxiv_collector import EnhancedArXivCollector
    from academic_cross_reference import get_cross_ref_resolver

    print("\n=== Academic Paper Research Workflow ===")

    arxiv_id = "2101.12345"  # 例

    # 1. 論文ダウンロード
    print("1. Downloading paper from arXiv...")
    collector = EnhancedArXivCollector()

    paper = await collector.download_paper_enhanced(
        arxiv_id=arxiv_id,
        extract_citations=True,
        generate_summary=True
    )

    if not paper:
        print(f"Failed to download paper {arxiv_id}")
        return

    print(f"   ✓ PDF: {paper['pdf_path']}")
    print(f"   ✓ Keywords: {len(paper.get('keywords', []))}")
    if paper.get('summary'):
        print(f"   ✓ AI Summary generated")

    # 2. クロスリファレンス解決
    print("\n2. Resolving cross-references...")
    resolver = get_cross_ref_resolver()

    cross_ref = resolver.resolve_arxiv(arxiv_id)

    if cross_ref:
        print(f"   ✓ Title: {cross_ref.title}")
        print(f"   ✓ Authors: {len(cross_ref.authors)}")
        print(f"   ✓ Citations: {cross_ref.citation_count}")

        if cross_ref.identifiers.doi:
            print(f"   ✓ DOI: {cross_ref.identifiers.doi}")
        if cross_ref.identifiers.semantic_scholar_id:
            print(f"   ✓ Semantic Scholar ID: {cross_ref.identifiers.semantic_scholar_id}")

    # 3. 関連論文検索
    print("\n3. Finding related papers...")
    related_papers = resolver.find_related_papers(cross_ref, max_results=10)

    print(f"   ✓ Found {len(related_papers)} related papers")
    for i, related in enumerate(related_papers[:5], 1):
        print(f"   {i}. {related.title[:60]}... ({related.year})")

    # 4. 引用ネットワーク構築
    print("\n4. Building citation network...")
    network = resolver.get_citation_network(cross_ref, depth=1)

    print(f"   ✓ Citations: {len(network['citations'])}")
    print(f"   ✓ References: {len(network['references'])}")

    # 5. 関連論文もダウンロード
    print("\n5. Downloading related papers...")
    downloaded_related = []

    for related in related_papers[:3]:  # 上位3件
        if related.identifiers.arxiv_id:
            print(f"   Downloading: {related.title[:50]}...")
            related_paper = await collector.download_paper_enhanced(
                arxiv_id=related.identifiers.arxiv_id,
                extract_citations=False
            )
            if related_paper:
                downloaded_related.append(related)
                print(f"   ✓ Downloaded")

    # 6. 書誌情報エクスポート
    print("\n6. Exporting bibliography...")

    all_papers = [cross_ref] + downloaded_related

    # BibTeX
    resolver.export_bibliography(
        cross_refs=all_papers,
        output_path='research_bibliography.bib',
        format='bibtex'
    )
    print("   ✓ BibTeX: research_bibliography.bib")

    # JSON
    resolver.export_bibliography(
        cross_refs=all_papers,
        output_path='research_papers.json',
        format='json'
    )
    print("   ✓ JSON: research_papers.json")

    # CSV
    resolver.export_bibliography(
        cross_refs=all_papers,
        output_path='research_papers.csv',
        format='csv'
    )
    print("   ✓ CSV: research_papers.csv")

    print("\n✅ Complete! Collection summary:")
    print(f"   - Main paper: {cross_ref.title}")
    print(f"   - Related papers downloaded: {len(downloaded_related)}")
    print(f"   - Total in bibliography: {len(all_papers)}")
    print(f"   - Citation network size: {len(network['citations']) + len(network['references'])}")


# ============================================================================
# 例3: 統合コレクターによる大規模収集
# ============================================================================

async def unified_large_scale_workflow():
    """
    統合コレクターによる大規模収集ワークフロー
    - 複数ソースから並列収集
    - 自動重複検出
    - 進捗追跡
    - レポート生成
    """
    from unified_collector import get_unified_collector

    print("\n=== Unified Large-Scale Collection Workflow ===")

    # 統合コレクター初期化
    collector = get_unified_collector({
        'base_path': 'downloads/unified_collection',
        'skip_duplicates': True,
        'max_retries': 3
    })

    # 1. YouTube動画追加
    print("1. Adding YouTube videos...")

    youtube_urls = [
        "https://www.youtube.com/watch?v=video1",
        "https://www.youtube.com/watch?v=video2",
        "https://www.youtube.com/watch?v=video3",
    ]

    for url in youtube_urls:
        task = await collector.add_youtube_video(
            url=url,
            quality='1080p',
            download_subtitles=True
        )
        if task:
            print(f"   ✓ Added: {url}")

    # 2. YouTubeプレイリスト追加
    print("\n2. Adding YouTube playlists...")

    playlist_url = "https://www.youtube.com/playlist?list=PLexample"
    task = await collector.add_youtube_playlist(
        url=playlist_url,
        quality='1080p',
        max_videos=20
    )
    if task:
        print(f"   ✓ Added playlist: {playlist_url}")

    # 3. 学術論文追加
    print("\n3. Adding academic papers...")

    arxiv_ids = ["2101.12345", "2102.23456", "2103.34567"]

    for arxiv_id in arxiv_ids:
        task = await collector.add_arxiv_paper(
            arxiv_id=arxiv_id,
            extract_citations=True
        )
        if task:
            print(f"   ✓ Added arXiv: {arxiv_id}")

    # 4. Webコンテンツ追加
    print("\n4. Adding web content...")

    web_urls = [
        "https://example.com/article1",
        "https://example.com/article2",
    ]

    for url in web_urls:
        task = await collector.add_web_content(
            url=url,
            scraper_type='playwright'
        )
        if task:
            print(f"   ✓ Added web: {url}")

    # 5. 並列処理開始
    print("\n5. Starting parallel processing (3 workers)...")
    print("   This may take a while...")

    await collector.start_workers(num_workers=3)

    # 6. 統計情報取得
    print("\n6. Collection statistics:")
    stats = collector.get_statistics()

    print(f"   Total tasks: {stats['total_tasks']}")
    print(f"   Completed: {stats['completed_tasks']}")
    print(f"   Failed: {stats['failed_tasks']}")
    print(f"   Skipped: {stats['skipped_tasks']}")
    print(f"   Total indexed: {stats['total_indexed']}")

    # 7. レポート生成
    print("\n7. Generating collection report...")
    collector.export_collection_report('collection_report.json')
    print("   ✓ Report: collection_report.json")

    print("\n✅ Complete! Large-scale collection finished")


# ============================================================================
# 例4: Webスクレイピング + 動画処理の統合
# ============================================================================

async def web_to_video_workflow():
    """
    Webスクレイピングと動画処理の統合ワークフロー
    - Webページから動画URL抽出
    - 動画ダウンロード
    - 自動処理・変換
    """
    from adaptive_scraper import AdaptiveScraper
    from youtube_downloader import EnhancedYouTubeDownloader
    from advanced_video_processor import get_video_processor

    print("\n=== Web to Video Processing Workflow ===")

    # 1. Webページスクレイピング
    print("1. Scraping web page for video URLs...")

    scraper = AdaptiveScraper()
    result = await scraper.scrape_with_playwright(
        url="https://example.com/videos",
        wait_for=".video-list"
    )

    if result and result['status'] == 'success':
        print("   ✓ Page scraped successfully")

        # 実際にはBeautifulSoupでURLを抽出
        # video_urls = extract_video_urls(result['content'])
        video_urls = ["https://www.youtube.com/watch?v=example"]  # 例

        print(f"   ✓ Found {len(video_urls)} video URLs")
    else:
        print("   ✗ Scraping failed")
        return

    # 2. 動画ダウンロード
    print("\n2. Downloading videos...")

    downloader = EnhancedYouTubeDownloader()
    processor = get_video_processor()

    for i, url in enumerate(video_urls[:3], 1):  # 最初の3件
        print(f"\n   Video {i}/{len(video_urls[:3])}")

        # ダウンロード
        result = await downloader.download_video_enhanced(
            url=url,
            quality='1080p'
        )

        if not result or 'error' in result:
            print(f"   ✗ Download failed")
            continue

        video_path = result['video_path']
        print(f"   ✓ Downloaded: {video_path}")

        # 自動処理
        print("   Processing...")

        # Web用変換
        web_path = video_path.replace('.mp4', '_web.mp4')
        processor.convert_format(video_path, web_path, preset='web_hd')

        # サムネイル生成
        thumb_path = video_path.replace('.mp4', '_thumb.jpg')
        processor.generate_thumbnail(video_path, thumb_path)

        # プレビューGIF
        gif_path = video_path.replace('.mp4', '_preview.gif')
        processor.create_gif(video_path, gif_path, duration=3.0)

        print(f"   ✓ Processed: web, thumbnail, preview")

    print("\n✅ Complete! Web to video workflow finished")


# ============================================================================
# 例5: すべてを統合した最終ワークフロー
# ============================================================================

async def complete_integrated_workflow():
    """
    すべての機能を統合した完全ワークフロー
    """
    print("\n" + "="*60)
    print("COMPLETE INTEGRATED WORKFLOW")
    print("="*60)

    # 1. YouTube研究動画処理
    await youtube_research_workflow()

    # 2. 学術論文収集・分析
    await academic_paper_workflow()

    # 3. 統合大規模収集
    await unified_large_scale_workflow()

    # 4. Webから動画処理
    await web_to_video_workflow()

    print("\n" + "="*60)
    print("✅ ALL WORKFLOWS COMPLETED SUCCESSFULLY!")
    print("="*60)


# ============================================================================
# メイン実行
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        workflow = sys.argv[1]

        if workflow == "youtube":
            asyncio.run(youtube_research_workflow())
        elif workflow == "paper":
            asyncio.run(academic_paper_workflow())
        elif workflow == "unified":
            asyncio.run(unified_large_scale_workflow())
        elif workflow == "web2video":
            asyncio.run(web_to_video_workflow())
        elif workflow == "all":
            asyncio.run(complete_integrated_workflow())
        else:
            print("Usage: python complete_workflow.py [youtube|paper|unified|web2video|all]")
    else:
        # デフォルト: 全ワークフロー実行
        asyncio.run(complete_integrated_workflow())
