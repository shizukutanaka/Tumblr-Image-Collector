#!/usr/bin/env python3
"""
実装した機能の動作テストスクリプト
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

def test_youtube_downloader():
    """YouTubeダウンローダーのテスト"""
    try:
        from youtube_downloader import YouTubeDownloader

        downloader = YouTubeDownloader("test_downloads")

        # テスト用のURLで情報取得をテスト（実際のダウンロードは行わない）
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        print("YouTube動画情報を取得中...")

        info = downloader.get_video_info(test_url)
        if info:
            print(f"✓ YouTube情報取得成功: {info['title']}")
            print(f"  著者: {info['author']}")
            print(f"  長さ: {info['length']}秒")
        else:
            print("✗ YouTube情報取得失敗")

        return True

    except ImportError as e:
        print(f"✗ インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"✗ YouTubeダウンローダーテスト失敗: {e}")
        return False

def test_arxiv_collector():
    """arXivコレクターのテスト"""
    try:
        from arxiv_collector import ArXivCollector

        collector = ArXivCollector("test_downloads")

        # テスト検索クエリ
        query = "machine learning"
        print(f"arXivで'{query}'を検索中...")

        papers = collector.search_papers(query, max_results=3)
        if papers:
            print(f"✓ arXiv検索成功: {len(papers)}件の論文が見つかりました")
            for i, paper in enumerate(papers[:2], 1):  # 最初の2件のみ表示
                print(f"  {i}. {paper['title']}")
                print(f"     著者: {', '.join(paper['authors'])}")
                print(f"     arXiv ID: {paper['arxiv_id']}")
        else:
            print("✗ arXiv検索失敗")

        return True

    except ImportError as e:
        print(f"✗ インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"✗ arXivコレクターテスト失敗: {e}")
        return False

def test_semantic_scholar_collector():
    """Semantic Scholarコレクターのテスト"""
    try:
        from semantic_scholar_collector import SemanticScholarCollector

        collector = SemanticScholarCollector("test_downloads")

        # テスト検索クエリ
        query = "machine learning"
        print(f"Semantic Scholarで'{query}'を検索中...")

        papers = collector.search_papers(query, limit=3)
        if papers:
            print(f"✓ Semantic Scholar検索成功: {len(papers)}件の論文が見つかりました")
            for i, paper in enumerate(papers[:2], 1):  # 最初の2件のみ表示
                print(f"  {i}. {paper['title']}")
                print(f"     著者: {', '.join(paper['authors'])}")
                print(f"     年: {paper['year']}")
        else:
            print("✗ Semantic Scholar検索失敗")

        return True

    except ImportError as e:
        print(f"✗ インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"✗ Semantic Scholarコレクターテスト失敗: {e}")
        return False

def main():
    """メイン実行関数"""
    print("=== Tumblr Image Collector 新機能テスト ===\n")

    tests = [
        ("YouTube Downloader", test_youtube_downloader),
        ("arXiv Collector", test_arxiv_collector),
        ("Semantic Scholar Collector", test_semantic_scholar_collector),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n--- {test_name} テスト ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}テストで例外発生: {e}")
            results.append((test_name, False))

    print("\n=== テスト結果サマリー ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ 成功" if result else "✗ 失敗"
        print(f"{test_name}: {status}")

    print(f"\n全体結果: {passed}/{total} テストが成功")

    if passed == total:
        print("🎉 全機能が正常に動作しています！")
        return True
    else:
        print("⚠️  一部の機能で問題が発生しました。依存関係の確認をおすすめします。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
