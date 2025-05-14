import pytest
import os
import sys
import json

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tumblr_image_collector import TumblrImageCollector

@pytest.fixture
def collector():
    """TumblrImageCollectorのインスタンスを作成"""
    return TumblrImageCollector()

def test_auto_image_collection(collector):
    """自動画像収集機能のテスト"""
    collection_params = {
        'blogs': ['example.tumblr.com'],
        'tags': ['test'],
        'max_images': 5,
        'min_resolution': (100, 100),
        'min_likes': 1,
        'download_options': {
            'output_directory': './test_downloads',
            'overwrite': True
        }
    }
    
    results = collector.auto_image_collection(collection_params)
    
    assert 'total_found' in results
    assert 'downloaded_images' in results
    assert 'skipped_images' in results
    assert 'errors' in results

def test_resume_image_collection(collector, tmp_path):
    """ダウンロード再開機能のテスト"""
    # テスト用の状態ファイルを作成
    test_state = {
        'total_found': 10,
        'downloaded_images': ['image1.jpg', 'image2.jpg'],
        'skipped_images': [],
        'errors': [],
        'collection_params': {
            'blogs': ['example.tumblr.com'],
            'tags': ['test']
        }
    }
    
    state_file = tmp_path / 'test_collection_state.json'
    with open(state_file, 'w') as f:
        json.dump(test_state, f)
    
    # モックされた状態ファイルのパスを設定
    with pytest.monkeypatch.context() as m:
        m.setattr(collector, '_load_last_collection_state', 
                  lambda state_file='last_collection_state.json': test_state)
        
        resume_params = {
            'extend_date_range': True,
            'skip_downloaded_images': True
        }
        
        results = collector.resume_image_collection(
            additional_params=resume_params
        )
    
    assert results is not None
    assert 'total_found' in results
    assert 'downloaded_images' in results

def test_image_details_analysis(collector):
    """画像詳細分析メソッドのテスト"""
    # テスト用の画像URLを指定（実際の環境に合わせて調整）
    test_image_url = 'https://example.com/test_image.jpg'
    
    image_details = collector._analyze_image_details(test_image_url)
    
    assert image_details is not None
    assert 'url' in image_details
    assert 'width' in image_details
    assert 'height' in image_details
    assert 'aspect_ratio' in image_details

def test_advanced_filters(collector):
    """高度な画像フィルタリングのテスト"""
    # テスト用の画像情報を作成
    image_info = {
        'width': 800,
        'height': 600,
        'aspect_ratio': 1.33,
        'local_path': './test_image.jpg'
    }
    
    params = {
        'min_resolution': (500, 500),
        'advanced_filters': {
            'aspect_ratio_range': (1.0, 1.5),
            'entropy_threshold': 2.0,
            'color_palette': None
        }
    }
    
    result = collector._apply_advanced_filters(image_info, params)
    
    assert result is True

def test_save_load_collection_state(collector, tmp_path):
    """収集状態の保存と読み込みのテスト"""
    test_results = {
        'total_found': 15,
        'downloaded_images': ['image1.jpg', 'image2.jpg', 'image3.jpg'],
        'skipped_images': ['image4.jpg'],
        'errors': []
    }
    
    # 状態を保存
    state_file = tmp_path / 'collection_state.json'
    with pytest.monkeypatch.context() as m:
        m.setattr(os, 'getcwd', lambda: str(tmp_path))
        
        collector._save_collection_state(test_results, str(state_file))
        
        # 状態を読み込む
        loaded_state = collector._load_last_collection_state(str(state_file))
    
    assert loaded_state is not None
    assert loaded_state['total_found'] == 15
    assert len(loaded_state['downloaded_images']) == 3

def test_merge_collection_results(collector):
    """収集結果のマージのテスト"""
    previous_results = {
        'total_found': 10,
        'downloaded_images': ['image1.jpg', 'image2.jpg'],
        'skipped_images': ['image3.jpg'],
        'errors': [],
        'collection_params': {'blogs': ['blog1.tumblr.com']}
    }
    
    new_results = {
        'total_found': 5,
        'downloaded_images': ['image4.jpg', 'image5.jpg'],
        'skipped_images': ['image6.jpg'],
        'errors': ['network error'],
        'collection_params': {'tags': ['art']}
    }
    
    merged_results = collector._merge_collection_results(previous_results, new_results)
    
    assert merged_results['total_found'] == 15
    assert len(merged_results['downloaded_images']) == 4
    assert len(merged_results['skipped_images']) == 2
    assert len(merged_results['errors']) == 1
    assert 'blogs' in merged_results['collection_params']
    assert 'tags' in merged_results['collection_params']

if __name__ == '__main__':
    pytest.main([__file__])
