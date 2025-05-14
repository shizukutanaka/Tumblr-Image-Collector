# Tumblr Image Collector

## 概要

高度な画像収集と分類を行うPythonベースのTumblrイメージコレクター。

## 主な機能

- 複数のTumblrブログから画像を自動収集
- 高度な画像フィルタリング
- 画像分類と重複排除
- メタデータ抽出と管理

## 必要要件

- Python 3.9+
- pip
- 仮想環境 (venv推奨)

## インストール

1. リポジトリをクローン
```bash
git clone https://github.com/yourusername/tumblr-image-collector.git
cd tumblr-image-collector
```

2. 仮想環境を作成・有効化
```bash
python -m venv venv
source venv/bin/activate  # Linuxの場合
venv\Scripts\activate    # Windowsの場合
```

3. 依存関係をインストール
```bash
pip install -r requirements.txt
```

4. 環境変数を設定
`.env.template`を`.env`にコピーし、必要な認証情報を入力

## 使用方法

```python
from tumblr_image_collector import TumblrImageCollector

collector = TumblrImageCollector()
collection_results = collector.auto_image_collection({
    'blogs': ['example.tumblr.com'],
    'tags': ['art'],
    'max_images': 100
})
```

## テスト

```bash
pytest test_tumblr_image_collector.py
```

## ライセンス

MIT License

## 貢献

プルリクエストは歓迎します。大きな変更を行う前に、まずissueで議論してください。

## 主な特徴 🚀

### 🤖 高度な画像インテリジェンス
- AI支援による画像スタイル分類
- セマンティック検索
- 色ベースの画像検索
- 高度な重複検出

### 🌐 マルチブログ/タグ検索
- 複数のブログを横断した画像収集
- 詳細なフィルタリングオプション
- カスタマイズ可能な検索パラメータ

### 🔒 セキュリティと設定管理
- 高度な設定ウィザード
- プロキシサポート
- 安全な認証情報管理
- 柔軟な接続設定

### 🖼️ 画像処理機能
- 並列ダウンロード
- サムネイル生成
- 画像品質評価
- 高度な重複検出

## インストール 📦

### 必要な依存関係
- Python 3.9+
- TensorFlow
- Keras
- Pillow
- Requests
- ImageHash
- NumPy

### 推奨環境
```bash
pip install -r requirements.txt
```

## 設定方法 🛠️

### 設定ウィザードの使用
```python
# 完全な設定
collector.advanced_configuration_wizard(config_type='full')

# 特定の設定のみ更新
collector.advanced_configuration_wizard(config_type='network')
```

### 画像検索の例
```python
# マルチブログ検索
results = collector.multi_blog_search(
    blogs=['blog1.tumblr.com', 'blog2.tumblr.com'],
    tags=['art', 'illustration'],
    search_params={
        'min_likes': 10,
        'date_range': {'start': datetime(2024, 1, 1)}
    }
)

# AI支援による画像検索
semantic_results = collector.advanced_image_search(
    query='beautiful landscape',
    search_type='semantic'
)
```

## セキュリティとプライバシー 🔐
- 認証情報は安全に管理
- プロキシ設定でネットワークセキュリティを強化
- 詳細なログ記録と監査

## 注意事項 ⚠️
- AIモデルの予測は100%正確ではありません
- 大量の画像処理には高性能なハードウェアを推奨
- 継続的な機能改善と更新を予定

## ライセンス 📄
MIT License

## コントリビューション 🤝
プルリクエストや機能提案を歓迎します！

## AI画像分類の詳細 🧠

### 画像分類の高度な機能
- 解像度フィルタリング
- NSFWコンテンツ検出
- 画像の種類と信頼度の推定
- マルチスタイル分類

### 使用技術
- TensorFlow
- Keras
- MobileNetV2モデル
- CLIP（Contrastive Language-Image Pre-training）

### 画像分類の主な機能
1. 画像の最小解像度チェック
2. NSFWコンテンツ検出
3. 画像の分類と信頼度スコア
4. 高度なメタデータ抽出
5. スタイルベースの分類

### 設定と拡張性
`image_classifier.py`と`tumblr_image_collector.py`で以下をカスタマイズ可能:
- `nsfw_threshold`: NSFWコンテンツ検出の閾値
- `min_resolution`: 最小許容解像度
- スタイル分類のカテゴリ

### 使用例

#### 画像スタイル分類
#### 自動画像収集
```python
# 高度な画像収集パラメータ
collection_params = {
    'blogs': ['art.tumblr.com', 'illustration.tumblr.com'],
    'tags': ['anime', 'digital art'],
    'max_images': 50,
    'min_resolution': (1024, 768),
    'min_likes': 10,
    'date_range': {
        'start': datetime(2024, 1, 1),
        'end': datetime(2024, 12, 31)
    },
    'style_filters': ['anime', 'digital_art'],
    'download_options': {
        'output_directory': './anime_art_collection',
        'naming_pattern': 'anime_art_{blog}_{timestamp}_{index}'
    },
    'advanced_filters': {
        'color_palette': [(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        'entropy_threshold': 3.0,
        'aspect_ratio_range': (0.75, 1.5)
    }
}

# 画像を自動収集
collection_results = collector.auto_image_collection(collection_params)

# 収集結果の分析
print(f"Total images found: {collection_results['total_found']}")
print(f"Downloaded images: {len(collection_results['downloaded_images'])}")
print(f"Skipped images: {len(collection_results['skipped_images'])}")

# 前回の収集結果を読み込んで再開
previous_results = collector._load_last_collection_state()

# カスタム再開パラメータ
resume_params = {
    'extend_date_range': True,
    'skip_downloaded_images': True,
    'max_retry_count': 5,
    'retry_delay': 120  # 2分
}

# 画像収集を再開
resumed_collection_results = collector.resume_image_collection(
    previous_collection_results=previous_results,
    additional_params=resume_params
)

# 再開された収集結果の分析
print(f"Total images found: {resumed_collection_results['total_found']}")
print(f"Total downloaded images: {len(resumed_collection_results['downloaded_images'])}")
print(f"Total errors: {len(resumed_collection_results['errors'])}")
