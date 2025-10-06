# Tumblr Image Collector - Personal Edition

**個人使用に最適化された、セキュリティと機能を最大化したバージョン**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Personal Edition](https://img.shields.io/badge/Edition-Personal-green.svg)]()

---

## 🎯 Personal Editionの特徴

個人使用に特化した、セキュリティと利便性を両立した最強バージョンです。

### 🔐 強化されたセキュリティ

- **AES-256暗号化**: 認証情報の完全暗号化
- **マスターパスワード**: PBKDF2による安全な鍵派生（100,000イテレーション）
- **システムキーリング統合**: Windows/macOS/Linuxの安全な認証情報管理
- **プライバシーモード**: ログの自動サニタイゼーションと古いデータの削除
- **セキュア削除**: ファイル上書き後の削除
- **整合性チェック**: ファイル改ざん検出

### ⚡ 個人用最適化

- **自動パフォーマンスチューニング**: システムリソースに基づく自動調整
- **アダプティブワーカー**: CPU・メモリに応じた動的ワーカー数調整
- **アグレッシブキャッシング**: 最大2GBのディスクキャッシュ
- **バックグラウンド監視**: リアルタイムリソース監視と自動最適化
- **メモリ最適化**: 自動ガベージコレクションとメモリ解放

### 🎨 便利な個人用機能

- **お気に入り管理**: よく使うブログを登録して一括管理
- **自動整理**: 日付別・タグ別の自動フォルダ分け
- **スマートコレクション**: 条件に基づく自動分類
- **スケジュール機能**: 毎日・毎週・毎月の自動ダウンロード
- **ブロックリスト**: 不要なブログを除外
- **サムネイル自動生成**: 高速プレビュー
- **重複自動処理**: 重複画像の自動検出と移動
- **自動バックアップ**: 24時間ごとの自動バックアップ
- **壁紙コレクション**: 高解像度画像の自動抽出
- **統計とレポート**: 詳細な分析とエクスポート

### 📚 パーソナルライブラリ

- **SQLiteデータベース**: 全画像のメタデータ管理
- **高度な検索**: タグ、品質、ブログ名での検索
- **お気に入り機能**: 画像単位のお気に入り
- **レーティング**: 画像の5段階評価
- **メモ機能**: 各画像にメモを追加
- **コレクション**: カスタムコレクションの作成

---

## 🚀 クイックスタート

### インストール

```bash
# 1. リポジトリのクローン
git clone https://github.com/shizukutanaka/Tumblr-Image-Collector.git
cd Tumblr-Image-Collector

# 2. 仮想環境の作成と有効化
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. 依存関係のインストール
pip install --upgrade pip
pip install -r requirements.txt

# 4. 個人用設定のコピー
cp config_personal.json config.json

# 5. セキュア設定ウィザードの実行
python config.py
```

### 初回セットアップ

```bash
# マスターパスワードの設定
# （初回実行時に自動的に促されます）
python3 -c "from personal_security import get_security_manager; \
            import json; \
            config = json.load(open('config.json')); \
            security = get_security_manager('.', config); \
            security.encrypt_credentials('YOUR_KEY', 'YOUR_SECRET')"
```

---

## 💻 基本的な使い方

### シンプルな使用例

```bash
# お気に入りブログからダウンロード
python tumblr_image_collector.py favorite-blog

# 高品質画像のみ（1920x1080以上、品質0.8以上）
python tumblr_image_collector.py art-blog \
  --min-resolution 1920x1080 \
  --min-quality 0.8

# タグフィルター付き
python tumblr_image_collector.py photography-blog \
  --tags landscape nature sunset
```

### Python APIの使用

```python
from personal_features import get_personal_manager
from personal_security import get_security_manager
from personal_optimizer import get_optimizer
from personal_convenience import get_convenience_features
import json

# 設定読み込み
with open('config.json') as f:
    config = json.load(f)

# 1. セキュリティマネージャー
security = get_security_manager("./tumblr_images", config)
credentials = security.decrypt_credentials()

# 2. パフォーマンス最適化
optimizer = get_optimizer(config)
optimizer.auto_tune()  # 自動チューニング

# 3. 便利機能
convenience = get_convenience_features("./tumblr_images", config)

# お気に入りブログを追加
convenience.add_favorite(
    blog_name="my-favorite-blog",
    tags=["art", "illustration"],
    auto_download=True,
    notes="素晴らしいイラストレーター"
)

# 毎日3時に自動ダウンロード
convenience.schedule_download(
    blog_name="my-favorite-blog",
    schedule_type="daily",
    time="03:00"
)

# 4. ライブラリ管理
manager = get_personal_manager("./tumblr_images", config)

# 画像を追加
manager.add_image_to_library(
    image_path=Path("downloaded_image.jpg"),
    metadata={
        'blog_name': 'my-favorite-blog',
        'tags': ['art', 'digital'],
        'quality_score': 0.95
    }
)

# 検索
results = manager.search_images(
    tags=['art'],
    min_quality=0.8
)

# 統計
stats = manager.get_statistics()
print(f"総画像数: {stats['total_images']}")
print(f"合計サイズ: {stats['total_storage_mb']}MB")
```

---

## 🔧 設定例

### 最高セキュリティ設定

```json
{
  "security": {
    "enable_encryption": true,
    "encrypt_credentials": true,
    "secure_delete": true,
    "enable_privacy_mode": true,
    "clear_logs_after_days": 7,
    "strip_metadata": true
  }
}
```

### 最高パフォーマンス設定

```json
{
  "max_download_workers": 20,
  "cache": {
    "enabled": true,
    "ttl_seconds": 604800,
    "max_entries": 10000,
    "disk_cache_size_mb": 4096,
    "enable_aggressive_caching": true
  }
}
```

### 最高品質のみ収集

```json
{
  "filters": {
    "min_resolution": [2560, 1440],
    "min_file_size_kb": 500,
    "max_file_size_mb": 100,
    "min_quality_score": 0.85,
    "enable_quality_filter": true
  }
}
```

### 完全自動化設定

```json
{
  "personal_features": {
    "auto_organize_by_date": true,
    "auto_organize_by_tags": true,
    "auto_backup": true,
    "backup_interval_hours": 12,
    "create_thumbnails": true,
    "auto_tag_images": true,
    "duplicate_action": "move_to_duplicates",
    "enable_smart_collections": true,
    "auto_cleanup_temp_files": true
  }
}
```

---

## 📊 生成されるフォルダ構造

```
tumblr_images/
├── images/                    # オリジナル画像
│   ├── blog1/
│   └── blog2/
├── by_date/                  # 日付別整理
│   ├── 2025/
│   │   ├── 01/
│   │   └── 02/
├── by_tags/                  # タグ別整理
│   ├── art/
│   ├── photography/
│   └── nature/
├── duplicates/               # 重複画像
├── thumbnails/               # サムネイル
│   └── (images配下と同じ構造)
├── backups/                  # 自動バックアップ
│   ├── backup_20250105_030000/
│   └── backup_20250106_030000/
├── wallpapers/               # 壁紙用高解像度画像
├── favorites/                # お気に入り画像
├── high_quality/             # 高品質画像（0.9以上）
├── metadata/                 # メタデータ
├── shortcuts/                # クイックアクセス
├── .security/                # セキュリティデータ（暗号化）
│   ├── master.key
│   └── credentials.enc
├── personal_library.db       # SQLiteデータベース
├── favorites.json            # お気に入りブログ
├── blocklist.json           # ブロックリスト
└── schedule.json            # スケジュール
```

---

## 🎯 実用的なユースケース

### ケース1: アートコレクター

```python
# お気に入りアーティストを登録
artists = [
    "artist1-blog",
    "artist2-blog",
    "digital-art-hub"
]

for artist in artists:
    convenience.add_favorite(
        blog_name=artist,
        tags=["illustration", "artwork"],
        auto_download=True
    )

    # 毎日深夜3時に自動ダウンロード
    convenience.schedule_download(
        blog_name=artist,
        schedule_type="daily",
        time="03:00"
    )

# 高品質フィルター設定
config['filters']['min_quality_score'] = 0.9
config['filters']['min_resolution'] = [1920, 1080]
```

### ケース2: 写真コレクター

```python
# 写真ブログを整理
photo_blogs = [
    {"name": "landscape-photos", "tags": ["landscape", "nature"]},
    {"name": "portrait-art", "tags": ["portrait", "people"]},
    {"name": "street-photography", "tags": ["street", "urban"]}
]

for blog in photo_blogs:
    manager.add_image_to_library(...)

# 壁紙コレクション作成（4K以上）
wallpapers = convenience.create_wallpaper_collection(
    min_resolution=(3840, 2160)
)
```

### ケース3: リサーチャー

```python
# 特定テーマの画像を収集
research_config = {
    'filters': {
        'tags': ['research-topic', 'academic'],
        'save_post_metadata': True,
        'save_comments': True
    }
}

# メタデータ付きで保存
manager.add_image_to_library(image, {
    'source_url': url,
    'blog_name': blog,
    'tags': tags,
    'notes': '研究資料として重要'
})

# 統計エクスポート
export_file = convenience.export_library_stats(format="json")
```

---

## 🔒 セキュリティベストプラクティス

### 1. 認証情報の保護

```bash
# 環境変数を使用（推奨）
export TUMBLR_CONSUMER_KEY="your_key"
export TUMBLR_CONSUMER_SECRET="your_secret"

# または暗号化保存
python3 -c "from personal_security import get_security_manager; \
            security = get_security_manager('.', config); \
            security.encrypt_credentials('key', 'secret')"
```

### 2. マスターパスワードの管理

- 8文字以上の強力なパスワードを使用
- システムキーリングに自動保存
- 定期的な変更を推奨

### 3. プライバシーモード

```json
{
  "security": {
    "enable_privacy_mode": true,
    "clear_logs_after_days": 30,
    "secure_delete": true
  }
}
```

### 4. 定期的な整合性チェック

```python
# 週次で実行
report = security.generate_integrity_report()
# 保存して次回比較
```

---

## ⚡ パフォーマンスチューニング

### 自動最適化（推奨）

```python
from personal_optimizer import get_optimizer

optimizer = get_optimizer(config)
optimizer.auto_tune()  # すべて自動設定

# パフォーマンスレポート
report = optimizer.get_performance_report()
print(f"最適ワーカー数: {report['current']['optimal_workers']}")
print(f"メモリ使用量: {report['current']['memory_mb']}MB")

# 推奨事項
for rec in report['recommendations']:
    print(f"💡 {rec}")
```

### 手動チューニング

```json
{
  "max_download_workers": 15,
  "cache": {
    "max_entries": 10000,
    "disk_cache_size_mb": 2048
  }
}
```

---

## 📈 統計と分析

### クイック統計

```python
stats = convenience.get_quick_stats()
# 結果:
# {
#   "total_images": 5420,
#   "total_size_mb": 12840.5,
#   "favorite_blogs": 15,
#   "blocked_blogs": 3,
#   "scheduled_tasks": 10
# }
```

### 詳細分析

```python
detailed = manager.get_statistics()
# 結果:
# {
#   "total_images": 5420,
#   "favorite_images": 432,
#   "duplicate_images": 156,
#   "average_quality": 0.82,
#   "total_storage_mb": 12840.5,
#   "total_tags": 234,
#   "total_collections": 8,
#   "recent_downloads": [...]
# }
```

---

## 🐛 トラブルシューティング

### パスワードを忘れた

```bash
rm -rf .security/
python config.py  # 再設定
```

### パフォーマンスが遅い

```python
optimizer = get_optimizer(config)
optimizer.auto_tune()
optimizer.cleanup_memory()
```

### ディスク容量不足

```python
# 古いバックアップ削除
manager.cleanup_old_backups(keep_count=5)

# 重複画像削除
duplicates_dir = Path("tumblr_images/duplicates")
# 確認後削除
```

---

## 📚 詳細ドキュメント

- [個人使用ガイド](PERSONAL_USER_GUIDE.md) - 完全な使用方法
- [API リファレンス](API_REFERENCE.md) - プログラミングAPI
- [インストールガイド](INSTALLATION_GUIDE.md) - 詳細なセットアップ
- [セキュリティガイド](SECURITY_IMPROVEMENTS.md) - セキュリティ詳細

---

## 🎁 Personal Edition 限定機能まとめ

| 機能 | 説明 |
|------|------|
| 🔐 AES-256暗号化 | 認証情報の完全暗号化 |
| 🔑 システムキーリング | OS統合の安全な認証情報管理 |
| 🛡️ プライバシーモード | ログのサニタイゼーションと自動削除 |
| ⚡ 自動最適化 | システムリソースに基づく自動調整 |
| 📚 パーソナルライブラリ | SQLiteによる全画像管理 |
| ⭐ お気に入り管理 | ブログと画像のお気に入り |
| 📅 スケジュール | 自動ダウンロードのスケジューリング |
| 🚫 ブロックリスト | 不要なブログの除外 |
| 📁 自動整理 | 日付・タグ別の自動分類 |
| 🖼️ サムネイル | 自動サムネイル生成 |
| 💾 自動バックアップ | 24時間ごとの自動バックアップ |
| 🎨 スマートコレクション | 条件ベースの自動分類 |
| 🖥️ 壁紙コレクション | 高解像度画像の自動抽出 |
| 📊 詳細統計 | 包括的な分析とレポート |
| 🔍 高度な検索 | タグ・品質・ブログでの検索 |

---

## 📞 サポート

- ドキュメント: プロジェクトのREADMEとガイドを参照
- Issues: バグ報告や機能リクエスト
- Community: 議論や質問

---

**Tumblr Image Collector Personal Edition - 個人使用に最適化された最強バージョン**

セキュリティを犠牲にせず、最大限の便利さと機能を提供します。
