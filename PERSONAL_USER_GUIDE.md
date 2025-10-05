# 個人使用ガイド - Tumblr Image Collector

**バージョン**: 2.0.0 Personal Edition
**対象**: 個人ユーザー向け最適化版

---

## 🎯 概要

このガイドは、個人使用に最適化されたTumblr Image Collectorの使用方法を説明します。セキュリティを最大限に保ちながら、便利な機能をフル活用できます。

---

## 🚀 クイックスタート

### 初回セットアップ

```bash
# 1. リポジトリのクローン
git clone https://github.com/shizukutanaka/Tumblr-Image-Collector.git
cd Tumblr-Image-Collector

# 2. 仮想環境の作成
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. 依存関係のインストール
pip install -r requirements.txt
pip install cryptography keyring  # 個人用セキュリティ機能

# 4. 個人用設定の作成
cp config_personal.json config.json

# 5. セキュア設定ウィザードの実行
python config.py
```

初回起動時、マスターパスワードの設定が求められます。このパスワードで認証情報が暗号化されます。

---

## 🔐 セキュリティ機能（個人用）

### 認証情報の暗号化

```python
from personal_security import get_security_manager

# セキュリティマネージャーの初期化
security = get_security_manager("./tumblr_images", config)

# 認証情報の暗号化保存
security.encrypt_credentials(
    consumer_key="your_key",
    consumer_secret="your_secret"
)

# 認証情報の復号化
credentials = security.decrypt_credentials()
```

**特徴**:
- AES-256暗号化
- PBKDF2によるパスワード派生（100,000イテレーション）
- システムキーリング統合（Windows/macOS/Linux）
- ファイルパーミッション自動設定（Unix系）

### プライバシーモード

```json
{
  "security": {
    "enable_privacy_mode": true,
    "clear_logs_after_days": 30,
    "secure_delete": true,
    "strip_metadata": false
  }
}
```

**機能**:
- ログの自動サニタイゼーション
- 30日後の古いログ自動削除
- セキュアな削除（上書き後削除）
- 制限的なファイルパーミッション

### ファイル整合性チェック

```python
# 整合性レポートの生成
report = security.generate_integrity_report()

# 後で検証
is_valid, issues = security.verify_integrity(report)
if not is_valid:
    print("警告: ファイルが改ざんされた可能性があります")
    for issue in issues:
        print(f"  - {issue}")
```

---

## 🎨 個人用機能

### お気に入りブログ管理

```python
from personal_convenience import get_convenience_features

convenience = get_convenience_features("./tumblr_images", config)

# お気に入りに追加
convenience.add_favorite(
    blog_name="example-blog",
    tags=["art", "photography"],
    auto_download=True,
    notes="素晴らしいアート作品"
)

# お気に入り一覧の取得
favorites = convenience.get_favorites()

# お気に入りブログの一括ダウンロード
for fav in favorites:
    if fav['auto_download']:
        download_from_blog(fav['blog_name'], fav['tags'])
```

### ブロックリスト

```python
# ブロックリストに追加
convenience.add_to_blocklist("unwanted-blog")

# ブロック確認
if convenience.is_blocked("unwanted-blog"):
    print("このブログはブロックされています")
```

### 自動整理

```json
{
  "personal_features": {
    "auto_organize_by_date": true,    // 日付別フォルダ
    "auto_organize_by_tags": true,    // タグ別フォルダ
    "create_thumbnails": true,        // サムネイル自動生成
    "thumbnail_size": [400, 400],
    "duplicate_action": "move_to_duplicates"  // 重複処理
  }
}
```

**フォルダ構造**:
```
tumblr_images/
├── images/              # オリジナル画像
├── by_date/            # 日付別
│   ├── 2025/
│   │   ├── 01/
│   │   └── 02/
├── by_tags/            # タグ別
│   ├── art/
│   ├── photography/
├── duplicates/         # 重複画像
├── thumbnails/         # サムネイル
└── backups/           # 自動バックアップ
```

### スケジュール機能

```python
# 毎日3時にダウンロード
convenience.schedule_download(
    blog_name="favorite-blog",
    schedule_type="daily",
    time="03:00",
    tags=["new", "updates"]
)

# スケジュール一覧
schedules = convenience.schedule
```

**静音時間**:
```json
{
  "personal_features": {
    "quiet_hours_start": "22:00",
    "quiet_hours_end": "08:00"
  }
}
```

---

## ⚡ パフォーマンス最適化（個人用）

### 自動チューニング

```python
from personal_optimizer import get_optimizer

optimizer = get_optimizer(config)

# 自動最適化
optimizer.auto_tune()

# 結果
# - 最適なワーカー数: 10 → 15
# - システム設定の最適化完了
# - バックグラウンド監視開始
```

**最適化内容**:
- CPU数に基づく自動ワーカー調整
- 利用可能メモリに基づくキャッシュサイズ
- ファイルディスクリプタ制限の増加
- プロセス優先度の最適化

### リソース監視

```python
# パフォーマンスレポート
report = optimizer.get_performance_report()

print(f"現在のメモリ使用量: {report['current']['memory_mb']}MB")
print(f"CPU使用率: {report['current']['cpu_percent']}%")
print(f"最適ワーカー数: {report['current']['optimal_workers']}")

# 推奨事項
for recommendation in report['recommendations']:
    print(f"💡 {recommendation}")
```

### アグレッシブキャッシング

```json
{
  "cache": {
    "enabled": true,
    "ttl_seconds": 604800,           // 7日間
    "max_entries": 10000,
    "disk_cache_size_mb": 2048,      // 2GB
    "enable_aggressive_caching": true
  }
}
```

---

## 📚 高度な機能

### パーソナルライブラリ

すべての画像がSQLiteデータベースで管理されます。

```python
from personal_features import get_personal_manager

manager = get_personal_manager("./tumblr_images", config)

# 画像をライブラリに追加
image_id = manager.add_image_to_library(
    image_path=Path("image.jpg"),
    metadata={
        'source_url': 'https://example.tumblr.com/post/123',
        'blog_name': 'example',
        'tags': ['art', 'digital'],
        'quality_score': 0.92
    }
)

# 検索
results = manager.search_images(
    tags=['art'],
    min_quality=0.8,
    favorites_only=False
)

# 統計
stats = manager.get_statistics()
print(f"総画像数: {stats['total_images']}")
print(f"お気に入り: {stats['favorite_images']}")
print(f"合計サイズ: {stats['total_storage_mb']}MB")
```

### スマートコレクション

```python
# コレクションの作成
manager.create_collection(
    name="Best of 2025",
    description="2025年のベスト画像",
    criteria={'min_quality': 0.9, 'tags': ['featured']}
)

# 自動追加ルール
# quality_score >= 0.9 の画像が自動的に追加されます
```

### 自動バックアップ

```json
{
  "personal_features": {
    "auto_backup": true,
    "backup_interval_hours": 24
  }
}
```

```python
# 手動バックアップ
backup_path = manager.create_backup()
print(f"バックアップ作成: {backup_path}")

# バックアップのクリーンアップ（最新10個を保持）
manager.cleanup_old_backups(keep_count=10)
```

### 壁紙コレクション

```python
# 高解像度画像から壁紙コレクションを作成
wallpaper_dir = convenience.create_wallpaper_collection(
    min_resolution=(1920, 1080)
)

# 4K壁紙
wallpaper_4k = convenience.create_wallpaper_collection(
    min_resolution=(3840, 2160)
)
```

---

## 🎯 実用例

### 例1: お気に入りブログの毎日ダウンロード

```python
# setup.py
from personal_convenience import get_convenience_features

convenience = get_convenience_features("./tumblr_images", config)

# お気に入りブログを登録
favorite_blogs = [
    {"name": "art-daily", "tags": ["illustration", "digital"]},
    {"name": "photo-blog", "tags": ["landscape", "nature"]},
    {"name": "design-hub", "tags": ["graphic", "ui"]}
]

for blog in favorite_blogs:
    convenience.add_favorite(
        blog_name=blog["name"],
        tags=blog["tags"],
        auto_download=True
    )

    # 毎日3時にダウンロード
    convenience.schedule_download(
        blog_name=blog["name"],
        schedule_type="daily",
        time="03:00",
        tags=blog["tags"]
    )
```

### 例2: セキュアな個人コレクション

```python
# secure_collection.py
from personal_security import get_security_manager
from personal_features import get_personal_manager

# セキュリティ設定
security = get_security_manager("./my_private_collection", config)

# 認証情報の暗号化
security.encrypt_credentials(
    consumer_key=os.getenv("TUMBLR_KEY"),
    consumer_secret=os.getenv("TUMBLR_SECRET")
)

# プライバシーモード有効化
security.enable_privacy_mode()

# ライブラリ管理
manager = get_personal_manager("./my_private_collection", config)

# 画像の自動整理
for image in downloaded_images:
    # ライブラリに追加
    manager.add_image_to_library(image, metadata)

    # 日付別整理
    manager.organize_by_date(image)

    # タグ別整理
    manager.organize_by_tags(image, metadata['tags'])

    # サムネイル作成
    manager.create_thumbnail(image)

# 毎日自動バックアップ
manager.create_backup()
```

### 例3: パフォーマンス最適化

```python
# optimized_download.py
from personal_optimizer import get_optimizer

optimizer = get_optimizer(config)

# 自動チューニング
optimizer.auto_tune()

# 監視開始
optimizer.start_monitoring()

# ダウンロード実行
# （ワーカー数は自動調整されます）
download_from_blog("large-blog")

# パフォーマンスレポート
report = optimizer.get_performance_report()
print(json.dumps(report, indent=2))

# 終了時
optimizer.stop_monitoring()
```

---

## 🔧 設定リファレンス

### 推奨個人用設定

```json
{
  "mode": "personal",
  "max_download_workers": 10,
  "enable_deep_model": true,

  "filters": {
    "max_file_size_mb": 50,
    "min_resolution": [1280, 720],
    "nsfw_threshold": 0.25,
    "enable_quality_filter": true,
    "min_quality_score": 0.6
  },

  "cache": {
    "enabled": true,
    "ttl_seconds": 604800,
    "max_entries": 10000,
    "disk_cache_size_mb": 2048,
    "enable_aggressive_caching": true
  },

  "personal_features": {
    "auto_organize_by_date": true,
    "auto_organize_by_tags": true,
    "auto_backup": true,
    "backup_interval_hours": 24,
    "create_thumbnails": true,
    "duplicate_action": "move_to_duplicates",
    "enable_smart_collections": true
  },

  "security": {
    "enable_encryption": true,
    "encrypt_credentials": true,
    "enable_privacy_mode": true,
    "clear_logs_after_days": 30
  },

  "ui": {
    "enable_interactive_mode": true,
    "enable_progress_bar": true,
    "enable_notifications": true,
    "color_output": true
  }
}
```

---

## 💡 ヒント＆テクニック

### ディスク容量の節約

```json
{
  "advanced": {
    "auto_enhance_images": false,
    "convert_to_format": "webp",    // WebP形式で30-50%削減
    "compression_quality": 85,
    "strip_metadata": true          // EXIFデータ削除
  }
}
```

### 最高品質の画像のみ

```json
{
  "filters": {
    "min_resolution": [2560, 1440],
    "min_quality_score": 0.8,
    "enable_quality_filter": true
  }
}
```

### バッテリー節約モード（ノートPC）

```json
{
  "max_download_workers": 3,
  "personal_features": {
    "quiet_hours_start": "09:00",
    "quiet_hours_end": "18:00"    // 日中は実行しない
  }
}
```

---

## 🚨 トラブルシューティング

### Q: マスターパスワードを忘れた

A: `.security/master.key`を削除して再設定してください。ただし、暗号化された認証情報は失われます。

```bash
rm -rf .security/
python config.py  # 再設定
```

### Q: パフォーマンスが遅い

A: 自動最適化を実行してください。

```python
from personal_optimizer import get_optimizer
optimizer = get_optimizer(config)
optimizer.auto_tune()

# レポート確認
report = optimizer.get_performance_report()
```

### Q: ディスク容量不足

A: 古いバックアップと重複画像を削除してください。

```python
# 古いバックアップ削除（最新5個保持）
manager.cleanup_old_backups(keep_count=5)

# 重複画像の確認
duplicates = Path("tumblr_images/duplicates")
print(f"重複画像: {len(list(duplicates.glob('*')))}件")
```

---

## 📊 統計とレポート

### クイック統計

```python
stats = convenience.get_quick_stats()
print(f"""
総画像数: {stats['total_images']}
合計サイズ: {stats['total_size_mb']}MB
お気に入りブログ: {stats['favorite_blogs']}
スケジュール: {stats['scheduled_tasks']}
""")
```

### 詳細統計

```python
detailed_stats = manager.get_statistics()
print(json.dumps(detailed_stats, indent=2, ensure_ascii=False))
```

### 統計のエクスポート

```python
# JSON形式
export_file = convenience.export_library_stats(format="json")

# テキスト形式
export_file = convenience.export_library_stats(format="txt")
```

---

## 🎓 ベストプラクティス

### 1. セキュリティ

- ✅ 認証情報は必ず暗号化
- ✅ プライバシーモードを有効化
- ✅ 定期的に整合性チェック
- ✅ 古いログは自動削除

### 2. パフォーマンス

- ✅ 自動チューニングを使用
- ✅ アグレッシブキャッシング有効化
- ✅ 重複画像は自動移動
- ✅ 定期的にメモリクリーンアップ

### 3. 整理

- ✅ 日付・タグ別の自動整理
- ✅ サムネイル自動生成
- ✅ スマートコレクション活用
- ✅ 定期的なバックアップ

### 4. 効率

- ✅ お気に入り機能で一括管理
- ✅ スケジュール機能で自動化
- ✅ ブロックリストで不要を除外
- ✅ クイック検索で即座にアクセス

---

## 📞 サポート

問題や質問がある場合:
- GitHub Issues: https://github.com/shizukutanaka/Tumblr-Image-Collector/issues
- GitHub Discussions: https://github.com/shizukutanaka/Tumblr-Image-Collector/discussions

---

**個人使用ガイド v2.0.0** - Tumblr Image Collector Personal Edition
