# APIリファレンス (API Reference)

## 目次 (Table of Contents)

1. [TumblrImageCollector クラス](#tumblrimagecollector-クラス)
2. [ImageClassifier クラス](#imageclassifier-クラス)
3. [ConfigWizard クラス](#configwizard-クラス)
4. [国際化システム](#国際化システム)
5. [ユーティリティ関数](#ユーティリティ関数)
6. [定数](#定数)
7. [例外クラス](#例外クラス)

## TumblrImageCollector クラス

### 概要
Tumblrブログから画像を収集・処理するメインクラスです。

### 初期化
```python
collector = TumblrImageCollector(
    config_file="config.json",
    output_dir_override=None,
    workers_override=None,
    proxy_config=None
)
```

#### パラメータ
- **config_file** (str): 設定ファイルのパス。デフォルトは"config.json"
- **output_dir_override** (str, optional): 出力ディレクトリのオーバーライド
- **workers_override** (int, optional): ワーカー数のオーバーライド
- **proxy_config** (dict, optional): プロキシ設定

### 主要メソッド (Core Methods)

#### `run(blog_name)`
指定されたブログから画像を収集します。

```python
collector.run("staff")
```

**パラメータ:**
- **blog_name** (str): 収集対象のブログ名

**戻り値:**
- None

#### `get_blog_posts(blog_name, limit=20, offset=0)`
ブログの投稿を取得します。

```python
posts = collector.get_blog_posts("staff", limit=50, offset=0)
```

**パラメータ:**
- **blog_name** (str): ブログ名
- **limit** (int): 取得する投稿数（デフォルト: 20）
- **offset** (int): オフセット（デフォルト: 0）

**戻り値:**
- list: 投稿データのリスト、またはNone（レート制限時）

#### `download_image(image_url, post_data=None)`
画像をダウンロードします。

```python
success = collector.download_image("https://example.com/image.jpg")
```

**パラメータ:**
- **image_url** (str): ダウンロードする画像のURL
- **post_data** (dict, optional): 投稿データ

**戻り値:**
- bool: ダウンロードの成功/失敗
- 保存先ディレクトリには、構造化ファイル名で保存された画像と同名の`*.json`メタデータファイルが作成されます。メタデータには分類結果、画質スコア、処理済み特徴量が含まれます。

#### `export_metadata(output_format='json')`
画像メタデータをエクスポートします。

```python
metadata_file = collector.export_metadata('json')
```
- **output_format** (str): エクスポート形式 ('json' または 'csv')

**戻り値:**
- Path: 生成されたメタデータファイルのパス

#### `_save_statistics()`
統計情報を保存します。

```python
collector._save_statistics()
```

**戻り値:**
- None

`download_statistics.json` と `download_statistics.csv` に書き出し、CSVには`metrics_summary`内の数値指標（例: `nsfw_score`）の件数・平均・最小・最大が含まれます。

### 設定メソッド (Configuration Methods)

{{ ... }}
#### `_setup_proxy()`
プロキシ設定を初期化します。

#### `_setup_logging()`
ログシステムを設定します。

#### `_create_requests_session(retries, backoff_factor)`
再試行機能付きのHTTPセッションを作成します。

### 画像処理メソッド (Image Processing Methods)

#### `_is_image_valid(image)`
画像の有効性をチェックします。

```python
is_valid = collector._is_image_valid(image)
```

**パラメータ:**
- **image** (PIL.Image): チェックする画像

**戻り値:**
- bool: 画像が有効かどうか

#### `_extract_image_metadata(image_path)`
画像メタデータを抽出します。

```python
metadata = collector._extract_image_metadata("path/to/image.jpg")
```

**パラメータ:**
- **image_path** (str): 画像ファイルのパス

**戻り値:**
- dict: 画像メタデータ（幅、高さ、ファイルサイズ、ファイル形式、知覚ハッシュ、AI分類結果など）。OpenCVとNumPyが利用可能な環境では`ai_classification.metrics.nsfw_score`が含まれ、`nsfw_threshold`以上の場合は`is_potentially_nsfw`が`True`になります。存在しないファイルを指定すると`None`を返します。

#### `_process_image_efficiently(image_path, max_dimension=2048)`
メモリ効率の良い画像処理を実行します。

```python
features = collector._process_image_efficiently("path/to/image.jpg")
```

**パラメータ:**
- **image_path** (str): 画像ファイルのパス
- **max_dimension** (int): 最大画像サイズ

**戻り値:**
- dict: 画像特徴量

#### `_calculate_image_quality(image)`
画像の品質を計算します。

```python
quality = collector._calculate_image_quality(image)
```

**パラメータ:**
- **image** (PIL.Image): 品質を計算する画像

**戻り値:**
- float: 品質スコア

### セキュリティメソッド (Security Methods)

#### `_validate_input(input_value, input_type, max_length=None, allowed_chars=None)`
入力値を検証します。

```python
is_valid = collector._validate_input("https://example.com", "url")
```

**パラメータ:**
- **input_value** (str): 検証する入力値
- **input_type** (str): 入力の種類 ('url', 'filename', 'path', 'text')
- **max_length** (int, optional): 最大長
- **allowed_chars** (str, optional): 許可する文字セット

**戻り値:**
- bool: 検証結果

#### `_sanitize_filename(filename)`
ファイル名をサニタイズします。

```python
safe_filename = collector._sanitize_filename("unsafe/file:name.jpg")
```

**パラメータ:**
- **filename** (str): サニタイズするファイル名

**戻り値:**
- str: サニタイズされたファイル名

#### `_check_rate_limit()`
レート制限をチェックします。

```python
can_request = collector._check_rate_limit()
```

**戻り値:**
- bool: リクエスト可能かどうか

### プロパティ (Properties)

#### `output_folder`
出力フォルダのパス。

```python
print(collector.output_folder)
```

#### `config`
設定辞書。

```python
print(collector.config)
```

#### `max_workers`
最大ワーカー数。

```python
print(collector.max_workers)
```

## ImageClassifier クラス

### 概要
画像分類を実行するクラスです。ヒューリスティックベースと深層学習ベースの両方をサポートします。

### 初期化
```python
classifier = ImageClassifier(enable_deep_model=False)
```

**パラメータ:**
- **enable_deep_model** (bool): 深層学習モデルを有効にするかどうか

### メソッド

#### `analyze_image(image_path)`
画像を分析します。

```python
result = classifier.analyze_image("path/to/image.jpg")
```

**パラメータ:**
- **image_path** (str): 分析する画像のパス

**戻り値:**
- dict: 分析結果

#### `is_valid_image(image_path)`
画像が有効かどうかをチェックします。

```python
is_valid = classifier.is_valid_image("path/to/image.jpg")
```

**パラメータ:**
- **image_path** (str): チェックする画像のパス

**戻り値:**
- bool: 画像が有効かどうか

#### `get_image_features(image_path)`
画像の特徴量を取得します。

```python
features = classifier.get_image_features("path/to/image.jpg")
```

**パラメータ:**
- **image_path** (str): 特徴量を取得する画像のパス

**戻り値:**
- dict: 画像特徴量

## ConfigWizard クラス

### 概要
対話型の設定ウィザードを提供するクラスです。

### 初期化
```python
wizard = ConfigWizard()
```

### メソッド

#### `run()`
設定ウィザードを実行し、設定を対話的に収集します。

```python
config = wizard.run()
```

**戻り値:**
- dict: 設定辞書

**対話項目:**
- Tumblr API 認証情報（Consumer Key / Consumer Secret）
- 出力フォルダ名
- 深層学習モデルの有効化
- 画像フィルタ（`max_file_size_mb`, `nsfw_threshold`）
- ネットワーク設定（タイムアウト／再試行）
- ログローテーション設定

`nsfw_threshold` は 0.0〜1.0 の範囲で入力し、NSFWヒューリスティック判定スコアが閾値を超える画像を除外します（初期値 0.35）。

#### `save_config(config, config_file)`
設定辞書をファイルに保存します。

```python
wizard.save_config(config, "config.json")
```

**パラメータ:**
- **config** (dict): 保存する設定
- **config_file** (str): 設定ファイルのパス

**戻り値:**
- None

#### `load_config(config_file)`
設定ファイルを読み込みます。

```python
config = wizard.load_config("config.json")
```

**パラメータ:**
- **config_file** (str): 設定ファイルのパス

**戻り値:**
- dict: 設定辞書

#### `prompt_string(prompt, default="", validator=None)`
文字列入力を求めます。

```python
value = wizard.prompt_string("Enter value", "default_value")
```

**パラメータ:**
- **prompt** (str): プロンプトメッセージ
- **default** (str): デフォルト値
- **validator** (callable, optional): 検証関数

**戻り値:**
- str: 入力された文字列

#### `prompt_int(prompt, default=0, min_value=None, max_value=None)`
整数入力を求めます。

```python
value = wizard.prompt_int("Enter number", 10, min_value=1, max_value=100)
```

**パラメータ:**
- **prompt** (str): プロンプトメッセージ
- **default** (int): デフォルト値
- **min_value** (int, optional): 最小値
- **max_value** (int, optional): 最大値

**戻り値:**
- int: 入力された整数

#### `prompt_bool(prompt, default=False)`
真偽値入力を求めます。

```python
value = wizard.prompt_bool("Enable feature", True)
```

**パラメータ:**
- **prompt** (str): プロンプトメッセージ
- **default** (bool): デフォルト値

**戻り値:**
- bool: 入力された真偽値

## 国際化システム (Internationalization System)

### 概要
50言語に対応した国際化システムです。

### 関数

#### `set_locale(locale)`
ロケールを設定します。

```python
from i18n import set_locale
set_locale('ja')  # 日本語に設定
set_locale('en')  # 英語に設定
```

**パラメータ:**
- **locale** (str): ロケールコード

**戻り値:**
- bool: 設定成功かどうか

#### `get_current_locale()`
現在のロケールを取得します。

```python
from i18n import get_current_locale
locale = get_current_locale()
print(locale)  # 'ja' など
```

**戻り値:**
- str: 現在のロケール

#### `_(key, default="", **kwargs)`
翻訳テキストを取得します。

```python
from i18n import _
message = _("welcome_message")
print(message)  # 現在のロケールでのウェルカムメッセージ
```

**パラメータ:**
- **key** (str): 翻訳キー
- **default** (str): デフォルトテキスト
- **kwargs**: プレースホルダー用のパラメータ

**戻り値:**
- str: 翻訳されたテキスト

### 対応言語
- **en**: English
- **ja**: 日本語
- **zh**: 中文
- **ko**: 한국어
- **es**: Español
- **fr**: Français
- **de**: Deutsch
- **it**: Italiano
- **pt**: Português
- **ru**: Русский
- その他40言語

## ユーティリティ関数 (Utility Functions)

### ダウンロード関連
#### `exponential_backoff(attempt, base_delay=1, max_delay=60)`
エクスポネンシャルバックオフを計算します。

```python
from tumblr_image_collector import exponential_backoff
delay = exponential_backoff(3)  # 3回目の再試行の遅延時間を計算
```

**パラメータ:**
- **attempt** (int): 再試行回数
- **base_delay** (float): 基本遅延時間
- **max_delay** (float): 最大遅延時間

**戻り値:**
- float: 計算された遅延時間

#### `download_with_retry(url, output_folder, max_retries=3, timeout=30)`
再試行機能付きのダウンロードを実行します。

```python
from tumblr_image_collector import download_with_retry
success, filepath = download_with_retry("https://example.com/image.jpg", Path("output"))
```

**パラメータ:**
- **url** (str): ダウンロードURL
- **output_folder** (Path): 出力フォルダ
- **max_retries** (int): 最大再試行回数
- **timeout** (int): タイムアウト時間

**戻り値:**
- tuple: (成功フラグ, ファイルパス)

#### `parallel_download(image_urls, output_folder, max_workers=5)`
並列ダウンロードを実行します。

```python
from tumblr_image_collector import parallel_download
files = parallel_download(urls, Path("output"), max_workers=10)
```

**パラメータ:**
- **image_urls** (list): ダウンロードするURLリスト
- **output_folder** (Path): 出力フォルダ
- **max_workers** (int): 最大ワーカー数

**戻り値:**
- list: ダウンロードされたファイルパスのリスト

### 画像処理関連
#### `calculate_image_hash(image_path, hash_size=8)`
画像のハッシュを計算します。

```python
from tumblr_image_collector import calculate_image_hash
hash_value = calculate_image_hash("path/to/image.jpg")
```

**パラメータ:**
- **image_path** (str): 画像ファイルのパス
- **hash_size** (int): ハッシュサイズ

**戻り値:**
- str: 計算されたハッシュ値

#### `compare_images(image1_path, image2_path, threshold=0.9)`
2つの画像を比較します。

```python
from tumblr_image_collector import compare_images
is_similar = compare_images("image1.jpg", "image2.jpg")
```

**パラメータ:**
- **image1_path** (str): 1つ目の画像パス
- **image2_path** (str): 2つ目の画像パス
- **threshold** (float): 類似度の閾値

**戻り値:**
- bool: 画像が類似しているかどうか

### ファイル操作関連
#### `ensure_directory(path)`
ディレクトリが存在することを確認します。

```python
from tumblr_image_collector import ensure_directory
ensure_directory("path/to/directory")
```

**パラメータ:**
- **path** (str): ディレクトリパス

**戻り値:**
- None

#### `get_file_size_mb(file_path)`
ファイルサイズをMB単位で取得します。

```python
from tumblr_image_collector import get_file_size_mb
size = get_file_size_mb("path/to/file.jpg")
```

**パラメータ:**
- **file_path** (str): ファイルパス

**戻り値:**
- float: ファイルサイズ（MB）

#### `is_image_file(file_path)`
ファイルが画像かどうかをチェックします。

```python
from tumblr_image_collector import is_image_file
is_image = is_image_file("path/to/file.jpg")
```

**パラメータ:**
- **file_path** (str): ファイルパス

**戻り値:**
- bool: ファイルが画像かどうか

## 定数 (Constants)

### ネットワーク定数 (Network Constants)
- `DEFAULT_TIMEOUT_SECONDS = 30`: デフォルトタイムアウト時間
- `DEFAULT_RETRY_ATTEMPTS = 3`: デフォルト再試行回数
- `DEFAULT_BACKOFF_FACTOR = 0.5`: デフォルトバックオフ係数
- `DEFAULT_CHUNK_SIZE = 8192`: デフォルトチャンクサイズ

### 画像処理定数 (Image Processing Constants)
- `MIN_IMAGE_DIMENSION = 500`: 最小画像サイズ
- `MIN_RESOLUTION_WIDTH = 300`: 最小解像度（幅）
- `MIN_RESOLUTION_HEIGHT = 300`: 最小解像度（高さ）
- `MAX_FILE_SIZE_MB = 10`: 最大ファイルサイズ（MB）
- `DEFAULT_THUMBNAIL_SIZE = (200, 200)`: デフォルトサムネイルサイズ
- `DEFAULT_QUALITY = 85`: デフォルト画質

### AIモデル定数 (AI Model Constants)
- `DEFAULT_MODEL_INPUT_SIZE = 224`: デフォルトモデル入力サイズ
- `DEFAULT_DENSE_LAYER_SIZE = 1024`: デフォルト密層サイズ
- `DEFAULT_EPOCHS = 50`: デフォルトエポック数
- `DEFAULT_BATCH_SIZE = 32`: デフォルトバッチサイズ

### 画像分析定数 (Image Analysis Constants)
- `CANNY_LOWER_THRESHOLD = 100`: Cannyエッジ検出の下限閾値
- `CANNY_UPPER_THRESHOLD = 200`: Cannyエッジ検出の上限閾値
- `BLUR_THRESHOLD = 50`: ぼかし閾値
- `FACE_MIN_SIZE = 30`: 顔検出の最小サイズ
- `FACE_MAX_SIZE = 300`: 顔検出の最大サイズ
- `DEFAULT_COLOR_CLUSTERS = 3`: デフォルトカラークラスタ数
- `QUALITY_THRESHOLD_LOW = 0.3`: 低品質閾値
- `BRIGHTNESS_THRESHOLD_LOW = 0.3`: 低輝度閾値
- `CONFIDENCE_THRESHOLD = 0.5`: 信頼度閾値
- `IOU_THRESHOLD = 0.3`: IoU閾値
- `RESIZE_THRESHOLD = 300`: リサイズ閾値
- `SCALE_FACTOR_FACE = 1.1`: 顔検出スケールファクター
- `MIN_NEIGHBORS_FACE = 3`: 顔検出最小隣接数

### キャッシュ・メモリ定数 (Cache and Memory Constants)
- `DEFAULT_CACHE_SIZE_MB = 500`: デフォルトキャッシュサイズ（MB）
- `CLEANUP_THRESHOLD = 0.8`: クリーンアップ閾値
- `MEMORY_CHUNK_SIZE = 1048576`: メモリチャンクサイズ
- `BYTES_TO_MB_DIVISOR = 1048576`: バイトからMBへの除数

### 日付・時間定数 (Date and Time Constants)
- `DEFAULT_DAYS_BACK = 30`: デフォルト遡及日数
- `DEFAULT_PAGE_LIMIT = 50`: デフォルトページ制限

### 色分析定数 (Color Analysis Constants)
- `COLOR_TOLERANCE = 30`: 色許容差
- `HISTOGRAM_BINS = 256`: ヒストグラムビン数
- `COLOR_HISTOGRAM_BINS = 16`: カラーヒストグラムビン数

## 例外クラス (Exception Classes)

### `DownloadError`
カスタムダウンロードエラー。

```python
from tumblr_image_collector import DownloadError

try:
    collector.download_image("invalid_url")
except DownloadError as e:
    print(f"Download failed: {e}")
```

### `ConfigurationError`
設定関連のエラー。

```python
from tumblr_image_collector import ConfigurationError

try:
    config = load_config("nonexistent.json")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

### `NetworkError`
ネットワーク関連のエラー。

```python
from tumblr_image_collector import NetworkError

try:
    response = requests.get("https://example.com")
except NetworkError as e:
    print(f"Network error: {e}")
```

### `ImageProcessingError`
画像処理関連のエラー。

```python
from tumblr_image_collector import ImageProcessingError

try:
    process_image("corrupted.jpg")
except ImageProcessingError as e:
    print(f"Image processing error: {e}")
```

## 使用例 (Usage Examples)

### 基本的な使用方法
```python
from tumblr_image_collector import TumblrImageCollector
from i18n import set_locale

# 日本語設定
set_locale('ja')

# コレクターの初期化
collector = TumblrImageCollector()

# ブログから画像を収集
collector.run("staff")

# 統計の表示
collector.print_download_stats()

# メタデータのエクスポート
collector.export_metadata('json')
```

### カスタム設定での使用
```python
from tumblr_image_collector import TumblrImageCollector
from pathlib import Path

# カスタム設定で初期化
collector = TumblrImageCollector(
    config_file="custom_config.json",
    output_dir_override="my_images",
    workers_override=10
)

# 複数のブログから収集
blogs = ["blog1", "blog2", "blog3"]
for blog in blogs:
    collector.run(blog)
```

### エラーハンドリング
```python
from tumblr_image_collector import TumblrImageCollector, DownloadError
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)

try:
    collector = TumblrImageCollector()
    collector.run("staff")
except DownloadError as e:
    logging.error(f"Download failed: {e}")
except Exception as e:
    logging.error(f"Unexpected error: {e}")
finally:
    # リソースのクリーンアップ
    if 'collector' in locals():
        collector._cleanup_resources()
```

### 高度な画像処理
```python
from tumblr_image_collector import TumblrImageCollector
from PIL import Image

collector = TumblrImageCollector()

# 画像の有効性チェック
with Image.open("image.jpg") as img:
    is_valid = collector._is_image_valid(img)
    if is_valid:
        # メタデータの抽出
        metadata = collector._extract_image_metadata("image.jpg")
        print(f"Image metadata: {metadata}")

        # 効率的な処理
        features = collector._process_image_efficiently("image.jpg")
        print(f"Image features: {features}")
```

---

**APIリファレンス v2.0.0** - Tumblr Image Collector
