# 開発者ガイド (Developer Guide)

## 目次 (Table of Contents)

1. [開発環境のセットアップ](#開発環境のセットアップ)
2. [プロジェクト構造](#プロジェクト構造)
3. [コーディング標準](#コーディング標準)
4. [テストの実行](#テストの実行)
5. [デバッグ](#デバッグ)
6. [パフォーマンス最適化](#パフォーマンス最適化)
7. [セキュリティ開発](#セキュリティ開発)
8. [国際化開発](#国際化開発)
9. [ドキュメント](#ドキュメント)
10. [リリースプロセス](#リリースプロセス)

## 開発環境のセットアップ (Development Environment Setup)

### システム要件 (System Requirements)
- **Python**: 3.9+
- **RAM**: 8GB以上（推奨16GB）
- **ストレージ**: 50GB以上の空き容量
- **OS**: Windows 10+, macOS 10.15+, Linux (Ubuntu 18.04+)

### 必要なツール (Required Tools)
- **Git**: バージョン管理
- **Python venv**: 仮想環境
- **IDE**: VSCode, PyCharm, または任意のPython対応エディタ
- **Docker**: コンテナ化された開発環境（オプション）

### 1. リポジトリのクローン
```bash
git clone <repository-url>
cd "tumblr image collector"
```

### 2. 仮想環境の作成
```bash
# Python venvを使用
python -m venv dev-env
source dev-env/bin/activate  # Linux/macOS
# dev-env\Scripts\activate  # Windows

# またはCondaを使用
conda create -n tumblr-dev python=3.9
conda activate tumblr-dev
```

### 3. 依存関係のインストール
```bash
# 基本的な開発依存関係
pip install -r requirements-dev.txt

# すべての依存関係（AI機能を含む）
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install tensorflow==2.12.0 keras==2.12.0 matplotlib==3.7.1 seaborn==0.12.2 scikit-learn==1.2.2
```

### 4. 開発ツールのインストール
```bash
# コード品質ツール
pip install black flake8 mypy isort bandit safety

# テストツール
pip install pytest pytest-cov pytest-html pytest-benchmark pytest-mock pytest-xdist

# ドキュメントツール
pip install mkdocs mkdocs-material

# その他の開発ツール
pip install pre-commit jupyter ipython
```

### 5. Pre-commitフックの設定
```bash
# Pre-commitのインストール
pip install pre-commit

# Pre-commitフックのインストール
pre-commit install

# Pre-commitフックの実行
pre-commit run --all-files
```

### 6. 開発環境の確認
```bash
# 基本的なインポートテスト
python -c "
import tumblr_image_collector
from image_classifier import ImageClassifier
from config import ConfigWizard
from i18n import set_locale
print('開発環境セットアップ完了')
"
```

## プロジェクト構造 (Project Structure)

```
tumblr-image-collector/
├── tumblr_image_collector/
│   ├── __init__.py              # パッケージ初期化
│   ├── tumblr_image_collector.py # メインアプリケーション
│   ├── image_classifier.py      # 画像分類システム
│   └── config.py               # 設定管理システム
├── i18n.py                     # 国際化システム
├── locales/                    # 翻訳ファイルディレクトリ
│   ├── en.json                # 英語翻訳
│   ├── ja.json                # 日本語翻訳
│   └── ...                    # その他の言語
├── tests/                      # テストディレクトリ
│   ├── __init__.py
│   ├── conftest.py            # pytest設定
│   ├── test_suite.py          # テストスイート
│   └── fixtures/              # テストフィクスチャ
├── docs/                      # ドキュメントディレクトリ
│   ├── api/                   # APIドキュメント
│   └── guides/                # ガイドドキュメント
├── .github/                   # GitHub設定
│   └── workflows/             # CI/CDワークフロー
├── requirements.txt           # 基本依存関係
├── requirements-dev.txt       # 開発依存関係
├── requirements-ai.txt        # AI機能依存関係
├── pyproject.toml            # Pythonプロジェクト設定
├── setup.py                  # パッケージ設定
├── MANIFEST.in               # パッケージマニフェスト
└── README.md                 # プロジェクトREADME
```

## コーディング標準 (Coding Standards)

### Pythonスタイルガイド
- **PEP 8**: 基本的なPythonコーディング標準
- **Black**: コードフォーマッタ（最大行長127文字）
- **isort**: import文のソート
- **flake8**: リンター（複雑度チェックを含む）

### 命名規則 (Naming Conventions)
- **クラス**: PascalCase (例: `TumblrImageCollector`)
- **関数/メソッド**: snake_case (例: `download_image`)
- **定数**: SCREAMING_SNAKE_CASE (例: `DEFAULT_TIMEOUT`)
- **プライベートメソッド**: _snake_case (例: `_validate_input`)
- **変数**: snake_case (例: `image_path`)

### ドキュメンテーション
- **Docstring**: GoogleスタイルまたはNumPyスタイル
- **型ヒント**: すべての関数とメソッドに型ヒントを付与
- **コメント**: 複雑なロジックには適切なコメント

### コード例 (Code Example)
```python
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """画像処理を行うクラス

    このクラスは、画像の読み込み、最適化、特徴量抽出を
    メモリ効率よく実行します。

    Attributes:
        max_dimension (int): 最大画像サイズ
        quality_threshold (float): 品質閾値
    """

    def __init__(self, max_dimension: int = 2048, quality_threshold: float = 0.7):
        """初期化

        Args:
            max_dimension: 最大画像サイズ
            quality_threshold: 品質判定の閾値
        """
        self.max_dimension = max_dimension
        self.quality_threshold = quality_threshold
        self._cache: Dict[str, Any] = {}

    def process_image(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """画像を処理する

        メモリ効率を考慮しつつ、画像の特徴量を抽出します。

        Args:
            image_path: 処理する画像のパス

        Returns:
            画像特徴量の辞書、またはNone（失敗時）

        Raises:
            FileNotFoundError: 画像ファイルが存在しない場合
            ValueError: 画像形式が不正な場合
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            # メモリ効率の良い画像処理
            with Image.open(image_path) as img:
                # サイズ最適化
                if img.size[0] > self.max_dimension or img.size[1] > self.max_dimension:
                    img.thumbnail((self.max_dimension, self.max_dimension), Image.LANCZOS)

                # 特徴量抽出
                features = self._extract_features(img)

            logger.info(f"Processed image: {image_path}")
            return features

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None

    def _extract_features(self, image: Image.Image) -> Dict[str, Any]:
        """特徴量を抽出する（プライベートメソッド）

        Args:
            image: 特徴量を抽出する画像

        Returns:
            特徴量の辞書
        """
        # 実装の詳細
        return {
            'dimensions': image.size,
            'mode': image.mode,
            'format': image.format
        }
```

## テストの実行 (Running Tests)

### 基本的なテスト実行
```bash
# すべてのテストを実行
pytest

# 詳細な出力で実行
pytest -v

# 特定のテストファイルを実行
pytest tests/test_suite.py

# 特定のテストクラスを実行
pytest tests/test_suite.py::TestTumblrImageCollector

# 特定のテストメソッドを実行
pytest tests/test_suite.py::TestTumblrImageCollector::test_initialization
```

### テストオプション
```bash
# カバレッジレポート付きで実行
pytest --cov=tumblr_image_collector --cov-report=html

# パフォーマンステストを実行
pytest tests/test_suite.py::TestPerformance -v

# セキュリティテストを実行
pytest -k "security" -v

# 失敗したテストのみ再実行
pytest --lf

# テストを並列実行（高速化）
pytest -n auto
```

### テストカバレッジの確認
```bash
# HTMLレポートの生成
pytest --cov=tumblr_image_collector --cov-report=html
open htmlcov/index.html

# ターミナルでのカバレッジレポート
pytest --cov=tumblr_image_collector --cov-report=term-missing

# カバレッジレポートの保存
pytest --cov=tumblr_image_collector --cov-report=xml
```

### ベンチマークテスト
```bash
# ベンチマークテストの実行
pytest tests/test_suite.py::TestPerformance::test_memory_efficiency --benchmark-only

# ベンチマーク結果の比較
pytest-benchmark compare
```

## デバッグ (Debugging)

### ログレベルの設定
```python
import logging

# デバッグレベルのログ設定
logging.basicConfig(level=logging.DEBUG)

# 特定のロガーのレベル設定
logger = logging.getLogger('tumblr_image_collector')
logger.setLevel(logging.DEBUG)
```

### デバッガーの使用
```python
import pdb
import ipdb

# PDBを使用
pdb.set_trace()

# IPDBを使用（より高度なデバッガー）
ipdb.set_trace()
```

### リモートデバッグ
```python
import debugpy

# VSCodeなどのデバッガーを待機
debugpy.listen(5678)
debugpy.wait_for_client()
```

### メモリデバッグ
```python
import psutil
import os
from memory_profiler import profile

# メモリ使用量の監視
process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss / 1024 / 1024  # MB
print(f"Memory usage: {memory_usage:.2f} MB")

# メモリプロファイリング
@profile
def memory_intensive_function():
    # メモリを多く使用する処理
    pass
```

## パフォーマンス最適化 (Performance Optimization)

### プロファイリング
```bash
# cProfileを使用したプロファイリング
python -m cProfile -s time tumblr_image_collector.py > profile_output.txt

# line_profilerを使用した行単位のプロファイリング
pip install line_profiler
kernprof -l tumblr_image_collector.py
python -m line_profiler tumblr_image_collector.py.lprof

# memory_profilerを使用したメモリプロファイリング
pip install memory_profiler
python -m memory_profiler tumblr_image_collector.py
```

### パフォーマンスメトリクスの収集
```python
import time
import functools

def timing_decorator(func):
    """関数の実行時間を測定するデコレーター"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# 使用例
@timing_decorator
def slow_function():
    time.sleep(1)
```

### メモリ最適化
```python
import gc
from typing import Generator

def memory_efficient_generator() -> Generator:
    """メモリ効率の良いジェネレーター"""
    for i in range(1000000):
        yield i

# 定期的なガベージコレクション
def process_large_dataset():
    for item in memory_efficient_generator():
        # 処理
        if item % 10000 == 0:
            gc.collect()  # メモリ解放
```

## セキュリティ開発 (Security Development)

### セキュリティテスト
```bash
# Banditを使用したセキュリティスキャン
bandit -r tumblr_image_collector/ -f json -o security-report.json

# Safetyを使用した依存関係の脆弱性チェック
safety check --json

# 両方を組み合わせたセキュリティチェック
bandit -r tumblr_image_collector/ && safety check
```

### セキュアコーディングの実践
```python
from typing import Optional
import secrets
import hashlib

def secure_password_hash(password: str, salt: Optional[str] = None) -> str:
    """セキュアなパスワードハッシュ"""
    if salt is None:
        salt = secrets.token_hex(16)

    # Argon2やbcryptなどの強力なハッシュ関数を使用
    hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return f"{salt}:{hash_obj.hex()}"

def validate_input(input_data: str, allowed_pattern: str) -> bool:
    """入力検証"""
    import re
    if not re.match(allowed_pattern, input_data):
        return False
    return True

# 使用例
user_input = "user_provided_data"
if validate_input(user_input, r'^[a-zA-Z0-9_]+$'):
    # 安全な入力として処理
    pass
```

## 国際化開発 (Internationalization Development)

### 翻訳ファイルの管理
```bash
# 新しい言語の追加
mkdir locales
python -c "
from i18n import Internationalization
i18n = Internationalization()
i18n.generate_locale_files()
"

# 翻訳の検証
python -c "
from i18n import set_locale, _
set_locale('ja')
print(_('welcome_message'))
"
```

### 翻訳キーの管理
```python
# 翻訳キーの使用
from i18n import _

# シンプルな翻訳
message = _("welcome_message")

# パラメータ付き翻訳
greeting = _("greeting", name="田中太郎")

# 複数形対応
count = 5
items_text = _("item_count", count=count, item="画像")
```

## ドキュメント (Documentation)

### ドキュメントのビルド
```bash
# MkDocsを使用したドキュメントビルド
mkdocs build

# ライブプレビュー
mkdocs serve

# デプロイ
mkdocs gh-deploy
```

### APIドキュメントの生成
```bash
# Sphinxを使用したAPIドキュメント生成
pip install sphinx sphinx-rtd-theme

# 設定ファイルの作成
sphinx-quickstart docs

# ドキュメントのビルド
sphinx-build docs _build
```

### コードドキュメントの生成
```python
# Docstringからのドキュメント生成
def documented_function(param: str) -> bool:
    """
    ドキュメント化された関数の例

    Args:
        param: パラメータの説明

    Returns:
        戻り値の説明

    Raises:
        ValueError: エラーの場合
    """
    pass
```

## リリースプロセス (Release Process)

### バージョン管理
```bash
# バージョン情報の更新
# pyproject.toml または setup.py の version を更新

# 変更ログの更新
# CHANGELOG.md に新しいバージョンの情報を追加

# タグの作成
git tag v2.0.0
git push origin v2.0.0
```

### リリースチェックリスト
- [ ] すべてのテストが通る
- [ ] コードカバレッジが90%以上
- [ ] セキュリティチェックが通る
- [ ] ドキュメントが更新されている
- [ ] バージョン番号が更新されている
- [ ] 変更ログが更新されている
- [ ] CI/CDが正常に動作する

### リリース後の作業
```bash
# PyPIへのアップロード
python -m build
twine upload dist/*

# Dockerイメージのビルドとプッシュ
docker build -t tumblr-image-collector:v2.0.0 .
docker push your-registry/tumblr-image-collector:v2.0.0
```

## トラブルシューティング (Troubleshooting)

### 一般的な開発時エラー

#### ImportError
```
ImportError: No module named 'tumblr_image_collector'
```
**解決策:**
- 仮想環境が有効化されていることを確認
- `PYTHONPATH`にプロジェクトディレクトリが含まれていることを確認
- `pip install -e .`で開発モードインストール

#### テストエラー
```
FAILED tests/test_suite.py::TestSomeClass::test_some_method
```
**解決策:**
- テストの依存関係を確認
- モックオブジェクトが適切に設定されていることを確認
- テストデータをクリーンアップ

#### メモリエラー
```
MemoryError
```
**解決策:**
- メモリ使用量を最適化
- テストデータを小さくする
- `gc.collect()`を適切な場所で呼び出し

### 開発ツールのトラブルシューティング

#### Blackフォーマッタ
```bash
# 設定ファイルの作成
echo "[tool.black]" > pyproject.toml
echo "line-length = 127" >> pyproject.toml
echo "target-version = ['py39']" >> pyproject.toml
```

#### MyPy型チェック
```bash
# 設定ファイルの作成
echo "[mypy]" > mypy.ini
echo "python_version = 3.9" >> mypy.ini
echo "ignore_missing_imports = True" >> mypy.ini
echo "strict_optional = True" >> mypy.ini
```

## ベストプラクティス (Best Practices)

### コード品質
- すべての関数に型ヒントを付与
- 複雑な関数は小さな関数に分割
- 意味のある変数名と関数名を使用
- 適切なエラーハンドリングを実装

### テスト
- 単体テスト、統合テスト、エンドツーエンドテストを作成
- テストカバレッジを90%以上維持
- モックとフィクスチャを適切に使用
- テストの実行時間を短く保つ

### パフォーマンス
- メモリ使用量を常に監視
- アルゴリズムの複雑度を考慮
- キャッシュを効果的に使用
- 並列処理を適切に実装

### セキュリティ
- すべての入力を検証
- 機密情報を適切に扱う
- 定期的なセキュリティレビューを実施
- 依存関係の脆弱性を監視

### ドキュメント
- すべての公開APIにドキュメンテーション
- 使用例を提供
- 変更点を常に更新
- 複数の言語でのドキュメント提供

---

**開発者ガイド v2.0.0** - Tumblr Image Collector
