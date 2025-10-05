# インストールガイド (Installation Guide)

## 目次 (Table of Contents)

1. [システム要件](#システム要件)
2. [クイックインストール](#クイックインストール)
3. [詳細インストール](#詳細インストール)
4. [設定](#設定)
5. [トラブルシューティング](#トラブルシューティング)
6. [アンインストール](#アンインストール)

## システム要件 (System Requirements)

### 必須要件 (Required)
- **Python**: 3.9以上
- **RAM**: 最低4GB（推奨8GB以上）
- **ストレージ**: 処理する画像数に応じた空き容量（最低1GB）
- **ネットワーク**: 安定したインターネット接続
- **OS**: Windows 10+, macOS 10.15+, Linux (Ubuntu 18.04+, CentOS 7+)

### 推奨要件 (Recommended)
- **RAM**: 16GB以上（大量の画像処理時）
- **CPU**: マルチコアプロセッサ（並列処理のため）
- **GPU**: NVIDIA GPU（AI分類機能使用時）
- **ストレージ**: SSD（高速な画像処理のため）

### ソフトウェア要件 (Software Requirements)
- **pip**: Pythonパッケージマネージャー
- **git**: バージョン管理（オプション）
- **仮想環境ツール**: venvまたはconda（推奨）

## クイックインストール (Quick Installation)

### 1. リポジトリのクローン
GitHub上の公式リポジトリから取得します。
```bash
git clone https://github.com/shizukutanaka/Tumblr-Image-Collector.git
cd Tumblr-Image-Collector
```

### 2. 仮想環境の作成と有効化
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

### 3. 依存関係のインストール
```bash
# 基本インストール（軽量モード）
pip install -r requirements.txt

# AI機能を含む完全インストール
pip install -r requirements.txt
pip install tensorflow==2.12.0 keras==2.12.0 matplotlib==3.7.1 seaborn==0.12.2 scikit-learn==1.2.2
```

### 4. 設定の実行
```bash
python config.py
```

### 5. 動作確認
```bash
python -c "from tumblr_image_collector import TumblrImageCollector; print('インストール成功')"
```
## 詳細インストール (Detailed Installation)

### ステップバイステップガイド (Step-by-Step Guide)

#### ステップ1: Pythonの確認
```bash
python --version
```
Python 3.9以上がインストールされていることを確認してください。

#### ステップ2: リポジトリの取得
公式リポジトリを `git clone` コマンドで取得します。
```bash
# HTTPS経由
git clone https://github.com/shizukutanaka/Tumblr-Image-Collector.git

# SSH経由（鍵認証済みの場合）
git clone git@github.com:shizukutanaka/Tumblr-Image-Collector.git

# ZIPダウンロードの場合
# GitHubの"Code"ボタンから"Download ZIP"を選択し、解凍
```

#### ステップ3: ディレクトリの移動
```bash
cd Tumblr-Image-Collector
```

#### ステップ4: 仮想環境の作成
仮想環境を使用することで、他のPythonプロジェクトとの依存関係の競合を避けることができます。

```bash
# venvを使用
python -m venv tumblr_collector_env

# Condaを使用（Anaconda/Minicondaの場合）
conda create -n tumblr_collector python=3.9
conda activate tumblr_collector
```

#### ステップ5: 仮想環境の有効化
```bash
# venvの場合
# Windows
tumblr_collector_env\Scripts\activate
# Linux/macOS
source tumblr_collector_env/bin/activate

# Condaの場合
conda activate tumblr_collector
```

#### ステップ6: 依存関係のインストール
```bash
# 基本パッケージのインストール
pip install --upgrade pip
pip install -r requirements.txt

# オプション: AI機能の追加
pip install tensorflow==2.12.0 keras==2.12.0
pip install matplotlib==3.7.1 seaborn==0.12.2 scikit-learn==1.2.2

# オプション: 開発ツールのインストール
pip install -r requirements-dev.txt
```

#### ステップ7: 設定の実行
```bash
# 対話型設定ウィザードの実行
python config.py
```

#### ステップ8: 動作確認
```bash
# 基本的なインポートテスト
python -c "import tumblr_image_collector; print('基本インポート: OK')"

# 詳細なテスト
python -c "
from tumblr_image_collector import TumblrImageCollector
from i18n import set_locale
set_locale('ja')
print('国際化対応: OK')
print('インストール完了')
"
```

## 設定 (Configuration)

### 設定ファイルの場所
設定は`config.json`ファイルに保存されます。通常、プロジェクトのルートディレクトリに作成されます。

### Tumblr API認証情報の管理 (Managing Tumblr API Credentials)

Tumblr APIのConsumer KeyとConsumer Secretは環境変数から安全に読み込まれます。環境変数`TUMBLR_CONSUMER_KEY`と`TUMBLR_CONSUMER_SECRET`を設定すると、`config.py`が自動的に読み込みます。環境変数が未設定または無効な場合は、設定ウィザードが非表示入力で再入力を求めます。保存された認証情報は`config.json`内に保持され、出力時には伏字化されます。

Set the environment variables before running the wizard:

```bash
export TUMBLR_CONSUMER_KEY="your_consumer_key"
export TUMBLR_CONSUMER_SECRET="your_consumer_secret"
```

```powershell
$Env:TUMBLR_CONSUMER_KEY = "your_consumer_key"
$Env:TUMBLR_CONSUMER_SECRET = "your_consumer_secret"
```

### 設定項目の詳細

#### ネットワーク設定 (Network Configuration)
```json
{
  "network": {
    "download_timeout_seconds": 30,
    "max_retries": 3,
    "backoff_factor": 0.5,
    "max_backoff_seconds": 60
  }
}
```

#### レート制限設定 (Rate Limiting)
```json
{
  "rate_limit": {
    "requests_per_minute": 30,
    "burst_limit": 5
  }
}
```

#### ログ設定 (Logging Configuration)
```json
{
  "logging": {
    "level": "INFO",
    "max_bytes": 10485760,
    "backup_count": 5
  }
}
```

#### AI設定 (AI Configuration)
```json
{
  "enable_deep_model": true,
  "ai_model": {
    "input_size": 224,
    "confidence_threshold": 0.8
  }
}
```

### 設定ウィザードの使用
```bash
python config.py
```

設定ウィザードでは以下の項目を設定できます：
- Tumblr APIキー
- 出力ディレクトリ
- プロキシ設定
- ネットワーク設定
- ログ設定
- AI機能の有効化

### CLIフィルタとLike収集の使い方 (CLI Filters & Liked Post Collection)

Tumblr Image Collectorはコマンドラインオプションでタグ、期間、Like収集を制御できます。

```bash
# タグと期間を指定してダウンロード
python tumblr_image_collector.py blogname \
  --tags illustration magazine \
  --start-date 2024-01-01 \
  --end-date 2024-03-31

# Like投稿のみをバックアップ
python tumblr_image_collector.py --include-likes

# ブログ収集とLikeバックアップを同時に実行
python tumblr_image_collector.py blogname \
  --tags photography \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --include-likes
```

`--tags` には複数タグを設定できます。内部では小文字比較を行い、投稿タグが一致する場合のみ処理します。`--start-date` / `--end-date` は `YYYY-MM-DD` 形式で指定し、投稿の `timestamp` が範囲内のものだけを対象にします。`--include-likes` を付与すると、OAuth認証済みユーザーの Like 投稿をバッチ単位で取得します。

**動作確認ポイント**

- **タグフィルタ:** 指定タグと異なる投稿が保存されていないか出力フォルダを確認します。
- **期間フィルタ:** `metadata/` ディレクトリの出力やログで期間外の投稿が含まれていないか確認します。
- **Like収集:** `logs/tumblr_collector.log` に `Finished liked-post collection.` が記録され、バックアップされたファイルが Like 投稿に限定されているかチェックします。

### 収集の再開ワークフロー (Resume Workflow)

Tumblr Image Collectorは `downloads/last_collection_state.json` に最新の収集状態を保存します。ファイルには以下が含まれます。

- **cli_filters**: CLIで指定した `tags`・`start_date`・`end_date`・`include_likes`。
- **offsets**: ブログごとのダウンロードオフセット（`offset_blogname` と同等）。
- **downloaded_images** 等の進捗情報。

`resume_image_collection()` を呼び出す、または次回 CLI を実行すると、保存されたフィルタとオフセットが自動的に復元されます。再開時の検証は以下を参照してください。

- **状態ファイル確認**: `last_collection_state.json` を開き、`cli_filters`・`offsets` が期待通りであることをチェック。
- **再実行テスト**: 中断後に同じコマンドを実行し、タグ・期間・Like設定が継続されているか出力とログで確認。

## トラブルシューティング (Troubleshooting)

### 一般的な問題と解決策

#### 問題1: Pythonが見つからない
```
エラー: 'python' は認識されていません
```
**解決策:**
- Pythonがインストールされていることを確認
- PATH環境変数にPythonのパスが含まれていることを確認
- 仮想環境を有効化していることを確認

#### 問題2: 依存関係のインストールエラー
```
エラー: Could not install packages due to an EnvironmentError
```
**解決策:**
- pipを最新版にアップデート: `pip install --upgrade pip`
- 仮想環境を使用していることを確認
- インターネット接続を確認
- プロキシ設定が必要な場合は設定

#### 問題3: TensorFlowのインストールエラー
```
エラー: Could not find a version that satisfies the requirement tensorflow
```
**解決策:**
- Python 3.9-3.11の範囲でTensorFlowをインストール
- GPU版が必要ない場合はCPU版を指定: `pip install tensorflow-cpu==2.12.0`
- システムの互換性を確認

#### 問題4: メモリ不足
```
エラー: MemoryError
```
**解決策:**
- システムのRAMを増設
- バッチサイズを小さくする
- 画像の最大サイズを制限
- 一時ファイルを定期的にクリーンアップ

#### 問題5: ネットワークタイムアウト
```
エラー: Connection timeout
```
**解決策:**
- `config.json`で`download_timeout_seconds`を増加
- 安定したネットワーク接続を使用
- プロキシ設定を確認

#### 問題6: 権限エラー
```
エラー: Permission denied
```
**解決策:**
- 管理者権限で実行
- 出力ディレクトリへの書き込み権限を確認
- アンチウイルスソフトの設定を確認

### デバッグモードの有効化

#### ログレベルの変更
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 詳細なエラーログ
```python
import traceback
try:
    # 処理コード
    pass
except Exception as e:
    traceback.print_exc()
```

### ログファイルの確認
ログファイルは`logs/tumblr_collector.log`に保存されます。

```bash
# ログファイルの確認
tail -f logs/tumblr_collector.log

# Windowsの場合
type logs\tumblr_collector.log | more
```

### システム情報の確認
```python
import sys
import platform

print(f"Pythonバージョン: {sys.version}")
print(f"OS: {platform.system()} {platform.release()}")
print(f"アーキテクチャ: {platform.architecture()}")
```

## アンインストール (Uninstallation)

### 完全アンインストール
```bash
# 仮想環境を削除
rm -rf tumblr_collector_env

# またはCondaの場合
conda remove -n tumblr_collector --all

# 生成されたファイルを削除
rm -rf output/
rm -rf logs/
rm -rf crash_reports/
rm -rf metadata/
rm -f config.json
rm -f tumblr_collector.log
```

### 設定ファイルの保持
```bash
# 設定ファイル以外を削除
rm -rf output/
rm -rf logs/
rm -rf crash_reports/
rm -rf metadata/
rm -f tumblr_collector.log
```

## 追加のリソース (Additional Resources)

### 公式ドキュメント
- [APIリファレンス](API_REFERENCE.md)
- [ユーザーガイド](USER_GUIDE.md)
- [開発者ガイド](DEVELOPER_GUIDE.md)

### コミュニティ
- [GitHub Issues](https://github.com/shizukutanaka/Tumblr-Image-Collector/issues)
- [ディスカッションフォーラム](https://github.com/shizukutanaka/Tumblr-Image-Collector/discussions)
- [コントリビューションガイド](CONTRIBUTING.md)

### サポート
- 技術サポート: support@example.com
- 商用サポート: enterprise@example.com

---

**インストールガイド v2.0.0** - Tumblr Image Collector
