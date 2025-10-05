# Dockerfile
# Tumblr Image Collector - 商用グレード版

FROM python:3.9-slim

# メタデータの設定
LABEL maintainer="Tumblr Image Collector Team"
LABEL version="2.0.0"
LABEL description="Commercial-grade Tumblr image collection and processing tool"

# 環境変数の設定
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=random
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# システム依存関係のインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /app

# Python依存関係のインストール
COPY requirements.txt .
COPY requirements-ai.txt .
RUN pip install --no-cache-dir -r requirements.txt

# AI機能のオプションインストール
COPY requirements-ai.txt .
RUN pip install --no-cache-dir -r requirements-ai.txt || echo "AI dependencies installation failed, continuing without AI features"

# 開発ツールのインストール
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt || echo "Dev dependencies installation failed, continuing without dev tools"

# アプリケーションコードのコピー
COPY tumblr_image_collector/ ./tumblr_image_collector/
COPY i18n.py .
COPY config.py .
COPY image_classifier.py .
COPY locales/ ./locales/
COPY tests/ ./tests/
COPY docs/ ./docs/

# 設定ファイルのコピー
COPY pyproject.toml setup.py MANIFEST.in ./

# アプリケーションのインストール（開発モード）
RUN pip install -e .

# ディレクトリの作成と権限設定
RUN mkdir -p /app/output /app/logs /app/crash_reports /app/metadata && \
    chmod -R 755 /app

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "from tumblr_image_collector import TumblrImageCollector; print('Health check passed')" || exit 1

# デフォルトのコマンド
CMD ["python", "-m", "tumblr_image_collector", "--interactive"]

# ポートの公開（オプション）
EXPOSE 8000

# ボリュームの設定
VOLUME ["/app/output", "/app/logs", "/app/crash_reports"]

# ユーザーの設定（セキュリティのため）
RUN useradd --create-home --shell /bin/bash tumblr && \
    chown -R tumblr:tumblr /app
USER tumblr

# デフォルトの設定ファイル
ENV TUMBLR_CONFIG_FILE=/app/config.json
ENV TUMBLR_LOG_LEVEL=INFO

# メモリ制限（オプション）
ENV MALLOC_TRIM_THRESHOLD_=0

# セキュリティ設定
RUN chmod 644 /app/requirements*.txt && \
    chmod 755 /app/tumblr_image_collector/*.py
