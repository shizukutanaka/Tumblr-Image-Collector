# デプロイメントチェックリスト

## 本番環境デプロイメント完全ガイド

**プロジェクト**: Tumblr Image Collector v2.0.0
**最終更新**: 2025-10-05
**ステータス**: Production Ready

---

## 📋 デプロイ前チェックリスト

### 1. コードベース準備

#### Git操作（WSL環境の制限により手動実行が必要）
```bash
# 削除ファイルのクリーンアップ
git rm advanced_image_ai.py test_advanced_image_ai.py

# 全変更をステージング
git add .

# コミット
git commit -m "Production-ready v2.0.0: Security, monitoring, and comprehensive documentation

- Implement production security (SSRF/XSS/SQLi/DDoS protection)
- Add error handling with circuit breaker and retry strategies
- Add monitoring system with health checks and metrics
- Implement multi-tier caching system
- Add image optimization and download management
- Complete comprehensive documentation
- Add Docker and Kubernetes support
- Add CI/CD pipeline with GitHub Actions
- Remove deprecated advanced_image_ai module

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

#### ブランチ戦略確認
- [ ] メインブランチ: `main`
- [ ] 開発ブランチ: `develop` (推奨)
- [ ] フィーチャーブランチ: `feature/*`
- [ ] ホットフィックスブランチ: `hotfix/*`

#### コード品質チェック
```bash
# Lintチェック
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=12 --max-line-length=120 --statistics

# 型チェック（オプション）
# mypy *.py --ignore-missing-imports
```

---

### 2. テスト検証

#### ユニットテスト
```bash
# 全テスト実行
pytest

# カバレッジ付き
pytest --cov=. --cov-report=html --cov-report=term

# 特定テストスイート
pytest test_image_classifier.py -v
pytest test_production_systems.py -v
pytest test_tumblr_image_collector.py -v
```

#### 統合テスト
```bash
# 本番システムテスト
pytest test_production_systems.py -v --tb=short

# パフォーマンステスト
pytest -m slow --durations=10
```

#### セキュリティテスト
```bash
# セキュリティ関連テスト
pytest -m security -v

# SQLインジェクションテスト
# XSS/SSRFテスト
# レート制限テスト
```

---

### 3. 環境設定

#### 環境変数設定
```bash
# Tumblr API認証情報
export TUMBLR_CONSUMER_KEY="your_consumer_key_here"
export TUMBLR_CONSUMER_SECRET="your_consumer_secret_here"
export TUMBLR_TOKEN="your_token_here"
export TUMBLR_TOKEN_SECRET="your_token_secret_here"

# レート制限設定
export RATE_LIMIT_REQUESTS_PER_MINUTE=60
export RATE_LIMIT_BURST_LIMIT=10

# 監視設定
export MONITORING_ENABLED=true
export METRICS_EXPORT_INTERVAL=300
export HEALTH_CHECK_INTERVAL=60

# エラー処理設定
export CIRCUIT_BREAKER_THRESHOLD=5
export CIRCUIT_BREAKER_TIMEOUT=60
export MAX_RETRY_ATTEMPTS=3
export RETRY_BACKOFF_FACTOR=2

# セキュリティ設定
export ENABLE_DDOS_PROTECTION=true
export MAX_CONNECTIONS_PER_IP=100
export SECURITY_AUDIT_LOG=true

# ログ設定
export LOG_LEVEL=INFO
export LOG_MAX_BYTES=10485760
export LOG_BACKUP_COUNT=5
```

#### 設定ファイル準備
```bash
# config.jsonの作成
cat > config.json <<EOF
{
  "consumer_key": "\${TUMBLR_CONSUMER_KEY}",
  "consumer_secret": "\${TUMBLR_CONSUMER_SECRET}",
  "output_folder_name": "tumblr_images",
  "max_download_workers": 10,
  "enable_deep_model": false,
  "filters": {
    "min_width": 500,
    "min_height": 500,
    "max_file_size_mb": 10,
    "nsfw_threshold": 0.35
  },
  "network": {
    "download_timeout_seconds": 30,
    "max_retries": 3,
    "backoff_factor": 1,
    "max_backoff_seconds": 60
  },
  "cache": {
    "memory_cache_size": 1000,
    "disk_cache_enabled": true,
    "cache_ttl_seconds": 3600
  }
}
EOF
```

---

### 4. 依存関係インストール

#### Python環境セットアップ
```bash
# Python 3.10以上推奨
python3 --version

# 仮想環境作成
python3 -m venv .venv

# 仮想環境アクティベート
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# pipアップグレード
pip install --upgrade pip setuptools wheel

# 依存パッケージインストール
pip install -r requirements.txt

# 開発環境の場合
# pip install -r requirements-dev.txt
```

#### 依存関係の検証
```bash
# インストール確認
pip list

# 必須パッケージチェック
python -c "import pytumblr; import requests; import PIL; import imagehash; import psutil; print('All dependencies OK')"
```

---

### 5. データベース・ストレージ準備

#### ディレクトリ構造作成
```bash
# 必要なディレクトリを作成
mkdir -p tumblr_images
mkdir -p crash_reports
mkdir -p cache
mkdir -p logs

# パーミッション設定（Linux/macOS）
chmod 755 tumblr_images crash_reports cache logs
```

#### データベース初期化
```bash
# SQLiteデータベースの初期化（自動作成）
# URL管理、エラーログ、メトリクスのDBが自動生成されます
python -c "from production_url_manager import ProductionURLManager; mgr = ProductionURLManager(); print('Database initialized')"
```

---

### 6. セキュリティ検証

#### セキュリティチェック
- [ ] API認証情報が環境変数に設定されている
- [ ] 設定ファイルに機密情報が含まれていない
- [ ] .gitignoreに機密ファイルが登録されている
- [ ] ファイルパーミッションが適切に設定されている
- [ ] SSRF/XSS/SQLi対策が有効化されている
- [ ] レート制限が設定されている
- [ ] DDoS保護が有効化されている

#### 脆弱性スキャン（オプション）
```bash
# Safety（依存関係の脆弱性チェック）
pip install safety
safety check

# Bandit（セキュリティ問題検出）
pip install bandit
bandit -r . -ll
```

---

### 7. パフォーマンス検証

#### リソース要件確認
- [ ] CPU: 最小2コア、推奨4コア以上
- [ ] メモリ: 最小4GB、推奨8GB以上
- [ ] ディスク: 20GB以上の空き容量
- [ ] ネットワーク: 安定したインターネット接続

#### ベンチマークテスト
```bash
# パフォーマンステスト実行
pytest tests/ -m performance --durations=10

# リソース監視
python -c "from production_monitoring import MonitoringDashboard; dash = MonitoringDashboard(); print(dash.get_dashboard_summary())"
```

---

## 🐳 Dockerデプロイメント

### Docker環境

#### イメージビルド
```bash
# Dockerイメージのビルド
docker build -t tumblr-collector:2.0.0 .
docker build -t tumblr-collector:latest .

# イメージ確認
docker images | grep tumblr-collector
```

#### コンテナ実行
```bash
# 環境変数ファイル作成
cat > .env <<EOF
TUMBLR_CONSUMER_KEY=your_key
TUMBLR_CONSUMER_SECRET=your_secret
EOF

# Docker Compose起動
docker-compose up -d

# ログ確認
docker-compose logs -f

# ステータス確認
docker-compose ps
```

#### コンテナヘルスチェック
```bash
# ヘルスチェック
docker exec tumblr-collector python -c "from production_monitoring import get_monitoring_dashboard; print(get_monitoring_dashboard().health_checker.get_health_summary())"

# リソース使用状況
docker stats tumblr-collector
```

---

## ☸️ Kubernetesデプロイメント

### Kubernetes環境

#### シークレット作成
```bash
# Tumblr APIシークレット
kubectl create secret generic tumblr-api-credentials \
  --from-literal=consumer-key=${TUMBLR_CONSUMER_KEY} \
  --from-literal=consumer-secret=${TUMBLR_CONSUMER_SECRET} \
  --from-literal=token=${TUMBLR_TOKEN} \
  --from-literal=token-secret=${TUMBLR_TOKEN_SECRET}

# シークレット確認
kubectl get secrets
```

#### ConfigMap作成
```bash
# 設定ファイルからConfigMap作成
kubectl create configmap tumblr-collector-config \
  --from-file=config.json

# ConfigMap確認
kubectl describe configmap tumblr-collector-config
```

#### デプロイメント
```bash
# 全リソースをデプロイ
kubectl apply -f kubernetes/

# デプロイメント確認
kubectl get deployments
kubectl get pods
kubectl get services

# Pod詳細確認
kubectl describe pod <pod-name>

# ログ確認
kubectl logs -f deployment/tumblr-collector
```

#### スケーリング
```bash
# 手動スケール
kubectl scale deployment tumblr-collector --replicas=3

# オートスケール設定
kubectl autoscale deployment tumblr-collector \
  --cpu-percent=70 \
  --min=2 \
  --max=10
```

---

## 📊 監視・ヘルスチェック

### ヘルスチェックエンドポイント

#### システムヘルスチェック
```python
from production_monitoring import get_monitoring_dashboard

dashboard = get_monitoring_dashboard()
health = dashboard.health_checker.get_health_summary()
print(f"Overall Status: {health['overall_status']}")
```

#### メトリクス確認
```python
# メトリクスサマリー
summary = dashboard.get_dashboard_summary()
print(f"Total Requests: {summary['metrics']['requests.total']}")
print(f"Error Rate: {summary['metrics']['requests.failed'] / summary['metrics']['requests.total'] * 100}%")
```

### ログ監視
```bash
# アプリケーションログ
tail -f tumblr_collector.log

# エラーログ
grep ERROR tumblr_collector.log

# セキュリティ監査ログ
grep SECURITY tumblr_collector.log
```

---

## 🚀 本番環境起動

### 標準実行
```bash
# 基本起動
python tumblr_image_collector.py <blog_name>

# フィルタ付き起動
python tumblr_image_collector.py <blog_name> \
  --tags illustration fanart \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --min-width 1000 \
  --min-height 1000

# インタラクティブモード
python tumblr_image_collector.py --interactive
```

### バックグラウンド実行
```bash
# systemdサービス（Linux）
cat > /etc/systemd/system/tumblr-collector.service <<EOF
[Unit]
Description=Tumblr Image Collector
After=network.target

[Service]
Type=simple
User=tumblr
WorkingDirectory=/opt/tumblr-collector
ExecStart=/opt/tumblr-collector/.venv/bin/python tumblr_image_collector.py --daemon
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
EOF

# サービス有効化
sudo systemctl daemon-reload
sudo systemctl enable tumblr-collector
sudo systemctl start tumblr-collector
sudo systemctl status tumblr-collector
```

---

## 🔧 トラブルシューティング

### 一般的な問題

#### 1. API認証エラー
```
問題: "API authentication failed"
解決策:
- 環境変数が正しく設定されているか確認
- Tumblr APIキーが有効か確認
- config.jsonの設定を確認
```

#### 2. レート制限エラー
```
問題: "Rate limit exceeded"
解決策:
- レート制限設定を調整（max_download_workers削減）
- リトライ戦略の設定を確認
- Tumblr APIのクォータを確認
```

#### 3. メモリ不足
```
問題: "Out of memory"
解決策:
- max_download_workersを削減
- メモリキャッシュサイズを削減
- システムメモリを増強
```

#### 4. ディスク容量不足
```
問題: "No space left on device"
解決策:
- 古い画像を削除
- ディスク容量を増強
- 画像最適化を有効化
```

### デバッグモード
```bash
# 詳細ログ有効化
export LOG_LEVEL=DEBUG
python tumblr_image_collector.py <blog_name>

# クラッシュレポート確認
ls -la crash_reports/

# データベース確認
sqlite3 url_manager.db "SELECT * FROM url_records LIMIT 10;"
```

---

## 📈 パフォーマンス最適化

### 推奨設定

#### 高速ダウンロード設定
```json
{
  "max_download_workers": 20,
  "network": {
    "download_timeout_seconds": 30,
    "max_retries": 2,
    "connection_pool_size": 50
  },
  "cache": {
    "memory_cache_size": 5000,
    "disk_cache_enabled": true
  }
}
```

#### 低リソース設定
```json
{
  "max_download_workers": 3,
  "network": {
    "download_timeout_seconds": 60,
    "max_retries": 5,
    "connection_pool_size": 10
  },
  "cache": {
    "memory_cache_size": 500,
    "disk_cache_enabled": false
  }
}
```

---

## ✅ デプロイ後検証

### 検証手順

1. **基本動作確認**
```bash
# テストブログから1枚ダウンロード
python tumblr_image_collector.py staff --limit 1

# 成功を確認
ls -la tumblr_images/
```

2. **セキュリティ検証**
```bash
# セキュリティテスト実行
pytest test_production_systems.py::test_security -v
```

3. **パフォーマンス検証**
```bash
# パフォーマンステスト実行
time python tumblr_image_collector.py <blog_name> --limit 100
```

4. **監視システム検証**
```bash
# メトリクス確認
python -c "from production_monitoring import get_monitoring_dashboard; print(get_monitoring_dashboard().get_dashboard_summary())"
```

---

## 📞 サポート・連絡先

### ドキュメント
- **README**: プロジェクト概要
- **INSTALLATION_GUIDE**: インストール詳細
- **DEVELOPER_GUIDE**: 開発者向け情報
- **API_REFERENCE**: API完全リファレンス

### トラブルシューティング
- **ログ**: `tumblr_collector.log`
- **クラッシュレポート**: `crash_reports/`
- **ヘルスチェック**: 監視ダッシュボード

### コミュニティ
- **GitHub Issues**: バグレポート・機能要望
- **Discussions**: 質問・議論

---

## 🎯 デプロイメント成功基準

### 必須要件
- [x] 全テストがパス
- [x] セキュリティチェック完了
- [x] ドキュメント完備
- [x] 環境変数設定完了
- [x] 依存関係インストール完了

### 推奨要件
- [ ] パフォーマンステスト実施
- [ ] 監視システム稼働確認
- [ ] バックアップ戦略策定
- [ ] ロールバック手順確認
- [ ] 運用マニュアル作成

### 本番環境基準
- [ ] 99.9%可用性達成
- [ ] セキュリティインシデント0件
- [ ] 平均復旧時間5分以内
- [ ] エラー率1%未満
- [ ] レスポンス時間1秒以内

---

## 🎉 デプロイメント完了

すべてのチェックリスト項目が完了したら、本番環境デプロイメントの準備が整っています。

**次のステップ:**
1. 段階的ロールアウト（カナリアデプロイメント）
2. モニタリング強化
3. ユーザーフィードバック収集
4. 継続的改善サイクル開始

---

**ドキュメントバージョン**: 1.0.0
**対象システムバージョン**: Tumblr Image Collector v2.0.0
**最終検証日**: 2025-10-05
