"""
監視・ヘルスチェックシステム
システムの健全性、パフォーマンス、エラー追跡を統合管理
"""

import json
import time
import logging
import threading
import queue
import sqlite3
import psutil
import platform
import socket
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import traceback
import sys
import os
import signal

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """ヘルスステータス"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """アラート重要度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """ヘルスチェック結果"""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0


@dataclass
class Alert:
    """アラート"""
    id: str
    title: str
    message: str
    severity: AlertSeverity
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MetricsCollector:
    """メトリクス収集システム"""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or Path.home() / ".tumblr_collector" / "metrics.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self.collection_interval = 60  # 秒
        self.retention_days = 30
        self.metrics_queue = queue.Queue()
        self.running = False

    def _init_database(self):
        """データベース初期化"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    tags TEXT,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
                ON metrics(timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_name
                ON metrics(metric_name)
            """)

            # アラートテーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    source TEXT,
                    details TEXT,
                    resolved BOOLEAN DEFAULT 0,
                    resolved_at DATETIME
                )
            """)

            # イベントログテーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT NOT NULL,
                    event_name TEXT NOT NULL,
                    event_data TEXT,
                    severity TEXT
                )
            """)

            conn.commit()

    def record_metric(
        self,
        name: str,
        value: float,
        tags: Dict[str, str] = None,
        metadata: Dict[str, Any] = None
    ):
        """メトリクス記録"""
        self.metrics_queue.put({
            'name': name,
            'value': value,
            'tags': json.dumps(tags or {}),
            'metadata': json.dumps(metadata or {}),
            'timestamp': datetime.utcnow()
        })

    def _persist_metrics(self):
        """メトリクスの永続化"""
        metrics_batch = []

        # キューから取得
        while not self.metrics_queue.empty():
            try:
                metric = self.metrics_queue.get_nowait()
                metrics_batch.append(metric)
            except queue.Empty:
                break

        if not metrics_batch:
            return

        # データベースに保存
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executemany("""
                INSERT INTO metrics (timestamp, metric_name, metric_value, tags, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, [
                (
                    m['timestamp'],
                    m['name'],
                    m['value'],
                    m['tags'],
                    m['metadata']
                )
                for m in metrics_batch
            ])
            conn.commit()

    def query_metrics(
        self,
        metric_name: str,
        start_time: datetime = None,
        end_time: datetime = None,
        aggregation: str = None
    ) -> List[Dict[str, Any]]:
        """メトリクス照会"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM metrics WHERE metric_name = ?"
            params = [metric_name]

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            if aggregation:
                if aggregation == 'avg':
                    query = f"SELECT AVG(metric_value) as value, DATE(timestamp) as date FROM metrics WHERE metric_name = ?"
                elif aggregation == 'sum':
                    query = f"SELECT SUM(metric_value) as value, DATE(timestamp) as date FROM metrics WHERE metric_name = ?"
                elif aggregation == 'max':
                    query = f"SELECT MAX(metric_value) as value, DATE(timestamp) as date FROM metrics WHERE metric_name = ?"
                elif aggregation == 'min':
                    query = f"SELECT MIN(metric_value) as value, DATE(timestamp) as date FROM metrics WHERE metric_name = ?"

                query += " GROUP BY DATE(timestamp)"

            cursor = conn.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]

        return results

    def cleanup_old_metrics(self):
        """古いメトリクスの削除"""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "DELETE FROM metrics WHERE timestamp < ?",
                (cutoff_date,)
            )
            conn.execute(
                "DELETE FROM events WHERE timestamp < ?",
                (cutoff_date,)
            )
            conn.commit()


class HealthMonitor:
    """ヘルスモニター"""

    def __init__(self, app_name: str = "TumblrImageCollector"):
        self.app_name = app_name
        self.checks = {}
        self.check_results = {}
        self.monitoring_thread = None
        self.running = False
        self.check_interval = 60  # 秒
        self.metrics_collector = MetricsCollector()

    def register_check(
        self,
        name: str,
        check_func: Callable[[], HealthCheck],
        critical: bool = False
    ):
        """ヘルスチェックの登録"""
        self.checks[name] = {
            'func': check_func,
            'critical': critical
        }

    def run_check(self, name: str) -> HealthCheck:
        """個別のヘルスチェック実行"""
        if name not in self.checks:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Check not found"
            )

        start_time = time.time()

        try:
            result = self.checks[name]['func']()
            result.latency_ms = (time.time() - start_time) * 1000

            # メトリクス記録
            self.metrics_collector.record_metric(
                f"health_check.{name}.latency",
                result.latency_ms
            )
            self.metrics_collector.record_metric(
                f"health_check.{name}.status",
                1 if result.status == HealthStatus.HEALTHY else 0
            )

            return result

        except Exception as e:
            logger.error(f"Health check {name} failed: {e}")
            return HealthCheck(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Check failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000
            )

    def run_all_checks(self) -> Dict[str, HealthCheck]:
        """すべてのヘルスチェック実行"""
        results = {}

        for name in self.checks:
            results[name] = self.run_check(name)
            self.check_results[name] = results[name]

        return results

    def get_overall_status(self) -> HealthStatus:
        """全体のヘルスステータス取得"""
        if not self.check_results:
            return HealthStatus.HEALTHY

        statuses = [result.status for result in self.check_results.values()]

        if any(s == HealthStatus.CRITICAL for s in statuses):
            return HealthStatus.CRITICAL
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def start_monitoring(self):
        """監視開始"""
        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """監視停止"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()

    def _monitoring_loop(self):
        """監視ループ"""
        while self.running:
            try:
                self.run_all_checks()
                self.metrics_collector._persist_metrics()
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")


class AlertManager:
    """アラート管理システム"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_handlers = []
        self.active_alerts = {}
        self.alert_rules = []

    def add_handler(self, handler: Callable[[Alert], None]):
        """アラートハンドラー追加"""
        self.alert_handlers.append(handler)

    def add_rule(
        self,
        name: str,
        condition: Callable[[], bool],
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING
    ):
        """アラートルール追加"""
        self.alert_rules.append({
            'name': name,
            'condition': condition,
            'message': message,
            'severity': severity
        })

    def trigger_alert(self, alert: Alert):
        """アラート発火"""
        # アクティブアラートに追加
        self.active_alerts[alert.id] = alert

        # データベースに記録
        with sqlite3.connect(str(self.metrics_collector.db_path)) as conn:
            conn.execute("""
                INSERT INTO alerts (id, title, message, severity, source, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                alert.id,
                alert.title,
                alert.message,
                alert.severity.value,
                alert.source,
                json.dumps(asdict(alert))
            ))
            conn.commit()

        # ハンドラー呼び出し
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")

    def resolve_alert(self, alert_id: str):
        """アラート解決"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()

            # データベース更新
            with sqlite3.connect(str(self.metrics_collector.db_path)) as conn:
                conn.execute("""
                    UPDATE alerts
                    SET resolved = 1, resolved_at = ?
                    WHERE id = ?
                """, (alert.resolved_at, alert_id))
                conn.commit()

            del self.active_alerts[alert_id]

    def check_rules(self):
        """アラートルールチェック"""
        for rule in self.alert_rules:
            try:
                if rule['condition']():
                    alert = Alert(
                        id=f"{rule['name']}_{int(time.time())}",
                        title=rule['name'],
                        message=rule['message'],
                        severity=rule['severity'],
                        source="rule_engine"
                    )
                    self.trigger_alert(alert)

            except Exception as e:
                logger.error(f"Rule check error for {rule['name']}: {e}")


class SystemHealthChecks:
    """システムヘルスチェック集"""

    @staticmethod
    def check_disk_space(threshold: float = 90.0) -> HealthCheck:
        """ディスク容量チェック"""
        disk_usage = psutil.disk_usage('/')
        usage_percent = disk_usage.percent

        if usage_percent > threshold:
            status = HealthStatus.CRITICAL
            message = f"Disk usage critical: {usage_percent:.1f}%"
        elif usage_percent > threshold * 0.9:
            status = HealthStatus.DEGRADED
            message = f"Disk usage high: {usage_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk usage normal: {usage_percent:.1f}%"

        return HealthCheck(
            name="disk_space",
            status=status,
            message=message,
            details={
                'usage_percent': usage_percent,
                'used_bytes': disk_usage.used,
                'free_bytes': disk_usage.free,
                'total_bytes': disk_usage.total
            }
        )

    @staticmethod
    def check_memory_usage(threshold: float = 80.0) -> HealthCheck:
        """メモリ使用率チェック"""
        memory = psutil.virtual_memory()
        usage_percent = memory.percent

        if usage_percent > threshold:
            status = HealthStatus.UNHEALTHY
            message = f"Memory usage high: {usage_percent:.1f}%"
        elif usage_percent > threshold * 0.8:
            status = HealthStatus.DEGRADED
            message = f"Memory usage elevated: {usage_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory usage normal: {usage_percent:.1f}%"

        return HealthCheck(
            name="memory_usage",
            status=status,
            message=message,
            details={
                'usage_percent': usage_percent,
                'used_bytes': memory.used,
                'available_bytes': memory.available,
                'total_bytes': memory.total
            }
        )

    @staticmethod
    def check_cpu_usage(threshold: float = 80.0) -> HealthCheck:
        """CPU使用率チェック"""
        cpu_percent = psutil.cpu_percent(interval=1)

        if cpu_percent > threshold:
            status = HealthStatus.DEGRADED
            message = f"CPU usage high: {cpu_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"CPU usage normal: {cpu_percent:.1f}%"

        return HealthCheck(
            name="cpu_usage",
            status=status,
            message=message,
            details={
                'usage_percent': cpu_percent,
                'cpu_count': psutil.cpu_count(),
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
        )

    @staticmethod
    def check_network_connectivity() -> HealthCheck:
        """ネットワーク接続チェック"""
        try:
            # DNSチェック
            socket.gethostbyname("api.tumblr.com")

            # 接続チェック
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(("api.tumblr.com", 443))
            sock.close()

            if result == 0:
                status = HealthStatus.HEALTHY
                message = "Network connectivity OK"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Cannot connect to Tumblr API"

        except socket.gaierror:
            status = HealthStatus.UNHEALTHY
            message = "DNS resolution failed"

        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Network check failed: {str(e)}"

        return HealthCheck(
            name="network_connectivity",
            status=status,
            message=message
        )

    @staticmethod
    def check_database(db_path: Path) -> HealthCheck:
        """データベース接続チェック"""
        try:
            with sqlite3.connect(str(db_path), timeout=5) as conn:
                cursor = conn.execute("SELECT 1")
                cursor.fetchone()

            status = HealthStatus.HEALTHY
            message = "Database connection OK"

            # データベースサイズチェック
            db_size = db_path.stat().st_size
            details = {'database_size': db_size}

            if db_size > 1024 * 1024 * 1024:  # 1GB
                status = HealthStatus.DEGRADED
                message = f"Database size large: {db_size / 1024 / 1024:.1f}MB"

        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Database check failed: {str(e)}"
            details = {}

        return HealthCheck(
            name="database",
            status=status,
            message=message,
            details=details
        )


class ErrorTracker:
    """エラー追跡システム"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.error_counts = {}
        self.error_patterns = {}

    def track_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None
    ):
        """エラー追跡"""
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()

        # エラーカウント更新
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1

        # パターン分析
        pattern_key = f"{error_type}:{error_message[:50]}"
        if pattern_key not in self.error_patterns:
            self.error_patterns[pattern_key] = {
                'count': 0,
                'first_seen': datetime.utcnow(),
                'last_seen': None,
                'contexts': []
            }

        self.error_patterns[pattern_key]['count'] += 1
        self.error_patterns[pattern_key]['last_seen'] = datetime.utcnow()

        if context:
            self.error_patterns[pattern_key]['contexts'].append(context)

        # データベースに記録
        with sqlite3.connect(str(self.metrics_collector.db_path)) as conn:
            conn.execute("""
                INSERT INTO events (event_type, event_name, event_data, severity)
                VALUES (?, ?, ?, ?)
            """, (
                'error',
                error_type,
                json.dumps({
                    'message': error_message,
                    'stack_trace': stack_trace,
                    'context': context
                }),
                'ERROR'
            ))
            conn.commit()

        # メトリクス記録
        self.metrics_collector.record_metric(
            f"error.{error_type}",
            1,
            tags={'error_message': error_message[:100]}
        )

    def get_error_summary(self) -> Dict[str, Any]:
        """エラーサマリー取得"""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_types': self.error_counts,
            'top_patterns': sorted(
                self.error_patterns.items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )[:10]
        }


class LivenessProbe:
    """生存性プローブ"""

    def __init__(self, port: int = 8080):
        self.port = port
        self.server_thread = None
        self.running = False

    def start(self):
        """プローブサーバー開始"""
        from http.server import HTTPServer, BaseHTTPRequestHandler

        class ProbeHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/health/live':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b'OK')
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass  # ログ抑制

        self.running = True
        server = HTTPServer(('', self.port), ProbeHandler)

        self.server_thread = threading.Thread(
            target=lambda: server.serve_forever() if self.running else None,
            daemon=True
        )
        self.server_thread.start()

        logger.info(f"Liveness probe started on port {self.port}")

    def stop(self):
        """プローブサーバー停止"""
        self.running = False


class DiagnosticTool:
    """診断ツール"""

    @staticmethod
    def system_info() -> Dict[str, Any]:
        """システム情報取得"""
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'processor': platform.processor(),
            'hostname': socket.gethostname(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'disk_partitions': [
                {
                    'device': p.device,
                    'mountpoint': p.mountpoint,
                    'fstype': p.fstype,
                    'usage': psutil.disk_usage(p.mountpoint)._asdict()
                }
                for p in psutil.disk_partitions()
            ],
            'network_interfaces': {
                name: [addr._asdict() for addr in addrs]
                for name, addrs in psutil.net_if_addrs().items()
            }
        }

    @staticmethod
    def process_info(pid: int = None) -> Dict[str, Any]:
        """プロセス情報取得"""
        if pid is None:
            pid = os.getpid()

        try:
            process = psutil.Process(pid)

            return {
                'pid': pid,
                'name': process.name(),
                'status': process.status(),
                'cpu_percent': process.cpu_percent(),
                'memory_info': process.memory_info()._asdict(),
                'num_threads': process.num_threads(),
                'open_files': [f.path for f in process.open_files()],
                'connections': [c._asdict() for c in process.connections()],
                'create_time': datetime.fromtimestamp(process.create_time()),
                'cmdline': process.cmdline()
            }

        except psutil.NoSuchProcess:
            return {'error': f'Process {pid} not found'}

    @staticmethod
    def dependency_check() -> Dict[str, Any]:
        """依存関係チェック"""
        dependencies = {}

        # Pythonパッケージチェック
        required_packages = [
            'requests', 'pytumblr', 'pillow', 'numpy',
            'scikit-image', 'psutil', 'cryptography'
        ]

        for package in required_packages:
            try:
                module = __import__(package.replace('-', '_'))
                version = getattr(module, '__version__', 'Unknown')
                dependencies[package] = {
                    'installed': True,
                    'version': version
                }
            except ImportError:
                dependencies[package] = {
                    'installed': False,
                    'version': None
                }

        # システムコマンドチェック
        system_commands = ['git', 'python3', 'pip']

        for cmd in system_commands:
            try:
                result = subprocess.run(
                    [cmd, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                dependencies[cmd] = {
                    'available': result.returncode == 0,
                    'version': result.stdout.strip()
                }
            except (subprocess.TimeoutExpired, FileNotFoundError):
                dependencies[cmd] = {
                    'available': False,
                    'version': None
                }

        return dependencies


# グローバルモニターインスタンス
_global_monitor = None


def get_monitor() -> HealthMonitor:
    """グローバルモニター取得"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = HealthMonitor()

        # デフォルトヘルスチェック登録
        _global_monitor.register_check(
            "disk_space",
            SystemHealthChecks.check_disk_space
        )
        _global_monitor.register_check(
            "memory_usage",
            SystemHealthChecks.check_memory_usage
        )
        _global_monitor.register_check(
            "cpu_usage",
            SystemHealthChecks.check_cpu_usage
        )
        _global_monitor.register_check(
            "network",
            SystemHealthChecks.check_network_connectivity,
            critical=True
        )

    return _global_monitor


# 使用例とテスト
if __name__ == "__main__":
    # モニター設定
    monitor = get_monitor()
    monitor.start_monitoring()

    # アラート設定
    alert_manager = AlertManager(monitor.metrics_collector)

    def email_handler(alert: Alert):
        print(f"Email Alert: {alert.title} - {alert.message}")

    alert_manager.add_handler(email_handler)

    # アラートルール追加
    alert_manager.add_rule(
        name="high_memory",
        condition=lambda: psutil.virtual_memory().percent > 80,
        message="Memory usage exceeded 80%",
        severity=AlertSeverity.WARNING
    )

    # エラートラッカー
    error_tracker = ErrorTracker(monitor.metrics_collector)

    try:
        # エラーのシミュレーション
        raise ValueError("Test error")
    except Exception as e:
        error_tracker.track_error(e, {'function': 'test', 'input': 'sample'})

    # 診断情報
    print("\nSystem Information:")
    print(json.dumps(DiagnosticTool.system_info(), indent=2, default=str))

    print("\nProcess Information:")
    print(json.dumps(DiagnosticTool.process_info(), indent=2, default=str))

    print("\nDependency Check:")
    print(json.dumps(DiagnosticTool.dependency_check(), indent=2))

    # ヘルスチェック実行
    print("\nHealth Checks:")
    results = monitor.run_all_checks()
    for name, result in results.items():
        print(f"- {name}: {result.status.value} - {result.message}")

    print(f"\nOverall Status: {monitor.get_overall_status().value}")

    # エラーサマリー
    print("\nError Summary:")
    print(json.dumps(error_tracker.get_error_summary(), indent=2, default=str))

    # 停止
    monitor.stop_monitoring()