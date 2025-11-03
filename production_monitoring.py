#!/usr/bin/env python3
"""
Production Monitoring and Health Check System
Comprehensive system monitoring, metrics collection, and alerting
"""

import logging
import time
import psutil
import threading
import json
import os
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import sqlite3

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    """Health check result"""
    component: str
    status: HealthStatus
    timestamp: float
    response_time_ms: float
    message: str
    metadata: Dict[str, Any]


@dataclass
class MetricSnapshot:
    """Metric snapshot"""
    timestamp: float
    metric_name: str
    value: float
    tags: Dict[str, str]


class MetricsCollector:
    """
    Metrics collection and aggregation
    """

    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment counter metric"""
        with self._lock:
            key = self._make_key(name, tags)
            self._counters[key] += value

            # Record snapshot
            snapshot = MetricSnapshot(
                timestamp=time.time(),
                metric_name=name,
                value=self._counters[key],
                tags=tags or {}
            )
            self._metrics[name].append(snapshot)

    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set gauge metric"""
        with self._lock:
            key = self._make_key(name, tags)
            self._gauges[key] = value

            snapshot = MetricSnapshot(
                timestamp=time.time(),
                metric_name=name,
                value=value,
                tags=tags or {}
            )
            self._metrics[name].append(snapshot)

    def timer(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """Record timer metric"""
        with self._lock:
            key = self._make_key(name, tags)
            self._timers[key].append(duration_ms)

            snapshot = MetricSnapshot(
                timestamp=time.time(),
                metric_name=name,
                value=duration_ms,
                tags=tags or {}
            )
            self._metrics[name].append(snapshot)

    def record_download_metrics(self, blog_name: str, download_time: float,
                               file_size: int, success: bool, error_type: str = None):
        """Record detailed download metrics"""
        tags = {
            "blog": blog_name,
            "success": "true" if success else "false",
            "file_size_mb": str(file_size / (1024 * 1024))
        }

        if not success and error_type:
            tags["error_type"] = error_type

        self.counter("downloads.total", 1, tags)
        self.histogram("download.duration", download_time, tags)
        self.gauge("download.file_size", file_size, tags)

        if success:
            self.counter("downloads.success", 1, tags)
        else:
            self.counter("downloads.failure", 1, tags)

    def record_system_metrics(self):
        """Record comprehensive system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_freq = psutil.cpu_freq()
            self.gauge("system.cpu.percent", cpu_percent)
            if cpu_freq:
                self.gauge("system.cpu.frequency", cpu_freq.current)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.gauge("system.memory.percent", memory.percent)
            self.gauge("system.memory.used_gb", memory.used / (1024**3))
            self.gauge("system.memory.available_gb", memory.available / (1024**3))

            # Disk metrics
            disk = psutil.disk_usage('/')
            self.gauge("system.disk.percent", disk.percent)
            self.gauge("system.disk.used_gb", disk.used / (1024**3))
            self.gauge("system.disk.free_gb", disk.free / (1024**3))

            # Network metrics
            net = psutil.net_io_counters()
            self.counter("system.network.bytes_sent", net.bytes_sent)
            self.counter("system.network.bytes_recv", net.bytes_recv)

            # Process metrics
            process = psutil.Process()
            self.gauge("process.memory.rss_mb", process.memory_info().rss / (1024**2))
            self.gauge("process.cpu.percent", process.cpu_percent())

        except Exception as e:
            logger.error(f"Failed to record system metrics: {e}")

    def record_cache_metrics(self, cache_hits: int, cache_misses: int, cache_size: int):
        """Record cache performance metrics"""
        total_requests = cache_hits + cache_misses
        if total_requests > 0:
            hit_rate = cache_hits / total_requests
        else:
            hit_rate = 0.0

        self.counter("cache.requests", total_requests)
        self.counter("cache.hits", cache_hits)
        self.counter("cache.misses", cache_misses)
        self.gauge("cache.hit_rate", hit_rate)
        self.gauge("cache.size", cache_size)

    def record_security_metrics(self, event_type: str, severity: str, details: Dict = None):
        """Record security-related metrics"""
        tags = {
            "event_type": event_type,
            "severity": severity
        }

        if details:
            for key, value in details.items():
                tags[f"detail_{key}"] = str(value)

        self.counter("security.events", 1, tags)

    def record_download_metrics(self, blog_name: str, download_time: float,
                               file_size: int, success: bool, error_type: str = None):
        """Record detailed download metrics"""
        tags = {
            "blog": blog_name,
            "success": "true" if success else "false",
            "file_size_mb": str(file_size / (1024 * 1024))
        }

        if not success and error_type:
            tags["error_type"] = error_type

        self.counter("downloads.total", 1, tags)
        self.histogram("download.duration", download_time, tags)
        self.gauge("download.file_size", file_size, tags)

        if success:
            self.counter("downloads.success", 1, tags)
        else:
            self.counter("downloads.failure", 1, tags)

    def record_system_metrics(self):
        """Record comprehensive system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_freq = psutil.cpu_freq()
            self.gauge("system.cpu.percent", cpu_percent)
            if cpu_freq:
                self.gauge("system.cpu.frequency", cpu_freq.current)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.gauge("system.memory.percent", memory.percent)
            self.gauge("system.memory.used_gb", memory.used / (1024**3))
            self.gauge("system.memory.available_gb", memory.available / (1024**3))

            # Disk metrics
            disk = psutil.disk_usage('/')
            self.gauge("system.disk.percent", disk.percent)
            self.gauge("system.disk.used_gb", disk.used / (1024**3))
            self.gauge("system.disk.free_gb", disk.free / (1024**3))

            # Network metrics
            net = psutil.net_io_counters()
            self.counter("system.network.bytes_sent", net.bytes_sent)
            self.counter("system.network.bytes_recv", net.bytes_recv)

            # Process metrics
            process = psutil.Process()
            self.gauge("process.memory.rss_mb", process.memory_info().rss / (1024**2))
            self.gauge("process.cpu.percent", process.cpu_percent())

        except Exception as e:
            logger.error(f"Failed to record system metrics: {e}")

    def record_cache_metrics(self, cache_hits: int, cache_misses: int, cache_size: int):
        """Record cache performance metrics"""
        total_requests = cache_hits + cache_misses
        if total_requests > 0:
            hit_rate = cache_hits / total_requests
        else:
            hit_rate = 0.0

        self.counter("cache.requests", total_requests)
        self.counter("cache.hits", cache_hits)
        self.counter("cache.misses", cache_misses)
        self.gauge("cache.hit_rate", hit_rate)
        self.gauge("cache.size", cache_size)

    def record_security_metrics(self, event_type: str, severity: str, details: Dict = None):
        """Record security-related metrics"""
        tags = {
            "event_type": event_type,
            "severity": severity
        }

        if details:
            for key, value in details.items():
                tags[f"detail_{key}"] = str(value)

        self.counter("security.events", 1, tags)

    def get_metric_stats(self, name: str, time_window_seconds: int = 3600) -> Dict[str, Any]:
        """Get statistics for metric"""
        cutoff_time = time.time() - time_window_seconds

        with self._lock:
            snapshots = [
                s for s in self._metrics.get(name, [])
                if s.timestamp > cutoff_time
            ]

            if not snapshots:
                return {
                    'count': 0,
                    'mean': 0,
                    'min': 0,
                    'max': 0,
                    'sum': 0
                }

            values = [s.value for s in snapshots]

            return {
                'count': len(values),
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'sum': sum(values),
                'latest': values[-1] if values else 0
            }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics"""
        with self._lock:
            return {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'timers': {
                    name: {
                        'count': len(values),
                        'mean': sum(values) / len(values) if values else 0,
                        'min': min(values) if values else 0,
                        'max': max(values) if values else 0
                    }
                    for name, values in self._timers.items()
                }
            }

    def cleanup_old_metrics(self):
        """Clean up old metric snapshots"""
        cutoff_time = time.time() - (self.retention_hours * 3600)

        with self._lock:
            for name in list(self._metrics.keys()):
                snapshots = self._metrics[name]
                # Remove old snapshots
                while snapshots and snapshots[0].timestamp < cutoff_time:
                    snapshots.popleft()

                if not snapshots:
                    del self._metrics[name]

    @staticmethod
    def _make_key(name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create unique key from name and tags"""
        if not tags:
            return name

        tag_str = ','.join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}:{tag_str}"


class SystemMonitor:
    """
    System resource monitoring
    """

    def __init__(self):
        self._process = psutil.Process(os.getpid())

    def get_cpu_usage(self) -> Dict[str, Any]:
        """Get CPU usage statistics"""
        try:
            return {
                'process_percent': self._process.cpu_percent(interval=0.1),
                'system_percent': psutil.cpu_percent(interval=0.1),
                'cpu_count': psutil.cpu_count(),
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
        except Exception as e:
            logger.error(f"Failed to get CPU usage: {e}")
            return {}

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        try:
            process_memory = self._process.memory_info()
            system_memory = psutil.virtual_memory()

            return {
                'process_rss_mb': process_memory.rss / 1024 / 1024,
                'process_vms_mb': process_memory.vms / 1024 / 1024,
                'system_total_mb': system_memory.total / 1024 / 1024,
                'system_available_mb': system_memory.available / 1024 / 1024,
                'system_percent': system_memory.percent
            }
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {}

    def get_disk_usage(self, path: str = "/") -> Dict[str, Any]:
        """Get disk usage statistics"""
        try:
            disk = psutil.disk_usage(path)

            return {
                'total_gb': disk.total / 1024 / 1024 / 1024,
                'used_gb': disk.used / 1024 / 1024 / 1024,
                'free_gb': disk.free / 1024 / 1024 / 1024,
                'percent': disk.percent
            }
        except Exception as e:
            logger.error(f"Failed to get disk usage: {e}")
            return {}

    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics"""
        try:
            net_io = psutil.net_io_counters()

            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errors_in': net_io.errin,
                'errors_out': net_io.errout,
                'drops_in': net_io.dropin,
                'drops_out': net_io.dropout
            }
        except Exception as e:
            logger.error(f"Failed to get network stats: {e}")
            return {}

    def get_thread_count(self) -> int:
        """Get active thread count"""
        try:
            return self._process.num_threads()
        except Exception:
            return threading.active_count()

    def get_open_files(self) -> int:
        """Get open file descriptor count"""
        try:
            return len(self._process.open_files())
        except Exception:
            return 0


class HealthChecker:
    """
    Component health checking
    """

    def __init__(self):
        self._checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._last_results: Dict[str, HealthCheckResult] = {}
        self._lock = threading.Lock()

    def register_check(self, name: str, check_func: Callable[[], HealthCheckResult]):
        """Register health check"""
        with self._lock:
            self._checks[name] = check_func

    def run_check(self, name: str) -> Optional[HealthCheckResult]:
        """Run specific health check"""
        check_func = self._checks.get(name)
        if not check_func:
            return None

        start_time = time.time()
        try:
            result = check_func()
            result.response_time_ms = (time.time() - start_time) * 1000

            with self._lock:
                self._last_results[name] = result

            return result
        except Exception as e:
            logger.error(f"Health check '{name}' failed: {e}")
            result = HealthCheckResult(
                component=name,
                status=HealthStatus.UNHEALTHY,
                timestamp=time.time(),
                response_time_ms=(time.time() - start_time) * 1000,
                message=f"Health check failed: {str(e)}",
                metadata={}
            )

            with self._lock:
                self._last_results[name] = result

            return result

    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks"""
        results = {}

        for name in list(self._checks.keys()):
            result = self.run_check(name)
            if result:
                results[name] = result

        return results

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status"""
        results = self.run_all_checks()

        if not results:
            return HealthStatus.HEALTHY

        statuses = [r.status for r in results.values()]

        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary"""
        results = self.run_all_checks()

        return {
            'overall_status': self.get_overall_status().value,
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {
                name: {
                    'status': result.status.value,
                    'message': result.message,
                    'response_time_ms': result.response_time_ms
                }
                for name, result in results.items()
            }
        }


class PerformanceMonitor:
    """
    Performance monitoring and profiling
    """

    def __init__(self, db_path: str = "performance_metrics.db"):
        self.db_path = Path(db_path)
        self._initialize_database()

    def _initialize_database(self):
        """Initialize metrics database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    operation TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    success INTEGER NOT NULL,
                    error_message TEXT,
                    metadata TEXT
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_operation ON performance_metrics(operation)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON performance_metrics(timestamp)")

            conn.commit()

    def record_operation(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record operation performance"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO performance_metrics (
                    timestamp, operation, duration_ms, success, error_message, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                time.time(),
                operation,
                duration_ms,
                1 if success else 0,
                error_message,
                json.dumps(metadata) if metadata else None
            ))
            conn.commit()

    def get_operation_stats(self, operation: str, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get statistics for operation"""
        cutoff_time = time.time() - (time_window_hours * 3600)

        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("""
                SELECT
                    COUNT(*) as total_count,
                    SUM(success) as success_count,
                    AVG(duration_ms) as avg_duration,
                    MIN(duration_ms) as min_duration,
                    MAX(duration_ms) as max_duration
                FROM performance_metrics
                WHERE operation = ? AND timestamp > ?
            """, (operation, cutoff_time)).fetchone()

            if not result or result[0] == 0:
                return {
                    'total_count': 0,
                    'success_count': 0,
                    'success_rate': 0,
                    'avg_duration_ms': 0,
                    'min_duration_ms': 0,
                    'max_duration_ms': 0
                }

            total, success, avg_dur, min_dur, max_dur = result

            return {
                'total_count': total,
                'success_count': success,
                'success_rate': (success / total * 100) if total > 0 else 0,
                'avg_duration_ms': avg_dur or 0,
                'min_duration_ms': min_dur or 0,
                'max_duration_ms': max_dur or 0
            }

    def get_slow_operations(self, threshold_ms: float = 1000, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest operations"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT operation, duration_ms, timestamp, metadata
                FROM performance_metrics
                WHERE duration_ms > ?
                ORDER BY duration_ms DESC
                LIMIT ?
            """, (threshold_ms, limit)).fetchall()

        return [
            {
                'operation': row[0],
                'duration_ms': row[1],
                'timestamp': datetime.fromtimestamp(row[2]).isoformat(),
                'metadata': json.loads(row[3]) if row[3] else {}
            }
            for row in rows
        ]

    def cleanup_old_metrics(self, max_age_days: int = 7):
        """Clean up old performance metrics"""
        cutoff_time = time.time() - (max_age_days * 86400)

        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "DELETE FROM performance_metrics WHERE timestamp < ?",
                (cutoff_time,)
            )
            removed_count = result.rowcount
            conn.commit()

        logger.info(f"Cleaned up {removed_count} old performance metrics")
        return removed_count


class MonitoringDashboard:
    """
    Centralized monitoring dashboard
    """

    def __init__(self):
        self.metrics = MetricsCollector()
        self.system_monitor = SystemMonitor()
        self.health_checker = HealthChecker()
        self.performance_monitor = PerformanceMonitor()

        # Register default health checks
        self._register_default_health_checks()

        # Start background monitoring
        self._start_background_monitoring()

    def _register_default_health_checks(self):
        """Register default health checks"""
        def check_memory() -> HealthCheckResult:
            memory = self.system_monitor.get_memory_usage()
            percent = memory.get('system_percent', 0)

            if percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Critical memory usage: {percent}%"
            elif percent > 85:
                status = HealthStatus.UNHEALTHY
                message = f"High memory usage: {percent}%"
            elif percent > 70:
                status = HealthStatus.DEGRADED
                message = f"Elevated memory usage: {percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {percent}%"

            return HealthCheckResult(
                component="memory",
                status=status,
                timestamp=time.time(),
                response_time_ms=0,
                message=message,
                metadata=memory
            )

        def check_disk() -> HealthCheckResult:
            disk = self.system_monitor.get_disk_usage()
            percent = disk.get('percent', 0)

            if percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Critical disk usage: {percent}%"
            elif percent > 85:
                status = HealthStatus.UNHEALTHY
                message = f"High disk usage: {percent}%"
            elif percent > 70:
                status = HealthStatus.DEGRADED
                message = f"Elevated disk usage: {percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {percent}%"

            return HealthCheckResult(
                component="disk",
                status=status,
                timestamp=time.time(),
                response_time_ms=0,
                message=message,
                metadata=disk
            )

        self.health_checker.register_check("memory", check_memory)
        self.health_checker.register_check("disk", check_disk)

    def _start_background_monitoring(self):
        """Start background monitoring thread"""
        def monitor_loop():
            while True:
                try:
                    # Collect system metrics
                    cpu = self.system_monitor.get_cpu_usage()
                    memory = self.system_monitor.get_memory_usage()

                    self.metrics.gauge("system.cpu.percent", cpu.get('system_percent', 0))
                    self.metrics.gauge("system.memory.percent", memory.get('system_percent', 0))
                    self.metrics.gauge("process.memory.rss_mb", memory.get('process_rss_mb', 0))
                    self.metrics.gauge("process.threads", self.system_monitor.get_thread_count())

                    # Cleanup old data
                    self.metrics.cleanup_old_metrics()

                except Exception as e:
                    logger.error(f"Background monitoring error: {e}")

                time.sleep(60)  # Run every minute

        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get comprehensive dashboard summary"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'health': self.health_checker.get_health_summary(),
            'system': {
                'cpu': self.system_monitor.get_cpu_usage(),
                'memory': self.system_monitor.get_memory_usage(),
                'disk': self.system_monitor.get_disk_usage(),
                'network': self.system_monitor.get_network_stats(),
                'threads': self.system_monitor.get_thread_count(),
                'open_files': self.system_monitor.get_open_files()
            },
            'metrics': self.metrics.get_all_metrics()
        }

    def export_metrics_json(self, output_file: str):
        """Export metrics to JSON file"""
        summary = self.get_dashboard_summary()

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Exported metrics to {output_file}")


# Global monitoring dashboard
_monitoring_dashboard = None


def get_monitoring_dashboard() -> MonitoringDashboard:
    """Get global monitoring dashboard"""
    global _monitoring_dashboard
    if _monitoring_dashboard is None:
        _monitoring_dashboard = MonitoringDashboard()
    return _monitoring_dashboard


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    dashboard = MonitoringDashboard()

    # Record some metrics
    dashboard.metrics.counter("requests.total", 1)
    dashboard.metrics.gauge("queue.size", 42)
    dashboard.metrics.timer("operation.duration", 123.45)

    # Get dashboard summary
    summary = dashboard.get_dashboard_summary()
    print(json.dumps(summary, indent=2))

    # Export metrics
    dashboard.export_metrics_json("metrics_snapshot.json")
