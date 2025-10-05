#!/usr/bin/env python3
"""
Personal Performance Optimizer
Optimizations specifically for single-user scenarios
"""

import os
import logging
import psutil
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import deque
import gc

logger = logging.getLogger(__name__)


class PersonalPerformanceOptimizer:
    """
    Performance optimizations for personal use:
    - Adaptive worker scaling
    - Smart resource management
    - Aggressive caching
    - Automatic optimization
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.personal_config = config.get('personal_features', {})

        # Performance metrics
        self.metrics = {
            'downloads_per_second': deque(maxlen=60),
            'memory_usage_mb': deque(maxlen=60),
            'cpu_usage_percent': deque(maxlen=60),
            'cache_hit_rate': deque(maxlen=60),
        }

        # Adaptive settings
        self.current_workers = config.get('max_download_workers', 10)
        self.optimal_workers = self.current_workers

        # Monitor thread
        self._monitor_thread = None
        self._monitoring = False

    def calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers based on system resources"""
        try:
            # Get system resources
            cpu_count = os.cpu_count() or 2
            memory = psutil.virtual_memory()
            available_mb = memory.available / 1024 / 1024

            # Calculate based on CPU
            cpu_based = min(cpu_count * 2, 20)

            # Calculate based on memory (assume 50MB per worker)
            memory_based = int(available_mb / 50)

            # Calculate based on network (estimate)
            network_based = 15

            # Take minimum to avoid overload
            optimal = min(cpu_based, memory_based, network_based)

            # Ensure at least 1, max 20
            return max(1, min(optimal, 20))

        except Exception as e:
            logger.error(f"Failed to calculate optimal workers: {e}")
            return 10

    def start_monitoring(self):
        """Start background performance monitoring"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                # Collect metrics
                process = psutil.Process()

                # Memory usage
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.metrics['memory_usage_mb'].append(memory_mb)

                # CPU usage
                cpu_percent = process.cpu_percent(interval=1)
                self.metrics['cpu_usage_percent'].append(cpu_percent)

                # Adaptive worker adjustment
                self._adjust_workers()

                # Automatic garbage collection if memory high
                if memory_mb > 1000:  # > 1GB
                    gc.collect()

                time.sleep(5)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(5)

    def _adjust_workers(self):
        """Automatically adjust worker count based on performance"""
        try:
            if len(self.metrics['cpu_usage_percent']) < 10:
                return

            avg_cpu = sum(self.metrics['cpu_usage_percent']) / len(self.metrics['cpu_usage_percent'])
            avg_memory_mb = sum(self.metrics['memory_usage_mb']) / len(self.metrics['memory_usage_mb'])

            # If CPU usage is low, we can increase workers
            if avg_cpu < 50 and avg_memory_mb < 800:
                self.optimal_workers = min(self.current_workers + 2, 20)

            # If CPU or memory is high, decrease workers
            elif avg_cpu > 80 or avg_memory_mb > 1500:
                self.optimal_workers = max(self.current_workers - 2, 2)

            # Gradual adjustment
            if self.optimal_workers != self.current_workers:
                logger.info(f"Adjusting workers: {self.current_workers} -> {self.optimal_workers}")
                self.current_workers = self.optimal_workers

        except Exception as e:
            logger.error(f"Failed to adjust workers: {e}")

    def optimize_system_settings(self):
        """Optimize system settings for downloads"""
        try:
            # Increase file descriptor limit (Unix-like systems)
            if hasattr(os, 'setrlimit'):
                import resource
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))

            # Set process priority to normal/high (Windows)
            if hasattr(psutil.Process(), 'nice'):
                process = psutil.Process()
                if os.name == 'nt':
                    process.nice(psutil.NORMAL_PRIORITY_CLASS)
                else:
                    process.nice(0)  # Normal priority

            logger.info("System settings optimized")

        except Exception as e:
            logger.warning(f"Could not optimize system settings: {e}")

    def cleanup_memory(self):
        """Aggressive memory cleanup"""
        try:
            # Force garbage collection
            gc.collect()

            # Clear Python's internal caches
            import sys
            sys.modules.clear()

            logger.info("Memory cleanup performed")

        except Exception as e:
            logger.error(f"Failed to cleanup memory: {e}")

    def optimize_disk_cache(self, cache_dir: Path):
        """Optimize disk cache for performance"""
        try:
            if not cache_dir.exists():
                return

            # Set optimal block size
            if hasattr(os, 'statvfs'):
                stats = os.statvfs(cache_dir)
                optimal_block_size = stats.f_bsize
                logger.info(f"Optimal block size: {optimal_block_size}")

            # Pre-allocate cache directories
            cache_subdirs = ['images', 'metadata', 'thumbnails']
            for subdir in cache_subdirs:
                (cache_dir / subdir).mkdir(parents=True, exist_ok=True)

        except Exception as e:
            logger.error(f"Failed to optimize disk cache: {e}")

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        try:
            process = psutil.Process()

            # Current metrics
            current_memory_mb = process.memory_info().rss / 1024 / 1024
            current_cpu = process.cpu_percent()

            # Average metrics
            avg_memory = sum(self.metrics['memory_usage_mb']) / len(self.metrics['memory_usage_mb']) if self.metrics['memory_usage_mb'] else 0
            avg_cpu = sum(self.metrics['cpu_usage_percent']) / len(self.metrics['cpu_usage_percent']) if self.metrics['cpu_usage_percent'] else 0

            # System info
            system_memory = psutil.virtual_memory()
            system_disk = psutil.disk_usage('/')

            return {
                'current': {
                    'memory_mb': round(current_memory_mb, 2),
                    'cpu_percent': round(current_cpu, 2),
                    'workers': self.current_workers,
                    'optimal_workers': self.optimal_workers
                },
                'average': {
                    'memory_mb': round(avg_memory, 2),
                    'cpu_percent': round(avg_cpu, 2)
                },
                'system': {
                    'total_memory_gb': round(system_memory.total / 1024 / 1024 / 1024, 2),
                    'available_memory_gb': round(system_memory.available / 1024 / 1024 / 1024, 2),
                    'memory_percent': system_memory.percent,
                    'disk_free_gb': round(system_disk.free / 1024 / 1024 / 1024, 2),
                    'disk_percent': system_disk.percent
                },
                'recommendations': self._get_recommendations()
            }

        except Exception as e:
            logger.error(f"Failed to get performance report: {e}")
            return {}

    def _get_recommendations(self) -> list[str]:
        """Get performance recommendations"""
        recommendations = []

        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()

            if memory_mb > 1000:
                recommendations.append("High memory usage detected. Consider reducing worker count or enabling cache cleanup.")

            if cpu_percent > 80:
                recommendations.append("High CPU usage detected. Consider reducing worker count.")

            system_memory = psutil.virtual_memory()
            if system_memory.percent > 85:
                recommendations.append("System memory is low. Close other applications or reduce worker count.")

            if self.current_workers < self.optimal_workers:
                recommendations.append(f"You can increase workers to {self.optimal_workers} for better performance.")

            if len(recommendations) == 0:
                recommendations.append("Performance is optimal. No changes needed.")

        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")

        return recommendations

    def auto_tune(self):
        """Automatically tune all performance settings"""
        logger.info("Starting auto-tune...")

        # Calculate optimal workers
        self.optimal_workers = self.calculate_optimal_workers()
        self.current_workers = self.optimal_workers

        # Optimize system settings
        self.optimize_system_settings()

        # Start monitoring
        self.start_monitoring()

        logger.info(f"Auto-tune complete. Workers set to {self.current_workers}")


# Global instance
_optimizer = None


def get_optimizer(config: Dict[str, Any]) -> PersonalPerformanceOptimizer:
    """Get global optimizer instance"""
    global _optimizer
    if _optimizer is None:
        _optimizer = PersonalPerformanceOptimizer(config)
    return _optimizer
