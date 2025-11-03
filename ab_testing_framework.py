"""
パフォーマンスメトリクスとA/Bテストフレームワーク
データ駆動型の最適化システム
"""

import logging
import json
import time
import random
import statistics
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """メトリクスタイプ"""
    COUNTER = "counter"      # カウント
    GAUGE = "gauge"         # 瞬間値
    HISTOGRAM = "histogram"  # 分布
    TIMER = "timer"         # 時間


@dataclass
class Metric:
    """メトリクスデータ"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None


class MetricsCollector:
    """メトリクス収集システム"""

    def __init__(self):
        self.metrics_buffer = defaultdict(list)
        self.flush_interval = 60  # 1分間隔でフラッシュ
        self.max_buffer_size = 10000

    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE,
                     tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """メトリクスを記録"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=time.time(),
            tags=tags or {},
            metadata=metadata or {}
        )

        self.metrics_buffer[name].append(metric)

        # バッファサイズチェック
        if sum(len(metrics) for metrics in self.metrics_buffer.values()) > self.max_buffer_size:
            self._flush_metrics()

    def increment_counter(self, name: str, value: float = 1, tags: Dict[str, str] = None):
        """カウンターをインクリメント"""
        self.record_metric(name, value, MetricType.COUNTER, tags)

    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """タイマーを記録"""
        self.record_metric(name, duration, MetricType.TIMER, tags)

    def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """ゲージを記録"""
        self.record_metric(name, value, MetricType.GAUGE, tags)

    def _flush_metrics(self):
        """メトリクスをフラッシュ"""
        # 実際の実装ではデータベースや外部システムに送信
        logger.info(f"メトリクスをフラッシュ: {len(self.metrics_buffer)} 種類のメトリクス")

        # バッファをクリア
        self.metrics_buffer.clear()

    def get_metrics_summary(self, metric_name: str = None, time_range: int = 3600) -> Dict[str, Any]:
        """メトリクスサマリーを取得"""
        if metric_name:
            metrics = self.metrics_buffer.get(metric_name, [])
        else:
            all_metrics = []
            for metrics_list in self.metrics_buffer.values():
                all_metrics.extend(metrics_list)
            metrics = all_metrics

        # 時間範囲でフィルタリング
        cutoff_time = time.time() - time_range
        recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return {}

        values = [m.value for m in recent_metrics]

        return {
            'count': len(recent_metrics),
            'min': min(values),
            'max': max(values),
            'avg': statistics.mean(values),
            'median': statistics.median(values),
            'p95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
            'p99': statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
        }


class ABTestVariant:
    """A/Bテストバリアント"""

    def __init__(self, name: str, config: Dict[str, Any], traffic_percentage: float = 50.0):
        self.name = name
        self.config = config
        self.traffic_percentage = traffic_percentage
        self.users_assigned = set()
        self.metrics = defaultdict(list)

    def assign_user(self, user_id: str):
        """ユーザーをバリアントに割り当て"""
        self.users_assigned.add(user_id)

    def record_metric(self, user_id: str, metric_name: str, value: float):
        """メトリクスを記録"""
        self.metrics[metric_name].append({
            'user_id': user_id,
            'value': value,
            'timestamp': time.time()
        })

    def get_metrics_summary(self, metric_name: str) -> Dict[str, Any]:
        """メトリクスのサマリーを取得"""
        if metric_name not in self.metrics:
            return {}

        values = [m['value'] for m in self.metrics[metric_name]]

        if not values:
            return {}

        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': statistics.mean(values),
            'median': statistics.median(values),
            'conversion_rate': sum(1 for v in values if v > 0) / len(values) if values else 0
        }


class ABTest:
    """A/Bテスト"""

    def __init__(self, test_id: str, name: str, variants: List[ABTestVariant],
                 primary_metric: str, minimum_sample_size: int = 1000):
        self.test_id = test_id
        self.name = name
        self.variants = {v.name: v for v in variants}
        self.primary_metric = primary_metric
        self.minimum_sample_size = minimum_sample_size
        self.start_time = time.time()
        self.end_time = None
        self.status = 'running'

    def assign_user_to_variant(self, user_id: str) -> str:
        """ユーザーをバリアントに割り当て"""
        # 既存の割り当てをチェック
        for variant_name, variant in self.variants.items():
            if user_id in variant.users_assigned:
                return variant_name

        # 新しい割り当てを決定
        total_percentage = sum(v.traffic_percentage for v in self.variants.values())

        # ランダム割り当て（実際にはより洗練された方法を使用）
        rand_value = random.random() * total_percentage
        cumulative = 0

        for variant_name, variant in self.variants.items():
            cumulative += variant.traffic_percentage
            if rand_value <= cumulative:
                variant.assign_user(user_id)
                return variant_name

        # デフォルトで最初のバリアントに割り当て
        first_variant = next(iter(self.variants.values()))
        first_variant.assign_user(user_id)
        return first_variant.name

    def record_metric(self, user_id: str, metric_name: str, value: float):
        """メトリクスを記録"""
        # ユーザーがどのバリアントに割り当てられているかを検索
        for variant in self.variants.values():
            if user_id in variant.users_assigned:
                variant.record_metric(user_id, metric_name, value)
                break

    def get_test_results(self) -> Dict[str, Any]:
        """テスト結果を取得"""
        results = {
            'test_id': self.test_id,
            'name': self.name,
            'status': self.status,
            'duration': time.time() - self.start_time,
            'total_users': sum(len(v.users_assigned) for v in self.variants.values()),
            'variants': {}
        }

        for variant_name, variant in self.variants.items():
            variant_results = {
                'name': variant_name,
                'traffic_percentage': variant.traffic_percentage,
                'user_count': len(variant.users_assigned),
                'metrics': {}
            }

            for metric_name in variant.metrics:
                variant_results['metrics'][metric_name] = variant.get_metrics_summary(metric_name)

            results['variants'][variant_name] = variant_results

        return results

    def calculate_statistical_significance(self) -> Dict[str, Any]:
        """統計的有意性を計算"""
        if not self._has_minimum_sample_size():
            return {'ready': False, 'message': 'サンプルサイズが不足しています'}

        primary_metric_values = {}
        for variant_name, variant in self.variants.items():
            summary = variant.get_metrics_summary(self.primary_metric)
            if summary:
                primary_metric_values[variant_name] = [m['value'] for m in variant.metrics[self.primary_metric]]

        if len(primary_metric_values) < 2:
            return {'ready': False, 'message': '比較可能なバリアントが不足しています'}

        # 簡易的な有意性テスト（実際にはt検定などを使用）
        significance_results = {}

        for i, (variant1, values1) in enumerate(primary_metric_values.items()):
            for variant2, values2 in list(primary_metric_values.items())[i+1:]:
                # 簡易的な効果量計算
                mean1 = statistics.mean(values1)
                mean2 = statistics.mean(values2)

                effect_size = abs(mean1 - mean2) / max(statistics.stdev(values1) if len(values1) > 1 else 1,
                                                       statistics.stdev(values2) if len(values2) > 1 else 1)

                significance_results[f'{variant1}_vs_{variant2}'] = {
                    'effect_size': effect_size,
                    'significant': effect_size > 0.2,  # 簡易的な閾値
                    'winner': variant1 if mean1 > mean2 else variant2
                }

        return {
            'ready': True,
            'results': significance_results,
            'confidence_level': 0.95
        }

    def _has_minimum_sample_size(self) -> bool:
        """最小サンプルサイズに達しているかチェック"""
        total_users = sum(len(v.users_assigned) for v in self.variants.values())
        return total_users >= self.minimum_sample_size

    def conclude_test(self) -> Dict[str, str]:
        """テストを結論づける"""
        significance = self.calculate_statistical_significance()

        if not significance['ready']:
            return {'status': 'inconclusive', 'reason': significance['message']}

        # 勝者を決定
        winners = []
        for comparison, result in significance['results'].items():
            if result['significant']:
                winners.append(result['winner'])

        if not winners:
            return {'status': 'inconclusive', 'reason': '有意な差が見つかりませんでした'}

        # 最も頻出する勝者を選択
        winner = max(set(winners), key=winners.count)

        self.status = 'concluded'
        self.end_time = time.time()

        return {
            'status': 'concluded',
            'winner': winner,
            'confidence': significance['confidence_level'],
            'reason': f'{winner}が有意に優れた結果を示しました'
        }


class ABTestingFramework:
    """A/Bテストフレームワーク"""

    def __init__(self):
        self.active_tests = {}
        self.completed_tests = []
        self.metrics_collector = MetricsCollector()

    def create_test(self, name: str, variants: List[Dict[str, Any]],
                   primary_metric: str, minimum_sample_size: int = 1000) -> str:
        """テストを作成"""
        test_id = f"test_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        # バリアントオブジェクトを作成
        variant_objects = []
        for variant_data in variants:
            variant = ABTestVariant(
                name=variant_data['name'],
                config=variant_data['config'],
                traffic_percentage=variant_data.get('traffic_percentage', 50.0)
            )
            variant_objects.append(variant)

        # テストオブジェクトを作成
        test = ABTest(
            test_id=test_id,
            name=name,
            variants=variant_objects,
            primary_metric=primary_metric,
            minimum_sample_size=minimum_sample_size
        )

        self.active_tests[test_id] = test
        logger.info(f"A/Bテストを作成: {test_id}")
        return test_id

    def assign_user_to_test(self, test_id: str, user_id: str) -> Optional[str]:
        """ユーザーをテストに割り当て"""
        if test_id not in self.active_tests:
            return None

        test = self.active_tests[test_id]
        variant_name = test.assign_user_to_variant(user_id)

        # 割り当てメトリクスを記録
        self.metrics_collector.increment_counter(
            f"test_{test_id}_assignment",
            tags={'test_id': test_id, 'variant': variant_name}
        )

        return variant_name

    def record_test_metric(self, test_id: str, user_id: str, metric_name: str, value: float):
        """テストメトリクスを記録"""
        if test_id not in self.active_tests:
            return

        test = self.active_tests[test_id]
        test.record_metric(user_id, metric_name, value)

        # メトリクス収集システムにも記録
        self.metrics_collector.record_gauge(
            f"test_{test_id}_{metric_name}",
            value,
            tags={'test_id': test_id, 'user_id': user_id}
        )

    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """テスト結果を取得"""
        if test_id not in self.active_tests:
            return {'error': 'テストが見つかりません'}

        test = self.active_tests[test_id]
        results = test.get_test_results()
        results['statistical_significance'] = test.calculate_statistical_significance()

        return results

    def conclude_test(self, test_id: str) -> Dict[str, Any]:
        """テストを結論づける"""
        if test_id not in self.active_tests:
            return {'error': 'テストが見つかりません'}

        test = self.active_tests[test_id]
        conclusion = test.conclude_test()

        if conclusion['status'] == 'concluded':
            # 完了リストに移動
            self.completed_tests.append(self.active_tests.pop(test_id))

        return conclusion

    def get_framework_status(self) -> Dict[str, Any]:
        """フレームワーク全体のステータスを取得"""
        return {
            'active_tests': len(self.active_tests),
            'completed_tests': len(self.completed_tests),
            'total_metrics_collected': sum(
                len(metrics) for test in self.active_tests.values()
                for variant in test.variants.values()
                for metrics in variant.metrics.values()
            ),
            'metrics_summary': self.metrics_collector.get_metrics_summary()
        }


class PerformanceOptimizer:
    """パフォーマンス最適化システム"""

    def __init__(self, ab_framework: ABTestingFramework):
        self.ab_framework = ab_framework
        self.optimization_history = []

    def create_optimization_test(self, optimization_name: str, variants: List[Dict[str, Any]]) -> str:
        """最適化テストを作成"""
        return self.ab_framework.create_test(
            name=f"optimization_{optimization_name}",
            variants=variants,
            primary_metric='performance_score',
            minimum_sample_size=500
        )

    def record_performance_metric(self, test_id: str, user_id: str, performance_score: float):
        """パフォーマンスメトリクスを記録"""
        self.ab_framework.record_test_metric(test_id, user_id, 'performance_score', performance_score)

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """最適化推奨を取得"""
        recommendations = []

        for test_id, test in self.ab_framework.active_tests.items():
            significance = test.calculate_statistical_significance()

            if significance['ready']:
                conclusion = test.conclude_test()

                if conclusion['status'] == 'concluded':
                    recommendations.append({
                        'test_id': test_id,
                        'test_name': test.name,
                        'recommended_variant': conclusion['winner'],
                        'confidence': conclusion['confidence'],
                        'reason': conclusion['reason']
                    })

        return recommendations

    def apply_optimization(self, recommendation: Dict[str, Any]) -> bool:
        """最適化を適用"""
        try:
            # 実際の実装ではシステム設定を変更
            self.optimization_history.append({
                **recommendation,
                'applied_at': time.time(),
                'status': 'applied'
            })

            logger.info(f"最適化を適用: {recommendation['recommended_variant']}")
            return True

        except Exception as e:
            logger.error(f"最適化適用エラー: {e}")
            return False


class DataDrivenOptimizer:
    """データ駆動型最適化統合システム"""

    def __init__(self):
        self.ab_framework = ABTestingFramework()
        self.performance_optimizer = PerformanceOptimizer(self.ab_framework)
        self.metrics_collector = self.ab_framework.metrics_collector

    def initialize(self):
        """システムを初期化"""
        logger.info("データ駆動型最適化システムを初期化しました")

    def record_user_interaction(self, user_id: str, action: str, value: float = 1.0,
                              metadata: Dict[str, Any] = None):
        """ユーザーインタラクションを記録"""
        # インタラクションメトリクスを記録
        self.metrics_collector.increment_counter(
            f"interaction_{action}",
            value,
            tags={'user_id': user_id, **(metadata or {})}
        )

    def create_performance_test(self, feature_name: str, variants: List[Dict[str, Any]]) -> str:
        """パフォーマンステストを作成"""
        return self.performance_optimizer.create_optimization_test(feature_name, variants)

    def record_performance_result(self, test_id: str, user_id: str, performance_score: float):
        """パフォーマンス結果を記録"""
        self.performance_optimizer.record_performance_metric(test_id, user_id, performance_score)

    def get_optimization_insights(self) -> Dict[str, Any]:
        """最適化インサイトを取得"""
        recommendations = self.performance_optimizer.get_optimization_recommendations()

        return {
            'framework_status': self.ab_framework.get_framework_status(),
            'recommendations': recommendations,
            'metrics_summary': self.metrics_collector.get_metrics_summary(),
            'optimization_history': self.performance_optimizer.optimization_history[-10:]  # 最新10件
        }

    def run_automated_optimization_cycle(self):
        """自動最適化サイクルを実行"""
        # 現在のテスト結果をチェック
        for test_id, test in self.ab_framework.active_tests.items():
            if test._has_minimum_sample_size():
                conclusion = test.conclude_test()

                if conclusion['status'] == 'concluded':
                    # 勝者のバリアントを適用
                    recommendation = {
                        'test_id': test_id,
                        'recommended_variant': conclusion['winner'],
                        'confidence': conclusion['confidence']
                    }

                    self.performance_optimizer.apply_optimization(recommendation)


# 使用例
def example_ab_testing():
    """A/Bテストの使用例"""

    optimizer = DataDrivenOptimizer()
    optimizer.initialize()

    # パフォーマンステストを作成
    test_id = optimizer.create_performance_test(
        'download_ui',
        [
            {'name': 'control', 'config': {'ui_version': 'v1'}, 'traffic_percentage': 50},
            {'name': 'variant_a', 'config': {'ui_version': 'v2'}, 'traffic_percentage': 50}
        ]
    )

    # ユーザー割り当てとメトリクス記録をシミュレーション
    users = [f'user_{i}' for i in range(100)]

    for user in users:
        variant = optimizer.ab_framework.assign_user_to_test(test_id, user)

        # パフォーマンススコアを記録（シミュレーション）
        performance_score = random.uniform(60, 90)  # 60-90のスコア
        optimizer.record_performance_result(test_id, user, performance_score)

    # 結果を取得
    results = optimizer.ab_framework.get_test_results(test_id)
    print("A/Bテスト結果:")
    print(json.dumps(results, indent=2, default=str))

    # 最適化インサイトを取得
    insights = optimizer.get_optimization_insights()
    print("\n最適化インサイト:")
    print(json.dumps(insights, indent=2, default=str))

if __name__ == "__main__":
    example_ab_testing()
