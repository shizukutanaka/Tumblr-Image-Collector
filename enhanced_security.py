import logging
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)


class ThreatDetector:
    """AI駆動の脅威検知システム"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.request_history = deque(maxlen=max_history)
        self.suspicious_patterns = {
            'rapid_requests': {'threshold': 10, 'window': 60},  # 1分間に10リクエスト以上
            'unusual_user_agents': ['bot', 'crawler', 'scraper'],
            'repeated_failures': {'threshold': 5, 'window': 300},  # 5分間に5失敗以上
        }
        self.ml_features = []

    def analyze_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """リクエストを分析して脅威を検知"""
        timestamp = time.time()
        ip = request_data.get('ip', 'unknown')
        user_agent = request_data.get('user_agent', '')
        url = request_data.get('url', '')

        # 基本的なパターン分析
        threat_score = 0
        reasons = []

        # 高速リクエストチェック
        if self._check_rapid_requests(ip, timestamp):
            threat_score += 30
            reasons.append('rapid_requests')

        # 異常なUser-Agentチェック
        if self._check_suspicious_user_agent(user_agent):
            threat_score += 20
            reasons.append('suspicious_user_agent')

        # 失敗パターン分析（履歴に基づく）
        if self._check_repeated_failures(ip, timestamp):
            threat_score += 25
            reasons.append('repeated_failures')

        # ML特徴の抽出（簡易版）
        features = self._extract_ml_features(request_data)
        self.ml_features.append(features)

        # 機械学習による予測（ここでは簡易ルールベース）
        ml_score = self._predict_threat_ml(features)
        threat_score += ml_score

        if ml_score > 10:
            reasons.append('ml_detected_anomaly')

        is_threat = threat_score >= 50

        return {
            'is_threat': is_threat,
            'threat_score': threat_score,
            'reasons': reasons,
            'timestamp': timestamp,
            'request_id': hashlib.md5(f"{ip}_{timestamp}".encode()).hexdigest()
        }

    def _check_rapid_requests(self, ip: str, timestamp: float) -> bool:
        """高速リクエストをチェック"""
        window = self.suspicious_patterns['rapid_requests']['window']
        threshold = self.suspicious_patterns['rapid_requests']['threshold']

        recent_requests = [
            req for req in self.request_history
            if req['ip'] == ip and timestamp - req['timestamp'] <= window
        ]

        return len(recent_requests) >= threshold

    def _check_suspicious_user_agent(self, user_agent: str) -> bool:
        """異常なUser-Agentをチェック"""
        user_agent_lower = user_agent.lower()
        return any(pattern in user_agent_lower for pattern in self.suspicious_patterns['unusual_user_agents'])

    def _check_repeated_failures(self, ip: str, timestamp: float) -> bool:
        """繰り返しの失敗をチェック"""
        window = self.suspicious_patterns['repeated_failures']['window']
        threshold = self.suspicious_patterns['repeated_failures']['threshold']

        recent_failures = [
            req for req in self.request_history
            if req['ip'] == ip and not req.get('success', True) and timestamp - req['timestamp'] <= window
        ]

        return len(recent_failures) >= threshold

    def _extract_ml_features(self, request_data: Dict[str, Any]) -> List[float]:
        """機械学習特徴を抽出"""
        features = [
            len(request_data.get('url', '')),
            len(request_data.get('user_agent', '')),
            request_data.get('response_time', 0),
            1 if request_data.get('success', True) else 0,
        ]
        return features

    def _predict_threat_ml(self, features: List[float]) -> float:
        """機械学習による脅威予測（簡易版）"""
        # ここではルールベースだが、実際にはモデルを使用
        if features[0] > 1000:  # 非常に長いURL
            return 15
        if features[1] < 10:  # 短いUser-Agent
            return 10
        if features[2] > 30:  # 長い応答時間
            return 5
        return 0

    def record_request(self, ip: str, success: bool, response_time: float = 0):
        """リクエストを記録"""
        self.request_history.append({
            'ip': ip,
            'timestamp': time.time(),
            'success': success,
            'response_time': response_time
        })


class AdaptiveRateLimiter:
    """適応型レート制限システム"""

    def __init__(self, base_limit: int = 100, adaptive_factor: float = 0.1):
        self.base_limit = base_limit
        self.adaptive_factor = adaptive_factor
        self.ip_limits = defaultdict(lambda: base_limit)
        self.threat_detector = ThreatDetector()

    def check_request_allowed(self, ip: str, request_data: Dict[str, Any]) -> bool:
        """リクエストが許可されるかをチェック"""
        threat_analysis = self.threat_detector.analyze_request(request_data)

        if threat_analysis['is_threat']:
            logger.warning(f"脅威検知: IP {ip}, スコア {threat_analysis['threat_score']}")
            return False

        # 適応型制限の適用
        current_limit = self.ip_limits[ip]
        if threat_analysis['threat_score'] > 20:
            current_limit = int(current_limit * (1 - self.adaptive_factor))

        # ここでは簡易実装（実際にはトークンバケットなどを使用）
        return True  # 実際の実装では適切な制限ロジックを追加

    def update_limits(self, ip: str, threat_score: float):
        """制限を更新"""
        if threat_score > 30:
            self.ip_limits[ip] = max(10, self.ip_limits[ip] - 10)
        elif threat_score < 10:
            self.ip_limits[ip] = min(self.base_limit, self.ip_limits[ip] + 5)


class SecurityEnhancer:
    """セキュリティ強化システムの統合"""

    def __init__(self):
        self.threat_detector = ThreatDetector()
        self.rate_limiter = AdaptiveRateLimiter()
        self.logger = logging.getLogger(__name__)

    def enhance_download_engine(self, download_engine):
        """ダウンロードエンジンにセキュリティを統合"""
        original_download = download_engine.download_image

        def secure_download(image_url: str, blog_name: str = "", post_data: Dict = None):
            # リクエストデータを構築
            request_data = {
                'url': image_url,
                'ip': 'localhost',  # 実際には適切なIPを取得
                'user_agent': 'TumblrImageCollector/1.0',
                'timestamp': time.time()
            }

            # 脅威チェック
            if not self.rate_limiter.check_request_allowed('localhost', request_data):
                raise SecurityError("リクエストが脅威としてブロックされました")

            # 元のダウンロードを実行
            result = original_download(image_url, blog_name, post_data)

            # 結果を記録
            self.threat_detector.record_request('localhost', result[0], 0)

            return result

        download_engine.download_image = secure_download
        self.logger.info("ダウンロードエンジンにセキュリティ強化を適用しました")


class SecurityError(Exception):
    """セキュリティ関連のエラー"""
    pass
