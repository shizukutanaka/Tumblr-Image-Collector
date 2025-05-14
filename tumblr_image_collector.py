import pytumblr
import requests
import os
import time
import random
from PIL import Image, ImageHash
from io import BytesIO
import datetime
import webbrowser
import json
import logging
import concurrent.futures
from pathlib import Path
import argparse
from functools import partial
import socket
import ssl
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.poolmanager import PoolManager
import imagehash
import urllib3
import socks
import socket
from urllib.parse import urlparse
import traceback
import platform
import uuid
import sys
from typing import Optional, Dict, Any
from image_classifier import ImageClassifier

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# クラッシュレポート設定
CRASH_REPORT_DIR = Path(__file__).parent / 'crash_reports'
CRASH_REPORT_DIR.mkdir(exist_ok=True)

# プロキシ設定のデフォルト値
DEFAULT_PROXY_CONFIG = {
    'type': None,  # 'http', 'socks4', 'socks5'
    'host': None,
    'port': None,
    'username': None,
    'password': None
}

# --- Constants ---
DEFAULT_CONFIG_FILE = "config.json"
DEFAULT_LOG_FILE = "tumblr_collector.log"

# --- Global Logger ---
# Logger will be configured in main() based on args
logger = logging.getLogger(__name__)

class TumblrImageCollector:
    """
    Tumblrブログから画像を収集し、関連タグの画像もダウンロードするクラス。
    設定管理、API認証、並列ダウンロード、重複防止、レート制限対応を行う。
    高度な画像フィルタリングと重複排除を実装。
    パフォーマンスと安定性を最適化。
    """

    # ダウンロード統計
    _download_stats = {
        'total_attempts': 0,
        'successful_downloads': 0,
        'failed_downloads': 0,
        
        # 基本ダウンロード統計
        'total_images_processed': 0,
        'total_images_downloaded': 0,
        'total_images_skipped': 0,
        
        # AI画像分類統計
        'ai_classification_stats': {
            'valid_images': 0,
            'invalid_images': 0,
            'high_resolution_images': 0,
            'low_resolution_images': 0,
            'potentially_nsfw_images': 0,
            'image_type_distribution': {}
        }
    }
        'total_attempts': 0,
        'successful_downloads': 0,
        'failed_downloads': 0,
        'skipped_duplicates': 0
    }

    # エラー種別
    class DownloadError(Exception):
        """カスタムダウンロードエラー"""
        pass

    # 画像ハッシュの類似度閾値
    IMAGE_HASH_THRESHOLD = 5

    # 画像フィルタリングオプション
    IMAGE_FILTERS = {
        'min_width': 500,
        'min_height': 500,
        'allowed_formats': ['jpg', 'jpeg', 'png', 'gif', 'webp'],
        'max_file_size_mb': 10,
        'aspect_ratio_range': (0.5, 2.0),  # 縦横比の制限
        'color_threshold': 0.1,  # カラー画像の判定閾値
        'blur_threshold': 50,  # ぼかし度の閾値
        'nsfw_detection': True  # NSFWコンテンツの検出
    }

    def __init__(self, config_file=DEFAULT_CONFIG_FILE, output_dir_override=None, workers_override=None, proxy_config=None):
        self.config_file = Path(config_file).resolve()
        self.config = self._load_config()
        self.script_dir = Path(__file__).parent.resolve()
        
        # ロギングシステムを初期化
        self._setup_logging()
        
        # プロキシ設定の初期化
        self.proxy_config = proxy_config or self.config.get('proxy', DEFAULT_PROXY_CONFIG)
        self._setup_proxy()

        # Determine output folder: CLI override > config > default
        output_folder_name = output_dir_override or self.config.get("output_folder_name", "tumblr_images")
        # Ensure output_folder is absolute path
        if Path(output_folder_name).is_absolute():
             self.output_folder = Path(output_folder_name)
        else:
             self.output_folder = self.script_dir / output_folder_name

        # Determine max workers: CLI override > config > default
        self.max_workers = workers_override or self.config.get("max_download_workers", 5)

        self.api_batch_sleep = self.config.get("api_batch_sleep_seconds", 2)
        self.api_wait_hours = self.config.get("api_wait_hours", 1)

        self.downloaded_files = set()
        self._setup_output_directory()
        self._load_downloaded_files()

        self.consumer_key = None
        self.consumer_secret = None
        self.token = None
        self.token_secret = None
        self._setup_credentials() # Reads from self.config

        self.client = self._initialize_client()
        self.executor = None # Will be managed by 'with' statement in run()

    def _setup_logging(self):
        """高度なロギングシステムを設定する"""
        # ログディレクトリを作成
        log_dir = self.script_dir / 'logs'
        log_dir.mkdir(exist_ok=True)

        # ログファイルのパスを設定
        log_filename = log_dir / f"tumblr_collector_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # カスタムログフォーマッター
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # ファイルハンドラー
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        # コンソールハンドラー
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

        # ロガーを設定
        logging.basicConfig(
            level=logging.DEBUG,
            handlers=[file_handler, console_handler]
        )

        # 未処理の例外をキャッチするハンドラーを追加
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
            self._generate_crash_report(exc_type, exc_value, exc_traceback)

        sys.excepthook = handle_exception

    def _load_config(self):
        """設定ファイルを読み込む"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"{self.config_file} is corrupted. Using defaults.")
                return {}
            except Exception as e:
                logger.error(f"Error loading config file {self.config_file}: {e}")
                return {}
        logger.info(f"Config file {self.config_file} not found. Using defaults.")
        return {}

    def _save_config(self):
        """設定ファイルに書き込む"""
        try:
            # Ensure parent directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4, sort_keys=True)
            logger.debug(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving config file {self.config_file}: {e}")

    def _setup_output_directory(self):
        """保存先ディレクトリを作成する"""
        try:
            self.output_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory set to: {self.output_folder}")
        except Exception as e:
            logger.error(f"Failed to create output directory {self.output_folder}: {e}")
            # Consider raising an exception instead of exiting directly
            raise IOError(f"Cannot create output directory: {self.output_folder}") from e

    def _load_downloaded_files(self):
        """出力フォルダから既存のファイル名を読み込み、セットに追加する"""
        if not self.output_folder.exists():
             logger.warning(f"Output folder {self.output_folder} does not exist yet.")
             return
        try:
            existing_files = {f.name for f in self.output_folder.iterdir() if f.is_file()}
            self.downloaded_files.update(existing_files)
            logger.info(f"Loaded {len(existing_files)} existing filenames from {self.output_folder}.")
        except Exception as e:
            logger.error(f"Error reading existing files from {self.output_folder}: {e}")

    def _get_oauth_token(self):
        """OAuthアクセストークンを取得する"""
        if not self.consumer_key or not self.consumer_secret:
             logger.error("Consumer key and secret must be set before getting OAuth token.")
             return None, None

        oauth_client = pytumblr.TumblrRestClient(self.consumer_key, self.consumer_secret)
        try:
            url = oauth_client.get_authorize_url()
            logger.info(f"Please visit the following URL in your browser to get the OAuth verifier: {url}")
            # Try to open browser, but don't fail if it doesn't work
            try:
                webbrowser.open(url)
            except Exception:
                 logger.warning("Could not automatically open browser.")

            verifier = input("Enter the OAuth verifier here: ")
            if not verifier:
                 logger.error("OAuth verifier is required.")
                 return None, None

            oauth_client.get_access_token(verifier)

            logger.info("OAuth access token obtained!")
            logger.info(f"OAuth Token: {oauth_client.token}")
            logger.info(f"OAuth Token Secret: {oauth_client.token_secret}")
            return oauth_client.token, oauth_client.token_secret

        except Exception as e:
            logger.error(f"Error obtaining OAuth token: {e}")
            return None, None

    def _setup_proxy(self):
        """プロキシ設定を初期化する"""
        proxy_type = self.proxy_config.get('type')
        if not proxy_type:
            logger.info("No proxy configuration found. Using direct connection.")
            return

        try:
            if proxy_type == 'socks4':
                socks.set_default_proxy(socks.SOCKS4, 
                    self.proxy_config['host'], 
                    int(self.proxy_config['port']),
                    username=self.proxy_config.get('username'))
            elif proxy_type == 'socks5':
                socks.set_default_proxy(socks.SOCKS5, 
                    self.proxy_config['host'], 
                    int(self.proxy_config['port']),
                    username=self.proxy_config.get('username'),
                    password=self.proxy_config.get('password'))
            elif proxy_type in ['http', 'https']:
                # urllib3のプロキシ設定
                proxy_url = f"{proxy_type}://{self.proxy_config['host']}:{self.proxy_config['port']}"
                if self.proxy_config.get('username') and self.proxy_config.get('password'):
                    proxy_url = f"{proxy_type}://{self.proxy_config['username']}:{self.proxy_config['password']}@{self.proxy_config['host']}:{self.proxy_config['port']}"
                
                self.proxy = urllib3.ProxyManager(proxy_url)
                logger.info(f"Proxy configured: {proxy_type.upper()} at {self.proxy_config['host']}:{self.proxy_config['port']}")
            
            # デフォルトソケットをSOCKSに変更
            socket.socket = socks.socksocket
            logger.info(f"Proxy type {proxy_type} initialized successfully.")
        except Exception as e:
            logger.error(f"Proxy configuration failed: {e}")
            # プロキシ設定をリセット
            self.proxy_config = DEFAULT_PROXY_CONFIG

    def _extract_image_metadata(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        画像メタデータを抽出する関数
        
        Args:
            image_path (str): 画像ファイルのパス
        
        Returns:
            Optional[Dict[str, Any]]: 画像メタデータ
        """
        try:
            with Image.open(image_path) as img:
                # 画像の基本情報
                width, height = img.size
                file_size = os.path.getsize(image_path)
                file_format = img.format
                color_mode = img.mode
                
                # 知覚的ハッシュによる重複検出
                phash = str(imagehash.phash(img))
                
                # 画質計算
                quality_score = self._calculate_image_quality(img)
                
                # AI画像分類器による追加分析
                classifier = ImageClassifier()
                classification_result = classifier.analyze_image(image_path)
                
                metadata = {
                    'width': width,
                    'height': height,
                    'file_size': file_size,
                    'format': file_format,
                    'color_mode': color_mode,
                    'phash': phash,
                    'quality_score': quality_score,
                    'ai_classification': {
                        'is_valid': classification_result.get('is_valid', False),
                        'is_high_resolution': classification_result.get('is_high_resolution', False),
                        'is_potentially_nsfw': classification_result.get('is_potentially_nsfw', False),
                        'top_predictions': classification_result.get('top_predictions', [])
                    }
                }
                
                return metadata
        except Exception as e:
            logger.error(f"メタデータ抽出エラー: {e}")
            return None

    def _advanced_image_analysis(self, image_path):
        """
        高度な画像分析を実行し、詳細なメタデータを生成
        
        Args:
            image_path (Path): 分析する画像のパス
        
        Returns:
            dict: 画像の詳細な分析結果
        """
        try:
            # OpenCVを使用した画像分析
            import cv2
            import numpy as np

            # 画像をロード
            img = cv2.imread(str(image_path))
            
            # 基本的な画像プロパティ
            height, width, channels = img.shape
            
            # カラーヒストグラム分析
            color_hist = {
                'red': cv2.calcHist([img], [2], None, [256], [0, 256]).flatten(),
                'green': cv2.calcHist([img], [1], None, [256], [0, 256]).flatten(),
                'blue': cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            }
            
            # エッジ検出
            edges = cv2.Canny(img, 100, 200)
            edge_density = np.sum(edges) / (height * width)
            
            # ぼかし度の検出
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # テクスチャ分析
            glcm = self._calculate_glcm(gray)
            texture_features = self._extract_glcm_features(glcm)
            
            # エッジ方向ヒストグラム
            edge_orientation_hist = self._calculate_edge_orientation_histogram(edges)
            
            # 色の均一性と変化
            color_uniformity = self._calculate_color_uniformity(img)
            
            # 画像の複雑さ指標
            image_entropy = self._calculate_image_entropy(gray)
            
            # 顔検出の簡易実装（オプション）
            face_detection_result = self._detect_faces(img)
            
            # 画像の構図分析
            composition_analysis = self._analyze_image_composition(img)
            
            # コントラスト分析
            contrast = np.std(gray)
            
            # 平均輝度
            mean_brightness = np.mean(gray)
            
            # NSFW検出プレースホルダー（将来の拡張用）
            nsfw_score = self._estimate_nsfw_content(img)

            return {
                'dimensions': {
                    'width': width,
                    'height': height,
                    'aspect_ratio': width / height
                },
                'color_analysis': {
                    'dominant_colors': self._extract_dominant_colors(img),
                    'color_histogram': color_hist
                },
                'quality_metrics': {
                    'blur_score': variance_of_laplacian,
                    'edge_density': edge_density,
                    'contrast': contrast,
                    'mean_brightness': mean_brightness
                },
                'nsfw_score': nsfw_score,
                'channels': channels
            }
        except Exception as e:
            logger.error(f"画像分析中にエラー: {e}")
            return {}

    def _extract_dominant_colors(self, image, top_n=5):
        """
        画像から支配的な色を抽出
        
        Args:
            image (numpy.ndarray): OpenCV形式の画像
            top_n (int): 抽出する上位の色の数
        
        Returns:
            list: 支配的な色のRGB値
        """
        try:
            # 画像をRGBに変換
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 画像をリシェイプ
            pixels = img_rgb.reshape((-1, 3))
            
            # float32に変換
            pixels = np.float32(pixels)
            
            # K-meansクラスタリング
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            k = min(top_n, pixels.shape[0])
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # 色をカウント
            unique, counts = np.unique(labels, return_counts=True)
            
            # 色を出現頻度でソート
            sorted_indices = np.argsort(counts)[::-1]
            
            # RGB値を返す
            return [centers[idx].astype(int).tolist() for idx in sorted_indices]
        except Exception as e:
            logger.error(f"支配的な色の抽出中にエラー: {e}")
            return []

    def _estimate_nsfw_content(self, image):
        """
        画像のNSFW（Not Safe For Work）スコアを推定
        
        Args:
            image (numpy.ndarray): OpenCV形式の画像
        
        Returns:
            float: NSFWスコア（0.0〜1.0）
        """
        # TODO: 機械学習モデルやAPIを使用した本格的なNSFW検出
        # 現在は簡易的な肌色検出を使用
        try:
            # HSV色空間に変換
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 肌色の色相範囲（おおよそ）
            lower_skin = np.array([0, 48, 80], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # 肌色マスクを作成
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # 肌色ピクセルの割合を計算
            skin_ratio = np.sum(skin_mask > 0) / (image.shape[0] * image.shape[1])
            
            # 簡易的なNSFWスコア（肌色の割合に基づく）
            return min(skin_ratio * 2, 1.0)  # 最大1.0に制限
        except Exception as e:
            logger.error(f"NSFW推定中にエラー: {e}")
            return 0.0

    def _generate_image_report(self, image_path):
        """
        画像の包括的な分析レポートを生成
        
        Args:
            image_path (Path): 分析する画像のパス
        
        Returns:
            dict: 画像分析レポート
        """
        try:
            # メタデータと高度な分析を統合
            with Image.open(image_path) as img:
                basic_metadata = self._extract_image_metadata(img)
            
            advanced_analysis = self._advanced_image_analysis(image_path)
            
            # レポートを統合
            comprehensive_report = {
                'basic_metadata': basic_metadata,
                'advanced_analysis': advanced_analysis,
                'file_info': {
                    'path': str(image_path),
                    'size_bytes': image_path.stat().st_size,
                    'last_modified': datetime.datetime.fromtimestamp(image_path.stat().st_mtime)
                }
            }
            
            # JSONファイルに保存
            report_path = image_path.with_suffix('.report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"画像レポートを生成: {report_path}")
            
            return comprehensive_report
        except Exception as e:
            logger.error(f"画像レポート生成中にエラー: {e}")
            return {}

    def _calculate_glcm(self, gray_image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """
        グレイレベル共起行列（GLCM）を計算
        
        Args:
            gray_image (numpy.ndarray): グレースケール画像
            distances (list): 画素間の距離
            angles (list): 画素間の角度
        
        Returns:
            numpy.ndarray: GLCM
        """
        try:
            import skimage.feature
            glcm = skimage.feature.graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
            return glcm
        except Exception as e:
            logger.error(f"GLCM計算中にエラー: {e}")
            return None

    def _extract_glcm_features(self, glcm):
        """
        GLCMから特徴量を抽出
        
        Args:
            glcm (numpy.ndarray): グレイレベル共起行列
        
        Returns:
            dict: テクスチャ特徴量
        """
        try:
            import skimage.feature
            features = {
                'contrast': skimage.feature.graycoprops(glcm, 'contrast')[0, 0],
                'dissimilarity': skimage.feature.graycoprops(glcm, 'dissimilarity')[0, 0],
                'homogeneity': skimage.feature.graycoprops(glcm, 'homogeneity')[0, 0],
                'energy': skimage.feature.graycoprops(glcm, 'energy')[0, 0],
                'correlation': skimage.feature.graycoprops(glcm, 'correlation')[0, 0]
            }
            return features
        except Exception as e:
            logger.error(f"GLCM特徴量抽出中にエラー: {e}")
            return {}

    def _calculate_edge_orientation_histogram(self, edges, num_bins=8):
        """
        エッジの方向ヒストグラムを計算
        
        Args:
            edges (numpy.ndarray): エッジ画像
            num_bins (int): 方向ビンの数
        
        Returns:
            numpy.ndarray: エッジ方向ヒストグラム
        """
        try:
            import cv2
            import numpy as np
            
            # エッジの勾配方向を計算
            gx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
            
            # 勾配の大きさと角度を計算
            mag, angle = cv2.cartToPolar(gx, gy)
            
            # 角度をビンに分割
            hist, _ = np.histogram(angle, bins=num_bins, range=(0, np.pi))
            
            return hist
        except Exception as e:
            logger.error(f"エッジ方向ヒストグラム計算中にエラー: {e}")
            return None

    def _calculate_color_uniformity(self, image):
        """
        画像の色の均一性を計算
        
        Args:
            image (numpy.ndarray): カラー画像
        
        Returns:
            float: 色の均一性スコア
        """
        try:
            import cv2
            import numpy as np
            
            # HSV色空間に変換
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 色相と彩度のヒストグラム
            h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            
            # エントロピーを計算
            h_entropy = -np.sum(h_hist * np.log2(h_hist + 1e-7))
            s_entropy = -np.sum(s_hist * np.log2(s_hist + 1e-7))
            
            return (h_entropy + s_entropy) / 2
        except Exception as e:
            logger.error(f"色の均一性計算中にエラー: {e}")
            return 0.0



    def _detect_faces(self, image, min_neighbors=3, scale_factor=1.1, advanced_analysis=True, detection_method='cascade'):
        """
        画像内の顔を高度に検出
        
        Args:
            image (numpy.ndarray): カラー画像
            min_neighbors (int): 最小隣接矩形数
            scale_factor (float): スケールファクター
            advanced_analysis (bool): 詳細な顔分析を有効化
            detection_method (str): 顔検出手法を指定
        
        Returns:
            dict: 顔検出結果
        """
        try:
            import cv2
            import numpy as np
            
            # 画像の前処理
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray)
            equalized = cv2.equalizeHist(denoised)
            
            # 顔検出結果格納用の変数
            faces = []
            face_details = []
            
            # 顔検出手法の選択
            if detection_method == 'cascade':
                # カスケード分類器を読み込み
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                
                # 正面とプロファイル顔を検出
                faces_frontal = face_cascade.detectMultiScale(equalized, scaleFactor=scale_factor, minNeighbors=min_neighbors)
                faces_profile_left = profile_face_cascade.detectMultiScale(equalized, scaleFactor=scale_factor, minNeighbors=min_neighbors)
                faces_profile_right = profile_face_cascade.detectMultiScale(cv2.flip(equalized, 1), scaleFactor=scale_factor, minNeighbors=min_neighbors)
                
                faces = np.concatenate([faces_frontal, faces_profile_left, faces_profile_right])
            
            elif detection_method == 'dnn':
                # DNNベースの顔検出（OpenCV深層学習モデル）
                net = cv2.dnn.readNetFromCaffe(
                    'path/to/deploy.prototxt', 
                    'path/to/res10_300x300_ssd_iter_140000.caffemodel'
                )
                blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
                net.setInput(blob)
                detections = net.forward()
                
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    
                    # 信頼度のしきい値を設定
                    if confidence > 0.5:  # 50%以上の信頼度
                        # 顔の位置座標を計算
                        box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                        (startX, startY, endX, endY) = box.astype('int')
                        
                        # 顔の幅と高さを計算
                        width = endX - startX
                        height = endY - startY
                        
                        # 小さすぎる顔や大きすぎる顔を除外
                        if 30 <= width <= 300 and 30 <= height <= 300:
                            faces.append([startX, startY, width, height])
                    
                    # 検出された顔の数が一定数を超えたら処理を中断
                    if len(faces) >= 10:  # 最大10個の顔を検出
                        break
            
            # 高度な顔分析
            if advanced_analysis:
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # 目の検出
                    eyes = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml').detectMultiScale(face_roi)
                    
                    # 顔の特徴量計算
                    face_brightness = np.mean(face_roi)
                    face_contrast = np.std(face_roi)
                    
                    # 顔の方向性
                    face_orientation = self._estimate_face_orientation(face_roi)
                    
                    # 顔の品質評価
                    face_quality = self._calculate_face_quality(face_roi, eyes)
                    
                    # 顔の特徴抽出
                    face_features = self._extract_face_features(face_roi)
                    
                    face_details.append({
                        'location': [x, y, w, h],
                        'brightness': float(face_brightness),
                        'contrast': float(face_contrast),
                        'eyes_count': len(eyes),
                        'quality_score': face_quality,
                        'features': face_features,
                        'orientation': face_orientation
                    })
            
            return {
                'detected_faces': len(faces),
                'face_locations': faces.tolist(),
                'face_details': face_details if advanced_analysis else []
            }
        except Exception as e:
            logger.error(f"顔検出中にエラー: {e}")
            return {'detected_faces': 0, 'face_locations': [], 'face_details': []}

    def _calculate_face_quality(self, face_roi, eyes):
        """
        顔の品質を評価するメソッド
        
        Args:
            face_roi (numpy.ndarray): 顔のローカルイメージ
            eyes (list): 検出された目のリスト
        
        Returns:
            float: 顔の品質スコア
        """
        try:
            import cv2
            import numpy as np
            
            # ノイズレベルの評価
            noise_level = cv2.meanStdDev(face_roi)[1][0][0]
            
            # シャープネスの評価
            laplacian = cv2.Laplacian(face_roi, cv2.CV_64F).var()
            
            # 目の存在を考慮
            eye_factor = 1.0 if len(eyes) > 0 else 0.5
            
            # 品質スコアの計算
            quality_score = (
                (1.0 / (1.0 + noise_level)) *  # ノイズの少なさ
                (laplacian / 100.0) *  # シャープな画像
                eye_factor  # 目の存在
            )
            
            return max(0.0, min(1.0, quality_score))
        except Exception as e:
            logger.error(f"顔品質評価中にエラー: {e}")
            return 0.5
    
    def _extract_face_features(self, face_roi):
        """
        顔の特徴を抽出するメソッド
        
        Args:
            face_roi (numpy.ndarray): 顔のローカルイメージ
        
        Returns:
            dict: 顔の特徴情報
        """
        try:
            import cv2
            import numpy as np
            
            # グレースケールに変換
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            
            # ヒストグラム均等化
            equalized_face = cv2.equalizeHist(gray_face)
            
            # 特徴量の計算
            features = {
                'mean_intensity': float(np.mean(equalized_face)),
                'std_intensity': float(np.std(equalized_face)),
                'entropy': self._calculate_image_entropy(equalized_face),
                'dominant_colors': self._extract_dominant_colors(face_roi)
            }
            
            return features
        except Exception as e:
            logger.error(f"顔特徴抽出中にエラー: {e}")
            return {}
    
    def _extract_dominant_colors(self, image, k=3):
        """
        画像から主要な色を抽出
        
        Args:
            image (numpy.ndarray): 入力画像
            k (int): 抽出する色の数
        
        Returns:
            list: 主要な色のリスト
        """
        try:
            import cv2
            import numpy as np
            from sklearn.cluster import MiniBatchKMeans
            
            # 画像のリサイズと前処理
            resized_image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
            pixels = resized_image.reshape(-1, 3)
            
            # メモリ効率的なミニバッチK-meansを使用
            kmeans = MiniBatchKMeans(
                n_clusters=k, 
                random_state=42, 
                batch_size=1000, 
                max_iter=10
            ).fit(pixels)
            
            # 主要な色を抽出
            dominant_colors = kmeans.cluster_centers_.astype(int).tolist()
            
            return dominant_colors
        except Exception as e:
            logger.error(f"主要色抽出中にエラー: {e}")
            return []
    
    def _parallel_face_detection(self, image, detection_methods=['cascade', 'dnn']):
        """
        並列処理による顔検出の高速化
        
        Args:
            image (numpy.ndarray): 入力画像
            detection_methods (list): 使用する顔検出手法
        
        Returns:
            dict: 顔検出結果
        """
        try:
            import concurrent.futures
            import numpy as np
            
            def detect_faces_method(method):
                return self._detect_faces(
                    image, 
                    detection_method=method, 
                    advanced_analysis=False
                )
            
            # 並列処理で顔検出
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(detection_methods)) as executor:
                results = list(executor.map(detect_faces_method, detection_methods))
            
            # 結果の統合
            total_faces = 0
            combined_locations = []
            
            for result in results:
                total_faces += result['detected_faces']
                combined_locations.extend(result['face_locations'])
            
            # 重複する顔領域を除去
            unique_faces = self._remove_duplicate_faces(combined_locations)
            
            return {
                'detected_faces': len(unique_faces),
                'face_locations': unique_faces
            }
        except Exception as e:
            logger.error(f"並列顔検出中にエラー: {e}")
            return {'detected_faces': 0, 'face_locations': []}
    
    def _remove_duplicate_faces(self, faces, iou_threshold=0.3, confidence_threshold=0.5):
        """
        重複する顔領域を除去し、高品質の顔を選択
        
        Args:
            faces (list): 顔の座標リスト
            iou_threshold (float): 重複判定の閾値
            confidence_threshold (float): 顔の信頼度閾値
        
        Returns:
            list: 重複を除去した高品質の顔の座標リスト
        """
        def calculate_iou(box1, box2):
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            
            # 交差領域の計算
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            # 交差領域の面積
            intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
            
            # 和集合の面積
            box1_area = w1 * h1
            box2_area = w2 * h2
            union_area = box1_area + box2_area - intersection_area
            
            # IoU (Intersection over Union)
            return intersection_area / union_area if union_area > 0 else 0
        
        def calculate_face_quality(face):
            """
            顔領域の品質を評価
            """
            x, y, w, h = face
            area = w * h
            aspect_ratio = w / h
            
            # 顔領域の大きさと形状に基づく品質スコア
            size_score = min(1.0, area / (300 * 300))
            aspect_score = 1.0 - abs(aspect_ratio - 1.0)  # 正方形に近いほど高いスコア
            
            return size_score * aspect_score
        
        # 品質スコアに基づく顔をソート
        sorted_faces = sorted(faces, key=calculate_face_quality, reverse=True)
        
        unique_faces = []
        for face in sorted_faces:
            is_duplicate = False
            for unique_face in unique_faces:
                if calculate_iou(face, unique_face) > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_faces.append(face)
                
                # 最大顔数に制限
                if len(unique_faces) >= 5:
                    break
        
        return unique_faces
    
    def _adaptive_face_detection(self, image, initial_methods=['cascade', 'dnn'], fallback_methods=['haar']):
        """
        適応的な顔検出手法を選択
        
        Args:
            image (numpy.ndarray): 入力画像
            initial_methods (list): 初期に試行する顔検出手法
            fallback_methods (list): フォールバック手法
        
        Returns:
            dict: 顔検出結果
        """
        try:
            # 初期手法で顔検出を試みる
            result = self._parallel_face_detection(image, detection_methods=initial_methods)
            
            # 顔が検出されなかった場合はフォールバック手法を使用
            if result['detected_faces'] == 0 and fallback_methods:
                result = self._parallel_face_detection(image, detection_methods=fallback_methods)
            
            return result
        except Exception as e:
            logger.error(f"適応的顔検出中にエラー: {e}")
            return {'detected_faces': 0, 'face_locations': []}
    
    def _face_detection_performance_monitor(self, detection_results, image_metadata=None):
        """
        顔検出性能をモニタリング
        
        Args:
            detection_results (dict): 顔検出結果
            image_metadata (dict, optional): 画像のメタデータ
        
        Returns:
            dict: 性能分析結果
        """
        try:
            import time
            
            performance_data = {
                'detected_faces': detection_results['detected_faces'],
                'detection_time': time.time(),
                'image_size': image_metadata.get('size', None) if image_metadata else None,
                'image_format': image_metadata.get('format', None) if image_metadata else None
            }
            
            # 性能ロギング
            if performance_data['detected_faces'] > 0:
                logger.info(f"Face Detection Performance: {performance_data}")
            
            return performance_data
        except Exception as e:
            logger.error(f"性能モニタリング中にエラー: {e}")
            return {}
    
    def extract_image_metadata(self, image_path, post_data=None):
        import os
        import cv2
        import numpy as np
        
        metadata = {
            'filename': os.path.basename(image_path),
            'file_size': os.path.getsize(image_path),
            'tags': post_data.get('tags', []) if post_data else [],
            'detected_faces': 0,
            'face_locations': [],
            'dominant_colors': [],
            'image_quality': 0.0
        }
        
        try:
            # 画像読み込み
            image = cv2.imread(image_path)
            
            # 顔検出
            face_result = self._adaptive_face_detection(image)
            metadata['detected_faces'] = face_result['detected_faces']
            metadata['face_locations'] = face_result['face_locations']
            
            # 主要色抽出
            metadata['dominant_colors'] = self._extract_dominant_colors(image)
            
            # 画像品質評価
            metadata['image_quality'] = self._calculate_image_quality(image)
            
            # タグに基づく分類フラグ
            metadata['is_person'] = self._classify_by_tags(metadata['tags'])
            
        except Exception as e:
            logger.error(f"{image_path}のメタデータ抽出中にエラー: {e}")
        
        return metadata
    
    def _translate_tags(self, tag):
        """
        タグの多言語翻訳と同義語マッピング
        
        Args:
            tag (str): 元のタグ
        
        Returns:
            list: 関連する多言語タグのリスト
        """
        # 多言語・同義語マッピング
        tag_translations = {
            # 人物関連タグ
            'portrait': ['肖像', 'ポートレート', 'プロフィール写真'],
            'selfie': ['自撮り', 'セルフポートレート', 'セルフショット'],
            'person': ['人', '人物', 'ヒト', 'individual'],
            'face': ['顔', '表情', 'フェイス', 'visage'],
            'people': ['人々', '集団', 'folks', 'crowd'],
            'human': ['人間', 'ホト', 'mankind', '人類'],
            'model': ['モデル', 'モデリング', 'mannequin', 'ポーザー'],
            
            # 詳細な人物カテゴリ
            'cosplay': ['コスプレ', 'costume play', '仮装', 'character dress'],
            'celebrity': ['セレブ', '有名人', 'スター', 'famous person'],
            'influencer': ['インフルエンサー', '影響力のある人', 'social media star'],
            'actor': ['俳優', '演技者', 'performer', 'artiste'],
            'actress': ['女優', '女性演技者', 'female performer'],
            'musician': ['音楽家', 'ミュージシャン', 'artist', '演奏家'],
            'artist': ['アーティスト', '芸術家', 'creator', '表現者'],
            'performer': ['パフォーマー', '実演家', 'entertainer', '舞台人'],
            'dancer': ['ダンサー', '踊り手', 'choreographer', '舞踊家'],
            
            # 感情・姿勢タグ
            'smile': ['笑顔', 'スマイル', '微笑', 'grin'],
            'laugh': ['笑う', '笑い', 'chuckle', 'giggle'],
            'pose': ['ポーズ', '姿勢', 'stance', 'posture'],
            'expression': ['表情', '感情表現', 'facial expression', 'look'],
            'emotion': ['感情', '気持ち', 'sentiment', 'feeling'],
            
            # NSFWタグ
            'nsfw': ['エロ', '不適切', 'センシティブ', 'mature content'],
            'adult': ['アダルト', '成人向け', 'mature', '18禁'],
            'sexy': ['セクシー', 'エロティック', 'provocative', '官能的'],
            'nude': ['ヌード', '裸', 'bare', 'undressed'],
            'provocative': ['挑発的', 'センシティブ', 'suggestive', 'risqué']
        }
        
        # 大文字小文字を区別しない検索
        tag_lower = tag.lower()
        
        # 完全一致と部分一致の両方を検索
        translations = []
        for key, values in tag_translations.items():
            if tag_lower == key.lower() or tag_lower in [v.lower() for v in values]:
                translations.append(key)
                translations.extend(values)
        
        return list(set(translations))
    
    def _classify_by_tags(self, tags):
        """
        タグに基づいて画像が人物関連かどうかを判定
        
        Args:
            tags (list): タグリスト
        
        Returns:
            dict: 画像分類結果の詳細情報
        """
        # 多言語タグの展開
        expanded_tags = []
        for tag in tags:
            expanded_tags.extend(self._translate_tags(tag))
        
        # 人物関連タグ（拡張版）
        person_tags = [
            # 一般的な人物タグ
            'portrait', 'selfie', 'person', 'face', 'people', 'human', 'model',
            
            # 詳細な人物カテゴリ
            'cosplay', 'celebrity', 'influencer', 'actor', 'actress', 
            'musician', 'artist', 'performer', 'dancer',
            
            # 感情や姿勢に関連するタグ
            'smile', 'laugh', 'pose', 'expression', 'emotion'
        ]
        
        # アダルト・センシティブコンテンツタグ
        nsfw_tags = [
            'nsfw', 'adult', 'sexy', 'nude', 'provocative'
        ]
        
        # タグを小文字化してマッチング
        normalized_tags = [tag.lower() for tag in expanded_tags]
        
        # 分類結果の詳細
        classification_result = {
            'is_person': False,
            'is_nsfw': False,
            'confidence': 0.0,
            'tags': []
        }
        
        # 人物関連タグの検出
        person_matches = [tag for tag in person_tags if tag.lower() in normalized_tags]
        if person_matches:
            classification_result['is_person'] = True
            classification_result['tags'] = person_matches
            classification_result['confidence'] = len(person_matches) / len(person_tags)
        
        # NSFWタグの検出
        nsfw_matches = [tag for tag in nsfw_tags if tag.lower() in normalized_tags]
        if nsfw_matches:
            classification_result['is_nsfw'] = True
        
        return classification_result
    
    def _generate_recommended_tags(self, image_path):
        """
        画像から推奨タグを自動生成
        
        Args:
            image_path (str): 画像ファイルパス
        
        Returns:
            list: 推奨タグのリスト
        """
        import cv2
        import numpy as np
        from PIL import Image
        
        recommended_tags = []
        
        try:
            # 画像読み込み
            image = cv2.imread(image_path)
            
            # 色分析
            dominant_colors = self._extract_dominant_colors(image)
            color_tags = [
                f'{color_name} tone' for color_name, _ in dominant_colors
            ]
            recommended_tags.extend(color_tags)
            
            # 画像の明るさと雰囲気
            brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) / 255.0
            if brightness < 0.3:
                recommended_tags.append('dark')
            elif brightness > 0.7:
                recommended_tags.append('bright')
            
            # アスペクト比分析
            height, width = image.shape[:2]
            aspect_ratio = width / height
            if aspect_ratio < 0.75:
                recommended_tags.append('portrait')
            elif aspect_ratio > 1.33:
                recommended_tags.append('landscape')
            
            # 画像の質感タグ
            quality_score = self._calculate_image_quality(image)
            if quality_score > 0.8:
                recommended_tags.append('high-quality')
            elif quality_score < 0.3:
                recommended_tags.append('low-quality')
            
            return list(set(recommended_tags))
        
        except Exception as e:
            logger.error(f'タグ生成中にエラー: {e}')
            return []
    
    def _calculate_image_quality(self, image, fast_mode=True, advanced_analysis=False):
        """
        画像品質の総合的な評価
        
        Args:
            image (numpy.ndarray): 入力画像
            fast_mode (bool): 高速モードを有効化する
            advanced_analysis (bool): 高度な画像分析を有効化
        
        Returns:
            dict or float: 画像品質スコアまたは詳細情報
        """
        import cv2
        import numpy as np
        
        try:
            # 画像サイズの縮小で計算間隔を前値する
            original_shape = image.shape
            if fast_mode and image.shape[0] > 300:
                scale_factor = 300 / image.shape[0]
                image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            
            # グレースケール変換
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # シャープネス評価
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000, 1.0)
            
            # 明度均一性評価
            brightness_mean = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness_mean - 0.5) * 2
            
            # ノイズレベル評価
            noise_score = 1.0 - np.std(gray) / 255.0
            
            # 総合品質スコア
            quality_score = (sharpness_score + brightness_score + noise_score) / 3
            
            # 高度な分析を有効化した場合の詳細情報
            if advanced_analysis:
                # コントラスト評価
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                contrast_enhanced = clahe.apply(gray)
                contrast_score = np.std(contrast_enhanced) / 255.0
                
                # エッジ検出
                edges = cv2.Canny(gray, 100, 200)
                edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
                
                return {
                    'quality_score': quality_score,
                    'sharpness': sharpness_score,
                    'brightness': brightness_score,
                    'noise_level': noise_score,
                    'contrast': contrast_score,
                    'edge_density': edge_density,
                    'original_size': original_shape[:2],
                    'resized': fast_mode and original_shape[0] > 300
                }
            
            return quality_score
        except Exception as e:
            logger.error(f"画像品質評価中にエラー: {e}")
            return 0.0
    
    def _robust_image_processing(self, image_path, processing_func, max_retries=3, timeout=10):
        """
        復復力とタイムアウトを考慮した画像処理
        
        Args:
            image_path (str): 画像ファイルパス
            processing_func (callable): 処理関数
            max_retries (int): 最大再試行回数
            timeout (int): タイムアウト秒数
        
        Returns:
            Any: 処理結果
        """
        import signal
        import functools
        
        class TimeoutError(Exception):
            pass
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Processing timed out")
        
        @functools.wraps(processing_func)
        def timeout_wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            try:
                result = processing_func(*args, **kwargs)
                signal.alarm(0)  # タイムアウト解除
                return result
            except TimeoutError:
                logger.warning(f"Processing timed out for {image_path}")
                return None
            finally:
                signal.alarm(0)
        
        for attempt in range(max_retries):
            try:
                # メモリ効率的な読み込みと処理
                result = self._memory_efficient_image_processing(
                    image_path, 
                    lambda img: timeout_wrapper(img)
                )
                
                if result is not None:
                    return result
                
                logger.warning(f"Attempt {attempt + 1} failed for {image_path}")
            
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
        
        logger.error(f"Failed to process {image_path} after {max_retries} attempts")
        return None
    
    def _image_cache_manager(self, cache_dir=None, max_cache_size_mb=500, cleanup_threshold=0.8):
        """
        画像キャッシュ管理システム
        
        Args:
            cache_dir (str, optional): キャッシュディレクトリ
            max_cache_size_mb (int): 最大キャッシュサイズ（MB）
            cleanup_threshold (float): クリーンアップ開始閾値
        
        Returns:
            dict: キャッシュ管理情報
        """
        import os
        import shutil
        
        # デフォルトキャッシュディレクトリ
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser('~'), '.tumblr_image_collector', 'cache')
        
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            # 現在のキャッシュサイズを評価
            total_size = sum(
                os.path.getsize(os.path.join(cache_dir, f)) 
                for f in os.listdir(cache_dir) 
                if os.path.isfile(os.path.join(cache_dir, f))
            ) / (1024 * 1024)  # MB単位
            
            # キャッシュクリーンアップ
            if total_size > max_cache_size_mb * cleanup_threshold:
                # 古い順にソートし、上限を超えるファイルを削除
                files = sorted(
                    [(f, os.path.getatime(os.path.join(cache_dir, f))) for f in os.listdir(cache_dir)],
                    key=lambda x: x[1]
                )
                
                while total_size > max_cache_size_mb * cleanup_threshold and files:
                    oldest_file, _ = files.pop(0)
                    file_path = os.path.join(cache_dir, oldest_file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)
                    os.remove(file_path)
                    total_size -= file_size
            
            return {
                'cache_dir': cache_dir,
                'total_size_mb': total_size,
                'max_size_mb': max_cache_size_mb,
                'cleanup_performed': total_size > max_cache_size_mb * cleanup_threshold
            }
        
        except Exception as e:
            logger.error(f"Cache management error: {e}")
            return None
    
    def _memory_efficient_image_processing(self, image_path, processing_func, chunk_size=1024*1024):
        """
        メモリ効率の高い画像処理方法
        
     def process_posts(self, posts, blog_name):
        """
        Tumblrの投稿を処理し、画像をダウンロード
        
        Args:
            posts (list): 処理する投稿のリスト
            blog_name (str): ブログ名
        
        Returns:
            dict: ダウンロード統計
        """
        import concurrent.futures
        import os
        import random
        
        # ダウンロード統計の初期化
        download_stats = {
            'total_posts': len(posts),
            'downloaded_images': 0,
            'skipped_images': 0,
            'error_images': 0,
            'tags_count': {},
            'tag_image_limits': 10  # 1タグあたりの目標画像数
        }
        
        # タグごとの画像数を追跡するディクショナリ
        tag_image_counts = {}
        
        # シャッフルして偏りを減らす
        random.shuffle(posts)
        
        def should_download_image(post_tags):
            """
            タグごとの画像数を管理し、バランスを取る
            
            Args:
                post_tags (list): 投稿のタグリスト
            
            Returns:
                bool: 画像をダウンロードするかどうか
            """
            # タグの優先順位を決定
            tag_priority = {}
            for tag in post_tags:
                tag_lower = tag.lower()
                if tag_lower not in tag_image_counts:
                    tag_image_counts[tag_lower] = 0
                tag_priority[tag_lower] = tag_image_counts[tag_lower]
            
            # 最も少ないタグを選択
            min_tag = min(tag_priority, key=tag_priority.get)
            
            # 選択されたタグの画像数をインクリメント
            if tag_image_counts[min_tag] < download_stats['tag_image_limits']:
                tag_image_counts[min_tag] += 1
                return True
            
            return False
        
        def get_balanced_tags(post_tags):
            """
            バランスの取れたタグを選択
            
            Args:
                post_tags (list): 投稿のタグリスト
            
            Returns:
                list: バランスの取れたタグリスト
            """
            tag_counts = {tag.lower(): tag_image_counts.get(tag.lower(), 0) for tag in post_tags}
            min_count = min(tag_counts.values(), default=0)
            balanced_tags = [tag for tag, count in tag_counts.items() if count == min_count]
            
            return balanced_tags
        
        try:
            # 画像ファイルのダウンロード
            image_paths = []
            for post in posts:
                if 'photos' in post:
                    for photo in post['photos']:
                        image_url = photo['original_size']['url']
                        image_path = os.path.join(os.path.expanduser('~'), '.tumblr_image_collector', 'images', os.path.basename(image_url))
                        image_paths.append(image_path)
            
            # ダウンロード処理
            def download_image(image_path):
                try:
                    # 画像のダウンロード
                    image_url = os.path.basename(image_path)
                    response = requests.get(image_url, stream=True)
                    if response.status_code == 200:
                        with open(image_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size):
                                f.write(chunk)
                        return True
                    else:
                        logger.error(f"Failed to download {image_url}")
                        return False
                except Exception as e:
                    logger.error(f"Error downloading {image_url}: {e}")
                    return False
            
            # 并列処理
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                # 各画像に対して処理関数を実行
                results = list(executor.map(download_image, image_paths))
            
            # ダウンロード統計の更新
            download_stats['downloaded_images'] = sum(results)
            download_stats['skipped_images'] = len(posts) - download_stats['downloaded_images']
            
            return download_stats
        
        except Exception as e:
            logger.error(f"Error processing posts: {e}")
            return {}
    
    def organize_images_with_metadata(self, source_folder, destination_folder=None):
        """
        タグ情報とメタデータに基づく画像管理
        
        Args:
            source_folder (str): 入力画像フォルダ
            destination_folder (str, optional): 出力先フォルダ
        
        Returns:
            dict: 分類結果
        """
        import os
        import json
        import shutil
        
        # 出力先フォルダの設定
        if destination_folder is None:
            destination_folder = os.path.join(source_folder, 'organized_images')
        
        # 出力先フォルダ作成
        os.makedirs(destination_folder, exist_ok=True)
        metadata_folder = os.path.join(destination_folder, 'metadata')
        os.makedirs(metadata_folder, exist_ok=True)
        
        # 結果記録用変数
        results = {
            'total_images': 0,
            'person_images': 0,
            'non_person_images': 0
        }
        
        # 画像ファイルの調査
        for filename in os.listdir(source_folder):
            file_path = os.path.join(source_folder, filename)
            
            # 画像ファイルのみ対象
            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                results['total_images'] += 1
                
                try:
                    # メタデータ抽出
                    metadata = self.extract_image_metadata(file_path)
                    
                    # メタデータ保存
                    metadata_path = os.path.join(metadata_folder, f"{os.path.splitext(filename)[0]}_metadata.json")
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                    
                    # 人物関連の画像をカウント
                    if metadata['is_person']:
                        results['person_images'] += 1
                    else:
                        results['non_person_images'] += 1
                    
                except Exception as e:
                    logger.error(f"{filename}の分類中にエラー: {e}")
        
        # 結果のロギング
        logger.info(f"Image Organization Results: {results}")
        return results

    def _estimate_face_orientation(self, face_image, orientation_method='gradient'):
        """
        顔の方向性を推定
        
        Args:
            face_image (numpy.ndarray): 顔領域のグレースケール画像
            orientation_method (str): 方向性推定手法
        
        Returns:
            dict: 顔の方向性情報
        """
        try:
            import cv2
            import numpy as np
            
            if orientation_method == 'gradient':
                # 勾配ベースの方向性推定
                gx = cv2.Sobel(face_image, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(face_image, cv2.CV_64F, 0, 1, ksize=3)
                
                # 勾配の大きさと角度
                mag, angle = cv2.cartToPolar(gx, gy)
                
                # 主要な勾配方向
                dominant_angle = np.mean(angle)
                dominant_magnitude = np.mean(mag)
                
                return {
                    'dominant_angle': float(dominant_angle),
                    'gradient_magnitude': float(dominant_magnitude)
                }
            
            return {}
        except Exception as e:
            logger.error(f"顔の方向性推定中にエラー: {e}")
            return {}

    def _analyze_image_composition(self, image):
        """
        画像の構図を分析
        
        Args:
            image (numpy.ndarray): カラー画像
        
        Returns:
            dict: 構図分析結果
        """
        try:
            import cv2
            import numpy as np
            
            height, width = image.shape[:2]
            
            # ルールオブサードの分析
            third_height = height // 3
            third_width = width // 3
            
            # 各セクションの平均色を計算
            sections = [
                image[0:third_height, 0:third_width],
                image[0:third_height, third_width:2*third_width],
                image[0:third_height, 2*third_width:width],
                image[third_height:2*third_height, 0:third_width],
                image[third_height:2*third_height, third_width:2*third_width],
                image[third_height:2*third_height, 2*third_width:width],
                image[2*third_height:height, 0:third_width],
                image[2*third_height:height, third_width:2*third_width],
                image[2*third_height:height, 2*third_width:width]
            ]
            
            # 各セクションの平均色を計算
            section_colors = [np.mean(section, axis=(0, 1)) for section in sections]
            
            return {
                'rule_of_thirds': {
                    'section_colors': section_colors,
                    'dominant_section': np.argmax([np.mean(color) for color in section_colors])
                },
                'aspect_ratio': width / height
            }
        except Exception as e:
            logger.error(f"構図分析中にエラー: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error extracting image metadata: {e}")
            return None

    def _analyze_color_distribution(self, image):
        """画像のカラー分布を分析"""
        try:
            import numpy as np
            from PIL import ImageStat, ImageColor

            # RGBヒストグラムを計算
            img_array = np.array(image)
            r_hist, g_hist, b_hist = [np.histogram(img_array[:,:,i], bins=256)[0] for i in range(3)]

            # 主色を検出
            dominant_color = self._get_dominant_color(img_array)

            return {
                'color_entropy': self._calculate_color_entropy(img_array),
                'dominant_color': dominant_color,
                'color_variance': np.var(img_array),
                'red_histogram': r_hist.tolist(),
                'green_histogram': g_hist.tolist(),
                'blue_histogram': b_hist.tolist()
            }
        except Exception as e:
            logger.error(f"Error analyzing color distribution: {e}")
            return {}

    def _get_dominant_color(self, img_array):
        """画像の主色を検出"""
        try:
            import numpy as np

            # ピクセルをグルーピング
            pixels = img_array.reshape(-1, 3)
            unique, counts = np.unique(pixels, axis=0, return_counts=True)
            dominant_color = unique[np.argmax(counts)]

            return {
                'r': int(dominant_color[0]),
                'g': int(dominant_color[1]),
                'b': int(dominant_color[2])
            }
        except Exception as e:
            logger.error(f"Error finding dominant color: {e}")
            return None

    def _calculate_color_entropy(self, img_array):
        """カラーエントロピーを計算"""
        try:
            import numpy as np

            # RGB値のヒストグラムを計算
            hist, _ = np.histogramdd(img_array.reshape(-1, 3), bins=(16, 16, 16))
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-10))

            return float(entropy)
        except Exception as e:
            logger.error(f"Error calculating color entropy: {e}")
            return None

    def _calculate_image_quality(self, image):
        """画像品質のメトリックを計算"""
        try:
            import cv2
            import numpy as np

            # OpenCVに変換
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

            # シープネスとコントラストを評価
            laplacian_var = cv2.Laplacian(cv_image, cv2.CV_64F).var()
            sobel_x = cv2.Sobel(cv_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(cv_image, cv2.CV_64F, 0, 1, ksize=3)
            edge_density = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))

            return {
                'sharpness': float(laplacian_var),
                'edge_density': float(edge_density),
                'dynamic_range': float(np.max(cv_image) - np.min(cv_image))
            }
        except Exception as e:
            logger.error(f"Error calculating image quality metrics: {e}")
            return {}

    def _setup_credentials(self):
        """APIキーとOAuthトークンを設定ファイルから読み込むか、ユーザーに要求する"""
        # Prioritize config file values
        self.consumer_key = self.config.get("consumer_key")
        self.consumer_secret = self.config.get("consumer_secret")

        if not (self.consumer_key and self.consumer_secret):
            logger.info("API keys not found in config. Please enter them:")
            # Use input only if keys are missing
            self.consumer_key = input("Enter your Tumblr Consumer Key: ").strip()
            self.consumer_secret = input("Enter your Tumblr Consumer Secret: ").strip()
            if not (self.consumer_key and self.consumer_secret):
                logger.error("Consumer Key and Secret are required.")
                raise ValueError("Missing API credentials") # Raise exception
            self.config["consumer_key"] = self.consumer_key
            self.config["consumer_secret"] = self.consumer_secret
            self._save_config() # Save immediately after getting them
            logger.info("API keys saved to config.")

        self.token = self.config.get("token")
        self.token_secret = self.config.get("token_secret")

        if not (self.token and self.token_secret):
            logger.info("OAuth tokens not found in config. Attempting to obtain them...")
            self.token, self.token_secret = self._get_oauth_token()
            if not (self.token and self.token_secret):
                logger.error("Failed to get OAuth token.")
                raise ValueError("Missing OAuth credentials") # Raise exception
            self.config["token"] = self.token
            self.config["token_secret"] = self.token_secret
            self._save_config() # Save immediately
            logger.info("OAuth tokens saved to config.")

    def _initialize_client(self):
        """Tumblr APIクライアントを初期化する"""
        if not all([self.consumer_key, self.consumer_secret, self.token, self.token_secret]):
            logger.error("Cannot initialize Tumblr client: Credentials missing.")
            raise ValueError("Cannot initialize client due to missing credentials")
        try:
            client = pytumblr.TumblrRestClient(
                self.consumer_key,
                self.consumer_secret,
                self.token,
                self.token_secret
            )
            # Optional: Test connection by fetching user info
            # user_info = client.info()
            # logger.debug(f"Connected to Tumblr API as user: {user_info.get('user', {}).get('name')}")
            logger.info("Tumblr client initialized successfully.")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Tumblr client: {e}")
            raise ConnectionError("Failed to initialize Tumblr client") from e

    def get_blog_posts(self, blog_name, limit=20, offset=0):
        """Tumblrブログの投稿を取得する"""
        try:
            if not self.client:
                logger.error("Tumblr client not initialized.")
                return [] # Return empty list, not None
            posts_data = self.client.posts(blog_name, limit=limit, offset=offset)
            # Check for API errors in the response if possible
            # meta = posts_data.get('meta', {})
            # if meta.get('status') != 200:
            #     logger.error(f"Tumblr API error: {meta.get('msg', 'Unknown error')}")
            #     if meta.get('status') == 429: # Too Many Requests
            #          return None # Indicate rate limit
            #     return []
            return posts_data.get('posts', [])
        except Exception as e:
            # Attempt to parse specific pytumblr exceptions if they exist
            # or check status codes if requests.exceptions are caught
            if "limit" in str(e).lower() or "429" in str(e) or "too many requests" in str(e).lower():
                 logger.warning(f"Rate limit likely hit while fetching posts for '{blog_name}'.")
                 return None # Indicate potential rate limit
            logger.error(f"Error fetching posts for {blog_name}: {e}")
            return [] # Return empty list for other errors

    def _is_image_valid(self, image):
        """画像が指定された条件を満たすかチェックする"""
        try:
            # ファイルサイズのチェック
            file_size_mb = len(image.getvalue()) / (1024 * 1024)
            if file_size_mb > self.IMAGE_FILTERS['max_file_size_mb']:
                logger.debug(f"Image exceeds max file size: {file_size_mb:.2f} MB")
                return False

            # 画像サイズのチェック
            width, height = image.size
            if width < self.IMAGE_FILTERS['min_width'] or height < self.IMAGE_FILTERS['min_height']:
                logger.debug(f"Image too small: {width}x{height}")
                return False

            # アスペクト比のチェック
            aspect_ratio = width / height
            min_ratio, max_ratio = self.IMAGE_FILTERS['aspect_ratio_range']
            if not (min_ratio <= aspect_ratio <= max_ratio):
                logger.debug(f"Image aspect ratio out of range: {aspect_ratio:.2f}")
                return False

            # ファイル形式のチェック
            img_format = image.format.lower()
            if img_format not in self.IMAGE_FILTERS['allowed_formats']:
                logger.debug(f"Unsupported image format: {img_format}")
                return False

            # カラー画像の判定
            if self.IMAGE_FILTERS.get('color_threshold'):
                grayscale = image.convert('L')
                color_ratio = self._calculate_color_ratio(image, grayscale)
                if color_ratio < self.IMAGE_FILTERS['color_threshold']:
                    logger.debug(f"Low color content: {color_ratio:.2f}")
                    return False

            # ぼかし度の判定
            if self.IMAGE_FILTERS.get('blur_threshold'):
                blur_score = self._calculate_blur_score(image)
                if blur_score > self.IMAGE_FILTERS['blur_threshold']:
                    logger.debug(f"Image too blurry: {blur_score:.2f}")
                    return False

            # NSFW検出
            if self.IMAGE_FILTERS.get('nsfw_detection'):
                if self._is_nsfw_content(image):
                    logger.debug("NSFW content detected")
                    return False

            return True
        except Exception as e:
            logger.error(f"Error validating image: {e}")
            return False

    def _calculate_color_ratio(self, color_image, grayscale_image):
        """画像のカラーコンテンツ比率を計算"""
        color_pixels = color_image.convert('RGB')
        grayscale_pixels = grayscale_image.convert('L')

        color_array = np.array(color_pixels)
        grayscale_array = np.array(grayscale_pixels)

        # RGBチャンネルの差異を計算
        color_diff = np.abs(color_array[:,:,0] - grayscale_array) + \
                     np.abs(color_array[:,:,1] - grayscale_array) + \
                     np.abs(color_array[:,:,2] - grayscale_array)

        # カラーコンテンツの比率を計算
        color_ratio = np.mean(color_diff) / 255.0
        return color_ratio

    def _calculate_blur_score(self, image):
        """画像のぼかし度を計算"""
        import cv2
        import numpy as np

        # OpenCVに変換
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # ラプラシアンフィルタを使用してぼかし度を評価
        blur_score = cv2.Laplacian(cv_image, cv2.CV_64F).var()
        return blur_score

    def _is_nsfw_content(self, image):
        """NSFWコンテンツを検出"""
        # TODO: 外部のNSFW検出APIや機械学習モデルを統合
        # 例: Google Cloud Vision API, Azure Content Moderator, 自作のML分類器など
        # 現時点では常にFalseを返す
        return False

    def _is_image_duplicate(self, image):
        """画像の重複をハッシュを使ってチェックする"""
        for existing_file in self.output_folder.glob('*'):
            if existing_file.is_file():
                try:
                    existing_image = Image.open(existing_file)
                    existing_hash = imagehash.phash(existing_image)
                    current_hash = imagehash.phash(image)
                    hash_diff = bin(existing_hash - current_hash).count('1')

                    if hash_diff <= self.IMAGE_HASH_THRESHOLD:
                        logger.debug(f"Similar image found: {existing_file.name}")
                        return True
                except Exception as e:
                    logger.warning(f"Error checking image hash: {e}")
        return False

    def _create_requests_session(self, retries=3, backoff_factor=0.3):
        """値下げと再試行機能付きのRequestsセッションを作成"""
        session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"],
            backoff_factor=backoff_factor
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)

    def download_image(self, image_url, post_data=None, max_retries=3):
        """高度な再試行メカニズムとキャッシュ管理を使用した画像ダウンロード
        
        Args:
            image_url (str): ダウンロードする画像のURL
            post_data (dict, optional): 画像に関連する投稿データ
            max_retries (int, optional): 最大再試行回数. デフォルトは3.
        
        Returns:
            bool: ダウンロードの成功/失敗
        """
        # ダウンロード統計を更新
        self._download_stats['total_attempts'] += 1

        # キャッシュをチェック
        cached_file = self._check_cache(image_url)
        if cached_file:
            try:
                # キャッシュからコピー
                filename = self._generate_filename_from_path(cached_file)
                filepath = self.output_folder / filename
                shutil.copy2(cached_file, filepath)
                self._download_stats['cache_hits'] += 1
                return True
            except Exception as e:
                logger.error(f"キャッシュファイルのコピー中にエラー: {e}")

        def exponential_backoff(retry_count):
            """エクスポネンシャルバックオフ戦略
            
            Args:
                retry_count (int): 現在の再試行回数
            """
            base_delay = 1  # 基本の待機時間
            max_delay = 30  # 最大待機時間
            jitter = random.uniform(0, 0.5)  # ランダム性を追加
            delay = min(max_delay, base_delay * (2 ** retry_count) + jitter)
            logger.info(f"再試行 {retry_count + 1}: {delay}秒待機")
            time.sleep(delay)

        def is_network_error(exception):
            """ネットワークエラーを判定
            
            Args:
                exception (Exception): 発生した例外
            
            Returns:
                bool: ネットワークエラーかどうか
            """
            network_error_types = (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ProxyError,
                requests.exceptions.SSLError
            )
            return isinstance(exception, network_error_types)

        last_exception = None
        for retry_count in range(max_retries):
            try:
                # プロキシとタイムアウトを考慮
                response = requests.get(
                    image_url, 
                    stream=True, 
                    proxies=self.proxies, 
                    timeout=self.download_timeout,
                    headers={
                        'User-Agent': 'TumblrImageCollector/1.0',
                        'Accept': 'image/*'
                    }
                )
                response.raise_for_status()

                # 画像を一時ファイルに保存
                with tempfile.NamedTemporaryFile(delete=False, suffix=self._get_file_extension(image_url)) as temp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        temp_file.write(chunk)

                # 画像をロード
                with Image.open(temp_file.name) as img:
                    # 画像の有効性をチェック
                    if not self._is_image_valid(img):
                        os.unlink(temp_file.name)
                        logger.warning(f"無効な画像: {image_url}")
                        return False

                    # 画像メタデータを抽出
                    metadata = self._extract_image_metadata(img, image_url, post_data)

                    # ファイル名を生成
                    filename = self._generate_filename(img, metadata)
                    filepath = self.output_folder / filename

                    # 重複ファイルをチェック
                    if filepath.exists():
                        self._download_stats['skipped_duplicates'] += 1
                        os.unlink(temp_file.name)
                        logger.info(f"重複画像をスキップ: {filename}")
                        return False

                    # 一時ファイルを最終ファイルに移動
                    shutil.move(temp_file.name, filepath)

                    # キャッシュに保存
                    self._save_to_cache(filepath, image_url)

                    # ダウンロード絅計を更新
                    self._download_stats['successful_downloads'] += 1
                    logger.info(f"画像ダウンロード成功: {filename}")

                return True

            except requests.RequestException as e:
                last_exception = e
                logger.warning(f"ダウンロード試行 {retry_count + 1} 失敗: {e}")
                
                # ネットワークエラーの場合のみ再試行
                if is_network_error(e):
                    if retry_count < max_retries - 1:
                        exponential_backoff(retry_count)
                    else:
                        logger.error(f"{image_url}のダウンロードに{max_retries}回失敗")
                        self._download_stats['failed_downloads'] += 1
                        self._update_download_stats('failure')
                        return False
                else:
                    # 回復不能なエラーは即座に失敗
                    logger.error(f"回復不能なエラー: {e}")
                    self._download_stats['failed_downloads'] += 1
                    self._update_download_stats('failure')
                    return False

            except IOError as e:
                logger.error(f"ファイル処理エラー: {e}")
                self._download_stats['failed_downloads'] += 1
                self._update_download_stats('failure')
                return False

        # 最終的な例外を記録
        if last_exception:
            self._log_download_failure(image_url, last_exception)

        return False

    def _log_download_failure(self, image_url, exception):
        """
        ダウンロード失敗の詳細をログに記録
        
        Args:
            image_url (str): ダウンロードに失敗したURL
            exception (Exception): 発生した例外
        """
        failure_log_path = self.output_folder / 'download_failures.log'
        with open(failure_log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"""
時刻: {datetime.datetime.now()}
URL: {image_url}
エラータイプ: {type(exception).__name__}
エラー詳細: {str(exception)}
---
""")

        for retry_count in range(max_retries):
            try:
                # プロキシとタイムアウトを考慮
                response = requests.get(
                    image_url, 
                    stream=True, 
                    proxies=self.proxies, 
                    timeout=self.download_timeout
                )
                response.raise_for_status()

                # 画像を一時ファイルに保存
                with tempfile.NamedTemporaryFile(delete=False, suffix=self._get_file_extension(image_url)) as temp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        temp_file.write(chunk)

                # 画像をロード
                with Image.open(temp_file.name) as img:
                    # 画像の有効性をチェック
                    if not self._is_image_valid(img):
                        os.unlink(temp_file.name)
                        return False

                    # 画像メタデータを抽出
                    metadata = self._extract_image_metadata(img, image_url, post_data)

                    # ファイル名を生成
                    filename = self._generate_filename(img, metadata)
                    filepath = self.output_folder / filename

                    # 重複ファイルをチェック
                    if filepath.exists():
                        self._download_stats['skipped_duplicates'] += 1
                        os.unlink(temp_file.name)
                        return False
    metadata = self._extract_image_metadata(image, image_url, post_data)
    metadata_file = filepath.with_suffix('.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # ダウンロード統計を更新
    self.downloaded_files.add(filename)
    self._download_stats['successful_downloads'] += 1
            metadata = self._extract_image_metadata(image, image_url, post_data)
            metadata_file = filepath.with_suffix('.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            # ダウンロード統計を更新
            self.downloaded_files.add(filename)
            self._download_stats['successful_downloads'] += 1

            logger.info(f"Downloaded: {filename}")
            return filepath

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error downloading {image_url}: {e}")
            self._download_stats['failed_downloads'] += 1
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing {image_url}: {e}")
            self._download_stats['failed_downloads'] += 1
            return None

    def print_download_stats(self):
        """ダウンロード統計を表示"""
        stats = self._download_stats
        total_attempts = stats['total_attempts']
        success_rate = (stats['successful_downloads'] / total_attempts * 100) if total_attempts > 0 else 0

        logger.info("\n=== ダウンロード統計 ===")
        logger.info(f"総試行回数: {total_attempts}")
        logger.info(f"成功ダウンロード数: {stats['successful_downloads']}")
        logger.info(f"失敗ダウンロード数: {stats['failed_downloads']}")
        logger.info(f"重複スキップ数: {stats['skipped_duplicates']}")
        logger.info(f"成功率: {success_rate:.2f}%")

    def _log_final_stats(self):
        """最終的なダウンロード統計をログに記録する"""
        logger.info("--- ダウンロード統計 ---")
        logger.info(f"総ダウンロード試行回数: {self._download_stats['total_attempts']}")
        logger.info(f"成功したダウンロード数: {self._download_stats['successful_downloads']}")
        logger.info(f"失敗したダウンロード数: {self._download_stats['failed_downloads']}")

        # AI画像分類統計
        logger.info("\n--- AI画像分類統計 ---")
        ai_stats = self._download_stats['ai_classification_stats']
        logger.info(f"有効な画像数: {ai_stats['valid_images']}")
        logger.info(f"無効な画像数: {ai_stats['invalid_images']}")
        logger.info(f"高解像度画像数: {ai_stats['high_resolution_images']}")
        logger.info(f"低解像度画像数: {ai_stats['low_resolution_images']}")
        logger.info(f"潜在的にNSFWな画像数: {ai_stats['potentially_nsfw_images']}")

        # 画像タイプ分布
        logger.info("\n画像タイプ分布:")
        for type_name, type_data in ai_stats['image_type_distribution'].items():
            avg_confidence = (type_data['total_confidence'] / type_data['count']) if type_data['count'] > 0 else 0
            logger.info(f"{type_name}: 数 = {type_data['count']}, 平均信頼度 = {avg_confidence:.2f}")

        return self._download_stats

    def generate_image_thumbnail(self, image_path, size=(200, 200), quality=85):
        """
        画像のサムネイルを自動生成

        Args:
            image_path (str): 元画像のパス
            size (tuple): サムネイルのサイズ (幅, 高さ)
            quality (int): JPEGエンコーディングの画質 (1-95)

        Returns:
            Path: 生成されたサムネイルのパス
        """
        from PIL import Image
        import hashlib

        try:
            with Image.open(image_path) as img:
                # アスペクト比を維持してリサイズ
                img.thumbnail(size, Image.LANCZOS)
                
                # サムネイル保存パスを生成
                filename = os.path.basename(image_path)
                base, ext = os.path.splitext(filename)
                
                # ハッシュを使用してユニークな名前を作成
                hash_object = hashlib.md5(open(image_path, 'rb').read())
                thumbnail_filename = f'{base}_thumb_{hash_object.hexdigest()[:8]}.jpg'
                thumbnail_path = self.output_folder / 'thumbnails' / thumbnail_filename
                
                # ディレクトリ作成
                thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
                
                # サムネイル保存
                img.convert('RGB').save(thumbnail_path, 'JPEG', quality=quality)
                
                return thumbnail_path
        
        except Exception as e:
            logger.error(f"サムネイル生成エラー: {image_path} - {e}")
            return None

    def evaluate_image_quality(self, image_path, min_resolution=(800, 600), min_entropy=4.0):
        """
        画像の品質を評価

        Args:
            image_path (str): 評価する画像のパス
            min_resolution (tuple): 最小許容解像度
            min_entropy (float): 最小エントロピー関値

        Returns:
            dict: 画像品質評価結果
        """
        from PIL import Image
        import numpy as np
        import math

        try:
            with Image.open(image_path) as img:
                # 解像度チェック
                width, height = img.size
                resolution_score = (
                    width >= min_resolution[0] and 
                    height >= min_resolution[1]
                )

                # エントロピー計算
                img_array = np.array(img.convert('L'))
                hist, _ = np.histogram(img_array, bins=256, range=(0, 256))
                hist = hist[hist > 0] / img_array.size
                entropy = -np.sum(hist * np.log2(hist))

                # 画像の鮮明さ評価
                laplacian = cv2.Laplacian(img_array, cv2.CV_64F).var()

                quality_metrics = {
                    'resolution': {
                        'width': width,
                        'height': height,
                        'meets_minimum': resolution_score
                    },
                    'entropy': {
                        'value': entropy,
                        'meets_threshold': entropy >= min_entropy
                    },
                    'sharpness': {
                        'laplacian_variance': laplacian,
                        'is_sharp': laplacian > 100  # 経験的な関値
                    },
                    'overall_quality': resolution_score and (entropy >= min_entropy) and (laplacian > 100)
                }

                return quality_metrics

        except Exception as e:
            logger.error(f"画像品質評価エラー: {image_path} - {e}")
            return None

    def detect_image_duplicates(self, image_paths, hash_size=8, threshold=0.9):
        """
        知覚的ハッシュを使用した画像重複検出

        Args:
            image_paths (list): 重複を検出する画像パスのリスト
            hash_size (int): ハッシュのサイズ
            threshold (float): 類似度の関値

        Returns:
            list: 重複グループのリスト
        """
        import imagehash
        from PIL import Image

        def calculate_hash(image_path):
            try:
                with Image.open(image_path) as img:
                    return imagehash.phash(img, hash_size=hash_size)
            except Exception as e:
                logger.error(f"ハッシュ計算エラー: {image_path} - {e}")
                return None

        # ハッシュ計算
        image_hashes = {path: calculate_hash(path) for path in image_paths if calculate_hash(path) is not None}

        # 重複検出
        duplicates = []
        processed = set()

        for path1, hash1 in image_hashes.items():
            if path1 in processed:
                continue

            duplicate_group = [path1]
            processed.add(path1)

            for path2, hash2 in image_hashes.items():
                if path1 != path2 and path2 not in processed:
                    similarity = 1 - (hash1 - hash2) / len(hash1.hash)**2
                    if similarity >= threshold:
                        duplicate_group.append(path2)
                        processed.add(path2)

            if len(duplicate_group) > 1:
                duplicates.append(duplicate_group)

        return duplicates

    def configure_network_proxy(self, proxy_config=None):
        """
        ネットワークプロキシの詳細設定

        Args:
            proxy_config (dict, optional): プロキシ設定辞書
                {
                    'type': 'http/https/socks4/socks5',
                    'host': 'プロキシホスト',
                    'port': プロキシポート,
                    'username': オプションのユーザー名,
                    'password': オプションのパスワード
                }

        Returns:
            dict: 設定されたプロキシ情報
        """
        import requests
        from urllib.parse import urlparse

        # デフォルトプロキシ設定
        default_proxy = {
            'type': None,
            'host': None,
            'port': None,
            'username': None,
            'password': None,
            'use_proxy': False
        }

        # プロキシ設定がない場合は対話型設定
        if proxy_config is None:
            try:
                use_proxy = input("プロキシを使用しますか？ (y/N): ").lower() == 'y'
                if use_proxy:
                    proxy_type = input("プロキシタイプ (http/https/socks4/socks5): ").lower()
                    proxy_host = input("プロキシホスト (例: 127.0.0.1): ")
                    proxy_port = int(input("プロキシポート (例: 8080): "))
                    
                    use_auth = input("プロキシ認証が必要ですか？ (y/N): ").lower() == 'y'
                    username = password = None
                    
                    if use_auth:
                        username = input("プロキシユーザー名: ")
                        password = getpass.getpass("プロキシパスワード: ")

                    proxy_config = {
                        'type': proxy_type,
                        'host': proxy_host,
                        'port': proxy_port,
                        'username': username,
                        'password': password
                    }
                else:
                    return default_proxy
            except Exception as e:
                logger.error(f"プロキシ設定エラー: {e}")
                return default_proxy

        # プロキシURLを構築
        proxy_url = f"{proxy_config['type']}://"
        if proxy_config.get('username') and proxy_config.get('password'):
            proxy_url += f"{proxy_config['username']}:{proxy_config['password']}@"
        proxy_url += f"{proxy_config['host']}:{proxy_config['port']}"

        # プロキシ設定を検証
        try:
            proxies = {
                'http': proxy_url,
                'https': proxy_url
            }
            
            # プロキシ接続テスト
            test_url = 'https://www.example.com'
            response = requests.get(test_url, proxies=proxies, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"プロキシ接続成功: {proxy_url}")
                proxy_config['use_proxy'] = True
                return proxy_config
            else:
                logger.warning(f"プロキシ接続テスト失敗: {response.status_code}")
                return default_proxy

        except requests.exceptions.RequestException as e:
            logger.error(f"プロキシ接続エラー: {e}")
            return default_proxy

    def advanced_connection_settings(self, timeout=30, max_retries=3, backoff_factor=0.3):
        """
        高度な接続設定とエラーハンドリング

        Args:
            timeout (int): デフォルト接続タイムアウト（秒）
            max_retries (int): 最大再試行回数
            backoff_factor (float): 再試行間のバックオフ促数

        Returns:
            requests.adapters.HTTPAdapter: カスタマイズされた接続アダプター
        """
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=backoff_factor
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        
        logger.info(f"接続設定: タイムアウト={timeout}秒, 最大再試行={max_retries}, バックオフ促数={backoff_factor}")
        
        return adapter

    def multi_blog_search(self, blogs=None, tags=None, search_params=None):
        """
        複数のブログとタグを横断した高度な画像検索

        Args:
            blogs (list): 検索対象のブログリスト
            tags (list): 検索対象のタグリスト
            search_params (dict): 追加の検索パラメータ
                {
                    'min_likes': 最小いいね数,
                    'min_notes': 最小ノート数,
                    'date_range': {'start': 開始日, 'end': 終了日},
                    'content_type': ['photo', 'gif', 'illustration'],
                    'nsfw_filter': ブール値
                }

        Returns:
            list: 検索結果の画像ウエブリ
        """
        import tumblr
        from datetime import datetime, timedelta

        # デフォルトパラメータ
        default_params = {
            'min_likes': 0,
            'min_notes': 0,
            'date_range': {
                'start': datetime.now() - timedelta(days=30),
                'end': datetime.now()
            },
            'content_type': ['photo'],
            'nsfw_filter': True
        }

        # パラメータのマージ
        if search_params:
            default_params.update(search_params)

        # 検索結果の保存リスト
        search_results = []

        # ブログとタグのデフォルト値
        blogs = blogs or [self.tumblr_blog]
        tags = tags or []

        try:
            # Tumblr APIクライアントの初期化
            client = tumblr.TumblrRestClient(
                self.consumer_key, 
                self.consumer_secret, 
                self.oauth_token, 
                self.oauth_token_secret
            )

            # 各ブログを横断して検索
            for blog in blogs:
                offset = 0
                limit = 50  # ページごとの取得数

                while True:
                    # タグ付き投稿を検索
                    if tags:
                        posts = client.tagged(
                            tag='|'.join(tags),  # タグの論理OR検索
                            before=default_params['date_range']['end'],
                            limit=limit,
                            offset=offset
                        )
                    else:
                        # ブログの投稿を検索
                        posts = client.posts(
                            blog,
                            type='photo',
                            limit=limit,
                            offset=offset
                        )

                    # 検索結果がない場合は終了
                    if not posts:
                        break

                    # 投稿をフィルタリング
                    for post in posts:
                        # 日付範囲チェック
                        post_date = datetime.strptime(post['date'], '%Y-%m-%d %H:%M:%S %Z')
                        if not (default_params['date_range']['start'] <= post_date <= default_params['date_range']['end']):
                            continue

                        # いいね数とノート数のフィルタリング
                        if (post.get('note_count', 0) < default_params['min_notes'] or 
                            post.get('likes', 0) < default_params['min_likes']):
                            continue

                        # NSFWフィルタ
                        if default_params['nsfw_filter'] and post.get('is_nsfw', False):
                            continue

                        # 画像URL抽出
                        for photo in post.get('photos', []):
                            search_results.append(photo['original_size']['url'])

                    offset += limit

            logger.info(f"検索結果: {len(search_results)}件の画像を発見")
            return search_results

        except Exception as e:
            logger.error(f"マルチブログ検索エラー: {e}")
            return []

    def advanced_image_search(self, query, search_type='semantic', max_results=100):
        """
        AI支援による高度な画像検索

        Args:
            query (str): 検索クエリ
            search_type (str): 検索タイプ ('semantic', 'tag', 'color', 'style')
            max_results (int): 最大検索結果数

        Returns:
            list: 検索結果の画像ウエブリ
        """
        from PIL import Image
        import numpy as np
        import torch
        from transformers import CLIPProcessor, CLIPModel

        try:
            # CLIP（Contrastive Language-Image Pre-training）モデルの読み込み
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            # 検索タイプに応じた処理
            if search_type == 'semantic':
                # テキストからの意味的検索
                inputs = processor(text=[query], return_tensors="pt", padding=True)
                text_features = model.get_text_features(**inputs)

                # 画像特徴量との類似度計算
                similarity_scores = []
                for image_path in self._get_local_images():
                    img = Image.open(image_path)
                    image_inputs = processor(images=img, return_tensors="pt", padding=True)
                    image_features = model.get_image_features(**image_inputs)
                    
                    # コサイン類似度計算
                    similarity = torch.nn.functional.cosine_similarity(text_features, image_features)
                    similarity_scores.append((image_path, similarity.item()))

                # 類似度でソートして上位を返す
                return [path for path, score in sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:max_results]]

            elif search_type == 'color':
                # 色ベースの検索
                from colorthief import ColorThief
                
                query_color = self._extract_dominant_color(query)
                color_matches = []

                for image_path in self._get_local_images():
                    dominant_color = self._extract_dominant_color(image_path)
                    color_distance = self._color_distance(query_color, dominant_color)
                    color_matches.append((image_path, color_distance))

                return [path for path, distance in sorted(color_matches, key=lambda x: x[1])[:max_results]]

            elif search_type == 'style':
                # スタイル分類による検索
                style_model = self._load_style_classification_model()
                style_matches = []

                for image_path in self._get_local_images():
                    predicted_style = style_model.predict(image_path)
                    if predicted_style == query:
                        style_matches.append(image_path)

                return style_matches[:max_results]

            else:
                logger.warning(f"サポートされていない検索タイプ: {search_type}")
                return []

        except Exception as e:
            logger.error(f"高度な画像検索エラー: {e}")
            return []

    def _get_local_images(self, directory=None, extensions=None):
        """
        ローカルの画像ファイルを取得

        Args:
            directory (str, optional): 検索するディレクトリ。指定しない場合はデフォルトのダウンロードディレクトリ
            extensions (list, optional): 対象とする画像拡張子

        Returns:
            list: 画像ファイルのパスリスト
        """
        import os
        import glob

        # デフォルトディレクトリ設定
        if directory is None:
            directory = os.path.join(os.getcwd(), 'downloads')

        # デフォルト画像拡張子
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']

        # 画像ファイルを再帰的に検索
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(directory, f'**/*{ext}'), recursive=True))

        logger.info(f"{len(image_paths)}件の画像ファイルを発見")
        return image_paths

    def _load_style_classification_model(self):
        """
        スタイル分類のための機械学習モデルを読み込む

        Returns:
            object: 学習済みスタイル分類モデル
        """
        import tensorflow as tf
        import numpy as np
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

        class StyleClassifier:
            def __init__(self, num_classes=10):
                """
                スタイル分類モデルの初期化

                Args:
                    num_classes (int): スタイルカテゴリの数
                """
                base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                x = base_model.output
                x = GlobalAveragePooling2D()(x)
                x = Dense(1024, activation='relu')(x)
                output = Dense(num_classes, activation='softmax', name='style_output')(x)
                
                self.model = Model(inputs=base_model.input, outputs=output)
                
                # 基本モデルの重みを凍結
                for layer in base_model.layers:
                    layer.trainable = False

                self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                
                # スタイルラベル
                self.style_labels = [
                    'anime', 'photorealistic', 'illustration', 
                    'digital_art', 'watercolor', 'sketch', 
                    'pixel_art', 'comic', 'abstract', 'minimalist'
                ]

            def preprocess_image(self, image_path):
                """
                画像の前処理

                Args:
                    image_path (str): 画像ファイルのパス

                Returns:
                    numpy.ndarray: 前処理された画像データ
                """
                from PIL import Image
                import numpy as np

                img = Image.open(image_path).convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img)
                img_array = preprocess_input(img_array)
                return np.expand_dims(img_array, axis=0)

            def predict(self, image_path):
                """
                画像のスタイルを予測

                Args:
                    image_path (str): 画像ファイルのパス

                Returns:
                    str: 予測されたスタイルラベル
                """
                try:
                    preprocessed_img = self.preprocess_image(image_path)
                    predictions = self.model.predict(preprocessed_img)
                    predicted_class_index = np.argmax(predictions)
                    return self.style_labels[predicted_class_index]
                except Exception as e:
                    logger.error(f"スタイル分類エラー: {e}")
                    return None

        # モデルのインスタンス化とキャッシュ
        if not hasattr(self, '_style_classifier'):
            self._style_classifier = StyleClassifier()

        return self._style_classifier

    def advanced_configuration_wizard(self, config_type='full'):
        """
        高度な設定ウィザード

        Args:
            config_type (str): 設定タイプ ('full', 'download', 'processing', 'network')

        Returns:
            dict: 設定された詳細な設定情報
        """
        import os
        import json
        import getpass
        from typing import Dict, Any

        # デフォルト設定テンプレート
        default_config = {
            'tumblr_credentials': {
                'consumer_key': None,
                'consumer_secret': None,
                'oauth_token': None,
                'oauth_token_secret': None
            },
            'download_settings': {
                'output_directory': os.path.join(os.getcwd(), 'downloads'),
                'max_concurrent_downloads': 5,
                'retry_attempts': 3,
                'download_timeout': 60,
                'file_naming_pattern': '{blog}_{timestamp}_{index}'
            },
            'image_processing': {
                'generate_thumbnails': True,
                'thumbnail_size': (200, 200),
                'quality_threshold': {
                    'min_resolution': (800, 600),
                    'min_entropy': 2.0,
                    'max_blur': 50
                },
                'duplicate_detection': {
                    'method': 'perceptual_hash',
                    'similarity_threshold': 0.9
                }
            },
            'network_settings': {
                'use_proxy': False,
                'proxy_config': {
                    'type': None,
                    'host': None,
                    'port': None,
                    'username': None,
                    'password': None
                },
                'connection_timeout': 30,
                'max_retries': 3
            },
            'logging': {
                'level': 'INFO',
                'file_path': os.path.join(os.getcwd(), 'logs', 'tumblr_collector.log')
            }
        }

        def prompt_tumblr_credentials() -> Dict[str, str]:
            print("\n🔐 Tumblr API 認証情報の設定")
            credentials = {}
            credentials['consumer_key'] = input("Consumer Key を入力: ")
            credentials['consumer_secret'] = getpass.getpass("Consumer Secret を入力: ")
            credentials['oauth_token'] = input("OAuth Token を入力: ")
            credentials['oauth_token_secret'] = getpass.getpass("OAuth Token Secret を入力: ")
            return credentials

        def prompt_download_settings() -> Dict[str, Any]:
            print("\n📥 ダウンロード設定")
            settings = {}
            settings['output_directory'] = input(f"出力ディレクトリ (デフォルト: {default_config['download_settings']['output_directory']}): ") or default_config['download_settings']['output_directory']
            settings['max_concurrent_downloads'] = int(input(f"最大同時ダウンロード数 (デフォルト: {default_config['download_settings']['max_concurrent_downloads']}): ） or default_config['download_settings']['max_concurrent_downloads'])
            settings['retry_attempts'] = int(input(f"ダウンロード再試行回数 (デフォルト: {default_config['download_settings']['retry_attempts']}): ") or default_config['download_settings']['retry_attempts'])
            return settings

        def prompt_image_processing() -> Dict[str, Any]:
            print("\n🖼️ 画像処理設定")
            settings = {}
            settings['generate_thumbnails'] = input("サムネイル生成を有効にしますか？ (y/N): ").lower() == 'y'
            if settings['generate_thumbnails']:
                settings['thumbnail_size'] = tuple(map(int, input("サムネイルサイズ (幅,高さ) (デフォルト: 200,200): ").split(',') or (200, 200)))
            return settings

        def prompt_network_settings() -> Dict[str, Any]:
            print("\n🌐 ネットワーク設定")
            settings = {}
            settings['use_proxy'] = input("プロキシを使用しますか？ (y/N): ").lower() == 'y'
            if settings['use_proxy']:
                settings['proxy_config'] = {
                    'type': input("プロキシタイプ (http/https/socks4/socks5): "),
                    'host': input("プロキシホスト: "),
                    'port': input("プロキシポート: "),
                    'username': input("プロキシユーザー名 (オプション): ") or None,
                    'password': getpass.getpass("プロキシパスワード (オプション): ") or None
                }
            return settings

        # 設定タイプに応じた処理
        if config_type == 'full':
            default_config['tumblr_credentials'] = prompt_tumblr_credentials()
            default_config['download_settings'].update(prompt_download_settings())
            default_config['image_processing'].update(prompt_image_processing())
            default_config['network_settings'].update(prompt_network_settings())

        elif config_type == 'download':
            default_config['download_settings'].update(prompt_download_settings())

        elif config_type == 'processing':
            default_config['image_processing'].update(prompt_image_processing())

        elif config_type == 'network':
            default_config['network_settings'].update(prompt_network_settings())

        # 設定の保存
        config_dir = os.path.join(os.getcwd(), 'config')
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, 'tumblr_collector_config.json')

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4, ensure_ascii=False)

        logger.info(f"設定を {config_path} に保存しました。")
        return default_config

    def validate_configuration(self, config_path=None):
        """
        設定ファイルの検証と修復

        Args:
            config_path (str, optional): 設定ファイルのパス

        Returns:
            bool: 設定が有効かどうか
        """
        import os
        import json
        import jsonschema

        # 設定スキーマ定義
        config_schema = {
            "type": "object",
            "properties": {
                "tumblr_credentials": {
                    "type": "object",
                    "required": ["consumer_key", "consumer_secret", "oauth_token", "oauth_token_secret"]
                },
                "download_settings": {
                    "type": "object",
                    "properties": {
                        "max_concurrent_downloads": {"type": "number", "minimum": 1, "maximum": 20},
                        "retry_attempts": {"type": "number", "minimum": 0, "maximum": 10}
                    }
                }
            },
            "required": ["tumblr_credentials", "download_settings"]
        }

        # デフォルトの設定ファイルパス
        if not config_path:
            config_path = os.path.join(os.getcwd(), 'config', 'tumblr_collector_config.json')

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 設定スキーマの検証
            jsonschema.validate(instance=config, schema=config_schema)

            # 追加の検証ロジック
            if not config['tumblr_credentials']['consumer_key']:
                logger.error("Consumer Key が設定されていません。")
                return False

            # ネットワーク設定の検証
            if config.get('network_settings', {}).get('use_proxy'):
                proxy_config = config['network_settings']['proxy_config']
                if not all([proxy_config['type'], proxy_config['host'], proxy_config['port']]):
                    logger.warning("プロキシ設定が不完全です。")

            return True

        except FileNotFoundError:
            logger.error(f"設定ファイルが見つかりません: {config_path}")
            return False
        except jsonschema.exceptions.ValidationError as e:
            logger.error(f"設定ファイルの検証エラー: {e}")
            return False
        except json.JSONDecodeError:
            logger.error(f"設定ファイルの形式が不正です: {config_path}")
            return False

    def auto_image_collection(self, collection_params=None):
        """
        高度な自動画像収集メソッド

        Args:
            collection_params (dict): 画像収集のパラメータ
                {
                    'blogs': ブログリスト,
                    'tags': タグリスト,
                    'max_images': 最大収集画像数,
                    'min_resolution': 最小解像度 (幅, 高さ),
                    'min_likes': 最小いいね数,
                    'min_notes': 最小ノート数,
                    'date_range': {'start': 開始日, 'end': 終了日},
                    'content_type': ['photo', 'illustration', 'art'],
                    'nsfw_filter': ブール値,
                    'style_filters': ['anime', 'photorealistic', 'digital_art'],
                    'download_options': {
                        'output_directory': 出力ディレクトリ,
                        'naming_pattern': ファイル名パターン,
                        'overwrite': 上書きフラグ
                    },
                    'advanced_filters': {
                        'color_palette': 色パレット,
                        'entropy_threshold': エントロピー閾値,
                        'aspect_ratio_range': (最小比率, 最大比率)
                    }
                }

        Returns:
            dict: 収集結果の詳細情報
        """
        import os
        import datetime
        import concurrent.futures
        from urllib.parse import urlparse

        # デフォルトパラメータ
        default_params = {
            'blogs': [self.tumblr_blog],
            'tags': [],
            'max_images': 100,
            'min_resolution': (800, 600),
            'min_likes': 5,
            'min_notes': 3,
            'date_range': {
                'start': datetime.datetime.now() - datetime.timedelta(days=30),
                'end': datetime.datetime.now()
            },
            'content_type': ['photo'],
            'nsfw_filter': True,
            'style_filters': [],
            'download_options': {
                'output_directory': os.path.join(os.getcwd(), 'downloads', 'auto_collection'),
                'naming_pattern': '{blog}_{timestamp}_{index}',
                'overwrite': False
            },
            'advanced_filters': {
                'color_palette': None,
                'entropy_threshold': 2.0,
                'aspect_ratio_range': (0.5, 2.0)
            }
        }

        # パラメータのマージ
        if collection_params:
            default_params = self._deep_merge(default_params, collection_params)

        # 出力ディレクトリ作成
        os.makedirs(default_params['download_options']['output_directory'], exist_ok=True)

        # 収集結果の初期化
        collection_results = {
            'total_found': 0,
            'downloaded_images': [],
            'skipped_images': [],
            'errors': []
        }

        try:
            # マルチブログ検索
            search_results = self.multi_blog_search(
                blogs=default_params['blogs'],
                tags=default_params['tags'],
                search_params={
                    'min_likes': default_params['min_likes'],
                    'min_notes': default_params['min_notes'],
                    'date_range': default_params['date_range'],
                    'content_type': default_params['content_type'],
                    'nsfw_filter': default_params['nsfw_filter']
                }
            )

            collection_results['total_found'] = len(search_results)

            # 画像処理と並列ダウンロード
            def process_image(image_url):
                try:
                    # 画像の詳細情報取得
                    image_info = self._analyze_image_details(image_url)

                    # 高度なフィルタリング
                    if not self._apply_advanced_filters(image_info, default_params):
                        collection_results['skipped_images'].append(image_url)
                        return None

                    # スタイルフィルタリング
                    if default_params['style_filters']:
                        style_model = self._load_style_classification_model()
                        predicted_style = style_model.predict(image_info['local_path'])
                        if predicted_style not in default_params['style_filters']:
                            collection_results['skipped_images'].append(image_url)
                            return None

                    # ダウンロードと保存
                    filename = self._generate_filename(
                        image_url, 
                        default_params['download_options']['naming_pattern'],
                        default_params['download_options']['output_directory']
                    )

                    if not default_params['download_options']['overwrite'] and os.path.exists(filename):
                        collection_results['skipped_images'].append(image_url)
                        return None

                    # 画像をダウンロード
                    downloaded_path = self.download_image(
                        image_url, 
                        filename, 
                        timeout=30
                    )

                    collection_results['downloaded_images'].append(downloaded_path)
                    return downloaded_path

                except Exception as e:
                    logger.error(f"画像処理エラー: {image_url}, エラー: {e}")
                    collection_results['errors'].append({
                        'url': image_url,
                        'error': str(e)
                    })
                    return None

            # 並列処理
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(process_image, image_url) 
                    for image_url in search_results[:default_params['max_images']]
                ]
                concurrent.futures.wait(futures)

            return collection_results

        except Exception as e:
            logger.error(f"自動画像収集エラー: {e}")
            return {
                'total_found': 0,
                'downloaded_images': [],
                'skipped_images': [],
                'errors': [{'error': str(e)}]
            }

    def _analyze_image_details(self, image_url):
        import requests
        from PIL import Image
        import io
        import numpy as np

        try:
            response = requests.get(image_url, timeout=10)
            img = Image.open(io.BytesIO(response.content))

            img_array = np.array(img)
            width, height = img.size

            return {
                'url': image_url,
                'width': width,
                'height': height,
                'aspect_ratio': width / height,
                'mode': img.mode,
                'format': img.format,
                'local_path': self._save_temp_image(img)
            }

        except Exception as e:
            logger.error(f"画像詳細分析エラー: {image_url}, エラー: {e}")
            return None

    def _save_temp_image(self, img):
        import os
        import tempfile

        temp_dir = os.path.join(tempfile.gettempdir(), 'tumblr_collector')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f'temp_image_{hash(img)}.png')
        img.save(temp_path)
        return temp_path

    def _apply_advanced_filters(self, image_info, params):
        # 解像度フィルタ
        min_width, min_height = params['min_resolution']
        if image_info['width'] < min_width or image_info['height'] < min_height:
            return False

        # アスペクト比フィルタ
        min_ratio, max_ratio = params['advanced_filters']['aspect_ratio_range']
        if not (min_ratio <= image_info['aspect_ratio'] <= max_ratio):
            return False

        # エントロピーフィルタ
        if params['advanced_filters']['entropy_threshold']:
            entropy = self._calculate_image_entropy(image_info['local_path'])
            if entropy < params['advanced_filters']['entropy_threshold']:
                return False

        # 色パレットフィルタ
        if params['advanced_filters']['color_palette']:
            dominant_color = self._extract_dominant_color(image_info['local_path'])
            if not self._color_matches_palette(dominant_color, params['advanced_filters']['color_palette']):
                return False

        return True

    def _generate_filename(self, image_url, naming_pattern, output_directory):
        import os
        from urllib.parse import urlparse
        from datetime import datetime

        # URLからブログ名を抽出
        parsed_url = urlparse(image_url)
        blog_name = parsed_url.netloc.split('.')[0]

        # パターンを置換
        filename = naming_pattern.format(
            blog=blog_name,
            timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'),
            index=hash(image_url) % 10000
        )

        # 拡張子を追加
        filename = f"{filename}.{image_url.split('.')[-1]}"

        return os.path.join(output_directory, filename)

    def _calculate_image_entropy(self, image_path):
        from PIL import Image
        import numpy as np
        from scipy.stats import entropy

        img = Image.open(image_path)
        img_array = np.array(img)
        
        # グレースケールに変換
        if len(img_array.shape) == 3:
            img_array = np.mean(img_array, axis=2)
        
        # ヒストグラムを計算
        hist, _ = np.histogram(img_array, bins=256)
        hist = hist[hist > 0]
        
        # エントロピーを計算
        return entropy(hist / hist.sum())

    def _extract_dominant_color(self, image_path):
        from PIL import Image
        import numpy as np
        from sklearn.cluster import KMeans

        img = Image.open(image_path)
        img_array = np.array(img)
        
        # 画像をリサイズして処理を高速化
        img_array = img_array.reshape(-1, 3)
        
        # K-meansで支配的な色を抽出
        kmeans = KMeans(n_clusters=1)
        kmeans.fit(img_array)
        
        return tuple(map(int, kmeans.cluster_centers_[0]))

    def _color_matches_palette(self, color, palette, tolerance=30):
        import numpy as np

        for palette_color in palette:
            distance = np.sqrt(np.sum((np.array(color) - np.array(palette_color))**2))
            if distance <= tolerance:
                return True
        
        return False

    def resume_image_collection(self, previous_collection_results=None, additional_params=None):
        """
        前回の画像収集を再開または継続するメソッド

        Args:
            previous_collection_results (dict): 前回の収集結果
            additional_params (dict): 追加または上書きするパラメータ

        Returns:
            dict: 更新された収集結果
        """
        import os
        import json
        from datetime import datetime, timedelta

        # デフォルトの再開パラメータ
        resume_params = {
            'resume_from_last_collection': True,
            'skip_downloaded_images': True,
            'extend_date_range': True,
            'max_retry_count': 3,
            'retry_delay': 60  # 秒
        }

        # ユーザーパラメータでデフォルトを上書き
        if additional_params:
            resume_params.update(additional_params)

        # 前回の収集結果がない場合は新規収集を実行
        if not previous_collection_results:
            previous_collection_results = self._load_last_collection_state()

        if not previous_collection_results:
            logger.warning("前回の収集結果が見つかりません。新規収集を開始します。")
            return self.auto_image_collection()

        # 収集パラメータを復元
        collection_params = previous_collection_results.get('collection_params', {})

        # 日付範囲を拡張
        if resume_params['extend_date_range']:
            collection_params['date_range'] = {
                'start': previous_collection_results.get('end_date', 
                    datetime.now() - timedelta(days=30)),
                'end': datetime.now()
            }

        # スキップするイメージのリストを作成
        if resume_params['skip_downloaded_images']:
            collection_params['skip_images'] = set(
                previous_collection_results.get('downloaded_images', []) +
                previous_collection_results.get('skipped_images', [])
            )

        # 再試行メカニズム
        retry_count = 0
        while retry_count < resume_params['max_retry_count']:
            try:
                # 画像収集を実行
                new_collection_results = self.auto_image_collection(collection_params)

                # 結果をマージ
                merged_results = self._merge_collection_results(
                    previous_collection_results, 
                    new_collection_results
                )

                # 状態を保存
                self._save_collection_state(merged_results)

                return merged_results

            except Exception as e:
                logger.error(f"収集中にエラーが発生: {e}")
                retry_count += 1
                
                if retry_count < resume_params['max_retry_count']:
                    logger.info(f"再試行 {retry_count}/{resume_params['max_retry_count']}")
                    time.sleep(resume_params['retry_delay'])
                else:
                    logger.error("最大再試行回数に達しました。")
                    return previous_collection_results

        return previous_collection_results

    def _load_last_collection_state(self, state_file='last_collection_state.json'):
        """
        最後の収集状態を読み込む

        Args:
            state_file (str): 状態ファイルのパス

        Returns:
            dict: 前回の収集状態、見つからない場合はNone
        """
        import os
        import json

        state_path = os.path.join(os.getcwd(), 'downloads', state_file)
        
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"状態ファイルの読み込みエラー: {e}")
        
        return None

    def _save_collection_state(self, collection_results, state_file='last_collection_state.json'):
        """
        収集状態を保存する

        Args:
            collection_results (dict): 収集結果
            state_file (str): 状態ファイルのパス
        """
        import os
        import json
        from datetime import datetime

        # ダウンロードディレクトリを作成
        os.makedirs(os.path.join(os.getcwd(), 'downloads'), exist_ok=True)
        
        state_path = os.path.join(os.getcwd(), 'downloads', state_file)
        
        try:
            # 追加のメタデータを含める
            collection_results['saved_timestamp'] = datetime.now().isoformat()
            
            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(collection_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"収集状態を {state_path} に保存しました。")
        
        except Exception as e:
            logger.error(f"状態ファイルの保存エラー: {e}")

    def _merge_collection_results(self, previous_results, new_results):
        """
        複数の収集結果をマージする

        Args:
            previous_results (dict): 以前の収集結果
            new_results (dict): 新しい収集結果

        Returns:
            dict: マージされた収集結果
        """
        merged_results = {
            'total_found': previous_results.get('total_found', 0) + new_results.get('total_found', 0),
            'downloaded_images': list(set(
                previous_results.get('downloaded_images', []) + 
                new_results.get('downloaded_images', [])
            )),
            'skipped_images': list(set(
                previous_results.get('skipped_images', []) + 
                new_results.get('skipped_images', [])
            )),
            'errors': (
                previous_results.get('errors', []) + 
                new_results.get('errors', [])
            ),
            'collection_params': previous_results.get('collection_params', {})
        }

        # 最新の収集パラメータで更新
        merged_results['collection_params'].update(new_results.get('collection_params', {}))

        return merged_results

    def export_metadata(self, output_format='json'):
        """画像メタデータをエクスポートする

        Args:
            output_format (str, optional): エクスポート形式. デフォルトは 'json'.
                サポートされる形式: 'json', 'csv'

        Returns:
            Path: 生成されたメタデータファイルのパス
        """
        import json
        import csv
        from pathlib import Path
        from PIL import Image
        import datetime
        import os

        # メタデータディレクトリを作成
        metadata_dir = self.output_folder / 'metadata'
        metadata_dir.mkdir(exist_ok=True)

        # メタデータファイル名を生成
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        metadata_filename = f'tumblr_metadata_{timestamp}'

        # メタデータリストを作成
        metadata_list = []
        for filepath in self.output_folder.glob('*.*'):
            if filepath.is_file() and filepath.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                try:
                    # 画像メタデータを抽出
                    metadata = self._extract_image_metadata(str(filepath))
                    
                    if metadata:
                        # 追加のファイル情報を含める
                        metadata['local_path'] = str(filepath)
                        metadata['file_size'] = filepath.stat().st_size
                        metadata['last_modified'] = filepath.stat().st_mtime
                        
                        # 画像の追加情報
                        with Image.open(filepath) as img:
                            metadata['width'], metadata['height'] = img.size
                            metadata['format'] = img.format
                        
                        metadata_list.append(metadata)
                
                except Exception as e:
                    logger.error(f"メタデータ抽出エラー: {filepath} - {e}")

        # エクスポート形式を選択
        if output_format == 'json':
            output_file = metadata_dir / f'{metadata_filename}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, ensure_ascii=False, indent=4)
            logger.info(f"メタデータをJSONファイルにエクスポート: {output_file}")
            return output_file

        elif output_format == 'csv':
            output_file = metadata_dir / f'{metadata_filename}.csv'
            # CSVに出力可能な形式に変換
            if metadata_list:
                # すべてのキーを取得（フラット化）
                keys = set()
                for item in metadata_list:
                    keys.update(item.keys())
                
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=sorted(list(keys)))
                    writer.writeheader()
                    writer.writerows(metadata_list)
            
            logger.info(f"メタデータをCSVファイルにエクスポート: {output_file}")
            return output_file

        else:
            logger.warning(f"サポートされていないフォーマット: {output_format}. JSONにフォールバックします。")
            output_file = metadata_dir / f'{metadata_filename}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, ensure_ascii=False, indent=4)
            return output_file

        metadata_dir = self.output_folder / 'metadata'
        metadata_dir.mkdir(exist_ok=True)

        # メタデータファイル名を生成
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        metadata_filename = f'tumblr_metadata_{timestamp}'

        # メタデータリストを作成
        metadata_list = []
        for filepath in self.output_folder.glob('*.*'):
            if filepath.is_file() and filepath.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                try:
                    with Image.open(filepath) as img:
                        metadata = self._extract_image_metadata(str(filepath))
                        if metadata:
                            metadata['local_path'] = str(filepath)
                            metadata['file_size'] = filepath.stat().st_size
                            metadata['last_modified'] = filepath.stat().st_mtime
                            metadata_list.append(metadata)
                except Exception as e:
                    logger.error(f"メタデータ抽出エラー: {filepath} - {e}")

        # エクスポート形式を選択
        if output_format == 'json':
            output_file = metadata_dir / f'{metadata_filename}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, ensure_ascii=False, indent=4)
            logger.info(f"メタデータをJSONファイルにエクスポート: {output_file}")
            return output_file

        elif output_format == 'csv':
            output_file = metadata_dir / f'{metadata_filename}.csv'
            # CSVに出力可能な形式に変換
            if metadata_list:
                keys = metadata_list[0].keys()
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(metadata_list)
            logger.info(f"メタデータをCSVファイルにエクスポート: {output_file}")
            return output_file

        else:
            logger.warning(f"サポートされていないフォーマット: {output_format}. JSONにフォールバックします。")
            output_file = metadata_dir / f'{metadata_filename}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, ensure_ascii=False, indent=4)
            return output_file
                try:
                    with Image.open(filepath) as img:
                        metadata = self._extract_image_metadata(str(filepath))
                        if metadata:
                            metadata['local_path'] = str(filepath)
                            metadata_list.append(metadata)
                except Exception as e:
                    logger.error(f"Error processing {filepath}: {e}")

        # エクスポート形式を選択
        if output_format == 'json':
            output_file = metadata_dir / f'{metadata_filename}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, indent=2, ensure_ascii=False)
        elif output_format == 'csv':
            output_file = metadata_dir / f'{metadata_filename}.csv'
            if metadata_list:
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=metadata_list[0].keys())
                    writer.writeheader()
                    writer.writerows(metadata_list)
        else:
            logger.error(f"Unsupported export format: {output_format}")
            return None

        logger.info(f"Metadata exported to {output_file}")
        return output_file

    def download_tagged_images(self, tag, count=3):
        """指定されたタグの画像をダウンロードする (高度な並列ダウンロード)"""
        if not self.executor:
            self._setup_executor()

        try:
            # Tumblrから指定タグの投稿を取得
            posts = self.client.tagged(tag, limit=count)

            # ダウンロードキューとスレッドの管理
            download_queue = queue.Queue()
            download_results = {
                'total_images': 0,
                'successful_downloads': 0,
                'failed_downloads': 0
            }

            def download_worker():
                while not download_queue.empty():
                    try:
                        post = download_queue.get(block=False)
                        if post.get('type') == 'photo':
                            success = self.download_image(post['photos'][0]['original_size']['url'], post)
                            with threading.Lock():
                                download_results['total_images'] += 1
                                if success:
                                    download_results['successful_downloads'] += 1
                                else:
                                    download_results['failed_downloads'] += 1
                    except queue.Empty:
                        break
                    except Exception as e:
                        logger.error(f"Error in download worker: {e}")
                    finally:
                        download_queue.task_done()

            # キューにポストを追加
            for post in posts:
                download_queue.put(post)

            # Wait for completion (optional, mainly for debugging or sequential logic)
            # concurrent.futures.wait(futures)

        except Exception as e:
            logger.error(f"Error during tagged post fetch/submission for tag '{tag}': {e}")
            if "limit" in str(e).lower() or "429" in str(e) or "too many requests" in str(e).lower():
                 logger.warning(f"Rate limit likely hit while fetching tagged posts for '{tag}'.")
                 rate_limit_suspected = True
                 for f in futures: f.cancel() # Attempt to cancel pending downloads for this tag

        return not rate_limit_suspected

    def process_posts(self, posts):
        """投稿から画像をダウンロードし、関連タグの画像もダウンロードする (並列ダウンロード)"""
        if not self.executor:
            logger.error("Executor not available for processing posts.")
            return True

        initial_image_futures = []
        rate_limit_hit = False

        # Submit initial image downloads
        submitted_count = 0
        for post in posts:
            if post.get('type') == 'photo':
                for photo in post.get('photos', []):
                    image_url = photo.get('original_size', {}).get('url')
                    if image_url:
                        future = self.executor.submit(self.download_image, image_url)
                        initial_image_futures.append((future, post.get('tags', [])))
                        submitted_count += 1
        logger.debug(f"Submitted {submitted_count} initial image download tasks for this batch.")

        # Process results as they complete
        processed_count = 0
        for future, tags in initial_image_futures:
            if rate_limit_hit:
                future.cancel()
                continue

            try:
                image_filename = future.result()  # Wait for initial download
                processed_count += 1

                if image_filename:
                    logger.info(f"({processed_count}/{submitted_count}) Downloaded initial image: {image_filename}")

                    # タグ処理の最適化と並列化
                    related_tags_successful = self._process_related_tags(image_filename, tags)

                    if not related_tags_successful:
                        rate_limit_hit = True
                        continue

            except concurrent.futures.CancelledError:
                logger.debug("Initial image download cancelled.")
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing result for an initial image download: {e}")
                processed_count += 1  # Count as processed even if errored

        logger.debug(f"Finished processing results for {submitted_count} initial images.")
        return not rate_limit_hit

    def _process_related_tags(self, image_filename, tags):
        """関連タグの処理を分離したメソッド"""
        if not tags:
            logger.info(f"No tags found for post containing image: {image_filename}")
            return True

        logger.debug(f"Processing {len(tags)} related tags for {image_filename}...")

                    result = future.result(timeout=30)  # 30秒のタイムアウト
                    if result:
                        tag_processing_results['successful_tags'] += 1
                        logger.debug(f"Successfully processed tag: {tag}")
                    else:
                        tag_processing_results['failed_tags'] += 1
                        logger.warning(f"Failed to process tag: {tag}")

                except concurrent.futures.TimeoutError:
                    tag_processing_results['failed_tags'] += 1
                    logger.warning(f"Tag processing timed out: {tag}")

            # 処理結果のサマリーをログ出力
            logger.info(
                f"Tag Processing Summary: "
                f"Successful: {tag_processing_results['successful_tags']}, "
                f"Failed: {tag_processing_results['failed_tags']}"
            )

            # レートリミットの可能性を検出
            if tag_processing_results['failed_tags'] > len(top_tags) // 2:
                logger.error("Potential rate limit detected: Too many tag processing failures")
                return False

        # ランダムタグの高度な選択
        if top_tags:
            random_tag = random.choice(top_tags)
            logger.debug(f"Selected random tag with potential relevance: {random_tag}")
            logger.debug(f"Processing random tag '{random_tag}' for {image_filename}")
            if not self.download_tagged_images(random_tag, 3):
                logger.warning(f"Rate limit suspected processing random tag '{random_tag}'.")
                return False

        return True

    def wait_and_resume(self):
        """API制限時の待機処理"""
        sleep_time = int(self.api_wait_hours * 60 * 60)
        logger.warning(f"API rate limit hit. Waiting for {self.api_wait_hours} hours ({sleep_time} seconds)...")

        # クラッシュレポートを生成
        self._generate_crash_report(None, None, "Rate limit hit")

        # 待機中のプログレスバーとカウントダウン
        for remaining in range(sleep_time, 0, -1):
            logger.debug(f"Waiting... {remaining} seconds remaining.")
            time.sleep(1)

        logger.info("Resuming download process after rate limit wait.")

    def run(self, blog_name):
        """メインの実行ループ"""
        offset_key = f"offset_{blog_name}"
        offset = self.config.get(offset_key, 0)
        logger.info(f"Starting download for blog '{blog_name}' from offset {offset}.")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as self.executor:
            while True:
                try:
                    logger.info(f"Fetching posts for '{blog_name}' with offset {offset}...")
                    posts = self.get_blog_posts(blog_name, limit=20, offset=offset)

                    if posts is None:  # Rate limit suspected
                        self.wait_and_resume()
                        continue

                    if not posts:  # No more posts or non-rate-limit error
                        logger.info("No more posts retrieved or error fetching posts.")
                        self.config[offset_key] = offset  # Save current offset on clean exit
                        self._save_config()
                        break

                    # Process posts
                    if not self.process_posts(posts):
                        # Rate limit hit during processing
                        self.wait_and_resume()
                        continue  # Retry same offset

                    # Success, advance offset
                    processed_count = len(posts)
                    offset += processed_count
                    self.config[offset_key] = offset  # Save progress
                    self._save_config()
                    logger.info(f"Successfully processed batch. New offset for '{blog_name}': {offset}")

                    # Wait before fetching next batch
                    logger.debug(f"Sleeping for {self.api_batch_sleep} seconds before next batch...")
                    time.sleep(self.api_batch_sleep)

                except Exception as e:
                    logger.error(f"Unexpected error in download process: {e}")
                    self._generate_crash_report(type(e), e, None)
                    break

        logger.info(f"Download process finished for blog '{blog_name}'.")


def main():
    """Parses arguments, sets up logging, creates collector, and runs it."""
    parser = argparse.ArgumentParser(description="Download images from a Tumblr blog and related tags.")
    parser.add_argument("blog_name", help="The name of the Tumblr blog (e.g., staff)")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG_FILE,
                        help=f"Path to the configuration file (default: {DEFAULT_CONFIG_FILE})")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory name or path (overrides config file setting)")
    parser.add_argument("-w", "--workers", type=int, default=None,
                        help="Number of download workers (overrides config file setting)")
    parser.add_argument("-l", "--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (default: INFO)")
    parser.add_argument("--log-file", default=DEFAULT_LOG_FILE,
                        help=f"Path to the log file (default: {DEFAULT_LOG_FILE})")

    args = parser.parse_args()

    # --- Configure Logging ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File Handler
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    logger.setLevel(log_level)

    # --- Run Collector ---
    collector = None # Initialize to None for finally block
    try:
        collector = TumblrImageCollector(
            config_file=args.config,
            output_dir_override=args.output,
            workers_override=args.workers
        )
        collector.run(args.blog_name)
    except (ValueError, ConnectionError, IOError) as e:
         logger.error(f"Initialization or runtime error: {e}")
         # No need to save config if initialization failed badly
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}") # Log full traceback
    finally:
        # Ensure config is saved on exit, especially if interrupted during run
        if collector and hasattr(collector, 'config'):
             logger.info("Saving final configuration state.")
             collector._save_config()
        logger.info("Exiting.")


import concurrent.futures
import time
import random
from functools import partial
from urllib.parse import urlparse

def exponential_backoff(attempt, base_delay=1, max_delay=60):
    """
    エクスポネンシャルバックオフ戦略
    
    Args:
        attempt (int): 現在の再試行回数
        base_delay (float): 基本遅延時間
        max_delay (float): 最大遅延時間
    
    Returns:
        float: 計算された遅延時間
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = random.uniform(0, 0.1 * delay)
    return delay + jitter

def download_with_retry(url, output_folder, max_retries=3, timeout=30):
    """
    高度な再試行メカニズムを持つダウンロード関数
    
    Args:
        url (str): ダウンロードするURL
        output_folder (Path): 出力フォルダ
        max_retries (int): 最大再試行回数
        timeout (int): タイムアウト時間（秒）
    
    Returns:
        tuple: (ダウンロード成功フラグ, ファイルパス)
    """
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    output_path = output_folder / filename

    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"ダウンロード成功: {url}")
            return True, output_path

        except (requests.RequestException, IOError) as e:
            delay = exponential_backoff(attempt)
            logger.warning(f"ダウンロード失敗 (試行 {attempt + 1}/{max_retries}): {url}")
            logger.warning(f"エラー: {e}")
            logger.info(f"次の再試行まで {delay:.2f}秒待機")
            time.sleep(delay)

    logger.error(f"最大再試行回数に達しました: {url}")
    return False, None

def parallel_download(image_urls, output_folder, max_workers=5):
    """
    画像を並列でダウンロード
    
    Args:
        image_urls (list): ダウンロードするURL一覧
        output_folder (Path): 出力フォルダ
        max_workers (int): 最大並列ワーカー数
    
    Returns:
        list: ダウンロードされたファイルパス
    """
    downloaded_files = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 部分適用された関数を作成
        download_func = partial(download_with_retry, output_folder=output_folder)
        
        # 並列ダウンロード
        futures = {executor.submit(download_func, url): url for url in image_urls}
        
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                success, file_path = future.result()
                if success:
                    downloaded_files.append(file_path)
            except Exception as e:
                logger.error(f"ダウンロード中に予期せぬエラー: {url} - {e}")
    
    return downloaded_files

def main():
    # 既存のmain関数の処理...
    
    # 並列ダウンロードの例
    image_urls = [...]  # 画像URLリスト
    downloaded_images = parallel_download(image_urls, output_folder)

if __name__ == "__main__":
    main()