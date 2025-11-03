"""
Image Processor

画像処理、分類、重複検出、NSFW検出を担当するモジュール
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import imagehash
import numpy as np

logger = logging.getLogger(__name__)

# 画像処理ライブラリのオプションインポート
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False

try:
    import skimage.feature
    _SKIMAGE_AVAILABLE = True
except ImportError:
    skimage = None
    _SKIMAGE_AVAILABLE = False

class ImageProcessor:
    """
    画像処理と分類を担当するクラス
    """

    # 画像フィルタリングオプション
    IMAGE_FILTERS = {
        'min_width': 500,
        'min_height': 500,
        'allowed_formats': ['jpg', 'jpeg', 'png', 'gif', 'webp'],
        'max_file_size_mb': 10,
        'aspect_ratio_range': (0.5, 2.0),
        'color_threshold': 0.1,
        'blur_threshold': 50,
        'nsfw_detection': True
    }

    # 画像ハッシュの類似度閾値
    IMAGE_HASH_THRESHOLD = 5

    # 各種定数
    BYTES_TO_MB_DIVISOR = 1048576
    MAX_METADATA_SIZE_BYTES = 1024 * 1024
    DEFAULT_QUALITY = 85
    MIN_IMAGE_DIMENSION = 500
    BLUR_THRESHOLD = 50

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nsfw_threshold = config.get('filters', {}).get('nsfw_threshold', 0.35)

        # ダウンロード済み画像のハッシュセット（重複検出用）
        self.downloaded_hashes = set()

    def is_image_valid(self, image: Image.Image) -> bool:
        """
        画像が指定された条件を満たすかチェック

        Args:
            image: PIL Imageオブジェクト

        Returns:
            画像が有効かどうか
        """
        try:
            # ファイルサイズのチェック
            file_size_mb = len(image.tobytes()) / self.BYTES_TO_MB_DIVISOR
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
            img_format = image.format.lower() if image.format else ''
            if img_format not in self.IMAGE_FILTERS['allowed_formats']:
                logger.debug(f"Unsupported image format: {img_format}")
                return False

            # カラー画像の判定
            if self.IMAGE_FILTERS.get('color_threshold'):
                color_ratio = self._calculate_color_ratio(image)
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

    def is_image_duplicate(self, image: Image.Image) -> bool:
        """
        画像の重複をハッシュを使ってチェック

        Args:
            image: PIL Imageオブジェクト

        Returns:
            重複しているかどうか
        """
        try:
            # 画像ハッシュを計算
            image_hash = imagehash.average_hash(image)

            # ダウンロード済みハッシュと比較
            for existing_hash in self.downloaded_hashes:
                if abs(image_hash - existing_hash) <= self.IMAGE_HASH_THRESHOLD:
                    logger.debug(f"Duplicate image detected (hash difference: {abs(image_hash - existing_hash)})")
                    return True

            # 新しいハッシュを追加
            self.downloaded_hashes.add(image_hash)
            return False

        except Exception as e:
            logger.error(f"Error checking image duplicate: {e}")
            return False

    def _calculate_color_ratio(self, image: Image.Image) -> float:
        """画像のカラーコンテンツ比率を計算"""
        try:
            # RGBに変換
            color_pixels = image.convert('RGB')
            grayscale_pixels = image.convert('L')

            color_array = np.array(color_pixels)
            grayscale_array = np.array(grayscale_pixels)

            # RGBチャンネルの差異を計算
            color_diff = np.abs(color_array[:,:,0] - grayscale_array) + \
                         np.abs(color_array[:,:,1] - grayscale_array) + \
                         np.abs(color_array[:,:,2] - grayscale_array)

            # カラーコンテンツの比率を計算
            color_ratio = np.mean(color_diff) / 255.0
            return color_ratio
        except Exception as e:
            logger.error(f"Error calculating color ratio: {e}")
            return 0.0

    def _calculate_blur_score(self, image: Image.Image) -> float:
        """画像のぼかし度を計算"""
        if not _CV2_AVAILABLE:
            return 0.0

        try:
            # OpenCVに変換
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

            # ラプラシアンフィルタを使用してぼかし度を評価
            blur_score = cv2.Laplacian(cv_image, cv2.CV_64F).var()
            return blur_score
        except Exception as e:
            logger.error(f"Error calculating blur score: {e}")
            return 0.0

    def _is_nsfw_content(self, image: Image.Image) -> bool:
        """NSFWコンテンツを検出"""
        if not _CV2_AVAILABLE:
            logger.debug("NSFW detection skipped: OpenCV not available")
            return False

        try:
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            nsfw_score = self._estimate_nsfw_content(image_np)
            return nsfw_score >= self.nsfw_threshold
        except Exception as e:
            logger.error(f"Error detecting NSFW content: {e}")
            return False

    def _estimate_nsfw_content(self, image: np.ndarray) -> float:
        """NSFWコンテンツの推定スコアを計算"""
        # 簡易的なNSFW検出（実際の実装ではより高度なモデルを使用）
        try:
            # 色の分布に基づく簡易判定
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # 肌色の範囲を定義（簡易版）
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 255, 255])

            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = cv2.countNonZero(skin_mask) / (image.shape[0] * image.shape[1])

            # 肌色が多すぎる場合はNSFWの可能性
            return min(skin_ratio * 2.0, 1.0)

        except Exception as e:
            logger.error(f"Error estimating NSFW content: {e}")
            return 0.0

    def calculate_image_quality(self, image_path: Path, fast_mode: bool = True) -> Dict[str, Any]:
        """
        画像品質の総合的な評価

        Args:
            image_path: 画像ファイルパス
            fast_mode: 高速モード

        Returns:
            品質評価結果
        """
        if not _CV2_AVAILABLE:
            return {'quality_score': 0.5, 'error': 'OpenCV not available'}

        try:
            # 画像読み込み
            image = cv2.imread(str(image_path))
            if image is None:
                return {'quality_score': 0.0, 'error': 'Failed to load image'}

            # 画像サイズの縮小（高速モード時）
            original_shape = image.shape
            if fast_mode and image.shape[0] > 800:
                scale_factor = 800 / image.shape[0]
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

            return {
                'quality_score': quality_score,
                'sharpness': sharpness_score,
                'brightness': brightness_score,
                'noise_level': noise_score,
                'original_size': original_shape[:2],
                'resized': fast_mode and original_shape[0] > 800
            }

        except Exception as e:
            logger.error(f"Error calculating image quality: {e}")
            return {'quality_score': 0.0, 'error': str(e)}

    def extract_image_metadata(self, image_path: Path) -> Dict[str, Any]:
        """
        画像からメタデータを抽出

        Args:
            image_path: 画像ファイルパス

        Returns:
            メタデータ辞書
        """
        try:
            with Image.open(image_path) as img:
                metadata = {
                    'filename': image_path.name,
                    'size': img.size,
                    'format': img.format,
                    'mode': img.mode,
                    'is_person': False,
                    'is_nsfw': False,
                    'quality_score': 0.0,
                    'tags': []
                }

                # 画像サイズ
                width, height = img.size
                metadata['width'] = width
                metadata['height'] = height
                metadata['aspect_ratio'] = width / height if height > 0 else 0

                # ファイルサイズ
                file_size = image_path.stat().st_size
                metadata['file_size_bytes'] = file_size
                metadata['file_size_mb'] = file_size / self.BYTES_TO_MB_DIVISOR

                # 画像品質評価
                quality_info = self.calculate_image_quality(image_path)
                metadata.update(quality_info)

                # 人物判定（簡易版）
                # 実際の実装ではより高度なAIモデルを使用
                if width > 400 and height > 400:
                    metadata['is_person'] = True
                    metadata['tags'].append('person')

                return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata from {image_path}: {e}")
            return {'error': str(e)}

    def generate_recommended_tags(self, image_path: Path) -> List[str]:
        """
        画像から推奨タグを自動生成

        Args:
            image_path: 画像ファイルパス

        Returns:
            推奨タグのリスト
        """
        if not _CV2_AVAILABLE:
            return []

        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return []

            recommended_tags = []

            # 色分析
            dominant_colors = self._extract_dominant_colors(image)
            color_tags = [f'{color_name} tone' for color_name, _ in dominant_colors]
            recommended_tags.extend(color_tags)

            # 画像の明るさ
            brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) / 255.0
            if brightness < 0.3:
                recommended_tags.append('dark')
            elif brightness > 0.7:
                recommended_tags.append('bright')

            # アスペクト比
            height, width = image.shape[:2]
            aspect_ratio = width / height
            if aspect_ratio < 0.75:
                recommended_tags.append('portrait')
            elif aspect_ratio > 1.33:
                recommended_tags.append('landscape')

            # 品質タグ
            quality_info = self.calculate_image_quality(image_path)
            quality_score = quality_info.get('quality_score', 0)
            if quality_score > 0.8:
                recommended_tags.append('high-quality')
            elif quality_score < 0.3:
                recommended_tags.append('low-quality')

            return list(set(recommended_tags))

        except Exception as e:
            logger.error(f'Error generating tags for {image_path}: {e}')
            return []

    def _extract_dominant_colors(self, image: np.ndarray, num_colors: int = 3) -> List[Tuple[str, float]]:
        """画像の主色を抽出"""
        try:
            # 画像を1D配列に変換
            pixels = image.reshape(-1, 3)

            # K-meansクラスタリングで主色を抽出
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_colors, n_init=10)
            kmeans.fit(pixels)

            # 色の名前付け（簡易版）
            colors = []
            for center in kmeans.cluster_centers_:
                r, g, b = center
                # 基本的な色名付け
                if r > 200 and g > 200 and b > 200:
                    color_name = 'white'
                elif r < 50 and g < 50 and b < 50:
                    color_name = 'black'
                elif r > g and r > b:
                    color_name = 'red'
                elif g > r and g > b:
                    color_name = 'green'
                elif b > r and b > g:
                    color_name = 'blue'
                else:
                    color_name = 'gray'

                colors.append((color_name, 1.0))  # 簡易的に信頼度1.0

            return colors

        except Exception as e:
            logger.error(f"Error extracting dominant colors: {e}")
            return []

    def load_downloaded_hashes(self, hash_file: Path) -> None:
        """ダウンロード済み画像のハッシュを読み込み"""
        if not hash_file.exists():
            return

        try:
            with open(hash_file, 'r', encoding='utf-8') as f:
                hash_strings = f.read().splitlines()

            for hash_str in hash_strings:
                if hash_str.strip():
                    try:
                        # ハッシュ文字列をimagehashオブジェクトに変換
                        hash_obj = imagehash.hex_to_hash(hash_str.strip())
                        self.downloaded_hashes.add(hash_obj)
                    except Exception as e:
                        logger.debug(f"Invalid hash format: {hash_str} - {e}")

            logger.info(f"Loaded {len(self.downloaded_hashes)} image hashes from {hash_file}")

        except Exception as e:
            logger.error(f"Error loading downloaded hashes: {e}")

    def save_downloaded_hashes(self, hash_file: Path) -> None:
        """ダウンロード済み画像のハッシュを保存"""
        try:
            hash_file.parent.mkdir(parents=True, exist_ok=True)
            with open(hash_file, 'w', encoding='utf-8') as f:
                for hash_obj in self.downloaded_hashes:
                    f.write(f"{hash_obj}\n")

            logger.debug(f"Saved {len(self.downloaded_hashes)} image hashes to {hash_file}")

        except Exception as e:
            logger.error(f"Error saving downloaded hashes: {e}")
