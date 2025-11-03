import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from skimage import color, measure

logger = logging.getLogger(__name__)

try:  # TensorFlowは任意依存に切り替え
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import (
        MobileNetV2,
        decode_predictions,
        preprocess_input,
    )
    from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as efficientnet_preprocess
    from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    # Vision Transformer support
    try:
        from tensorflow.keras.applications import vit
        _VIT_AVAILABLE = True
    except ImportError:
        try:
            import tf_keras_vis
            _VIT_AVAILABLE = True
        except ImportError:
            _VIT_AVAILABLE = False

    # YOLO support
    try:
        import yolov9
        _YOLO_AVAILABLE = True
    except ImportError:
        try:
            from ultralytics import YOLO
            _YOLO_AVAILABLE = True
        except ImportError:
            _YOLO_AVAILABLE = False

    # OCR support
    try:
        import pytesseract
        from PIL import Image
        _OCR_AVAILABLE = True
    except ImportError:
        _OCR_AVAILABLE = False

    _TENSORFLOW_AVAILABLE = True
    _EFFICIENTNET_AVAILABLE = True
except ImportError:
    _EFFICIENTNET_AVAILABLE = False
    _VIT_AVAILABLE = False
    _YOLO_AVAILABLE = False
    _OCR_AVAILABLE = False
    try:  # EfficientNetが失敗した場合のフォールバック
        from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as efficientnet_preprocess
        _EFFICIENTNET_AVAILABLE = True
    except ImportError:
        _EFFICIENTNET_AVAILABLE = False
        logger.warning("EfficientNetが利用できないため、MobileNetV2のみを使用します。")
except Exception:  # pragma: no cover - 環境によってはTensorFlowが存在しない
    tf = None
    MobileNetV2 = None
    decode_predictions = None
    preprocess_input = None
    img_to_array = None
    load_img = None
    ImageDataGenerator = None
    Dense = None
    GlobalAveragePooling2D = None
    Model = None
    load_model = None
    Adam = None
    EarlyStopping = None
    ModelCheckpoint = None
    _TENSORFLOW_AVAILABLE = False
    _EFFICIENTNET_AVAILABLE = False
    _VIT_AVAILABLE = False
    _YOLO_AVAILABLE = False
    _OCR_AVAILABLE = False


class ImageClassifier:
    """軽量ヒューリスティックと任意の深層学習モデルを両立する分類器。"""

    def __init__(
        self,
        nsfw_threshold: float = 0.25,
        min_resolution: Tuple[int, int] = (300, 300),
        enable_deep_model: bool = False,
        model_type: str = 'mobilenet',  # 'mobilenet' or 'efficientnet'
        model_input_size: int = 224,
        num_classes: int = 10,
        learning_rate: float = 1e-4,
        model_save_path: str = 'image_classifier.h5',
        pretrained_weights: Optional[str] = None,
    ) -> None:
        self.nsfw_threshold = nsfw_threshold
        self.min_resolution = min_resolution
        self.model_type = model_type
        self.model_input_size = model_input_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model_save_path = model_save_path

        self._deep_model_requested = enable_deep_model and _TENSORFLOW_AVAILABLE
        self._deep_model_loaded = False
        self._classification_model: Optional[Any] = None

        if enable_deep_model and not _TENSORFLOW_AVAILABLE:
            logger.warning(
                "TensorFlowが利用できないため、ヒューリスティックモードで動作します。"
            )
        
        # 高度なモデル機能の初期化
        if self._deep_model_requested:
            try:
                if pretrained_weights and os.path.exists(pretrained_weights):
                    self._classification_model = self._load_pretrained_model(pretrained_weights)
                    self._deep_model_loaded = True
                    logger.info(f'事前学習済みモデルをロード: {pretrained_weights}')
                else:
                    self._classification_model = self._build_advanced_model()
                    self._deep_model_loaded = True
                    logger.info(f'新規モデルを構築: クラス数={num_classes}, 画像サイズ={model_input_size}, モデルタイプ={model_type}')
            except Exception as e:
                logger.error(f'高度なモデル初期化中にエラー発生: {e}')
                self._deep_model_requested = False

    # ------------------------------------------------------------------
    # 公開API
    # ------------------------------------------------------------------
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """画像の品質判定と簡易分類を実施する。"""

        # 入力検証
        if not image_path or not isinstance(image_path, str):
            logger.error("Invalid image path provided")
            return self._error_response("Invalid image path")

        image_path_obj = Path(image_path)

        # パス検証
        if not image_path_obj.exists():
            logger.error(f"Image file does not exist: {image_path}")
            return self._error_response("File not found")

        # ファイルサイズ検証
        try:
            file_size = image_path_obj.stat().st_size
            max_size = 200 * 1024 * 1024  # 200MB
            if file_size > max_size:
                logger.error(f"Image file too large: {file_size / 1024 / 1024:.2f}MB")
                return self._error_response("File too large")
            if file_size == 0:
                logger.error("Image file is empty")
                return self._error_response("Empty file")
        except OSError as e:
            logger.error(f"Failed to get file stats: {e}")
            return self._error_response(f"File access error: {e}")

        try:
            with Image.open(image_path) as img:
                # 画像形式検証
                if img.format not in ('JPEG', 'PNG', 'GIF', 'WEBP', 'BMP'):
                    logger.warning(f"Unsupported image format: {img.format}")

                img = img.convert("RGB")
                width, height = img.size

                # サイズ検証（爆弾画像対策）
                max_pixels = 178956970
                if width * height > max_pixels:
                    logger.error(f"Image dimensions too large: {width}x{height}")
                    return self._error_response("Image dimensions exceed limit")

                resolution_ok = width >= self.min_resolution[0] and height >= self.min_resolution[1]

                np_image = np.asarray(img, dtype=np.float32)
                entropy = self._compute_entropy(img)
                brightness = float(np_image.mean() / 255.0)
                contrast = float(np_image.std() / 128.0)

                skin_ratio = self._estimate_skin_ratio(np_image)
                nsfw_flag = skin_ratio >= self.nsfw_threshold

                # OCRによるテキスト抽出（オプション）
                ocr_result = {}
                if _OCR_AVAILABLE:
                    try:
                        ocr_result = self.extract_text_from_image(image_path)
                    except Exception as e:
                        logger.debug(f"OCR処理をスキップ: {e}")

                # YOLOによる物体検出（オプション）
                object_detection = {}
                if _YOLO_AVAILABLE:
                    try:
                        object_detection = self.detect_objects_yolo(image_path)
                    except Exception as e:
                        logger.debug(f"物体検出をスキップ: {e}")

                # 高度な分類予測
                predictions: List[Dict[str, Any]] = []
                if self._deep_model_requested and self._ensure_deep_model():
                    predictions = self._predict_with_deep_model(img)
                    nsfw_flag = nsfw_flag or self._contains_nsfw_keyword(predictions)

                return {
                    "is_valid": bool(resolution_ok and not nsfw_flag),
                    "resolution": (width, height),
                    "is_high_resolution": bool(resolution_ok),
                    "is_potentially_nsfw": bool(nsfw_flag),
                    "top_predictions": predictions,
                    "ocr_result": ocr_result,
                    "object_detection": object_detection,
                    "metrics": {
                        "entropy": float(entropy),
                        "brightness": brightness,
                        "contrast": contrast,
                        "skin_ratio": float(skin_ratio),
                    },
                }

        except IOError as io_exc:
            logger.error(f"Failed to open or read image: {image_path} - {io_exc}")
            return self._error_response(f"Image read error: {io_exc}")
        except Exception as exc:
            logger.exception("画像解析に失敗しました: %s", image_path)
            return self._error_response(str(exc))

    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """エラーレスポンスを生成"""
        return {
            "is_valid": False,
            "is_high_resolution": False,
            "is_potentially_nsfw": False,
            "top_predictions": [],
            "error": error_message,
            "metrics": {}
        }

    # ------------------------------------------------------------------
    # 内部処理
    # ------------------------------------------------------------------
    def _ensure_deep_model(self) -> bool:
        if self._deep_model_loaded:
            return True
        if not self._deep_model_requested:
            return False
        try:
            if self.model_type == 'efficientnet' and _EFFICIENTNET_AVAILABLE:
                self._classification_model = EfficientNetB0(weights="imagenet")
            else:
                self._classification_model = MobileNetV2(weights="imagenet")
            self._deep_model_loaded = True
            return True
        except Exception as exc:  # pragma: no cover - 重量依存の失敗を記録
            logger.warning("深層モデルの初期化に失敗しました: %s", exc)
            self._deep_model_requested = False
            return False

    def _predict_with_deep_model(self, image: Image.Image) -> List[Dict[str, Any]]:
        if not self._classification_model:
            return []

        resized = image.resize((self.model_input_size, self.model_input_size))
        if img_to_array is not None:
            array = img_to_array(resized)
        else:  # TensorFlowが存在しない環境向けの防御策
            array = np.asarray(resized, dtype=np.float32)

        array = np.expand_dims(array, axis=0)
        if self.model_type == 'efficientnet' and _EFFICIENTNET_AVAILABLE:
            if 'efficientnet_preprocess' in globals() and efficientnet_preprocess:
                array = efficientnet_preprocess(array)
        elif preprocess_input is not None:
            array = preprocess_input(array)

        predictions = self._classification_model.predict(array, verbose=0)
        if decode_predictions is None:
            return []

        decoded = decode_predictions(predictions, top=3)[0]
        return [
            {"label": pred[1], "confidence": float(pred[2])}
            for pred in decoded
        ]

    @staticmethod
    def _contains_nsfw_keyword(predictions: List[Dict[str, Any]]) -> bool:
        nsfw_keywords = {"nude", "lingerie", "bikini", "underwear", "brassiere"}
        for item in predictions:
            if any(keyword in item["label"].lower() for keyword in nsfw_keywords):
                return True
        return False

    @staticmethod
    def _compute_entropy(image: Image.Image) -> float:
        grayscale = image.convert("L")
        return float(measure.shannon_entropy(np.asarray(grayscale)))

    @staticmethod
    def _estimate_skin_ratio(np_image: np.ndarray) -> float:
        if np_image.size == 0:
            return 0.0
        normalized = np.clip(np_image / 255.0, 0.0, 1.0)
        ycbcr = color.rgb2ycbcr(normalized)
        cb = ycbcr[:, :, 1]
        cr = ycbcr[:, :, 2]

        skin_mask = (cb >= 0.25) & (cb <= 0.38) & (cr >= 0.25) & (cr <= 0.4)
        ratio = skin_mask.sum() / float(skin_mask.size)
        return float(ratio)

    # ------------------------------------------------------------------
    # 高度なモデル機能（advanced_image_ai.pyから統合）
    # ------------------------------------------------------------------
    
    def _load_pretrained_model(self, weights_path: str):
        """事前学習済みモデルの重みをロードする"""
        if not _TENSORFLOW_AVAILABLE:
            return None
            
        try:
            base_model = MobileNetV2(
                weights='imagenet', 
                include_top=False, 
                input_shape=(self.model_input_size, self.model_input_size, 3)
            )
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(self.num_classes, activation='softmax')(x)
            
            model = Model(inputs=base_model.input, outputs=predictions)
            model.load_weights(weights_path)
            
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        except Exception as e:
            logger.error(f'モデルのロード中にエラー: {e}')
            return self._build_advanced_model()
    
    def _build_advanced_model(self):
        """転移学習を用いたニューラルネットワークモデルの構築"""
        if not _TENSORFLOW_AVAILABLE:
            return None
            
        if self.model_type == 'efficientnet' and _EFFICIENTNET_AVAILABLE:
            base_model = EfficientNetB0(
                weights='imagenet', 
                include_top=False, 
                input_shape=(self.model_input_size, self.model_input_size, 3)
            )
        else:
            base_model = MobileNetV2(
                weights='imagenet', 
                include_top=False, 
                input_shape=(self.model_input_size, self.model_input_size, 3)
            )
        
        base_model.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        output = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_training_data(self, data_dir: str, validation_split: float = 0.2):
        """画像データの前処理と分割"""
        if not _TENSORFLOW_AVAILABLE or not ImageDataGenerator:
            logger.error("TensorFlowが利用できないため、データ準備ができません")
            return None, None
            
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=validation_split
        )
        
        train_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(self.model_input_size, self.model_input_size),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(self.model_input_size, self.model_input_size),
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )
        
        return train_generator, validation_generator
    
    def train_model(self, data_dir: str, epochs: int = 50, early_stopping_patience: int = 10):
        """モデルのトレーニング"""
        if not self._deep_model_loaded or not self._classification_model:
            logger.error("深層学習モデルが初期化されていません")
            return None
            
        train_generator, validation_generator = self.prepare_training_data(data_dir)
        if train_generator is None:
            return None
            
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=early_stopping_patience,
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            self.model_save_path, 
            monitor='val_accuracy', 
            save_best_only=True
        )
        
        history = self._classification_model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[early_stopping, model_checkpoint]
        )
        
        return history.history
    
    def predict_single_image(self, image_path: str, threshold: float = 0.5) -> Dict[str, Any]:
        """単一画像の高度な予測"""
        if not self._deep_model_loaded or not self._classification_model:
            logger.warning("深層学習モデルが利用できません。基本分析のみ実行します。")
            return self.analyze_image(image_path)
            
        try:
            if not load_img:
                return self.analyze_image(image_path)
                
            img = load_img(image_path, target_size=(self.model_input_size, self.model_input_size))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            predictions = self._classification_model.predict(img_array)[0]
            
            top_3_indices = predictions.argsort()[-3:][::-1]
            top_predictions = [
                {
                    'class_index': int(idx),
                    'confidence': float(predictions[idx]),
                    'is_confident': float(predictions[idx]) >= threshold
                } for idx in top_3_indices
            ]
            
            with Image.open(image_path) as img:
                width, height = img.size
                file_size = os.path.getsize(image_path)
            
            result = {
                'predictions': top_predictions,
                'image_details': {
                    'path': image_path,
                    'width': width,
                    'height': height,
                    'file_size': file_size,
                    'aspect_ratio': width / height
                },
                'metadata': {
                    'model_classes': self.num_classes,
                    'confidence_threshold': threshold
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f'高度な画像予測中にエラー発生: {e}')
            return self.analyze_image(image_path)
    
    def save_model(self, path: Optional[str] = None):
        """モデルを保存"""
        if not self._deep_model_loaded or not self._classification_model:
            logger.error("保存するモデルがありません")
            return False
            
        save_path = path or self.model_save_path
        try:
            self._classification_model.save(save_path)
            logger.info(f"モデルを保存しました: {save_path}")
            return True
        except Exception as e:
            logger.error(f"モデル保存中にエラー: {e}")
            return False

    def extract_text_from_image(self, image_path: str, lang: str = 'eng') -> Dict[str, Any]:
        """画像からテキストを抽出するOCR機能"""
        if not _OCR_AVAILABLE:
            logger.warning("OCRが利用できないため、テキスト抽出をスキップします")
            return {"extracted_text": "", "confidence": 0.0, "text_boxes": []}

def classify_tumblr_images(
    image_paths: List[str], classifier: Optional[ImageClassifier] = None
) -> Dict[str, Dict[str, Any]]:
    if classifier is None:
        classifier = ImageClassifier()

    results: Dict[str, Dict[str, Any]] = {}
    for image_path in image_paths:
        results[image_path] = classifier.analyze_image(image_path)

    return results


def extract_text_from_image(image_path: str, lang: str = 'eng') -> Dict[str, Any]:
    """画像からテキストを抽出するOCR機能（スタンドアローン関数）"""
    if not _OCR_AVAILABLE:
        logger.warning("OCRが利用できないため、テキスト抽出をスキップします")
        return {"extracted_text": "", "confidence": 0.0, "text_boxes": []}

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")

            # pytesseractでテキスト抽出
            text = pytesseract.image_to_string(img, lang=lang)

            # 詳細なデータ取得
            data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)

            # テキストボックスの情報
            text_boxes = []
            for i in range(len(data['text'])):
                if data['text'][i].strip():
                    text_boxes.append({
                        'text': data['text'][i],
                        'confidence': data['conf'][i],
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i]
                    })

            avg_confidence = sum(tb['confidence'] for tb in text_boxes) / len(text_boxes) if text_boxes else 0

            return {
                "extracted_text": text.strip(),
                "confidence": float(avg_confidence),
                "text_boxes": text_boxes,
                "text_length": len(text.strip()),
                "has_text": bool(text.strip())
            }

    except Exception as e:
        logger.error(f"OCR処理中にエラー: {e}")
        return {"extracted_text": "", "confidence": 0.0, "text_boxes": [], "error": str(e)}


def detect_objects_yolo(image_path: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """YOLOを使った物体検出（スタンドアローン関数）"""
    if not _YOLO_AVAILABLE:
        logger.warning("YOLOが利用できないため、物体検出をスキップします")
        return {"detected_objects": [], "object_count": 0}

    try:
        # YOLOモデルをロード（キャッシュ）
        if not hasattr(detect_objects_yolo, '_yolo_model'):
            try:
                detect_objects_yolo._yolo_model = YOLO('yolov9c.pt')  # YOLOv9
            except:
                try:
                    detect_objects_yolo._yolo_model = YOLO('yolov8n.pt')  # YOLOv8 fallback
                except:
                    detect_objects_yolo._yolo_model = YOLO('yolov5s.pt')  # YOLOv5 fallback

        results = detect_objects_yolo._yolo_model(image_path, conf=confidence_threshold)

        detected_objects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detected_objects.append({
                    'class': result.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist(),
                    'class_id': int(box.cls)
                })

        return {
            "detected_objects": detected_objects,
            "object_count": len(detected_objects),
            "model_used": getattr(detect_objects_yolo._yolo_model, 'model_name', 'YOLO')
        }

    except Exception as e:
        logger.error(f"YOLO物体検出中にエラー: {e}")
        return {"detected_objects": [], "object_count": 0, "error": str(e)}


def compute_image_similarity(image_path1: str, image_path2: str) -> float:
    """2つの画像の類似度を計算（スタンドアローン関数）"""
    try:
        with Image.open(image_path1) as img1, Image.open(image_path2) as img2:
            # 画像を同じサイズにリサイズ
            size = (224, 224)
            img1 = img1.resize(size).convert('RGB')
            img2 = img2.resize(size).convert('RGB')

            # ヒストグラム比較
            hist1 = img1.histogram()
            hist2 = img2.histogram()

            # ヒストグラム相関係数
            correlation = 0
            if sum(hist1) > 0 and sum(hist2) > 0:
                correlation = sum((a * b) for a, b in zip(hist1, hist2))
                correlation /= (sum(hist1) * sum(hist2)) ** 0.5

            # SSIM計算（より正確な類似度）
            try:
                from skimage.metrics import structural_similarity as ssim
                np_img1 = np.array(img1)
                np_img2 = np.array(img2)

                ssim_score = ssim(np_img1, np_img2, multichannel=True)
                return float((correlation + ssim_score) / 2)
            except ImportError:
                return float(correlation)

    except Exception as e:
        logger.error(f"画像類似度計算中にエラー: {e}")
        return 0.0


def find_similar_images(target_image: str, image_directory: str, threshold: float = 0.8) -> List[Dict[str, Any]]:
    """指定画像と類似した画像を検索（スタンドアローン関数）"""
    try:
        similar_images = []
        target_path = Path(target_image)

        for image_path in Path(image_directory).glob("*"):
            if image_path.is_file() and image_path != target_path:
                try:
                    similarity = compute_image_similarity(target_image, str(image_path))
                    if similarity >= threshold:
                        similar_images.append({
                            'path': str(image_path),
                            'similarity': similarity,
                            'filename': image_path.name
                        })
                except Exception as e:
                    logger.debug(f"画像比較をスキップ: {image_path} - {e}")

        # 類似度順にソート
        similar_images.sort(key=lambda x: x['similarity'], reverse=True)

        return similar_images

    except Exception as e:
        logger.error(f"類似画像検索中にエラー: {e}")
        return []


def analyze_image_comprehensive(image_path: str, include_ocr: bool = True, include_objects: bool = True) -> Dict[str, Any]:
    """包括的な画像分析（OCR、物体検出、分類を含む）"""
    classifier = ImageClassifier()
    result = classifier.analyze_image(image_path)

    if include_ocr and _OCR_AVAILABLE:
        try:
            ocr_result = extract_text_from_image(image_path)
            result["ocr_result"] = ocr_result
        except Exception as e:
            logger.debug(f"OCR分析をスキップ: {e}")

    if include_objects and _YOLO_AVAILABLE:
        try:
            object_result = detect_objects_yolo(image_path)
            result["object_detection"] = object_result
        except Exception as e:
            logger.debug(f"物体検出をスキップ: {e}")

    # 画像の特徴量を追加
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            np_image = np.asarray(img, dtype=np.float32)

            # 追加の特徴量
            result["features"] = {
                "dominant_colors": _extract_dominant_colors(np_image),
                "texture_complexity": _compute_texture_complexity(np_image),
                "edge_density": _compute_edge_density(np_image),
                "color_histogram": _compute_color_histogram(np_image)
            }
    except Exception as e:
        logger.debug(f"特徴量抽出をスキップ: {e}")

    return result


def _extract_dominant_colors(np_image: np.ndarray, num_colors: int = 5) -> List[Dict[str, Any]]:
    """画像の主な色を抽出"""
    try:
        from sklearn.cluster import KMeans

        # 画像を2D配列に変換
        pixels = np_image.reshape(-1, 3)

        # K-meansクラスタリング
        kmeans = KMeans(n_clusters=num_colors, random_state=42)
        kmeans.fit(pixels)

        colors = []
        for center in kmeans.cluster_centers_:
            colors.append({
                'r': int(center[0]),
                'g': int(center[1]),
                'b': int(center[2]),
                'hex': f'#{int(center[0]):02x}{int(center[1]):02x}{int(center[2]):02x}'
            })

        return colors
    except ImportError:
        return []
    except Exception as e:
        logger.debug(f"色抽出をスキップ: {e}")
        return []


def _compute_texture_complexity(np_image: np.ndarray) -> float:
    """画像のテクスチャ複雑さを計算"""
    try:
        from skimage.feature import graycomatrix, graycoprops

        gray = color.rgb2gray(np_image)
        glcm = graycomatrix((gray * 255).astype(np.uint8), [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])

        contrast = graycoprops(glcm, 'contrast')[0, 0]
        return float(contrast / 1000.0)  # 正規化
    except Exception as e:
        logger.debug(f"テクスチャ計算をスキップ: {e}")
        return 0.0


def _compute_edge_density(np_image: np.ndarray) -> float:
    """画像のエッジ密度を計算"""
    try:
        from skimage import filters

        gray = color.rgb2gray(np_image)
        edges = filters.sobel(gray)
        edge_density = np.sum(edges > 0.1) / edges.size
        return float(edge_density)
    except Exception as e:
        logger.debug(f"エッジ密度計算をスキップ: {e}")
        return 0.0


def _compute_color_histogram(np_image: np.ndarray) -> Dict[str, List[float]]:
    """色ヒストグラムを計算"""
    try:
        hist_r = np.histogram(np_image[:,:,0], bins=32, range=(0,255))[0]
        hist_g = np.histogram(np_image[:,:,1], bins=32, range=(0,255))[0]
        hist_b = np.histogram(np_image[:,:,2], bins=32, range=(0,255))[0]

        return {
            'red': hist_r.tolist(),
            'green': hist_g.tolist(),
            'blue': hist_b.tolist()
        }
    except Exception as e:
        logger.debug(f"色ヒストグラム計算をスキップ: {e}")
        return {'red': [], 'green': [], 'blue': []}


if __name__ == "__main__":  # pragma: no cover - 手動確認用
    sample_images = [
        "path/to/image1.jpg",
        "path/to/image2.png",
    ]

    for image_path, analysis in classify_tumblr_images(sample_images).items():
        print(f"Image: {image_path}")
        print(json.dumps(analysis, ensure_ascii=False, indent=2))
