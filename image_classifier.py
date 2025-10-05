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
    from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    _TENSORFLOW_AVAILABLE = True
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


class ImageClassifier:
    """軽量ヒューリスティックと任意の深層学習モデルを両立する分類器。"""

    def __init__(
        self,
        nsfw_threshold: float = 0.25,
        min_resolution: Tuple[int, int] = (300, 300),
        enable_deep_model: bool = False,
        model_input_size: int = 224,
        num_classes: int = 10,
        learning_rate: float = 1e-4,
        model_save_path: str = 'image_classifier.h5',
        pretrained_weights: Optional[str] = None,
    ) -> None:
        self.nsfw_threshold = nsfw_threshold
        self.min_resolution = min_resolution
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
                    logger.info(f'新規モデルを構築: クラス数={num_classes}, 画像サイズ={model_input_size}')
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
        if preprocess_input is not None:
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


def classify_tumblr_images(
    image_paths: List[str], classifier: Optional[ImageClassifier] = None
) -> Dict[str, Dict[str, Any]]:
    if classifier is None:
        classifier = ImageClassifier()

    results: Dict[str, Dict[str, Any]] = {}
    for image_path in image_paths:
        results[image_path] = classifier.analyze_image(image_path)

    return results


if __name__ == "__main__":  # pragma: no cover - 手動確認用
    sample_images = [
        "path/to/image1.jpg",
        "path/to/image2.png",
    ]

    for image_path, analysis in classify_tumblr_images(sample_images).items():
        print(f"Image: {image_path}")
        print(json.dumps(analysis, ensure_ascii=False, indent=2))
