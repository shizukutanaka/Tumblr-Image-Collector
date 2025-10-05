import os
import unittest
import numpy as np
try:
    import tensorflow as tf
    _TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None
    _TENSORFLOW_AVAILABLE = False
from image_classifier import ImageClassifier

class TestImageClassifier(unittest.TestCase):
    def setUp(self):
        """
        各テストの前に実行されるセットアップメソッド
        """
        self.model_input_size = 224
        self.num_classes = 10
        self.learning_rate = 1e-4
        self.enable_deep_model = _TENSORFLOW_AVAILABLE

    def test_model_initialization(self):
        """
        モデルの初期化テスト
        """
        classifier = ImageClassifier(
            model_input_size=self.model_input_size,
            num_classes=self.num_classes,
            learning_rate=self.learning_rate,
            enable_deep_model=self.enable_deep_model
        )

        # モデルが正しく初期化されていることを確認
        if _TENSORFLOW_AVAILABLE:
            self.assertIsNotNone(classifier._classification_model)
        self.assertEqual(classifier.num_classes, self.num_classes)
        self.assertEqual(classifier.model_input_size, self.model_input_size)

    @unittest.skipUnless(_TENSORFLOW_AVAILABLE, "TensorFlow not available")
    def test_model_compilation(self):
        """
        モデルのコンパイル設定テスト
        """
        classifier = ImageClassifier(
            model_input_size=self.model_input_size,
            num_classes=self.num_classes,
            learning_rate=self.learning_rate,
            enable_deep_model=True
        )

        # モデルのコンパイル設定を確認
        if classifier._classification_model:
            self.assertEqual(
                classifier._classification_model.optimizer.__class__.__name__,
                'Adam'
            )
            self.assertEqual(
                classifier._classification_model.loss,
                'categorical_crossentropy'
            )

    @unittest.skipUnless(_TENSORFLOW_AVAILABLE, "TensorFlow not available")
    def test_model_input_shape(self):
        """
        モデルの入力形状テスト
        """
        classifier = ImageClassifier(
            model_input_size=self.model_input_size,
            num_classes=self.num_classes,
            learning_rate=self.learning_rate,
            enable_deep_model=True
        )

        # 入力形状が正しいことを確認
        if classifier._classification_model:
            expected_input_shape = (None, self.model_input_size, self.model_input_size, 3)
            self.assertEqual(
                classifier._classification_model.input_shape,
                expected_input_shape
            )

    @unittest.skipUnless(_TENSORFLOW_AVAILABLE, "TensorFlow not available")
    def test_model_output_shape(self):
        """
        モデルの出力形状テスト
        """
        classifier = ImageClassifier(
            model_input_size=self.model_input_size,
            num_classes=self.num_classes,
            learning_rate=self.learning_rate,
            enable_deep_model=True
        )

        # 出力形状が正しいことを確認
        if classifier._classification_model:
            test_input = np.random.random((1, self.model_input_size, self.model_input_size, 3))
            predictions = classifier._classification_model.predict(test_input, verbose=0)

            self.assertEqual(predictions.shape, (1, self.num_classes))

    @unittest.skipUnless(_TENSORFLOW_AVAILABLE, "TensorFlow not available")
    def test_pretrained_weights_loading(self):
        """
        事前学習済み重みのロードテスト
        """
        # テスト用の一時的な重みファイルを作成
        temp_weights_path = 'temp_test_weights.h5'

        try:
            # ダミーモデルを作成して重みを保存
            base_model = tf.keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(self.model_input_size, self.model_input_size, 3)
            )
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(1024, activation='relu')(x)
            predictions = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
            
            model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
            model.save_weights(temp_weights_path)

            # 事前学習済み重みを使用してモデルを初期化
            classifier = ImageClassifier(
                model_input_size=self.model_input_size,
                num_classes=self.num_classes,
                learning_rate=self.learning_rate,
                enable_deep_model=True,
                pretrained_weights=temp_weights_path
            )

            # モデルが正しくロードされたことを確認
            self.assertIsNotNone(classifier._classification_model)
        
        finally:
            # テスト用の重みファイルを削除
            if os.path.exists(temp_weights_path):
                os.remove(temp_weights_path)

if __name__ == '__main__':
    unittest.main()
