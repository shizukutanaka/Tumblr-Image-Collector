import os
import unittest
import numpy as np
import tensorflow as tf
from advanced_image_ai import Konohana

class TestKonohanaAI(unittest.TestCase):
    def setUp(self):
        """
        各テストの前に実行されるセットアップメソッド
        """
        self.image_size = (224, 224)
        self.num_classes = 10
        self.learning_rate = 1e-4

    def test_model_initialization(self):
        """
        モデルの初期化テスト
        """
        konohana = Konohana(
            image_size=self.image_size, 
            num_classes=self.num_classes, 
            learning_rate=self.learning_rate
        )
        
        # モデルが正しく初期化されていることを確認
        self.assertIsNotNone(konohana.model)
        self.assertEqual(konohana.num_classes, self.num_classes)
        self.assertEqual(konohana.image_size, self.image_size)

    def test_model_compilation(self):
        """
        モデルのコンパイル設定テスト
        """
        konohana = Konohana(
            image_size=self.image_size, 
            num_classes=self.num_classes, 
            learning_rate=self.learning_rate
        )
        
        # モデルのコンパイル設定を確認
        self.assertEqual(
            konohana.model.optimizer.__class__.__name__, 
            'Adam'
        )
        self.assertEqual(
            konohana.model.loss, 
            'categorical_crossentropy'
        )

    def test_model_input_shape(self):
        """
        モデルの入力形状テスト
        """
        konohana = Konohana(
            image_size=self.image_size, 
            num_classes=self.num_classes, 
            learning_rate=self.learning_rate
        )
        
        # 入力形状が正しいことを確認
        expected_input_shape = (None, *self.image_size, 3)
        self.assertEqual(
            konohana.model.input_shape, 
            expected_input_shape
        )

    def test_model_output_shape(self):
        """
        モデルの出力形状テスト
        """
        konohana = Konohana(
            image_size=self.image_size, 
            num_classes=self.num_classes, 
            learning_rate=self.learning_rate
        )
        
        # 出力形状が正しいことを確認
        test_input = np.random.random((1, *self.image_size, 3))
        predictions = konohana.model.predict(test_input)
        
        self.assertEqual(predictions.shape, (1, self.num_classes))

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
                input_shape=self.image_size + (3,)
            )
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(1024, activation='relu')(x)
            predictions = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
            
            model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
            model.save_weights(temp_weights_path)

            # 事前学習済み重みを使用してモデルを初期化
            konohana = Konohana(
                image_size=self.image_size, 
                num_classes=self.num_classes, 
                learning_rate=self.learning_rate,
                pretrained_weights=temp_weights_path
            )
            
            # モデルが正しくロードされたことを確認
            self.assertIsNotNone(konohana.model)
        
        finally:
            # テスト用の重みファイルを削除
            if os.path.exists(temp_weights_path):
                os.remove(temp_weights_path)

if __name__ == '__main__':
    unittest.main()
