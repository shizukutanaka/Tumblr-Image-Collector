import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import shutil
import logging
from PIL import Image

class Konohana:
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='image_ai.log'
    )
    def __init__(self, 
                 image_size=(224, 224), 
                 num_classes=10, 
                 learning_rate=1e-4,
                 model_save_path='best_image_model.h5'):
        """
        高度な画像識別・学習AIクラス
        
        Args:
            image_size (tuple): 画像のリサイズサイズ
            num_classes (int): 分類するクラス数
            learning_rate (float): 学習率
            model_save_path (str): モデル保存パス
        """
        self.image_size = image_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model_save_path = model_save_path
        self.model = self._build_model()
        logging.info(f'Konohana AI初期化: クラス数={num_classes}, 画像サイズ={image_size}')
    
    def _build_model(self):
        """
        転移学習を用いたニューラルネットワークモデルの構築
        
        Returns:
            tf.keras.Model: 学習可能なモデル
        """
        # 事前学習済みMobileNetV2モデルの読み込み
        base_model = MobileNetV2(
            weights='imagenet', 
            include_top=False, 
            input_shape=(*self.image_size, 3)
        )
        
        # ベースモデルの重みを凍結
        base_model.trainable = False
        
        # カスタム分類層の追加
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        output = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        
        # モデルのコンパイル
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_data(self, data_dir, validation_split=0.2):
        """
        画像データの前処理と分割
        
        Args:
            data_dir (str): 画像が保存されているディレクトリ
            validation_split (float): 検証用データの割合
        
        Returns:
            tuple: トレーニングデータとラベル、検証データとラベル
        """
        # データ拡張
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
        
        # トレーニングデータジェネレータ
        train_generator = datagen.flow_from_directory(
            data_dir,
            target_size=self.image_size,
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        
        # 検証データジェネレータ
        validation_generator = datagen.flow_from_directory(
            data_dir,
            target_size=self.image_size,
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )
        
        return train_generator, validation_generator
    
    def train(self, data_dir, epochs=50, early_stopping_patience=10):
        """
        モデルのトレーニング
        
        Args:
            data_dir (str): 学習データが保存されているディレクトリ
            epochs (int): 学習エポック数
            early_stopping_patience (int): 早期停止のパターン数
        
        Returns:
            dict: 学習履歴
        """
        # データの準備
        train_generator, validation_generator = self.prepare_data(data_dir)
        
        # コールバックの設定
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=early_stopping_patience,
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            'best_model.h5', 
            monitor='val_accuracy', 
            save_best_only=True
        )
        
        # モデルの学習
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[early_stopping, model_checkpoint]
        )
        
        return history.history
    
    def evaluate(self, data_dir):
        """
        モデルの評価と詳細な分析
        
        Args:
            data_dir (str): 評価データが保存されているディレクトリ
        
        Returns:
            dict: 評価結果と分析情報
        """
        # データの準備
        _, validation_generator = self.prepare_data(data_dir)
        
        # モデル評価
        test_loss, test_accuracy = self.model.evaluate(validation_generator)
        
        # 予測の生成
        predictions = self.model.predict(validation_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = validation_generator.classes
        
        # 分類レポート
        class_report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=validation_generator.class_indices.keys()
        )
        
        # 混同行列
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # 可視化
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混同行列')
        plt.xlabel('予測ラベル')
        plt.ylabel('真のラベル')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        return {
            'loss': test_loss,
            'accuracy': test_accuracy,
            'classification_report': class_report,
            'confusion_matrix': cm
        }
    
    def predict(self, image_path, threshold=0.5):
        """
        単一画像の予測
        
        Args:
            image_path (str): 予測する画像のパス
            threshold (float): 信頼度の閾値
        
        Returns:
            dict: 予測結果と詯細情報
        """
        try:
            logging.info(f'画像予測開始: {image_path}')
            
            # 画像の前処理
            img = load_img(image_path, target_size=self.image_size)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            # 予測の実行
            predictions = self.model.predict(img_array)[0]
            class_names = list(self.model.get_config()['layers'][-1]['config']['units'])
            
            # 上位3つの予測結果
            top_3_indices = predictions.argsort()[-3:][::-1]
            top_predictions = [
                {
                    'class': class_names[idx],
                    'confidence': float(predictions[idx]),
                    'is_confident': float(predictions[idx]) >= threshold
                } for idx in top_3_indices
            ]
            
            # 画像の詯細情報を収集
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
                    'model_classes': class_names,
                    'confidence_threshold': threshold
                }
            }
            
            logging.info(f'画像予測完了: {image_path}')
            return result
        
        except Exception as e:
            logging.error(f'画像予測中にエラー発生: {e}')
            return {
                'error': str(e),
                'image_path': image_path
            }
        

def cli():
    """
    コマンドラインインターフェース
    """
    parser = argparse.ArgumentParser(
        description='高度な画像分類AIツール'
    )
    
    # サブコマンドの追加
    subparsers = parser.add_subparsers(dest='command', help='サブコマンド')
    
    # 学習コマンド
    train_parser = subparsers.add_parser('train', help='画像データセットから学習')
    train_parser.add_argument('directory', type=str, help='学習画像が保存されているディレクトリ')
    train_parser.add_argument('--model', type=str, default='image_classifier.h5', help='保存するモデルファイル名')
    train_parser.add_argument('--classes', type=int, default=3, help='分類するクラス数')
    
    # 予測コマンド
    predict_parser = subparsers.add_parser('predict', help='画像を分類')
    predict_parser.add_argument('image', type=str, help='予測する画像ファイル')
    predict_parser.add_argument('--model', type=str, default='image_classifier.h5', help='使用するモデルファイル')
    predict_parser.add_argument('--threshold', type=float, default=0.5, help='信頼度の閾値')
    predict_parser.add_argument('--output', type=str, default='predictions.json', help='出力ファイル名')
    
    # ディレクトリ分類コマンド
    classify_dir_parser = subparsers.add_parser('classify-dir', help='ディレクトリ内の画像を分類')
    classify_dir_parser.add_argument('directory', type=str, help='分類する画像が保存されているディレクトリ')
    classify_dir_parser.add_argument('--model', type=str, default='image_classifier.h5', help='使用するモデルファイル')
    classify_dir_parser.add_argument('--threshold', type=float, default=0.5, help='信頼度の閾値')
    classify_dir_parser.add_argument('--output', type=str, default='directory_predictions.json', help='出力ファイル名')
    
    # 引数の解析
    args = parser.parse_args()
    
    # モデルの初期化
    image_ai = Konohana(num_classes=args.classes if hasattr(args, 'classes') else 3)
    
    try:
        if args.command == 'train':
            # 画像ファイルとラベルの取得
            image_files, labels = image_ai.generate_labels_from_directory(args.directory)
            
            # モデルの学習
            training_result = image_ai.train_from_files(image_files, labels)
            
            # モデルの保存
            image_ai.model.save(args.model)
            print(f"モデルを {args.model} に保存しました")
        
        elif args.command == 'predict':
            # 単一画像の予測
            prediction_result = image_ai.predict(args.image, threshold=args.threshold)
            
            # 結果の出力
            image_ai.export_predictions(prediction_result, args.output)
            print(f"予測結果を {args.output} にエクスポートしました")
        
        elif args.command == 'classify-dir':
            # ディレクトリ内の画像を分類
            image_files = image_ai.list_images_in_directory(args.directory)
            
            # 各画像の予測
            all_predictions = {}
            for image_path in image_files:
                prediction = image_ai.predict(image_path, threshold=args.threshold)
                all_predictions[image_path] = prediction
            
            # 結果のエクスポート
            image_ai.export_predictions(all_predictions, args.output)
            print(f"ディレクトリ分類結果を {args.output} にエクスポートしました")
    
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        logging.error(f"エラー発生: {e}")

def main():
    """デモンストレーション用のメイン関数"""
    try:
        # AIの初期化
        image_ai = Konohana(num_classes=3, model_save_path='tumblr_image_classifier.h5')
        
        # Tumblr画像コレクター用の画像データセット
        image_files = [
            './tumblr_images/cat1.jpg',
            './tumblr_images/cat2.jpg',
            './tumblr_images/dog1.jpg',
            './tumblr_images/bird1.jpg'
        ]
        
        # ラベル
        labels = ['cat', 'cat', 'dog', 'bird']
        
        # 指定されたファイルから直接学習
        training_result = image_ai.train_from_files(image_files, labels)
        
        # 単一画像の予測デモ
        test_image_path = './test_image.jpg'
        prediction_result = image_ai.predict(test_image_path, threshold=0.7)
        
        # 結果の詳細出力
        print("学習結果:", training_result)
        print("画像予測:", prediction_result)
        
        # 学習したモデルを保存
        image_ai.model.save('tumblr_image_classifier.h5')
        print("モデルを保存しました: tumblr_image_classifier.h5")
    
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        logging.error(f"エラー発生: {e}")
        
if __name__ == '__main__':
    cli()
