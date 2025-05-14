import json
import os
import sys
import getpass
from pathlib import Path

class ConfigWizard:
    def __init__(self, config_path='config.json'):
        self.config_path = Path(config_path)
        self.config = self._load_existing_config()

    def _load_existing_config(self):
        """既存の設定ファイルを読み込む"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _validate_tumblr_credentials(self, consumer_key, consumer_secret):
        """Tumblr APIの認証情報を簡易的に検証"""
        # TODO: 実際の認証ロジックを実装
        return len(consumer_key) > 10 and len(consumer_secret) > 10

    def _get_proxy_config(self):
        """プロキシ設定を対話的に取得"""
        print("\n🌐 プロキシ設定 (オプション)")
        use_proxy = input("プロキシを使用しますか？ (y/N): ").lower() == 'y'
        
        if not use_proxy:
            return None

        proxy_types = ['http', 'https', 'socks4', 'socks5']
        while True:
            proxy_type = input(f"プロキシタイプを選択 {proxy_types}: ").lower()
            if proxy_type in proxy_types:
                break
            print("無効なプロキシタイプです。")

        host = input("プロキシホスト (例: 127.0.0.1): ")
        port = input("プロキシポート (例: 8080): ")
        
        use_auth = input("認証が必要ですか？ (y/N): ").lower() == 'y'
        username = password = None
        
        if use_auth:
            username = input("ユーザー名: ")
            password = getpass.getpass("パスワード: ")

        return {
            'type': proxy_type,
            'host': host,
            'port': port,
            'username': username,
            'password': password
        }

    def run(self):
        """設定ウィザードを実行"""
        print("🚀 Tumblr Image Collector 設定ウィザード")
        
        # Tumblr API認証情報
        print("\n🔑 Tumblr API 認証情報")
        while True:
            consumer_key = input("Consumer Key: ")
            consumer_secret = input("Consumer Secret: ")
            
            if self._validate_tumblr_credentials(consumer_key, consumer_secret):
                self.config['consumer_key'] = consumer_key
                self.config['consumer_secret'] = consumer_secret
                break
            print("無効な認証情報です。再入力してください。")

        # 出力フォルダ
        print("\n📂 出力設定")
        output_folder = input("画像の保存先フォルダ (デフォルト: tumblr_images): ") or "tumblr_images"
        self.config['output_folder_name'] = output_folder

        # プロキシ設定
        proxy_config = self._get_proxy_config()
        if proxy_config:
            self.config['proxy'] = proxy_config

        # 高度なフィルタリング設定
        print("\n🖼️ 画像フィルタリング設定")
        max_file_size = input("最大ファイルサイズ (MB, デフォルト: 10): ") or 10
        self.config['filters'] = {
            'max_file_size_mb': int(max_file_size)
        }

        # 設定を保存
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

        print("\n✅ 設定が正常に保存されました！")
        return self.config

def main():
    wizard = ConfigWizard()
    config = wizard.run()
    print("\n設定の詳細:")
    print(json.dumps(config, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
