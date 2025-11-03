#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
クラウドストレージ統合モジュール
Google DriveやDropboxなどのクラウドストレージとの連携機能を提供
"""

import os
import sys
import logging
import json
from typing import Optional, Dict, List, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class CloudStorageProvider(ABC):
    """クラウドストレージプロバイダーの基底クラス"""

    @abstractmethod
    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """ファイルをクラウドストレージにアップロード"""
        pass

    @abstractmethod
    def download_file(self, remote_path: str, local_path: str) -> bool:
        """クラウドストレージからファイルをダウンロード"""
        pass

    @abstractmethod
    def is_authenticated(self) -> bool:
        """認証状態をチェック"""
        pass

    @abstractmethod
    def get_storage_info(self) -> Dict[str, Any]:
        """ストレージ情報を取得"""
        pass

class GoogleDriveProvider(CloudStorageProvider):
    """Google Drive統合プロバイダー"""

    def __init__(self, credentials_file: str = "credentials.json"):
        """
        Google Driveプロバイダーを初期化

        Args:
            credentials_file (str): Google API認証情報ファイルパス
        """
        self.credentials_file = credentials_file
        self.service = None
        self._authenticated = False
        self._initialize_service()

    def _initialize_service(self):
        """Google Driveサービスを初期化"""
        try:
            # Google APIライブラリが利用可能かチェック
            try:
                from google.oauth2.credentials import Credentials
                from googleapiclient.discovery import build
                from googleapiclient.http import MediaFileUpload
            except ImportError:
                logger.warning("Google APIライブラリがインストールされていません。pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib を実行してください。")
                return

            # 認証情報の読み込み
            if os.path.exists(self.credentials_file):
                try:
                    from google.oauth2 import service_account
                    credentials = service_account.Credentials.from_service_account_file(
                        self.credentials_file,
                        scopes=['https://www.googleapis.com/auth/drive']
                    )
                    self.service = build('drive', 'v3', credentials=credentials)
                    self._authenticated = True
                    logger.info("Google Drive認証成功")
                except Exception as e:
                    logger.error(f"Google Drive認証エラー: {e}")
                    self._authenticated = False
            else:
                logger.warning(f"Google Drive認証ファイルが見つかりません: {self.credentials_file}")

        except Exception as e:
            logger.error(f"Google Driveサービス初期化エラー: {e}")
            self._authenticated = False

    def is_authenticated(self) -> bool:
        return self._authenticated and self.service is not None

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """ファイルをGoogle Driveにアップロード"""
        if not self.is_authenticated():
            logger.error("Google Drive認証が必要です")
            return False

        if not os.path.exists(local_path):
            logger.error(f"アップロードファイルが見つかりません: {local_path}")
            return False

        try:
            from googleapiclient.http import MediaFileUpload

            file_metadata = {
                'name': os.path.basename(remote_path),
                'parents': []  # 特定のフォルダを指定する場合はフォルダIDを追加
            }

            media = MediaFileUpload(local_path, resumable=True)

            # ファイルのアップロード
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()

            logger.info(f"Google Driveにアップロード成功: {local_path} -> {file.get('id')}")
            return True

        except Exception as e:
            logger.error(f"Google Driveアップロードエラー: {e}")
            return False

    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Google Driveからファイルをダウンロード"""
        if not self.is_authenticated():
            logger.error("Google Drive認証が必要です")
            return False

        try:
            # ファイル名からファイルIDを検索（簡易実装）
            # 実際の実装ではファイルIDを直接使用するか、検索APIを使用
            logger.warning("Google Driveダウンロード機能は開発中です")
            return False

        except Exception as e:
            logger.error(f"Google Driveダウンロードエラー: {e}")
            return False

    def get_storage_info(self) -> Dict[str, Any]:
        """Google Driveのストレージ情報を取得"""
        if not self.is_authenticated():
            return {"error": "認証が必要です"}

        try:
            # アカウント情報を取得
            about = self.service.about().get(fields='storageQuota').execute()

            quota = about.get('storageQuota', {})
            limit = int(quota.get('limit', 0))
            usage = int(quota.get('usage', 0))

            return {
                "provider": "Google Drive",
                "total_space_gb": limit / (1024**3),
                "used_space_gb": usage / (1024**3),
                "free_space_gb": (limit - usage) / (1024**3),
                "usage_percent": (usage / limit) * 100 if limit > 0 else 0
            }

        except Exception as e:
            logger.error(f"Google Drive情報取得エラー: {e}")
            return {"error": str(e)}

class DropboxProvider(CloudStorageProvider):
    """Dropbox統合プロバイダー"""

    def __init__(self, access_token: Optional[str] = None, token_file: str = "dropbox_token.json"):
        """
        Dropboxプロバイダーを初期化

        Args:
            access_token (Optional[str]): Dropboxアクセストークン
            token_file (str): トークン保存ファイルパス
        """
        self.token_file = token_file
        self.access_token = access_token or self._load_token()
        self.client = None
        self._authenticated = False
        self._initialize_client()

    def _load_token(self) -> Optional[str]:
        """トークンファイルからアクセストークンを読み込み"""
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, 'r') as f:
                    data = json.load(f)
                    return data.get('access_token')
            except Exception as e:
                logger.warning(f"トークンファイル読み込みエラー: {e}")

        return None

    def _save_token(self, token: str):
        """アクセストークンをファイルに保存"""
        try:
            with open(self.token_file, 'w') as f:
                json.dump({'access_token': token}, f)
            logger.info("Dropboxトークンを保存しました")
        except Exception as e:
            logger.error(f"トークン保存エラー: {e}")

    def _initialize_client(self):
        """Dropboxクライアントを初期化"""
        if not self.access_token:
            logger.warning("Dropboxアクセストークンが設定されていません")
            return

        try:
            # Dropbox SDKが利用可能かチェック
            try:
                import dropbox
            except ImportError:
                logger.warning("Dropboxライブラリがインストールされていません。pip install dropbox を実行してください。")
                return

            self.client = dropbox.Dropbox(self.access_token)
            # 認証状態を確認
            try:
                self.client.users_get_current_account()
                self._authenticated = True
                logger.info("Dropbox認証成功")
            except Exception as e:
                logger.error(f"Dropbox認証エラー: {e}")
                self._authenticated = False

        except Exception as e:
            logger.error(f"Dropboxクライアント初期化エラー: {e}")
            self._authenticated = False

    def is_authenticated(self) -> bool:
        return self._authenticated and self.client is not None

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """ファイルをDropboxにアップロード"""
        if not self.is_authenticated():
            logger.error("Dropbox認証が必要です")
            return False

        if not os.path.exists(local_path):
            logger.error(f"アップロードファイルが見つかりません: {local_path}")
            return False

        try:
            with open(local_path, 'rb') as f:
                # Dropboxにアップロード
                self.client.files_upload(
                    f.read(),
                    remote_path,
                    mode=dropbox.files.WriteMode('overwrite')
                )

            logger.info(f"Dropboxにアップロード成功: {local_path} -> {remote_path}")
            return True

        except Exception as e:
            logger.error(f"Dropboxアップロードエラー: {e}")
            return False

    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Dropboxからファイルをダウンロード"""
        if not self.is_authenticated():
            logger.error("Dropbox認証が必要です")
            return False

        try:
            # ファイルのメタデータを取得
            metadata = self.client.files_download_to_file(local_path, remote_path)

            logger.info(f"Dropboxからダウンロード成功: {remote_path} -> {local_path}")
            return True

        except Exception as e:
            logger.error(f"Dropboxダウンロードエラー: {e}")
            return False

    def get_storage_info(self) -> Dict[str, Any]:
        """Dropboxのストレージ情報を取得"""
        if not self.is_authenticated():
            return {"error": "認証が必要です"}

        try:
            # ユーザー情報を取得
            account = self.client.users_get_current_account()

            # ストレージ情報はDropbox APIで直接取得できないため、簡易情報を返す
            return {
                "provider": "Dropbox",
                "account_name": account.name.display_name,
                "account_email": account.email,
                "note": "Dropboxのストレージ情報は制限により取得できません"
            }

        except Exception as e:
            logger.error(f"Dropbox情報取得エラー: {e}")
            return {"error": str(e)}

class CloudStorageManager:
    """クラウドストレージマネージャークラス"""

    def __init__(self):
        """クラウドストレージプロバイダーを初期化"""
        self.providers = {
            'google_drive': GoogleDriveProvider(),
            'dropbox': DropboxProvider()
        }

    def get_provider(self, provider_name: str) -> Optional[CloudStorageProvider]:
        """指定されたプロバイダーを取得"""
        return self.providers.get(provider_name.lower())

    def upload_to_cloud(self, provider_name: str, local_path: str,
                       remote_path: str) -> bool:
        """
        指定されたクラウドストレージにファイルをアップロード

        Args:
            provider_name (str): プロバイダー名（google_drive, dropbox）
            local_path (str): ローカルファイルパス
            remote_path (str): リモートパス

        Returns:
            bool: アップロード成功フラグ
        """
        provider = self.get_provider(provider_name)
        if not provider:
            logger.error(f"不明なプロバイダー: {provider_name}")
            return False

        if not provider.is_authenticated():
            logger.error(f"{provider_name}認証が必要です")
            return False

        return provider.upload_file(local_path, remote_path)

    def download_from_cloud(self, provider_name: str, remote_path: str,
                           local_path: str) -> bool:
        """
        指定されたクラウドストレージからファイルをダウンロード

        Args:
            provider_name (str): プロバイダー名
            remote_path (str): リモートファイルパス
            local_path (str): ローカル保存先パス

        Returns:
            bool: ダウンロード成功フラグ
        """
        provider = self.get_provider(provider_name)
        if not provider:
            logger.error(f"不明なプロバイダー: {provider_name}")
            return False

        if not provider.is_authenticated():
            logger.error(f"{provider_name}認証が必要です")
            return False

        return provider.download_file(remote_path, local_path)

    def get_storage_info(self, provider_name: str) -> Dict[str, Any]:
        """指定されたプロバイダーのストレージ情報を取得"""
        provider = self.get_provider(provider_name)
        if not provider:
            return {"error": f"不明なプロバイダー: {provider_name}"}

        if not provider.is_authenticated():
            return {"error": f"{provider_name}認証が必要です"}

        return provider.get_storage_info()

    def list_available_providers(self) -> List[str]:
        """利用可能なプロバイダーのリストを取得"""
        return [name for name, provider in self.providers.items() if provider.is_authenticated()]

    def setup_google_drive(self, credentials_file: str) -> bool:
        """Google Drive認証を設定"""
        try:
            provider = GoogleDriveProvider(credentials_file)
            if provider.is_authenticated():
                self.providers['google_drive'] = provider
                return True
            return False
        except Exception as e:
            logger.error(f"Google Drive設定エラー: {e}")
            return False

    def setup_dropbox(self, access_token: str) -> bool:
        """Dropbox認証を設定"""
        try:
            provider = DropboxProvider(access_token)
            if provider.is_authenticated():
                self.providers['dropbox'] = provider
                return True
            return False
        except Exception as e:
            logger.error(f"Dropbox設定エラー: {e}")
            return False

# グローバルクラウドストレージマネージャー
_cloud_manager = None

def get_cloud_storage_manager() -> CloudStorageManager:
    """クラウドストレージマネージャーのシングルトンインスタンスを取得"""
    global _cloud_manager
    if _cloud_manager is None:
        _cloud_manager = CloudStorageManager()
    return _cloud_manager

# 便利関数
def upload_to_cloud(provider: str, local_file: str, remote_path: str) -> bool:
    """ファイルをクラウドストレージにアップロード"""
    return get_cloud_storage_manager().upload_to_cloud(provider, local_file, remote_path)

def download_from_cloud(provider: str, remote_path: str, local_file: str) -> bool:
    """クラウドストレージからファイルをダウンロード"""
    return get_cloud_storage_manager().download_from_cloud(provider, remote_path, local_file)

def get_cloud_storage_info(provider: str) -> Dict[str, Any]:
    """クラウドストレージ情報を取得"""
    return get_cloud_storage_manager().get_storage_info(provider)

if __name__ == "__main__":
    # サンプル使用例
    cloud_manager = get_cloud_storage_manager()

    # 利用可能なプロバイダーを表示
    available = cloud_manager.list_available_providers()
    print(f"利用可能なクラウドストレージ: {available}")

    # 各プロバイダーの情報を取得
    for provider_name in ['google_drive', 'dropbox']:
        info = cloud_manager.get_storage_info(provider_name)
        print(f"{provider_name}: {info}")

    print("クラウドストレージ統合テスト完了")
