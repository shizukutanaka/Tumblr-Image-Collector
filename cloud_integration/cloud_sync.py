# Tumblr Image Collector Cloud Integration
# Cloud storage sync and backup functionality

import os
import sys
import json
import time
import threading
from pathlib import Path
from typing import Optional, Dict, List, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tumblr_image_collector import TumblrImageCollector

class CloudSyncManager:
    """Manages cloud storage synchronization and backup"""

    def __init__(self, collector=None):
        self.collector = collector or TumblrImageCollector()
        self.providers = {}
        self.sync_status = {}
        self.load_providers()

    def load_providers(self):
        """Load available cloud providers"""
        try:
            from .providers.dropbox_sync import DropboxSync
            self.providers['dropbox'] = DropboxSync()
        except ImportError:
            pass

        try:
            from .providers.google_drive_sync import GoogleDriveSync
            self.providers['google_drive'] = GoogleDriveSync()
        except ImportError:
            pass

        try:
            from .providers.onedrive_sync import OneDriveSync
            self.providers['onedrive'] = OneDriveSync()
        except ImportError:
            pass

    def authenticate_provider(self, provider_name: str, credentials: Dict[str, Any]) -> bool:
        """Authenticate with a cloud provider"""
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not available")

        provider = self.providers[provider_name]
        return provider.authenticate(credentials)

    def upload_file(self, provider_name: str, local_path: str, remote_path: str = None) -> bool:
        """Upload a file to cloud storage"""
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not available")

        provider = self.providers[provider_name]

        if remote_path is None:
            # Generate remote path based on local path
            remote_path = f"tumblr_collector/{Path(local_path).name}"

        return provider.upload_file(local_path, remote_path)

    def download_file(self, provider_name: str, remote_path: str, local_path: str) -> bool:
        """Download a file from cloud storage"""
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not available")

        provider = self.providers[provider_name]
        return provider.download_file(remote_path, local_path)

    def sync_directory(self, provider_name: str, local_dir: str, remote_dir: str = None,
                      mode: str = 'sync') -> Dict[str, Any]:
        """
        Sync a local directory with cloud storage

        Args:
            provider_name: Cloud provider name
            local_dir: Local directory path
            remote_dir: Remote directory path (optional)
            mode: 'sync', 'upload', or 'download'

        Returns:
            Dict with sync results
        """
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not available")

        if remote_dir is None:
            remote_dir = f"tumblr_collector/{Path(local_dir).name}"

        provider = self.providers[provider_name]

        if mode == 'upload':
            return provider.upload_directory(local_dir, remote_dir)
        elif mode == 'download':
            return provider.download_directory(remote_dir, local_dir)
        elif mode == 'sync':
            return provider.sync_directories(local_dir, remote_dir)
        else:
            raise ValueError(f"Invalid sync mode: {mode}")

    def backup_collection(self, provider_name: str, collection_path: str,
                         backup_name: str = None) -> bool:
        """Create a backup of Tumblr collection to cloud storage"""
        if backup_name is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            backup_name = f"tumblr_backup_{timestamp}"

        try:
            # Create zip archive of collection
            import zipfile
            zip_path = f"{backup_name}.zip"

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for root, dirs, files in os.walk(collection_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, collection_path)
                        zip_file.write(file_path, arcname)

            # Upload zip to cloud
            remote_path = f"backups/{backup_name}.zip"
            success = self.upload_file(provider_name, zip_path, remote_path)

            # Clean up local zip
            os.remove(zip_path)

            return success

        except Exception as e:
            print(f"Backup failed: {e}")
            return False

    def restore_collection(self, provider_name: str, backup_name: str,
                          restore_path: str) -> bool:
        """Restore a Tumblr collection from cloud backup"""
        try:
            remote_path = f"backups/{backup_name}.zip"
            local_zip = f"{backup_name}_temp.zip"

            # Download backup
            if not self.download_file(provider_name, remote_path, local_zip):
                return False

            # Extract zip
            import zipfile
            with zipfile.ZipFile(local_zip, 'r') as zip_file:
                zip_file.extractall(restore_path)

            # Clean up
            os.remove(local_zip)

            return True

        except Exception as e:
            print(f"Restore failed: {e}")
            return False

    def list_backups(self, provider_name: str) -> List[str]:
        """List available backups in cloud storage"""
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not available")

        provider = self.providers[provider_name]
        backups = provider.list_files("backups/")

        return [name for name in backups if name.endswith('.zip')]

    def get_sync_status(self, provider_name: str) -> Dict[str, Any]:
        """Get synchronization status for a provider"""
        if provider_name not in self.providers:
            return {'error': 'Provider not available'}

        provider = self.providers[provider_name]
        return provider.get_sync_status()

    def schedule_sync(self, provider_name: str, local_dir: str, interval_hours: int = 24):
        """Schedule automatic synchronization"""
        def sync_job():
            while True:
                try:
                    self.sync_directory(provider_name, local_dir, mode='sync')
                    print(f"Auto-sync completed for {provider_name}")
                except Exception as e:
                    print(f"Auto-sync failed: {e}")

                time.sleep(interval_hours * 3600)  # Sleep for specified hours

        thread = threading.Thread(target=sync_job, daemon=True)
        thread.start()

        return f"Auto-sync scheduled for {provider_name} every {interval_hours} hours"

class BaseCloudProvider:
    """Base class for cloud storage providers"""

    def __init__(self):
        self.authenticated = False
        self.credentials = None

    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate with the cloud provider"""
        raise NotImplementedError

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload a file to cloud storage"""
        raise NotImplementedError

    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download a file from cloud storage"""
        raise NotImplementedError

    def upload_directory(self, local_dir: str, remote_dir: str) -> Dict[str, Any]:
        """Upload a directory to cloud storage"""
        raise NotImplementedError

    def download_directory(self, remote_dir: str, local_dir: str) -> Dict[str, Any]:
        """Download a directory from cloud storage"""
        raise NotImplementedError

    def sync_directories(self, local_dir: str, remote_dir: str) -> Dict[str, Any]:
        """Sync local and remote directories"""
        raise NotImplementedError

    def list_files(self, remote_dir: str) -> List[str]:
        """List files in remote directory"""
        raise NotImplementedError

    def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync status"""
        raise NotImplementedError

# Placeholder for provider implementations
# These will be implemented in separate files
