# Dropbox Cloud Sync Provider
# Implementation for Dropbox API integration

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import dropbox
    from dropbox.files import WriteMode
    from dropbox.exceptions import ApiError, AuthError
    DROPBOX_AVAILABLE = True
except ImportError:
    DROPBOX_AVAILABLE = False
    dropbox = None

from .cloud_sync import BaseCloudProvider

class DropboxSync(BaseCloudProvider):
    """Dropbox cloud storage provider implementation"""

    def __init__(self):
        super().__init__()
        self.dbx = None
        self.account_info = None

    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate with Dropbox using access token"""
        if not DROPBOX_AVAILABLE:
            raise ImportError("Dropbox SDK not installed. Install with: pip install dropbox")

        try:
            access_token = credentials.get('access_token')
            if not access_token:
                raise ValueError("Access token required for Dropbox authentication")

            self.dbx = dropbox.Dropbox(access_token)

            # Test authentication
            self.account_info = self.dbx.users_get_current_account()
            self.authenticated = True

            print(f"Authenticated with Dropbox as: {self.account_info.name.display_name}")
            return True

        except AuthError as e:
            print(f"Dropbox authentication failed: {e}")
            self.authenticated = False
            return False
        except Exception as e:
            print(f"Dropbox connection error: {e}")
            self.authenticated = False
            return False

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload a file to Dropbox"""
        if not self.authenticated or not self.dbx:
            raise ValueError("Not authenticated with Dropbox")

        try:
            with open(local_path, 'rb') as f:
                self.dbx.files_upload(
                    f.read(),
                    remote_path,
                    mode=WriteMode('overwrite')
                )
            print(f"Uploaded {local_path} to Dropbox: {remote_path}")
            return True

        except ApiError as e:
            print(f"Dropbox upload error: {e}")
            return False
        except Exception as e:
            print(f"Upload failed: {e}")
            return False

    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download a file from Dropbox"""
        if not self.authenticated or not self.dbx:
            raise ValueError("Not authenticated with Dropbox")

        try:
            metadata, response = self.dbx.files_download(remote_path)

            with open(local_path, 'wb') as f:
                f.write(response.content)

            print(f"Downloaded from Dropbox: {remote_path} -> {local_path}")
            return True

        except ApiError as e:
            if e.error.is_path() and e.error.get_path().is_not_found():
                print(f"File not found in Dropbox: {remote_path}")
            else:
                print(f"Dropbox download error: {e}")
            return False
        except Exception as e:
            print(f"Download failed: {e}")
            return False

    def upload_directory(self, local_dir: str, remote_dir: str) -> Dict[str, Any]:
        """Upload a directory to Dropbox"""
        results = {
            'uploaded': 0,
            'failed': 0,
            'skipped': 0,
            'total': 0,
            'errors': []
        }

        local_path = Path(local_dir)
        if not local_path.exists() or not local_path.is_dir():
            results['errors'].append(f"Local directory not found: {local_dir}")
            return results

        try:
            for root, dirs, files in os.walk(local_dir):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file_path, local_dir)
                    remote_file_path = f"{remote_dir}/{relative_path}".replace('\\', '/')

                    results['total'] += 1

                    try:
                        if self.upload_file(local_file_path, remote_file_path):
                            results['uploaded'] += 1
                        else:
                            results['failed'] += 1
                    except Exception as e:
                        results['failed'] += 1
                        results['errors'].append(f"Failed to upload {relative_path}: {e}")

        except Exception as e:
            results['errors'].append(f"Directory upload failed: {e}")

        return results

    def download_directory(self, remote_dir: str, local_dir: str) -> Dict[str, Any]:
        """Download a directory from Dropbox"""
        results = {
            'downloaded': 0,
            'failed': 0,
            'total': 0,
            'errors': []
        }

        try:
            # List all files in remote directory
            files = self.list_files(remote_dir)
            results['total'] = len(files)

            os.makedirs(local_dir, exist_ok=True)

            for remote_file in files:
                local_file_path = os.path.join(local_dir, os.path.basename(remote_file))

                try:
                    if self.download_file(remote_file, local_file_path):
                        results['downloaded'] += 1
                    else:
                        results['failed'] += 1
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"Failed to download {remote_file}: {e}")

        except Exception as e:
            results['errors'].append(f"Directory download failed: {e}")

        return results

    def sync_directories(self, local_dir: str, remote_dir: str) -> Dict[str, Any]:
        """Sync local and remote directories"""
        # This is a basic implementation
        # A full sync would compare file timestamps, sizes, and hashes

        results = {
            'uploaded': 0,
            'downloaded': 0,
            'skipped': 0,
            'errors': []
        }

        try:
            # For now, just upload the entire local directory
            upload_results = self.upload_directory(local_dir, remote_dir)
            results['uploaded'] = upload_results['uploaded']
            results['errors'].extend(upload_results['errors'])

        except Exception as e:
            results['errors'].append(f"Sync failed: {e}")

        return results

    def list_files(self, remote_dir: str) -> List[str]:
        """List files in remote directory"""
        if not self.authenticated or not self.dbx:
            return []

        files = []
        try:
            result = self.dbx.files_list_folder(remote_dir, recursive=True)

            while True:
                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FileMetadata):
                        files.append(entry.path_lower)

                if not result.has_more:
                    break

                result = self.dbx.files_list_folder_continue(result.cursor)

        except ApiError as e:
            if e.error.is_path() and e.error.get_path().is_not_found():
                print(f"Directory not found in Dropbox: {remote_dir}")
            else:
                print(f"Failed to list Dropbox files: {e}")
        except Exception as e:
            print(f"Error listing files: {e}")

        return files

    def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync status"""
        if not self.authenticated:
            return {
                'status': 'disconnected',
                'account': None,
                'usage': None
            }

        try:
            # Get account usage
            usage = self.dbx.users_get_space_usage()

            return {
                'status': 'connected',
                'account': {
                    'name': self.account_info.name.display_name,
                    'email': self.account_info.email
                },
                'usage': {
                    'used': usage.used,
                    'allocated': usage.allocation.get_individual().allocated,
                    'unit': 'bytes'
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'account': None,
                'usage': None
            }

    @staticmethod
    def get_auth_instructions() -> str:
        """Get instructions for obtaining Dropbox access token"""
        return """
To use Dropbox sync:

1. Go to https://www.dropbox.com/developers/apps
2. Click "Create app"
3. Choose "Scoped access" and "App folder"
4. Name your app (e.g., "Tumblr Collector")
5. Go to Permissions tab and enable:
   - files.content.write
   - files.content.read
   - files.metadata.write
   - files.metadata.read
6. Generate access token from Settings tab
7. Use the token in the app settings

Note: For production use, implement OAuth 2.0 flow instead of using access tokens directly.
        """
