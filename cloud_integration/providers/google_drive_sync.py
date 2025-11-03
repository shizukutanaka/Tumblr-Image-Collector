# Google Drive Cloud Sync Provider
# Implementation for Google Drive API integration

import os
import sys
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False

from ..cloud_sync import BaseCloudProvider

class GoogleDriveSync(BaseCloudProvider):
    """Google Drive cloud storage provider implementation"""

    SCOPES = ['https://www.googleapis.com/auth/drive.file']

    def __init__(self):
        super().__init__()
        self.service = None
        self.creds = None

    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate with Google Drive"""
        if not GOOGLE_API_AVAILABLE:
            raise ImportError("Google API client not installed. Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")

        try:
            creds_data = credentials.get('credentials')
            if not creds_data:
                raise ValueError("Google credentials required")

            # Load credentials from token data
            if isinstance(creds_data, dict):
                self.creds = Credentials.from_authorized_user_info(creds_data, self.SCOPES)
            elif isinstance(creds_data, str):
                # Assume it's a token file path or token JSON string
                try:
                    token_data = json.loads(creds_data)
                    self.creds = Credentials.from_authorized_user_info(token_data, self.SCOPES)
                except json.JSONDecodeError:
                    # Assume it's a file path
                    with open(creds_data, 'r') as f:
                        token_data = json.load(f)
                    self.creds = Credentials.from_authorized_user_info(token_data, self.SCOPES)

            # Refresh credentials if expired
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())

            # Build the service
            self.service = build('drive', 'v3', credentials=self.creds)
            self.authenticated = True

            # Test authentication
            about = self.service.about().get(fields="user").execute()
            print(f"Authenticated with Google Drive as: {about.get('user', {}).get('displayName', 'Unknown')}")

            return True

        except Exception as e:
            print(f"Google Drive authentication failed: {e}")
            self.authenticated = False
            return False

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload a file to Google Drive"""
        if not self.authenticated or not self.service:
            raise ValueError("Not authenticated with Google Drive")

        try:
            # Create folder structure if needed
            folder_id = self._ensure_folder_path(remote_path)

            # Prepare file metadata
            file_metadata = {
                'name': Path(remote_path).name,
                'parents': [folder_id] if folder_id != 'root' else []
            }

            # Upload file
            media = MediaFileUpload(local_path, resumable=True)
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()

            print(f"Uploaded {local_path} to Google Drive: {remote_path} (ID: {file.get('id')})")
            return True

        except HttpError as e:
            print(f"Google Drive upload error: {e}")
            return False
        except Exception as e:
            print(f"Upload failed: {e}")
            return False

    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download a file from Google Drive"""
        if not self.authenticated or not self.service:
            raise ValueError("Not authenticated with Google Drive")

        try:
            # Find file by path
            file_id = self._get_file_id_by_path(remote_path)
            if not file_id:
                print(f"File not found in Google Drive: {remote_path}")
                return False

            # Download file
            request = self.service.files().get_media(fileId=file_id)
            with open(local_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()

            print(f"Downloaded from Google Drive: {remote_path} -> {local_path}")
            return True

        except HttpError as e:
            print(f"Google Drive download error: {e}")
            return False
        except Exception as e:
            print(f"Download failed: {e}")
            return False

    def upload_directory(self, local_dir: str, remote_dir: str) -> Dict[str, Any]:
        """Upload a directory to Google Drive"""
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
        """Download a directory from Google Drive"""
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
                # Remove remote_dir prefix to get relative path
                if remote_file.startswith(remote_dir):
                    relative_path = remote_file[len(remote_dir):].lstrip('/')
                    local_file_path = os.path.join(local_dir, relative_path)
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                else:
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
        # Basic implementation - upload local directory
        results = {
            'uploaded': 0,
            'downloaded': 0,
            'skipped': 0,
            'errors': []
        }

        try:
            upload_results = self.upload_directory(local_dir, remote_dir)
            results['uploaded'] = upload_results['uploaded']
            results['errors'].extend(upload_results['errors'])

        except Exception as e:
            results['errors'].append(f"Sync failed: {e}")

        return results

    def list_files(self, remote_dir: str) -> List[str]:
        """List files in remote directory"""
        if not self.authenticated or not self.service:
            return []

        files = []
        try:
            # Get folder ID
            folder_id = self._get_folder_id_by_path(remote_dir)
            if not folder_id:
                return []

            # List files in folder
            query = f"'{folder_id}' in parents and trashed = false"
            results = self.service.files().list(
                q=query,
                fields="files(id, name, mimeType)"
            ).execute()

            for file in results.get('files', []):
                if file['mimeType'] != 'application/vnd.google-apps.folder':
                    files.append(f"{remote_dir}/{file['name']}")

        except HttpError as e:
            print(f"Failed to list Google Drive files: {e}")
        except Exception as e:
            print(f"Error listing files: {e}")

        return files

    def _ensure_folder_path(self, file_path: str) -> str:
        """Ensure the folder path exists in Google Drive and return the parent folder ID"""
        path_parts = Path(file_path).parent.parts
        if len(path_parts) == 0 or (len(path_parts) == 1 and path_parts[0] == '.'):
            return 'root'

        current_parent = 'root'

        for part in path_parts:
            if part == '.' or part == '':
                continue

            folder_id = self._get_or_create_folder(part, current_parent)
            if not folder_id:
                raise Exception(f"Failed to create folder: {part}")
            current_parent = folder_id

        return current_parent

    def _get_or_create_folder(self, name: str, parent_id: str) -> Optional[str]:
        """Get or create a folder in Google Drive"""
        try:
            # Check if folder exists
            query = f"name = '{name}' and '{parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            results = self.service.files().list(q=query, fields="files(id, name)").execute()

            if results.get('files'):
                return results['files'][0]['id']

            # Create folder
            folder_metadata = {
                'name': name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_id]
            }

            folder = self.service.files().create(
                body=folder_metadata,
                fields='id'
            ).execute()

            return folder.get('id')

        except Exception as e:
            print(f"Error creating/getting folder {name}: {e}")
            return None

    def _get_file_id_by_path(self, file_path: str) -> Optional[str]:
        """Get file ID by path"""
        try:
            path_parts = Path(file_path).parts
            if len(path_parts) == 0:
                return None

            filename = path_parts[-1]
            parent_path = str(Path(*path_parts[:-1])) if len(path_parts) > 1 else ''

            parent_id = self._ensure_folder_path(parent_path) if parent_path else 'root'

            # Find file
            query = f"name = '{filename}' and '{parent_id}' in parents and trashed = false"
            results = self.service.files().list(q=query, fields="files(id, name)").execute()

            files = results.get('files', [])
            if files:
                return files[0]['id']

        except Exception as e:
            print(f"Error finding file {file_path}: {e}")

        return None

    def _get_folder_id_by_path(self, folder_path: str) -> Optional[str]:
        """Get folder ID by path"""
        try:
            return self._ensure_folder_path(folder_path)
        except Exception:
            return None

    def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync status"""
        if not self.authenticated or not self.service:
            return {
                'status': 'disconnected',
                'account': None,
                'usage': None
            }

        try:
            # Get account info
            about = self.service.about().get(fields="user, storageQuota").execute()

            user = about.get('user', {})
            quota = about.get('storageQuota', {})

            return {
                'status': 'connected',
                'account': {
                    'name': user.get('displayName', 'Unknown'),
                    'email': user.get('emailAddress', '')
                },
                'usage': {
                    'used': int(quota.get('usage', 0)),
                    'allocated': int(quota.get('limit', 0)),
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
        """Get instructions for Google Drive authentication"""
        return """
To use Google Drive sync:

1. Go to https://console.cloud.google.com/
2. Create a new project or select existing one
3. Enable Google Drive API
4. Create credentials (OAuth 2.0 Client IDs)
5. Download client_secret.json
6. Run local OAuth flow to get refresh token
7. Use the refresh token in the app

For development:
- Install google-auth-oauthlib
- Use InstalledAppFlow for local authentication
- Store credentials securely

Note: For production apps, implement proper OAuth flow with secure token storage.
        """
