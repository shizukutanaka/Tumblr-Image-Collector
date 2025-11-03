#!/usr/bin/env python3
"""
Tumblr Image Collector Cloud Integration Demo
Demonstrates cloud storage sync functionality
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cloud_integration.cloud_sync import CloudSyncManager

def main():
    parser = argparse.ArgumentParser(description="Tumblr Collector Cloud Sync Demo")
    parser.add_argument('--provider', choices=['dropbox', 'google_drive', 'onedrive'],
                       default='dropbox', help='Cloud provider to use')
    parser.add_argument('--action', choices=['upload', 'download', 'sync', 'backup', 'restore', 'list'],
                       default='list', help='Action to perform')
    parser.add_argument('--local-path', help='Local file or directory path')
    parser.add_argument('--remote-path', help='Remote file or directory path')
    parser.add_argument('--config', default='cloud_config.json',
                       help='Configuration file path')

    args = parser.parse_args()

    # Initialize cloud sync manager
    sync_manager = CloudSyncManager()

    # Check if provider is available
    if args.provider not in sync_manager.providers:
        print(f"Provider {args.provider} is not available. Available providers:")
        for provider_name in sync_manager.providers.keys():
            print(f"  - {provider_name}")
        return

    # Load configuration
    config = load_config(args.config)
    if not config:
        print("No configuration found. Please set up authentication first.")
        show_setup_instructions(args.provider)
        return

    # Authenticate
    provider_config = config.get(args.provider, {})
    if not provider_config:
        print(f"No configuration found for {args.provider}")
        show_setup_instructions(args.provider)
        return

    try:
        if not sync_manager.authenticate_provider(args.provider, provider_config):
            print(f"Failed to authenticate with {args.provider}")
            return

        print(f"Successfully authenticated with {args.provider}")

        # Perform action
        if args.action == 'upload':
            if not args.local_path:
                print("Local path required for upload")
                return

            remote_path = args.remote_path or f"tumblr_collector/{Path(args.local_path).name}"

            if os.path.isfile(args.local_path):
                success = sync_manager.upload_file(args.provider, args.local_path, remote_path)
                print(f"Upload {'successful' if success else 'failed'}")
            else:
                results = sync_manager.upload_directory(args.provider, args.local_path, remote_path)
                print(f"Directory upload: {results['uploaded']}/{results['total']} files uploaded")

        elif args.action == 'download':
            if not args.remote_path or not args.local_path:
                print("Both remote and local paths required for download")
                return

            if args.remote_path.endswith('/'):
                # Directory download
                results = sync_manager.download_directory(args.provider, args.remote_path, args.local_path)
                print(f"Directory download: {results['downloaded']}/{results['total']} files downloaded")
            else:
                # File download
                success = sync_manager.download_file(args.provider, args.remote_path, args.local_path)
                print(f"Download {'successful' if success else 'failed'}")

        elif args.action == 'sync':
            if not args.local_path:
                print("Local path required for sync")
                return

            remote_path = args.remote_path or f"tumblr_collector/{Path(args.local_path).name}"
            results = sync_manager.sync_directory(args.provider, args.local_path, remote_path)
            print(f"Sync completed: {results}")

        elif args.action == 'backup':
            if not args.local_path:
                print("Local collection path required for backup")
                return

            backup_name = args.remote_path or f"tumblr_backup_{Path(args.local_path).name}"
            success = sync_manager.backup_collection(args.provider, args.local_path, backup_name)
            print(f"Backup {'successful' if success else 'failed'}")

        elif args.action == 'restore':
            if not args.local_path or not args.remote_path:
                print("Both local path and backup name required for restore")
                return

            success = sync_manager.restore_collection(args.provider, args.remote_path, args.local_path)
            print(f"Restore {'successful' if success else 'failed'}")

        elif args.action == 'list':
            remote_path = args.remote_path or "tumblr_collector"
            if args.provider == 'dropbox':
                files = sync_manager.providers[args.provider].list_files(remote_path)
            else:
                files = sync_manager.list_backups(args.provider) if remote_path == "backups" else []

            print(f"Files in {remote_path}:")
            for file in files:
                print(f"  {file}")

        # Show sync status
        status = sync_manager.get_sync_status(args.provider)
        print(f"\nSync Status for {args.provider}:")
        print(f"  Status: {status.get('status', 'unknown')}")
        if status.get('account'):
            print(f"  Account: {status['account'].get('name', 'Unknown')}")
        if status.get('usage'):
            usage = status['usage']
            used_mb = usage.get('used', 0) / (1024 * 1024)
            allocated_mb = usage.get('allocated', 0) / (1024 * 1024)
            print(".1f")

    except Exception as e:
        print(f"Error: {e}")

def load_config(config_path):
    """Load configuration from file"""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
    return {}

def show_setup_instructions(provider):
    """Show setup instructions for a provider"""
    if provider == 'dropbox':
        instructions = """
Dropbox Setup Instructions:

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
7. Create cloud_config.json:
   {
     "dropbox": {
       "access_token": "YOUR_ACCESS_TOKEN"
     }
   }
        """
    elif provider == 'google_drive':
        instructions = """
Google Drive Setup Instructions:

1. Go to https://console.cloud.google.com/
2. Create a new project or select existing one
3. Enable Google Drive API
4. Create OAuth 2.0 credentials
5. Download client_secret.json
6. Run OAuth flow to get tokens
7. Create cloud_config.json:
   {
     "google_drive": {
       "credentials": {
         "token": "ACCESS_TOKEN",
         "refresh_token": "REFRESH_TOKEN",
         "client_id": "CLIENT_ID",
         "client_secret": "CLIENT_SECRET"
       }
     }
   }
        """
    else:
        instructions = f"Setup instructions for {provider} are not available yet."

    print(instructions)

if __name__ == '__main__':
    main()
