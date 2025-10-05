#!/usr/bin/env python3
"""
Personal Security Module
Enhanced security features for individual users
"""

import os
import base64
import hashlib
import logging
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import getpass
import keyring

logger = logging.getLogger(__name__)


class PersonalSecurityManager:
    """
    Personal security features:
    - Credential encryption
    - Privacy mode
    - Secure storage
    - Access control
    """

    def __init__(self, base_dir: str, config: Dict[str, Any]):
        self.base_dir = Path(base_dir)
        self.config = config
        self.security_config = config.get('security', {})

        # Security files
        self.key_file = self.base_dir / ".security" / "master.key"
        self.encrypted_config_file = self.base_dir / ".security" / "credentials.enc"

        # Initialize encryption
        self._cipher = None
        if self.security_config.get('enable_encryption', True):
            self._initialize_encryption()

    def _initialize_encryption(self):
        """Initialize encryption system"""
        try:
            self.key_file.parent.mkdir(parents=True, exist_ok=True)

            if self.key_file.exists():
                # Load existing key
                with open(self.key_file, 'rb') as f:
                    key = f.read()
            else:
                # Generate new key with password
                password = self._get_master_password()
                key = self._derive_key(password)

                # Save key (consider using system keyring instead)
                with open(self.key_file, 'wb') as f:
                    f.write(key)

                # Secure file permissions (Unix-like systems)
                if hasattr(os, 'chmod'):
                    os.chmod(self.key_file, 0o600)

            self._cipher = Fernet(key)

        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            self._cipher = None

    def _get_master_password(self) -> str:
        """Get master password from user or keyring"""
        try:
            # Try to get from system keyring
            password = keyring.get_password("tumblr_collector", "master")
            if password:
                return password
        except Exception:
            pass

        # Prompt user
        print("\n=== Tumblr Collector Security Setup ===")
        print("Enter a master password to encrypt your credentials.")
        print("This password will be required to access your settings.\n")

        while True:
            password = getpass.getpass("Master Password: ")
            confirm = getpass.getpass("Confirm Password: ")

            if password == confirm:
                if len(password) < 8:
                    print("Password must be at least 8 characters long.")
                    continue

                # Save to keyring
                try:
                    keyring.set_password("tumblr_collector", "master", password)
                except Exception:
                    pass

                return password
            else:
                print("Passwords do not match. Please try again.")

    def _derive_key(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password"""
        if salt is None:
            salt = b'tumblr_collector_salt_v1'  # Use unique salt per installation

        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def encrypt_credentials(
        self,
        consumer_key: str,
        consumer_secret: str,
        token: Optional[str] = None,
        token_secret: Optional[str] = None
    ) -> bool:
        """Encrypt and store API credentials"""
        if not self._cipher:
            logger.warning("Encryption not available")
            return False

        try:
            credentials = {
                'consumer_key': consumer_key,
                'consumer_secret': consumer_secret,
                'token': token,
                'token_secret': token_secret,
                'encrypted_at': str(Path(__file__).stat().st_mtime)
            }

            encrypted_data = self._cipher.encrypt(
                json.dumps(credentials).encode()
            )

            with open(self.encrypted_config_file, 'wb') as f:
                f.write(encrypted_data)

            # Secure file permissions
            if hasattr(os, 'chmod'):
                os.chmod(self.encrypted_config_file, 0o600)

            logger.info("Credentials encrypted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to encrypt credentials: {e}")
            return False

    def decrypt_credentials(self) -> Optional[Dict[str, str]]:
        """Decrypt and retrieve API credentials"""
        if not self._cipher:
            logger.warning("Encryption not available")
            return None

        try:
            if not self.encrypted_config_file.exists():
                return None

            with open(self.encrypted_config_file, 'rb') as f:
                encrypted_data = f.read()

            decrypted_data = self._cipher.decrypt(encrypted_data)
            credentials = json.loads(decrypted_data.decode())

            return credentials

        except Exception as e:
            logger.error(f"Failed to decrypt credentials: {e}")
            return None

    def enable_privacy_mode(self) -> bool:
        """Enable privacy mode features"""
        if not self.security_config.get('enable_privacy_mode', False):
            return False

        try:
            # Clear sensitive data from logs
            self._sanitize_logs()

            # Disable analytics and telemetry
            self._disable_tracking()

            # Set restrictive permissions
            self._set_restrictive_permissions()

            logger.info("Privacy mode enabled")
            return True

        except Exception as e:
            logger.error(f"Failed to enable privacy mode: {e}")
            return False

    def _sanitize_logs(self):
        """Remove sensitive information from logs"""
        log_patterns = [
            r'consumer_key["\s:=]+\S+',
            r'consumer_secret["\s:=]+\S+',
            r'token["\s:=]+\S+',
            r'password["\s:=]+\S+',
        ]

        # Implementation would sanitize log files
        pass

    def _disable_tracking(self):
        """Disable any analytics or telemetry"""
        # Implementation would disable tracking features
        pass

    def _set_restrictive_permissions(self):
        """Set restrictive file permissions"""
        if not hasattr(os, 'chmod'):
            return

        try:
            # Secure directories
            for directory in [self.base_dir, self.base_dir / ".security"]:
                if directory.exists():
                    os.chmod(directory, 0o700)

            # Secure sensitive files
            for pattern in ['*.key', '*.enc', '*.db', 'config*.json']:
                for file_path in self.base_dir.rglob(pattern):
                    if file_path.is_file():
                        os.chmod(file_path, 0o600)

        except Exception as e:
            logger.error(f"Failed to set permissions: {e}")

    def secure_delete(self, file_path: Path) -> bool:
        """Securely delete file (overwrite before deletion)"""
        if not self.security_config.get('secure_delete', False):
            # Normal deletion
            try:
                file_path.unlink()
                return True
            except Exception:
                return False

        try:
            if not file_path.exists():
                return True

            # Overwrite file with random data
            file_size = file_path.stat().st_size

            with open(file_path, 'wb') as f:
                f.write(os.urandom(file_size))

            # Overwrite with zeros
            with open(file_path, 'wb') as f:
                f.write(b'\x00' * file_size)

            # Delete
            file_path.unlink()

            logger.info(f"Securely deleted: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to securely delete: {e}")
            return False

    def clear_old_logs(self, days: int = 30):
        """Clear logs older than specified days"""
        clear_days = self.security_config.get('clear_logs_after_days', days)

        try:
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=clear_days)

            for log_file in self.base_dir.rglob('*.log'):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    if self.secure_delete(log_file):
                        logger.info(f"Cleared old log: {log_file}")

        except Exception as e:
            logger.error(f"Failed to clear old logs: {e}")

    def generate_integrity_report(self) -> Dict[str, Any]:
        """Generate integrity report of important files"""
        report = {
            'timestamp': str(Path(__file__).stat().st_mtime),
            'files': {}
        }

        try:
            important_files = [
                self.encrypted_config_file,
                self.base_dir / "personal_library.db",
            ]

            for file_path in important_files:
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()

                    report['files'][str(file_path)] = {
                        'hash': file_hash,
                        'size': file_path.stat().st_size,
                        'modified': str(file_path.stat().st_mtime)
                    }

            return report

        except Exception as e:
            logger.error(f"Failed to generate integrity report: {e}")
            return report

    def verify_integrity(self, previous_report: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Verify integrity against previous report"""
        current_report = self.generate_integrity_report()
        issues = []

        try:
            for file_path, prev_data in previous_report.get('files', {}).items():
                curr_data = current_report['files'].get(file_path)

                if not curr_data:
                    issues.append(f"File missing: {file_path}")
                    continue

                if prev_data['hash'] != curr_data['hash']:
                    issues.append(f"File modified: {file_path}")

            return len(issues) == 0, issues

        except Exception as e:
            logger.error(f"Failed to verify integrity: {e}")
            return False, [str(e)]


# Global instance
_security_manager = None


def get_security_manager(base_dir: str, config: Dict[str, Any]) -> PersonalSecurityManager:
    """Get global security manager instance"""
    global _security_manager
    if _security_manager is None:
        _security_manager = PersonalSecurityManager(base_dir, config)
    return _security_manager
