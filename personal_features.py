#!/usr/bin/env python3
"""
Personal Use Features Module
Advanced features optimized for individual users
"""

import os
import json
import shutil
import logging
import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import time
from collections import defaultdict
from PIL import Image
import threading

logger = logging.getLogger(__name__)


class PersonalFeatureManager:
    """
    Advanced features for personal use
    - Auto organization by date/tags
    - Smart collections
    - Backup management
    - Metadata generation
    - Statistics and analytics
    """

    def __init__(self, base_dir: str, config: Dict[str, Any]):
        self.base_dir = Path(base_dir)
        self.config = config
        self.personal_config = config.get('personal_features', {})

        # Create organized directories
        self.images_dir = self.base_dir / "images"
        self.by_date_dir = self.base_dir / "by_date"
        self.by_tags_dir = self.base_dir / "by_tags"
        self.duplicates_dir = self.base_dir / "duplicates"
        self.thumbnails_dir = self.base_dir / "thumbnails"
        self.backups_dir = self.base_dir / "backups"
        self.metadata_dir = self.base_dir / "metadata"

        # Database for metadata and organization
        self.db_path = self.base_dir / "personal_library.db"
        self._initialize_database()

        # Statistics
        self.stats = {
            'total_images': 0,
            'organized_images': 0,
            'duplicates_found': 0,
            'thumbnails_created': 0,
            'backups_created': 0,
            'storage_used_mb': 0
        }

        self._lock = threading.Lock()

    def _initialize_database(self):
        """Initialize personal library database"""
        self.base_dir.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # Images table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER,
                    width INTEGER,
                    height INTEGER,
                    format TEXT,
                    created_date TEXT,
                    download_date TEXT,
                    source_url TEXT,
                    blog_name TEXT,
                    post_id TEXT,
                    tags TEXT,
                    quality_score REAL,
                    is_duplicate BOOLEAN DEFAULT 0,
                    duplicate_of INTEGER,
                    favorite BOOLEAN DEFAULT 0,
                    rating INTEGER DEFAULT 0,
                    notes TEXT
                )
            """)

            # Tags table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tag_name TEXT UNIQUE NOT NULL,
                    usage_count INTEGER DEFAULT 0
                )
            """)

            # Image-Tag relationship
            conn.execute("""
                CREATE TABLE IF NOT EXISTS image_tags (
                    image_id INTEGER,
                    tag_id INTEGER,
                    FOREIGN KEY (image_id) REFERENCES images(id),
                    FOREIGN KEY (tag_id) REFERENCES tags(id),
                    PRIMARY KEY (image_id, tag_id)
                )
            """)

            # Collections
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    created_date TEXT,
                    image_count INTEGER DEFAULT 0
                )
            """)

            # Collection-Image relationship
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collection_images (
                    collection_id INTEGER,
                    image_id INTEGER,
                    added_date TEXT,
                    FOREIGN KEY (collection_id) REFERENCES collections(id),
                    FOREIGN KEY (image_id) REFERENCES images(id),
                    PRIMARY KEY (collection_id, image_id)
                )
            """)

            # Statistics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS statistics (
                    date TEXT PRIMARY KEY,
                    images_downloaded INTEGER DEFAULT 0,
                    images_organized INTEGER DEFAULT 0,
                    duplicates_found INTEGER DEFAULT 0,
                    storage_used_mb REAL DEFAULT 0
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON images(file_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_blog_name ON images(blog_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_download_date ON images(download_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tags ON tags(tag_name)")

            conn.commit()

    def organize_by_date(self, image_path: Path) -> Path:
        """Organize image by date"""
        if not self.personal_config.get('auto_organize_by_date', False):
            return image_path

        try:
            # Get image date (EXIF or file modification)
            img = Image.open(image_path)
            exif_data = img._getexif() if hasattr(img, '_getexif') else None

            if exif_data and 36867 in exif_data:  # DateTimeOriginal
                date_str = exif_data[36867]
                date_obj = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
            else:
                date_obj = datetime.fromtimestamp(image_path.stat().st_mtime)

            # Create date-based directory
            year_dir = self.by_date_dir / str(date_obj.year)
            month_dir = year_dir / f"{date_obj.month:02d}"
            month_dir.mkdir(parents=True, exist_ok=True)

            # Create symlink or copy
            target_path = month_dir / image_path.name
            if not target_path.exists():
                os.symlink(image_path, target_path)

            return target_path

        except Exception as e:
            logger.error(f"Failed to organize by date: {e}")
            return image_path

    def organize_by_tags(self, image_path: Path, tags: List[str]) -> List[Path]:
        """Organize image by tags"""
        if not self.personal_config.get('auto_organize_by_tags', False):
            return []

        organized_paths = []

        try:
            for tag in tags[:10]:  # Limit to 10 tags
                # Sanitize tag name
                safe_tag = "".join(c for c in tag if c.isalnum() or c in (' ', '-', '_'))
                if not safe_tag:
                    continue

                tag_dir = self.by_tags_dir / safe_tag
                tag_dir.mkdir(parents=True, exist_ok=True)

                target_path = tag_dir / image_path.name
                if not target_path.exists():
                    os.symlink(image_path, target_path)
                    organized_paths.append(target_path)

        except Exception as e:
            logger.error(f"Failed to organize by tags: {e}")

        return organized_paths

    def create_thumbnail(self, image_path: Path) -> Optional[Path]:
        """Create thumbnail for image"""
        if not self.personal_config.get('create_thumbnails', False):
            return None

        try:
            thumb_size = tuple(self.personal_config.get('thumbnail_size', [400, 400]))

            # Create thumbnail directory structure
            rel_path = image_path.relative_to(self.images_dir) if self.images_dir in image_path.parents else image_path
            thumb_path = self.thumbnails_dir / rel_path.parent / f"thumb_{rel_path.name}"
            thumb_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate thumbnail
            img = Image.open(image_path)
            img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            img.save(thumb_path, optimize=True, quality=85)

            self.stats['thumbnails_created'] += 1
            return thumb_path

        except Exception as e:
            logger.error(f"Failed to create thumbnail: {e}")
            return None

    def add_image_to_library(
        self,
        image_path: Path,
        metadata: Dict[str, Any]
    ) -> int:
        """Add image to personal library database"""
        try:
            file_hash = self._calculate_file_hash(image_path)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT OR REPLACE INTO images (
                        file_path, file_hash, file_size, width, height, format,
                        download_date, source_url, blog_name, post_id, tags,
                        quality_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(image_path),
                    file_hash,
                    metadata.get('file_size', 0),
                    metadata.get('width', 0),
                    metadata.get('height', 0),
                    metadata.get('format', ''),
                    datetime.now().isoformat(),
                    metadata.get('source_url', ''),
                    metadata.get('blog_name', ''),
                    metadata.get('post_id', ''),
                    json.dumps(metadata.get('tags', [])),
                    metadata.get('quality_score', 0.0)
                ))

                image_id = cursor.lastrowid

                # Add tags
                tags = metadata.get('tags', [])
                for tag in tags:
                    self._add_tag(conn, image_id, tag)

                conn.commit()
                return image_id

        except Exception as e:
            logger.error(f"Failed to add image to library: {e}")
            return -1

    def _add_tag(self, conn: sqlite3.Connection, image_id: int, tag: str):
        """Add tag to database"""
        cursor = conn.execute(
            "INSERT OR IGNORE INTO tags (tag_name, usage_count) VALUES (?, 0)",
            (tag,)
        )

        conn.execute(
            "UPDATE tags SET usage_count = usage_count + 1 WHERE tag_name = ?",
            (tag,)
        )

        tag_id = conn.execute(
            "SELECT id FROM tags WHERE tag_name = ?",
            (tag,)
        ).fetchone()[0]

        conn.execute(
            "INSERT OR IGNORE INTO image_tags (image_id, tag_id) VALUES (?, ?)",
            (image_id, tag_id)
        )

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def handle_duplicate(self, image_path: Path, duplicate_of: Path) -> str:
        """Handle duplicate image based on configuration"""
        action = self.personal_config.get('duplicate_action', 'skip')

        if action == 'skip':
            return 'skipped'

        elif action == 'move_to_duplicates':
            try:
                self.duplicates_dir.mkdir(parents=True, exist_ok=True)
                target = self.duplicates_dir / image_path.name
                shutil.move(str(image_path), str(target))
                self.stats['duplicates_found'] += 1
                return 'moved'
            except Exception as e:
                logger.error(f"Failed to move duplicate: {e}")
                return 'error'

        elif action == 'delete':
            try:
                image_path.unlink()
                self.stats['duplicates_found'] += 1
                return 'deleted'
            except Exception as e:
                logger.error(f"Failed to delete duplicate: {e}")
                return 'error'

        return 'unknown'

    def create_backup(self) -> Optional[Path]:
        """Create backup of library"""
        if not self.personal_config.get('auto_backup', False):
            return None

        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"backup_{timestamp}"
            backup_path = self.backups_dir / backup_name

            # Create backup directory
            backup_path.mkdir(parents=True, exist_ok=True)

            # Backup database
            db_backup = backup_path / "library.db"
            shutil.copy2(self.db_path, db_backup)

            # Backup metadata
            if self.metadata_dir.exists():
                shutil.copytree(
                    self.metadata_dir,
                    backup_path / "metadata",
                    dirs_exist_ok=True
                )

            # Create backup manifest
            manifest = {
                'created': datetime.now().isoformat(),
                'stats': self.stats.copy(),
                'config': self.personal_config
            }

            with open(backup_path / "manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)

            self.stats['backups_created'] += 1
            logger.info(f"Backup created: {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Image statistics
                result = conn.execute("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(CASE WHEN favorite = 1 THEN 1 END) as favorites,
                        COUNT(CASE WHEN is_duplicate = 1 THEN 1 END) as duplicates,
                        AVG(quality_score) as avg_quality,
                        SUM(file_size) as total_size
                    FROM images
                """).fetchone()

                # Tag statistics
                tag_stats = conn.execute("""
                    SELECT COUNT(*) as total_tags, MAX(usage_count) as max_usage
                    FROM tags
                """).fetchone()

                # Collection statistics
                collection_stats = conn.execute("""
                    SELECT COUNT(*) as total_collections, SUM(image_count) as total_in_collections
                    FROM collections
                """).fetchone()

                # Recent downloads
                recent = conn.execute("""
                    SELECT DATE(download_date) as date, COUNT(*) as count
                    FROM images
                    WHERE download_date >= date('now', '-7 days')
                    GROUP BY DATE(download_date)
                    ORDER BY date DESC
                """).fetchall()

                return {
                    'total_images': result[0] or 0,
                    'favorite_images': result[1] or 0,
                    'duplicate_images': result[2] or 0,
                    'average_quality': round(result[3] or 0, 2),
                    'total_storage_mb': round((result[4] or 0) / 1024 / 1024, 2),
                    'total_tags': tag_stats[0] or 0,
                    'most_used_tag_count': tag_stats[1] or 0,
                    'total_collections': collection_stats[0] or 0,
                    'images_in_collections': collection_stats[1] or 0,
                    'recent_downloads': [{'date': r[0], 'count': r[1]} for r in recent],
                    'thumbnails_created': self.stats['thumbnails_created'],
                    'backups_created': self.stats['backups_created']
                }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def search_images(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        blog_name: Optional[str] = None,
        min_quality: Optional[float] = None,
        favorites_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Search images in library"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                sql = "SELECT * FROM images WHERE 1=1"
                params = []

                if blog_name:
                    sql += " AND blog_name = ?"
                    params.append(blog_name)

                if min_quality:
                    sql += " AND quality_score >= ?"
                    params.append(min_quality)

                if favorites_only:
                    sql += " AND favorite = 1"

                if tags:
                    tag_placeholders = ','.join(['?' for _ in tags])
                    sql += f"""
                        AND id IN (
                            SELECT image_id FROM image_tags
                            JOIN tags ON tags.id = image_tags.tag_id
                            WHERE tags.tag_name IN ({tag_placeholders})
                        )
                    """
                    params.extend(tags)

                results = conn.execute(sql, params).fetchall()
                return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to search images: {e}")
            return []

    def cleanup_old_backups(self, keep_count: int = 10):
        """Keep only the most recent backups"""
        try:
            if not self.backups_dir.exists():
                return

            backups = sorted(self.backups_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)

            for backup in backups[keep_count:]:
                shutil.rmtree(backup)
                logger.info(f"Removed old backup: {backup}")

        except Exception as e:
            logger.error(f"Failed to cleanup backups: {e}")


# Global instance
_personal_manager = None

def get_personal_manager(base_dir: str, config: Dict[str, Any]) -> PersonalFeatureManager:
    """Get global personal feature manager instance"""
    global _personal_manager
    if _personal_manager is None:
        _personal_manager = PersonalFeatureManager(base_dir, config)
    return _personal_manager
