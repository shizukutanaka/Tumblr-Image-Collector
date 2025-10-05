#!/usr/bin/env python3
"""
Personal Convenience Features
User-friendly features for individual use
"""

import os
import logging
import json
import webbrowser
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import subprocess
import shutil

logger = logging.getLogger(__name__)


class ConvenienceFeatures:
    """
    Convenience features for personal use:
    - Quick commands
    - Shortcuts
    - Automation
    - Smart defaults
    """

    def __init__(self, base_dir: str, config: Dict[str, Any]):
        self.base_dir = Path(base_dir)
        self.config = config
        self.ui_config = config.get('ui', {})

        # Favorite blogs
        self.favorites_file = self.base_dir / "favorites.json"
        self.favorites = self._load_favorites()

        # Blocklist
        self.blocklist_file = self.base_dir / "blocklist.json"
        self.blocklist = self._load_blocklist()

        # Scheduled tasks
        self.schedule_file = self.base_dir / "schedule.json"
        self.schedule = self._load_schedule()

    def _load_favorites(self) -> List[Dict[str, Any]]:
        """Load favorite blogs"""
        if self.favorites_file.exists():
            try:
                with open(self.favorites_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load favorites: {e}")
        return []

    def _save_favorites(self):
        """Save favorite blogs"""
        try:
            with open(self.favorites_file, 'w') as f:
                json.dump(self.favorites, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save favorites: {e}")

    def _load_blocklist(self) -> List[str]:
        """Load blocked blogs"""
        if self.blocklist_file.exists():
            try:
                with open(self.blocklist_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load blocklist: {e}")
        return []

    def _save_blocklist(self):
        """Save blocked blogs"""
        try:
            with open(self.blocklist_file, 'w') as f:
                json.dump(self.blocklist, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save blocklist: {e}")

    def _load_schedule(self) -> List[Dict[str, Any]]:
        """Load scheduled tasks"""
        if self.schedule_file.exists():
            try:
                with open(self.schedule_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load schedule: {e}")
        return []

    def _save_schedule(self):
        """Save scheduled tasks"""
        try:
            with open(self.schedule_file, 'w') as f:
                json.dump(self.schedule, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save schedule: {e}")

    def add_favorite(
        self,
        blog_name: str,
        tags: Optional[List[str]] = None,
        auto_download: bool = True,
        notes: str = ""
    ) -> bool:
        """Add blog to favorites"""
        try:
            # Check if already exists
            for fav in self.favorites:
                if fav['blog_name'] == blog_name:
                    logger.warning(f"Blog already in favorites: {blog_name}")
                    return False

            self.favorites.append({
                'blog_name': blog_name,
                'tags': tags or [],
                'auto_download': auto_download,
                'notes': notes,
                'added_date': datetime.now().isoformat(),
                'last_checked': None,
                'total_downloads': 0
            })

            self._save_favorites()
            logger.info(f"Added to favorites: {blog_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to add favorite: {e}")
            return False

    def remove_favorite(self, blog_name: str) -> bool:
        """Remove blog from favorites"""
        try:
            self.favorites = [f for f in self.favorites if f['blog_name'] != blog_name]
            self._save_favorites()
            logger.info(f"Removed from favorites: {blog_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove favorite: {e}")
            return False

    def get_favorites(self) -> List[Dict[str, Any]]:
        """Get all favorite blogs"""
        return self.favorites

    def add_to_blocklist(self, blog_name: str) -> bool:
        """Add blog to blocklist"""
        try:
            if blog_name not in self.blocklist:
                self.blocklist.append(blog_name)
                self._save_blocklist()
                logger.info(f"Added to blocklist: {blog_name}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to add to blocklist: {e}")
            return False

    def is_blocked(self, blog_name: str) -> bool:
        """Check if blog is blocked"""
        return blog_name in self.blocklist

    def schedule_download(
        self,
        blog_name: str,
        schedule_type: str = "daily",  # daily, weekly, monthly
        time: str = "03:00",
        tags: Optional[List[str]] = None,
        enabled: bool = True
    ) -> bool:
        """Schedule automatic download"""
        try:
            task = {
                'id': len(self.schedule) + 1,
                'blog_name': blog_name,
                'schedule_type': schedule_type,
                'time': time,
                'tags': tags or [],
                'enabled': enabled,
                'created': datetime.now().isoformat(),
                'last_run': None,
                'next_run': self._calculate_next_run(schedule_type, time)
            }

            self.schedule.append(task)
            self._save_schedule()
            logger.info(f"Scheduled download for {blog_name}: {schedule_type} at {time}")
            return True

        except Exception as e:
            logger.error(f"Failed to schedule download: {e}")
            return False

    def _calculate_next_run(self, schedule_type: str, time: str) -> str:
        """Calculate next run time"""
        try:
            hour, minute = map(int, time.split(':'))
            now = datetime.now()
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

            if schedule_type == "daily":
                if next_run <= now:
                    next_run += timedelta(days=1)
            elif schedule_type == "weekly":
                days_ahead = 7 - now.weekday()
                next_run = now + timedelta(days=days_ahead)
                next_run = next_run.replace(hour=hour, minute=minute)
            elif schedule_type == "monthly":
                if now.day >= 1:
                    next_run = now.replace(day=1, hour=hour, minute=minute)
                    next_run += timedelta(days=32)
                    next_run = next_run.replace(day=1)

            return next_run.isoformat()

        except Exception:
            return datetime.now().isoformat()

    def open_download_folder(self) -> bool:
        """Open download folder in file explorer"""
        try:
            output_folder = self.base_dir / self.config.get('output_folder_name', 'tumblr_images')
            output_folder.mkdir(parents=True, exist_ok=True)

            if os.name == 'nt':  # Windows
                os.startfile(output_folder)
            elif os.name == 'posix':  # macOS, Linux
                subprocess.run(['xdg-open', str(output_folder)])
            else:
                logger.warning("Unable to open folder on this OS")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to open folder: {e}")
            return False

    def create_quick_access_shortcuts(self) -> bool:
        """Create quick access shortcuts"""
        try:
            shortcuts_dir = self.base_dir / "shortcuts"
            shortcuts_dir.mkdir(parents=True, exist_ok=True)

            # Create shortcuts for common folders
            shortcuts = {
                'Latest Downloads': self.base_dir / "images",
                'By Date': self.base_dir / "by_date",
                'By Tags': self.base_dir / "by_tags",
                'Favorites': self.base_dir / "favorites",
                'High Quality': self.base_dir / "high_quality"
            }

            for name, target in shortcuts.items():
                target.mkdir(parents=True, exist_ok=True)
                shortcut = shortcuts_dir / name

                if not shortcut.exists():
                    if os.name == 'nt':
                        # Windows shortcut
                        pass  # Would create .lnk file
                    else:
                        # Unix symlink
                        os.symlink(target, shortcut)

            logger.info("Quick access shortcuts created")
            return True

        except Exception as e:
            logger.error(f"Failed to create shortcuts: {e}")
            return False

    def export_library_stats(self, format: str = "json") -> Optional[Path]:
        """Export library statistics"""
        try:
            from personal_features import get_personal_manager
            manager = get_personal_manager(str(self.base_dir), self.config)
            stats = manager.get_statistics()

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            if format == "json":
                export_file = self.base_dir / f"stats_{timestamp}.json"
                with open(export_file, 'w') as f:
                    json.dump(stats, f, indent=2)

            elif format == "txt":
                export_file = self.base_dir / f"stats_{timestamp}.txt"
                with open(export_file, 'w') as f:
                    f.write("=== Tumblr Collection Statistics ===\n\n")
                    for key, value in stats.items():
                        f.write(f"{key}: {value}\n")

            logger.info(f"Statistics exported to {export_file}")
            return export_file

        except Exception as e:
            logger.error(f"Failed to export statistics: {e}")
            return None

    def quick_search(self, query: str) -> List[Path]:
        """Quick search for images"""
        results = []
        try:
            images_dir = self.base_dir / "images"
            if not images_dir.exists():
                return results

            query_lower = query.lower()

            for image_path in images_dir.rglob('*'):
                if image_path.is_file():
                    if query_lower in image_path.name.lower():
                        results.append(image_path)

                    if len(results) >= 100:  # Limit results
                        break

            return results

        except Exception as e:
            logger.error(f"Quick search failed: {e}")
            return results

    def create_wallpaper_collection(self, min_resolution: tuple = (1920, 1080)) -> Optional[Path]:
        """Create collection of wallpaper-quality images"""
        try:
            wallpaper_dir = self.base_dir / "wallpapers"
            wallpaper_dir.mkdir(parents=True, exist_ok=True)

            from PIL import Image
            images_dir = self.base_dir / "images"
            count = 0

            for image_path in images_dir.rglob('*'):
                if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    try:
                        img = Image.open(image_path)
                        if img.width >= min_resolution[0] and img.height >= min_resolution[1]:
                            # Create symlink or copy
                            target = wallpaper_dir / image_path.name
                            if not target.exists():
                                shutil.copy2(image_path, target)
                                count += 1

                    except Exception:
                        continue

            logger.info(f"Created wallpaper collection with {count} images")
            return wallpaper_dir

        except Exception as e:
            logger.error(f"Failed to create wallpaper collection: {e}")
            return None

    def get_quick_stats(self) -> Dict[str, Any]:
        """Get quick statistics summary"""
        try:
            images_dir = self.base_dir / "images"

            stats = {
                'total_images': 0,
                'total_size_mb': 0,
                'favorite_blogs': len(self.favorites),
                'blocked_blogs': len(self.blocklist),
                'scheduled_tasks': len(self.schedule)
            }

            if images_dir.exists():
                for img in images_dir.rglob('*'):
                    if img.is_file():
                        stats['total_images'] += 1
                        stats['total_size_mb'] += img.stat().st_size / 1024 / 1024

            stats['total_size_mb'] = round(stats['total_size_mb'], 2)
            return stats

        except Exception as e:
            logger.error(f"Failed to get quick stats: {e}")
            return {}


# Global instance
_convenience = None


def get_convenience_features(base_dir: str, config: Dict[str, Any]) -> ConvenienceFeatures:
    """Get global convenience features instance"""
    global _convenience
    if _convenience is None:
        _convenience = ConvenienceFeatures(base_dir, config)
    return _convenience
