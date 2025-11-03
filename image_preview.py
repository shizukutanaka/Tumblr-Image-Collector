"""
Image and Video Preview System for Tumblr Image Collector

Provides advanced preview capabilities including:
- Image thumbnail generation and display
- Video thumbnail extraction and preview
- Full-size image popup viewer
- Slideshow functionality
- Cross-platform compatibility

Features inspired by TumblThree's preview system.
"""

import os
import logging
import threading
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import tempfile
import time

# Image processing libraries
try:
    from PIL import Image, ImageTk, ImageFilter, ImageEnhance
    _PIL_AVAILABLE = True
except ImportError:
    Image = None
    ImageTk = None
    _PIL_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    _TKINTER_AVAILABLE = True
except ImportError:
    tk = None
    _TKINTER_AVAILABLE = False

# Platform-specific imports for video processing
if _CV2_AVAILABLE:
    try:
        import numpy as np
        _NUMPY_AVAILABLE = True
    except ImportError:
        np = None
        _NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class ImagePreview:
    """
    Image preview system with thumbnail generation and display capabilities.
    """

    def __init__(self, cache_dir: str = "preview_cache", max_cache_size: int = 1000):
        """
        Initialize image preview system.

        Args:
            cache_dir: Directory for caching thumbnails
            max_cache_size: Maximum number of cached thumbnails
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, str] = {}  # URL -> thumbnail path mapping

        # Thumbnail settings
        self.thumbnail_size = (200, 200)
        self.full_preview_size = (800, 600)

        # Slideshow settings
        self.slideshow_delay = 3.0  # seconds
        self.slideshow_running = False

        logger.info(f"Image preview system initialized with cache dir: {self.cache_dir}")

    def generate_thumbnail(self, image_path: str, url: str = "") -> Optional[str]:
        """
        Generate thumbnail for an image.

        Args:
            image_path: Path to the original image
            url: URL of the image (for caching)

        Returns:
            Path to thumbnail or None if failed
        """
        if not _PIL_AVAILABLE or not os.path.exists(image_path):
            return None

        try:
            # Check cache first
            cache_key = url or image_path
            if cache_key in self.cache:
                cached_path = self.cache[cache_key]
                if os.path.exists(cached_path):
                    return cached_path

            # Generate thumbnail
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')

                # Calculate thumbnail size maintaining aspect ratio
                img_ratio = img.width / img.height
                thumb_ratio = self.thumbnail_size[0] / self.thumbnail_size[1]

                if img_ratio > thumb_ratio:
                    # Image is wider
                    new_width = self.thumbnail_size[0]
                    new_height = int(new_width / img_ratio)
                else:
                    # Image is taller
                    new_height = self.thumbnail_size[1]
                    new_width = int(new_height * img_ratio)

                # Create thumbnail
                thumbnail = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Add padding if necessary to match thumbnail size
                if new_width != self.thumbnail_size[0] or new_height != self.thumbnail_size[1]:
                    padded = Image.new('RGB', self.thumbnail_size, (255, 255, 255))
                    x = (self.thumbnail_size[0] - new_width) // 2
                    y = (self.thumbnail_size[1] - new_height) // 2
                    padded.paste(thumbnail, (x, y))
                    thumbnail = padded

                # Save thumbnail
                if url:
                    thumb_filename = f"thumb_{hash(url) % 10000}.jpg"
                else:
                    thumb_filename = f"thumb_{hash(image_path) % 10000}.jpg"

                thumb_path = self.cache_dir / thumb_filename
                thumbnail.save(thumb_path, 'JPEG', quality=85)

                # Update cache
                self.cache[cache_key] = str(thumb_path)
                self._cleanup_cache()

                return str(thumb_path)

        except Exception as e:
            logger.error(f"Thumbnail generation failed for {image_path}: {e}")
            return None

    def generate_video_thumbnail(self, video_path: str, url: str = "", timestamp: float = 1.0) -> Optional[str]:
        """
        Generate thumbnail for a video at specified timestamp.

        Args:
            video_path: Path to the video file
            url: URL of the video (for caching)
            timestamp: Time in seconds to capture thumbnail

        Returns:
            Path to thumbnail or None if failed
        """
        if not _CV2_AVAILABLE or not os.path.exists(video_path):
            return None

        try:
            # Check cache first
            cache_key = url or video_path
            if cache_key in self.cache:
                cached_path = self.cache[cache_key]
                if os.path.exists(cached_path):
                    return cached_path

            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            # Seek to timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()

            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)

                # Generate thumbnail same as image
                return self.generate_thumbnail_from_pil(pil_image, cache_key)
            else:
                # If seeking fails, use first frame
                cap.set(cv2.CAP_PROP_POS_MSEC, 0)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    return self.generate_thumbnail_from_pil(pil_image, cache_key)

            cap.release()

        except Exception as e:
            logger.error(f"Video thumbnail generation failed for {video_path}: {e}")

        return None

    def generate_thumbnail_from_pil(self, pil_image: Image.Image, cache_key: str) -> Optional[str]:
        """Generate thumbnail from PIL Image."""
        if not _PIL_AVAILABLE:
            return None

        try:
            # Calculate thumbnail size maintaining aspect ratio
            img_ratio = pil_image.width / pil_image.height
            thumb_ratio = self.thumbnail_size[0] / self.thumbnail_size[1]

            if img_ratio > thumb_ratio:
                new_width = self.thumbnail_size[0]
                new_height = int(new_width / img_ratio)
            else:
                new_height = self.thumbnail_size[1]
                new_width = int(new_height * img_ratio)

            # Create thumbnail
            thumbnail = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Add padding if necessary
            if new_width != self.thumbnail_size[0] or new_height != self.thumbnail_size[1]:
                padded = Image.new('RGB', self.thumbnail_size, (255, 255, 255))
                x = (self.thumbnail_size[0] - new_width) // 2
                y = (self.thumbnail_size[1] - new_height) // 2
                padded.paste(thumbnail, (x, y))
                thumbnail = padded

            # Save thumbnail
            thumb_filename = f"thumb_{hash(cache_key) % 10000}.jpg"
            thumb_path = self.cache_dir / thumb_filename
            thumbnail.save(thumb_path, 'JPEG', quality=85)

            # Update cache
            self.cache[cache_key] = str(thumb_path)
            self._cleanup_cache()

            return str(thumb_path)

        except Exception as e:
            logger.error(f"Thumbnail generation from PIL failed: {e}")
            return None

    def _cleanup_cache(self):
        """Clean up old cache files."""
        try:
            # Remove excess cache entries
            while len(self.cache) > self.max_cache_size:
                oldest_key = next(iter(self.cache))
                cache_path = self.cache.pop(oldest_key)
                if os.path.exists(cache_path):
                    os.remove(cache_path)

            # Clean up orphaned files
            cached_files = set(self.cache.values())
            for file_path in self.cache_dir.glob("thumb_*.jpg"):
                if str(file_path) not in cached_files:
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass

        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")

    def show_full_preview(self, image_path: str, title: str = "Image Preview") -> bool:
        """
        Show full-size image preview in a popup window.

        Args:
            image_path: Path to the image file
            title: Window title

        Returns:
            True if preview was shown successfully
        """
        if not _TKINTER_AVAILABLE or not _PIL_AVAILABLE or not os.path.exists(image_path):
            return False

        try:
            # Load image
            image = Image.open(image_path)
            if image.mode not in ('RGB', 'RGBA'):
                image = image.convert('RGB')

            # Calculate display size
            display_size = self._calculate_display_size(image.size, self.full_preview_size)

            # Resize for display
            display_image = image.resize(display_size, Image.Resampling.LANCZOS)

            # Create preview window
            preview_window = tk.Toplevel()
            preview_window.title(title)
            preview_window.geometry(f"{display_size[0]}x{display_size[1]}")
            preview_window.resizable(True, True)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(display_image)

            # Display image
            label = ttk.Label(preview_window, image=photo)
            label.image = photo  # Keep a reference
            label.pack(fill=tk.BOTH, expand=True)

            # Add info label
            info_text = f"Size: {image.size[0]}x{image.size[1]} | Mode: {image.mode}"
            info_label = ttk.Label(preview_window, text=info_text, font=("Arial", 8))
            info_label.pack(side=tk.BOTTOM, fill=tk.X)

            # Center window
            preview_window.update_idletasks()
            x = (preview_window.winfo_screenwidth() - display_size[0]) // 2
            y = (preview_window.winfo_screenheight() - display_size[1]) // 2
            preview_window.geometry(f"+{x}+{y}")

            return True

        except Exception as e:
            logger.error(f"Full preview failed: {e}")
            return False

    def _calculate_display_size(self, original_size: Tuple[int, int], max_size: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate display size maintaining aspect ratio."""
        orig_width, orig_height = original_size
        max_width, max_height = max_size

        ratio = min(max_width / orig_width, max_height / orig_height)

        return (int(orig_width * ratio), int(orig_height * ratio))

    def start_slideshow(self, image_paths: List[str], delay: float = 3.0, window_title: str = "Slideshow") -> bool:
        """
        Start slideshow of images.

        Args:
            image_paths: List of image file paths
            delay: Delay between slides in seconds
            window_title: Window title

        Returns:
            True if slideshow started successfully
        """
        if not _TKINTER_AVAILABLE or not _PIL_AVAILABLE or not image_paths:
            return False

        try:
            self.slideshow_running = True
            self.slideshow_delay = delay

            # Create slideshow window
            self.slideshow_window = tk.Toplevel()
            self.slideshow_window.title(window_title)
            self.slideshow_window.attributes('-fullscreen', True)
            self.slideshow_window.configure(bg='black')

            # Image display label
            self.slideshow_label = ttk.Label(self.slideshow_window, background='black')
            self.slideshow_label.pack(fill=tk.BOTH, expand=True)

            # Info label
            self.slideshow_info = ttk.Label(
                self.slideshow_window,
                background='black',
                foreground='white',
                font=("Arial", 12)
            )
            self.slideshow_info.pack(side=tk.BOTTOM, pady=20)

            # Progress label
            self.slideshow_progress = ttk.Label(
                self.slideshow_window,
                text="",
                background='black',
                foreground='white',
                font=("Arial", 10)
            )
            self.slideshow_progress.pack(side=tk.BOTTOM, pady=5)

            # Start slideshow thread
            self.slideshow_thread = threading.Thread(
                target=self._run_slideshow,
                args=(image_paths,),
                daemon=True
            )
            self.slideshow_thread.start()

            # Bind escape key to stop slideshow
            self.slideshow_window.bind('<Escape>', lambda e: self.stop_slideshow())
            self.slideshow_window.bind('<space>', lambda e: self._pause_slideshow())
            self.slideshow_window.protocol("WM_DELETE_WINDOW", self.stop_slideshow)

            return True

        except Exception as e:
            logger.error(f"Slideshow start failed: {e}")
            return False

    def _run_slideshow(self, image_paths: List[str]):
        """Run slideshow in background thread."""
        current_index = 0
        paused = False

        while self.slideshow_running and current_index < len(image_paths):
            if not paused:
                try:
                    image_path = image_paths[current_index]

                    if os.path.exists(image_path):
                        # Load and display image
                        image = Image.open(image_path)
                        if image.mode not in ('RGB', 'RGBA'):
                            image = image.convert('RGB')

                        # Calculate display size for fullscreen
                        screen_width = self.slideshow_window.winfo_screenwidth()
                        screen_height = self.slideshow_window.winfo_screenheight()
                        display_size = self._calculate_display_size(image.size, (screen_width, screen_height))

                        # Resize image
                        display_image = image.resize(display_size, Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(display_image)

                        # Update display
                        self.slideshow_label.config(image=photo)
                        self.slideshow_label.image = photo

                        # Update info
                        filename = Path(image_path).name
                        self.slideshow_info.config(
                            text=f"{filename} ({current_index + 1}/{len(image_paths)}) - {image.size[0]}x{image.size[1]}"
                        )
                        self.slideshow_progress.config(text=f"Press SPACE to pause, ESC to exit")

                        # Center image
                        self.slideshow_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

                    current_index += 1

                    # Wait for delay or pause
                    start_time = time.time()
                    while time.time() - start_time < self.slideshow_delay:
                        if not self.slideshow_running:
                            break
                        time.sleep(0.1)

                except Exception as e:
                    logger.error(f"Slideshow display error: {e}")
                    current_index += 1

        # Slideshow finished
        self.slideshow_window.after(0, self.slideshow_window.destroy)

    def _pause_slideshow(self):
        """Pause/unpause slideshow."""
        # This would need additional state management for pause functionality
        pass

    def stop_slideshow(self):
        """Stop slideshow."""
        self.slideshow_running = False
        if hasattr(self, 'slideshow_window'):
            self.slideshow_window.destroy()

    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """
        Get detailed information about an image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with image information
        """
        if not _PIL_AVAILABLE or not os.path.exists(image_path):
            return {}

        try:
            with Image.open(image_path) as img:
                info = {
                    'size': img.size,
                    'mode': img.mode,
                    'format': img.format,
                    'filename': Path(image_path).name,
                    'file_size': os.path.getsize(image_path),
                    'has_transparency': img.mode in ('RGBA', 'LA', 'P'),
                }

                # Additional metadata if available
                if hasattr(img, '_getexif') and img._getexif():
                    info['exif'] = img._getexif()

                return info

        except Exception as e:
            logger.error(f"Image info extraction failed for {image_path}: {e}")
            return {}

    def batch_generate_thumbnails(self, image_paths: List[str], progress_callback: Optional[Any] = None) -> List[str]:
        """
        Generate thumbnails for multiple images.

        Args:
            image_paths: List of image file paths
            progress_callback: Callback for progress updates

        Returns:
            List of thumbnail paths
        """
        thumbnails = []

        for i, image_path in enumerate(image_paths):
            thumb_path = self.generate_thumbnail(image_path)
            if thumb_path:
                thumbnails.append(thumb_path)

            # Progress update
            if progress_callback:
                progress = (i + 1) / len(image_paths)
                progress_callback(progress, len(image_paths), i + 1)

        return thumbnails


def create_preview_system(cache_dir: str = "preview_cache") -> ImagePreview:
    """
    Factory function to create image preview system.

    Args:
        cache_dir: Directory for caching thumbnails

    Returns:
        ImagePreview instance
    """
    return ImagePreview(cache_dir)


if __name__ == "__main__":
    # Example usage
    preview = create_preview_system()

    # Test with sample image if available
    test_image = "test_image.jpg"
    if os.path.exists(test_image):
        thumb = preview.generate_thumbnail(test_image)
        if thumb:
            print(f"Thumbnail generated: {thumb}")
        else:
            print("Thumbnail generation failed")

    print("Image preview system ready")
