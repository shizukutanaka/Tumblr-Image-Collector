#!/usr/bin/env python3
"""Image Optimization and Processing.

Lightweight utilities for image manipulation and optimization.
Supports resizing, format conversion, and quality optimization.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from PIL import Image, ImageFile

logger = logging.getLogger(__name__)

# Allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageOptimizer:
    """Optimize images for storage and bandwidth.

    Provides image resizing, format conversion, and quality optimization
    with minimal memory footprint.

    Attributes:
        max_dimension: Maximum width or height in pixels
        quality: JPEG quality (1-100)
        progressive: Enable progressive JPEG encoding
    """

    def __init__(
        self,
        max_dimension: int = 2048,
        quality: int = 85,
        progressive: bool = True
    ):
        """Initialize image optimizer.

        Args:
            max_dimension: Maximum dimension in pixels (default: 2048)
            quality: JPEG quality 1-100 (default: 85)
            progressive: Progressive JPEG encoding (default: True)
        """
        if max_dimension < 1:
            raise ValueError("max_dimension must be positive")
        if not 1 <= quality <= 100:
            raise ValueError("quality must be between 1 and 100")

        self.max_dimension = max_dimension
        self.quality = quality
        self.progressive = progressive

    def optimize_file(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        resize: bool = True,
        convert_format: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Optimize image file
        Returns: (success, message, output_path)
        """
        try:
            with Image.open(input_path) as img:
                # Convert RGBA to RGB if needed
                if img.mode == 'RGBA' and convert_format in ['JPEG', 'jpg']:
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[3])
                    img = rgb_img
                elif img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')

                original_size = input_path.stat().st_size

                # Resize if needed
                if resize and max(img.size) > self.max_dimension:
                    img = self._resize_keep_aspect(img, self.max_dimension)

                # Determine output path and format
                if output_path is None:
                    output_path = input_path

                save_format = convert_format or img.format or 'JPEG'

                # Save optimized image
                save_kwargs = {
                    'quality': self.quality,
                    'optimize': True
                }

                if save_format.upper() in ['JPEG', 'JPG']:
                    save_kwargs['progressive'] = self.progressive

                img.save(output_path, format=save_format, **save_kwargs)

                new_size = output_path.stat().st_size
                reduction = ((original_size - new_size) / original_size * 100) if original_size > 0 else 0

                logger.info(
                    f"Optimized {input_path.name}: "
                    f"{original_size / 1024:.1f}KB -> {new_size / 1024:.1f}KB "
                    f"({reduction:.1f}% reduction)"
                )

                return True, f"Optimized {reduction:.1f}%", output_path

        except Exception as e:
            logger.error(f"Failed to optimize {input_path}: {e}")
            return False, str(e), None

    def _resize_keep_aspect(self, img: Image.Image, max_dimension: int) -> Image.Image:
        """Resize image keeping aspect ratio"""
        width, height = img.size

        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))

        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def create_thumbnail(
        self,
        input_path: Path,
        output_path: Path,
        size: Tuple[int, int] = (300, 300)
    ) -> Tuple[bool, str]:
        """
        Create thumbnail
        Returns: (success, message)
        """
        try:
            with Image.open(input_path) as img:
                # Convert to RGB if needed
                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')

                # Create thumbnail
                img.thumbnail(size, Image.Resampling.LANCZOS)

                # Save
                output_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(output_path, quality=self.quality, optimize=True)

                logger.info(f"Created thumbnail: {output_path.name}")
                return True, "Thumbnail created"

        except Exception as e:
            logger.error(f"Failed to create thumbnail: {e}")
            return False, str(e)

    def get_image_info(self, image_path: Path) -> Optional[dict]:
        """Get image information"""
        try:
            with Image.open(image_path) as img:
                return {
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'width': img.width,
                    'height': img.height,
                    'file_size': image_path.stat().st_size,
                    'aspect_ratio': round(img.width / img.height, 2) if img.height > 0 else 0
                }
        except Exception as e:
            logger.error(f"Failed to get image info: {e}")
            return None

    def is_valid_image(self, image_path: Path) -> bool:
        """Check if file is a valid image"""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def convert_format(
        self,
        input_path: Path,
        output_format: str,
        output_path: Optional[Path] = None
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Convert image to different format
        Returns: (success, message, output_path)
        """
        if output_path is None:
            output_path = input_path.with_suffix(f'.{output_format.lower()}')

        try:
            with Image.open(input_path) as img:
                # Convert mode if needed
                if output_format.upper() in ['JPEG', 'JPG'] and img.mode == 'RGBA':
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[3])
                    img = rgb_img

                img.save(output_path, format=output_format.upper())

                logger.info(f"Converted {input_path.name} to {output_format}")
                return True, f"Converted to {output_format}", output_path

        except Exception as e:
            logger.error(f"Failed to convert image: {e}")
            return False, str(e), None

    def batch_optimize(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        pattern: str = "*.jpg",
        inplace: bool = False
    ) -> dict:
        """
        Optimize all images in directory
        Returns: statistics dict
        """
        if output_dir is None:
            output_dir = input_dir if inplace else input_dir / "optimized"

        output_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'original_size': 0,
            'optimized_size': 0
        }

        for image_path in input_dir.glob(pattern):
            if not image_path.is_file():
                continue

            stats['total'] += 1
            stats['original_size'] += image_path.stat().st_size

            output_path = output_dir / image_path.name if not inplace else image_path

            success, message, result_path = self.optimize_file(
                image_path,
                output_path,
                resize=True
            )

            if success and result_path:
                stats['success'] += 1
                stats['optimized_size'] += result_path.stat().st_size
            else:
                stats['failed'] += 1

        # Calculate total reduction
        if stats['original_size'] > 0:
            reduction = (stats['original_size'] - stats['optimized_size']) / stats['original_size'] * 100
            stats['reduction_percent'] = round(reduction, 2)
        else:
            stats['reduction_percent'] = 0

        logger.info(
            f"Batch optimization complete: {stats['success']}/{stats['total']} "
            f"({stats['reduction_percent']}% size reduction)"
        )

        return stats


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    optimizer = ImageOptimizer(max_dimension=1920, quality=85)

    # Example: optimize single image
    test_image = Path("test_image.jpg")
    if test_image.exists():
        success, message, output = optimizer.optimize_file(test_image)
        print(f"Optimization: {success} - {message}")

        # Get info
        info = optimizer.get_image_info(test_image)
        print(f"Image info: {info}")
