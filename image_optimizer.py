#!/usr/bin/env python3
"""Image Optimization and Processing.

Lightweight utilities for image manipulation and optimization.
Supports resizing, format conversion, and quality optimization.
"""

import logging
import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Callable, Iterator, Generator
from PIL import Image, ImageFile
import time
import psutil
import io
import tempfile
import os

# GPU acceleration support
try:
    import cv2
    import numpy as np
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)

# Allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageOptimizer:
    """Optimize images for storage and bandwidth.

    Provides image resizing, format conversion, and quality optimization
    with minimal memory footprint and parallel processing support.

    Attributes:
        max_dimension: Maximum width or height in pixels
        quality: JPEG quality (1-100)
        progressive: Enable progressive JPEG encoding
        parallel_workers: Number of parallel workers (auto-detected if None)
    """

    def __init__(
        self,
        max_dimension: int = 2048,
        quality: int = 85,
        progressive: bool = True,
        parallel_workers: Optional[int] = None,
        use_streaming: bool = True,
        chunk_size: int = 8192,
        temp_dir: Optional[Path] = None,
        preferred_format: str = "AUTO"
    ):
        """Initialize image optimizer.

        Args:
            max_dimension: Maximum dimension in pixels (default: 2048)
            quality: JPEG quality 1-100 (default: 85)
            progressive: Progressive JPEG encoding (default: True)
            parallel_workers: Number of parallel workers (default: auto-detect)
            use_streaming: Enable streaming processing (default: True)
            chunk_size: Chunk size for streaming in bytes (default: 8192)
            temp_dir: Temporary directory for streaming (default: system temp)
            preferred_format: Preferred output format (JPEG, WEBP, AVIF, AUTO)
        """
        if max_dimension < 1:
            raise ValueError("max_dimension must be positive")
        if not 1 <= quality <= 100:
            raise ValueError("quality must be between 1 and 100")

        self.max_dimension = max_dimension
        self.quality = quality
        self.progressive = progressive
        self.parallel_workers = parallel_workers or self._detect_optimal_workers()
        self.use_streaming = use_streaming
        self.chunk_size = chunk_size
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "image_optimizer"
        self.preferred_format = preferred_format.upper()

        # GPU acceleration setup
        self.gpu_available = self._check_gpu_availability()
        self.use_gpu = self.gpu_available and self._should_use_gpu()

        # Performance monitoring
        self.processing_stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'total_original_size': 0,
            'total_optimized_size': 0
        }

        # Ensure temp directory exists
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
        if self.gpu_available and not self.use_gpu:
            logger.info("GPU available but disabled (likely insufficient GPU memory or compatibility)")

    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        if not GPU_AVAILABLE:
            return False

        try:
            # Check for CUDA-enabled OpenCV
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                return True

            # Check for basic OpenCV GPU support
            if hasattr(cv2, 'cuda'):
                return True

        except Exception as e:
            logger.debug(f"GPU availability check failed: {e}")

        return False

    def _should_use_gpu(self) -> bool:
        """Determine if GPU should be used based on system resources."""
        if not self.gpu_available:
            return False

        try:
            # Check GPU memory (if available)
            if hasattr(cv2, 'cuda') and hasattr(cv2.cuda, 'getDevice'):
                try:
                    gpu_memory = cv2.cuda.getCudaEnabledDeviceCount()
                    return gpu_memory > 0
                except:
                    pass

            # For systems without detailed GPU info, use conservative approach
            return True

        except Exception as e:
            logger.debug(f"GPU usage determination failed: {e}")
            return False

    def _select_optimal_format(self, original_format: str, file_size: int) -> str:
        """Select optimal output format based on original format and file size."""
        if self.preferred_format != "AUTO":
            return self.preferred_format

        # Check if WebP is supported (Pillow 5.0+)
        webp_supported = hasattr(Image, 'WEBP') or 'WEBP' in Image.EXTENSION

        # Check if AVIF is supported (requires pillow-avif plugin)
        try:
            avif_supported = hasattr(Image, 'AVIF') or 'AVIF' in Image.EXTENSION
        except:
            avif_supported = False

        # Format selection logic
        if original_format and original_format.upper() in ['PNG', 'WEBP', 'AVIF']:
            # Keep original format if it's already modern
            return original_format.upper()

        # For large files (>2MB), prefer WebP if supported
        if file_size > 2 * 1024 * 1024 and webp_supported:
            return 'WEBP'

        # For very large files (>5MB), prefer JPEG for compatibility
        if file_size > 5 * 1024 * 1024:
            return 'JPEG'

        # Default to WebP if supported, otherwise JPEG
        return 'WEBP' if webp_supported else 'JPEG'

    def _get_save_kwargs(self, format: str) -> dict:
        """Get format-specific save parameters."""
        kwargs = {
            'quality': self.quality,
            'optimize': True
        }

        if format.upper() in ['JPEG', 'JPG']:
            kwargs['progressive'] = self.progressive
        elif format.upper() == 'WEBP':
            # WebP specific settings
            kwargs['method'] = 6  # Compression method (0-6)
            kwargs['lossless'] = False if self.quality < 90 else True
        elif format.upper() == 'AVIF':
            # AVIF specific settings
            kwargs['quality'] = self.quality

        return kwargs

    def _detect_optimal_workers(self) -> int:
        """Detect optimal number of parallel workers based on system resources."""
        try:
            cpu_count = psutil.cpu_count(logical=True) or mp.cpu_count() or 1
            memory_gb = psutil.virtual_memory().total / (1024**3) if psutil else 4

            # Conservative approach: use fewer cores for image processing
            # to avoid memory issues with large images
            if memory_gb > 8:
                workers = max(1, min(cpu_count - 1, 8))
            elif memory_gb > 4:
                workers = max(1, min(cpu_count - 1, 4))
            else:
                workers = max(1, min(cpu_count - 1, 2))

            logger.info(f"Detected optimal workers: {workers} (CPU: {cpu_count}, Memory: {memory_gb:.1f}GB)")
            return workers

        except Exception as e:
            logger.warning(f"Failed to detect optimal workers, using 2: {e}")
            return 2

    def optimize_file(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        resize: bool = True,
        convert_format: Optional[str] = None,
        use_streaming: Optional[bool] = None
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Optimize image file with optional streaming processing
        Returns: (success, message, output_path)
        """
        use_streaming = use_streaming if use_streaming is not None else self.use_streaming

        if use_streaming and input_path.stat().st_size > 1024 * 1024:  # 1MB以上のファイル
            return self._optimize_file_streaming(input_path, output_path, resize, convert_format)
        else:
            return self._optimize_file_standard(input_path, output_path, resize, convert_format)

    def _resize_keep_aspect(self, img: Image.Image, max_dimension: int) -> Image.Image:
        """Resize image keeping aspect ratio"""
        width, height = img.size

        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))

    def _resize_keep_aspect(self, img: Image.Image, max_dimension: int) -> Image.Image:
        """Resize image keeping aspect ratio (with optional GPU acceleration)"""
        if self.use_gpu and GPU_AVAILABLE and max(img.size) > 1000:
            return self._resize_gpu(img, max_dimension)
        else:
            return self._resize_cpu(img, max_dimension)

    def _resize_cpu(self, img: Image.Image, max_dimension: int) -> Image.Image:
        """CPU-based resize using PIL"""
        width, height = img.size

        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))

        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _resize_gpu(self, img: Image.Image, max_dimension: int) -> Image.Image:
        """GPU-accelerated resize using OpenCV"""
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(img)

            # Convert RGB to BGR for OpenCV
            if img_array.shape[2] == 3:  # RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

            height, width = img_array.shape[:2]

            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))

            # Use GPU-accelerated resize if available
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                # GPU resize
                gpu_img = cv2.cuda.GpuMat()
                gpu_img.upload(img_array)

                resized_gpu = cv2.cuda.resize(gpu_img, (new_width, new_height))
                resized_array = resized_gpu.download()
            else:
                # CPU fallback with OpenCV
                resized_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

            # Convert back to RGB
            resized_array = cv2.cvtColor(resized_array, cv2.COLOR_BGR2RGB)

            # Convert back to PIL Image
            return Image.fromarray(resized_array)

        except Exception as e:
            logger.warning(f"GPU resize failed, falling back to CPU: {e}")
            return self._resize_cpu(img, max_dimension)

    def _optimize_file_standard(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        resize: bool = True,
        convert_format: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Path]]:
        """Standard image optimization (load full image into memory)"""
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

                save_format = self._select_optimal_format(img.format or 'JPEG', original_size)
                if convert_format:
                    save_format = convert_format.upper()

                # Get format-specific save parameters
                save_kwargs = self._get_save_kwargs(save_format)

                # Save optimized image
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

    def _optimize_file_streaming(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        resize: bool = True,
        convert_format: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Path]]:
        """Streaming image optimization (memory efficient for large files)"""
        try:
            # Use temporary file for intermediate processing
            with tempfile.NamedTemporaryFile(suffix='.jpg', dir=self.temp_dir, delete=False) as temp_file:
                temp_path = Path(temp_file.name)

            try:
                # Read image in chunks and write to temp file
                with input_path.open('rb') as input_file:
                    with Image.open(input_file) as img:
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

                        save_format = self._select_optimal_format(img.format or 'JPEG', original_size)
                        if convert_format:
                            save_format = convert_format.upper()

                        # Get format-specific save parameters
                        save_kwargs = self._get_save_kwargs(save_format)

                        # Save optimized image
                        img.save(output_path, format=save_format, **save_kwargs)

                        new_size = output_path.stat().st_size
                        reduction = ((original_size - new_size) / original_size * 100) if original_size > 0 else 0

                        logger.info(
                            f"Streaming optimized {input_path.name}: "
                            f"{original_size / 1024:.1f}KB -> {new_size / 1024:.1f}KB "
                            f"({reduction:.1f}% reduction)"
                        )

                        return True, f"Streaming optimized {reduction:.1f}%", output_path

            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()

        except Exception as e:
            logger.error(f"Failed to streaming optimize {input_path}: {e}")
            return False, str(e), None

    def optimize_image_memory_efficient(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        resize: bool = True,
        convert_format: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Path]]:
        """Memory-efficient image optimization using streaming"""
        return self._optimize_file_streaming(input_path, output_path, resize, convert_format)

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

    def batch_optimize_parallel(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        pattern: str = "*.jpg",
        inplace: bool = False,
        use_multiprocessing: bool = True
    ) -> dict:
        """
        Optimize all images in directory using parallel processing
        Returns: statistics dict
        """
        # Find all image files
        image_paths = [p for p in input_dir.glob(pattern) if p.is_file()]
        if not image_paths:
            logger.warning(f"No images found matching pattern: {pattern}")
            return {'total': 0, 'success': 0, 'failed': 0}

        if output_dir is None:
            output_dir = input_dir if inplace else input_dir / "optimized"

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting parallel optimization of {len(image_paths)} images with {self.parallel_workers} workers")

        start_time = time.time()

        # Choose executor type
        if use_multiprocessing and len(image_paths) > 10:
            executor_class = ProcessPoolExecutor
            logger.info("Using ProcessPoolExecutor for CPU-intensive tasks")
        else:
            executor_class = ThreadPoolExecutor
            logger.info("Using ThreadPoolExecutor for I/O-bound tasks")

        stats = {
            'total': len(image_paths),
            'success': 0,
            'failed': 0,
            'original_size': 0,
            'optimized_size': 0,
            'processing_time': 0
        }

        # Prepare tasks
        tasks = []
        for image_path in image_paths:
            output_path = output_dir / image_path.name if not inplace else image_path
            tasks.append((image_path, output_path))

        # Process in parallel
        with executor_class(max_workers=self.parallel_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self._optimize_single_task, input_path, output_path): (input_path, output_path)
                for input_path, output_path in tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_path):
                input_path, output_path = future_to_path[future]
                try:
                    success, original_size, optimized_size = future.result()

                    stats['original_size'] += original_size
                    if success and optimized_size > 0:
                        stats['success'] += 1
                        stats['optimized_size'] += optimized_size
                    else:
                        stats['failed'] += 1

                except Exception as e:
                    logger.error(f"Task failed for {input_path}: {e}")
                    stats['failed'] += 1

        # Calculate performance metrics
        stats['processing_time'] = time.time() - start_time
        if stats['original_size'] > 0:
            reduction = (stats['original_size'] - stats['optimized_size']) / stats['original_size'] * 100
            stats['reduction_percent'] = round(reduction, 2)
        else:
            stats['reduction_percent'] = 0

        # Update global statistics
        self.processing_stats['total_processed'] += stats['total']
        self.processing_stats['total_time'] += stats['processing_time']
        self.processing_stats['total_original_size'] += stats['original_size']
        self.processing_stats['total_optimized_size'] += stats['optimized_size']

        # Calculate throughput
        if stats['processing_time'] > 0:
            stats['images_per_second'] = stats['total'] / stats['processing_time']
            stats['avg_processing_time'] = stats['processing_time'] / stats['total']

        logger.info(
            f"Parallel optimization complete: {stats['success']}/{stats['total']} "
            f"({stats['reduction_percent']}% size reduction) "
            f"in {stats['processing_time']:.2f}s "
            f"({stats.get('images_per_second', 0):.2f} img/s)"
        )

        return stats

    def _optimize_single_task(self, input_path: Path, output_path: Path) -> Tuple[bool, int, int]:
        """Optimize a single image (for parallel processing)"""
        try:
            original_size = input_path.stat().st_size
            success, message, result_path = self.optimize_file(input_path, output_path, resize=True)

            if success and result_path and result_path.exists():
                optimized_size = result_path.stat().st_size
                return True, original_size, optimized_size
            else:
                return False, original_size, 0

        except Exception as e:
            logger.error(f"Failed to optimize {input_path}: {e}")
            return False, input_path.stat().st_size if input_path.exists() else 0, 0

    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        stats = self.processing_stats.copy()

        # Calculate averages
        if stats['total_processed'] > 0:
            stats['avg_time_per_image'] = stats['total_time'] / stats['total_processed']
            stats['avg_original_size'] = stats['total_original_size'] / stats['total_processed']
            stats['avg_optimized_size'] = stats['total_optimized_size'] / stats['total_processed']

            if stats['total_original_size'] > 0:
                stats['overall_reduction'] = (
                    (stats['total_original_size'] - stats['total_optimized_size']) /
                    stats['total_original_size'] * 100
                )

        return stats

    def reset_stats(self):
        """Reset performance statistics"""
        self.processing_stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'total_original_size': 0,
            'total_optimized_size': 0
        }

    def optimize_with_quality_tuning(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        resize: bool = True,
        target_reduction: float = 0.7,
        max_quality_loss: float = 0.1
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Optimize image with quality tuning to achieve target file size reduction.
        Uses multiple quality settings to find optimal balance.
        """
        try:
            with Image.open(input_path) as img:
                # Convert to RGB if needed
                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')

                original_size = input_path.stat().st_size
                target_size = int(original_size * (1 - target_reduction))

                # Resize first if needed
                if resize and max(img.size) > self.max_dimension:
                    img = self._resize_keep_aspect(img, self.max_dimension)

                if output_path is None:
                    output_path = input_path

                # Try different quality settings
                best_result = None
                best_score = float('inf')

                quality_range = range(max(10, self.quality - 20), min(95, self.quality + 10), 5)

                for quality in quality_range:
                    # Test save with current quality
                    temp_path = self.temp_dir / f"temp_{quality}_{input_path.name}"
                    save_kwargs = self._get_save_kwargs('JPEG')
                    save_kwargs['quality'] = quality

                    img.save(temp_path, **save_kwargs)
                    test_size = temp_path.stat().st_size

                    # Calculate score (closer to target size is better, but maintain quality)
                    size_diff = abs(test_size - target_size)
                    quality_penalty = (self.quality - quality) * 1000  # Penalize quality loss
                    score = size_diff + quality_penalty

                    if score < best_score:
                        best_score = score
                        best_result = (quality, test_size)

                    # Clean up temp file
                    if temp_path.exists():
                        temp_path.unlink()

                    # Early exit if we're close enough to target
                    if abs(test_size - target_size) / target_size < 0.05:  # Within 5%
                        break

                if best_result:
                    quality, final_size = best_result
                    save_kwargs = self._get_save_kwargs('JPEG')
                    save_kwargs['quality'] = quality

                    img.save(output_path, **save_kwargs)

                    reduction = ((original_size - final_size) / original_size * 100) if original_size > 0 else 0

                    logger.info(
                        f"Quality-tuned optimization {input_path.name}: "
                        f"{original_size / 1024:.1f}KB -> {final_size / 1024:.1f}KB "
                        f"(Q{quality}, {reduction:.1f}% reduction)"
                    )

                    return True, f"Quality-tuned {reduction:.1f}% (Q{quality})", output_path
                else:
                    # Fallback to standard optimization
                    return self._optimize_file_standard(input_path, output_path, resize, 'JPEG')

        except Exception as e:
            logger.error(f"Failed to quality-tune optimize {input_path}: {e}")
            return False, str(e), None

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test different format preferences
    formats_to_test = ["AUTO", "WEBP", "JPEG"]

    for format_pref in formats_to_test:
        print(f"\n=== Testing {format_pref} format ===")
        optimizer = ImageOptimizer(
            max_dimension=1920,
            quality=85,
            preferred_format=format_pref
        )

        # Example: optimize single image
        test_image = Path("test_image.jpg")
        if test_image.exists():
            success, message, output = optimizer.optimize_file(test_image)
            print(f"Optimization: {success} - {message}")

            # Get info
            info = optimizer.get_image_info(test_image)
            print(f"Image info: {info}")

        # Example: parallel batch optimization
        test_dir = Path("test_images")
        if test_dir.exists() and test_dir.is_dir():
            print("\n=== Parallel Batch Optimization ===")
            parallel_stats = optimizer.batch_optimize_parallel(
                test_dir,
                pattern="*.jpg",
                use_multiprocessing=True
            )
            print(f"Parallel stats: {parallel_stats}")

            print("\n=== Performance Statistics ===")
            perf_stats = optimizer.get_performance_stats()
            print(f"Performance stats: {perf_stats}")

        print(f"\n=== Format Support Check ===")
        print(f"WebP supported: {hasattr(Image, 'WEBP') or 'WEBP' in getattr(Image, 'EXTENSION', {})}")
        print(f"GPU available: {optimizer.gpu_available}")
        print(f"GPU enabled: {optimizer.use_gpu}")
        if GPU_AVAILABLE:
            print(f"OpenCV version: {cv2.__version__}")
            try:
                print(f"CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
            except:
                print("CUDA not available")
        print(f"Available formats: {getattr(Image, 'EXTENSION', {})}")

    # Test quality tuning optimization
    print(f"\n=== Quality Tuning Optimization ===")
    optimizer_quality = ImageOptimizer(max_dimension=1920, quality=85)

    if test_image.exists():
        success, message, output = optimizer_quality.optimize_with_quality_tuning(
            test_image, target_reduction=0.6
        )
        print(f"Quality tuning: {success} - {message}")

    if test_dir.exists() and test_dir.is_dir():
        print("\n=== Quality-Tuned Batch Optimization ===")
        quality_stats = optimizer_quality.batch_optimize_quality_tuned(
            test_dir, target_reduction=0.6
        )
        print(f"Quality batch stats: {quality_stats}")
