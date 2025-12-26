"""
Image loading utilities with RAW format support via dcraw.
"""

import os
import subprocess
from io import BytesIO
from typing import Optional, Tuple

from PIL import Image

from .config import MAX_FILE_SIZE, MAX_IMAGE_PIXELS, RAW_EXTENSIONS


def load_image_pil(path: str) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Load image using PIL.

    Returns:
        (image, None) on success
        (None, error_message) on failure
    """
    try:
        file_size = os.path.getsize(path)
        if file_size > MAX_FILE_SIZE:
            return None, f"File too large: {file_size / 1e9:.1f}GB"

        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        img = Image.open(path)

        width, height = img.size
        if width * height > MAX_IMAGE_PIXELS:
            return None, f"Image too large: {width}x{height} ({width*height/1e6:.0f}MP)"

        if img.mode != 'RGB':
            img = img.convert('RGB')

        return img, None

    except Exception as e:
        return None, str(e)


def load_via_dcraw(path: str) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Load RAW file using dcraw.

    Tries embedded preview first (fast), falls back to full demosaic.

    Returns:
        (image, None) on success
        (None, error_message) on failure
    """
    # Try extracting embedded JPEG preview first (fast: ~0.1s vs ~8s for demosaic)
    try:
        result = subprocess.run(
            ['dcraw', '-e', '-c', path],  # -e = extract thumbnail, -c = stdout
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0 and len(result.stdout) > 0:
            img = Image.open(BytesIO(result.stdout))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img, None
    except Exception:
        pass  # Fall through to demosaic

    # Fallback: full demosaic (slow but works for files without embedded preview)
    try:
        result = subprocess.run(
            ['dcraw', '-c', '-w', '-h', path],  # -w = white balance, -h = half-size
            capture_output=True,
            timeout=30
        )
        if result.returncode != 0:
            return None, f"dcraw failed: {result.stderr.decode()[:100]}"

        img = Image.open(BytesIO(result.stdout))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img, None

    except subprocess.TimeoutExpired:
        return None, "dcraw timeout"
    except Exception as e:
        return None, f"dcraw error: {str(e)}"


def load_image(path: str) -> Tuple[Optional[Image.Image], Optional[str]]:
    """Load image, using dcraw fallback for RAW files.

    Returns:
        (image, None) on success
        (None, error_message) on failure
    """
    ext = os.path.splitext(path)[1].lower()

    # Try PIL first (fast for JPEGs, works for many formats)
    img, error = load_image_pil(path)
    if img is not None:
        return img, None

    # For RAW files, try dcraw as fallback
    if ext in RAW_EXTENSIONS:
        return load_via_dcraw(path)

    # Not a RAW file and PIL failed
    return None, error


def load_and_preprocess_image(path: str, preprocess):
    """Load image and apply CLIP preprocessing.

    Args:
        path: Path to image file
        preprocess: CLIP preprocessing transform

    Returns:
        (preprocessed_tensor, None) on success
        (None, error_message) on failure
    """
    img, error = load_image(path)
    if img is None:
        return None, error

    try:
        return preprocess(img), None
    except Exception as e:
        return None, f"Preprocessing failed: {str(e)}"
