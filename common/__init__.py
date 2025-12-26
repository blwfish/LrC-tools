"""
Shared infrastructure for image processing tools.
Used by imageSearch (CLIP embeddings) and racing_tagger (vision AI tagging).
"""

from .config import (
    ARCHIVE_ROOT,
    IMAGE_EXTENSIONS,
    RAW_EXTENSIONS,
    DATABASE_PATH,
    get_device,
)
from .db import Database
from .files import (
    content_hash,
    find_changed_directories,
    scan_directory_for_images,
)
from .image_load import load_and_preprocess_image, load_image_pil

__all__ = [
    'ARCHIVE_ROOT',
    'IMAGE_EXTENSIONS',
    'RAW_EXTENSIONS',
    'DATABASE_PATH',
    'get_device',
    'Database',
    'content_hash',
    'find_changed_directories',
    'scan_directory_for_images',
    'load_and_preprocess_image',
    'load_image_pil',
]
