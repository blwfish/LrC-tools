"""
Shared infrastructure for image processing tools.
Used by imageSearch (CLIP embeddings) and racing_tagger (vision AI tagging).
"""

# Core imports that don't require heavy dependencies
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

# Image loading requires PIL - import lazily
def load_and_preprocess_image(path, preprocess):
    """Lazy import wrapper for image loading."""
    from .image_load import load_and_preprocess_image as _load
    return _load(path, preprocess)

def load_image_pil(path):
    """Lazy import wrapper for PIL image loading."""
    from .image_load import load_image_pil as _load
    return _load(path)

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
