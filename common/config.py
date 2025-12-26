"""
Shared configuration for image processing tools.
"""

import os
import torch

# Archive location
ARCHIVE_ROOT = "/Volumes/archive2/images/"

# Supported image extensions
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.dng', '.raw', '.tiff', '.tif',
    '.nef', '.png', '.raf', '.cr2', '.cr3', '.arw', '.orf', '.rw2'
}

# RAW formats that need dcraw fallback
RAW_EXTENSIONS = {
    '.nef', '.raw', '.dng', '.raf', '.cr2', '.cr3', '.arw', '.orf', '.rw2'
}

# Size limits
MAX_IMAGE_PIXELS = 200_000_000  # Skip images over 200MP
MAX_FILE_SIZE = 500_000_000  # Skip files over 500MB

# Database location
DATABASE_PATH = os.path.expanduser("~/.local/share/photo_tools/catalog.db")

# Qdrant settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "images_full"


def get_device():
    """Select best available device: CUDA (NVIDIA), MPS (Apple Silicon), or CPU."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"
