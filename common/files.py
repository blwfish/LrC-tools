"""
File discovery and hashing utilities.
"""

import hashlib
import os
from typing import Generator

from .config import ARCHIVE_ROOT, IMAGE_EXTENSIONS


def content_hash(path: str) -> str:
    """Compute fast hash for move detection.

    Uses file size + first 8KB of content. This is sufficient for
    identifying identical files - two different photos with the same
    size AND identical first 8KB is essentially impossible.

    ~0.1ms per file vs ~100ms for full file hash.
    """
    stat = os.stat(path)
    h = hashlib.md5()
    h.update(str(stat.st_size).encode())
    with open(path, 'rb') as f:
        h.update(f.read(8192))
    return h.hexdigest()


def find_changed_directories(archive_root: str, stored_mtimes: dict) -> list:
    """Find directories that have changed since last scan.

    Args:
        archive_root: Root directory to scan
        stored_mtimes: Dict of {path: mtime} from database

    Returns:
        List of directory paths that need scanning
    """
    changed = []

    for root, dirs, files in os.walk(archive_root):
        try:
            current_mtime = os.stat(root).st_mtime
        except OSError:
            continue

        stored = stored_mtimes.get(root)
        if stored is None or current_mtime > stored:
            changed.append(root)

    return changed


def scan_directory_for_images(directory: str) -> Generator[str, None, None]:
    """Scan a single directory (non-recursive) for image files.

    Args:
        directory: Directory to scan

    Yields:
        Full paths to image files
    """
    try:
        entries = os.scandir(directory)
    except OSError:
        return

    for entry in entries:
        if entry.is_file():
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                yield entry.path


def scan_all_images(archive_root: str = ARCHIVE_ROOT) -> Generator[str, None, None]:
    """Walk entire archive and yield all image paths.

    Args:
        archive_root: Root directory to scan

    Yields:
        Full paths to image files
    """
    for root, dirs, files in os.walk(archive_root):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                yield os.path.join(root, fname)


def get_file_info(path: str) -> dict:
    """Get file metadata for database storage.

    Returns:
        Dict with path, file_size, file_mtime, extension
    """
    stat = os.stat(path)
    return {
        'path': path,
        'file_size': stat.st_size,
        'file_mtime': stat.st_mtime,
        'extension': os.path.splitext(path)[1].lower(),
    }
