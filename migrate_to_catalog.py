#!/usr/bin/env python3
"""
One-time migration to populate the SQLite catalog from existing Qdrant data.

This script:
1. Creates the SQLite database with schema
2. Queries all points from Qdrant
3. For each path that exists on disk:
   - Computes content hash
   - Adds to files table
   - Adds to clip_embeddings table
4. Scans archive directories and populates directory mtimes

No CLIP model needed - just file operations.
Expected runtime: ~30-60 minutes for 546k files.
"""

import os
import sys
from datetime import datetime
from tqdm import tqdm
from qdrant_client import QdrantClient

# Add parent directory to path for common module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
    ARCHIVE_ROOT,
    DATABASE_PATH,
    Database,
    content_hash,
)
from common.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME


def log(msg: str):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


def fetch_all_qdrant_points(client: QdrantClient, collection: str) -> list:
    """Fetch all points from Qdrant collection using scroll."""
    log(f"Fetching all points from Qdrant collection '{collection}'...")

    all_points = []
    offset = None
    batch_size = 1000

    # Get total count for progress bar
    info = client.get_collection(collection)
    total = info.points_count
    log(f"Collection has {total} points")

    with tqdm(total=total, desc="Fetching from Qdrant") as pbar:
        while True:
            results, next_offset = client.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,  # Don't need vectors, just paths
            )

            if not results:
                break

            all_points.extend(results)
            pbar.update(len(results))

            if next_offset is None:
                break
            offset = next_offset

    log(f"Fetched {len(all_points)} points")
    return all_points


def scan_archive_directories(archive_root: str) -> list:
    """Scan archive and collect all directory mtimes."""
    log(f"Scanning directories in {archive_root}...")

    directories = []
    for root, dirs, files in os.walk(archive_root):
        try:
            mtime = os.stat(root).st_mtime
            directories.append((root, mtime))
        except OSError:
            continue

    log(f"Found {len(directories)} directories")
    return directories


def main():
    log("=" * 60)
    log("Migration to SQLite Catalog")
    log("=" * 60)
    log(f"Database: {DATABASE_PATH}")
    log(f"Archive: {ARCHIVE_ROOT}")
    log(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT} / {COLLECTION_NAME}")
    log("")

    # Check if database already exists
    if os.path.exists(DATABASE_PATH):
        response = input(f"Database already exists at {DATABASE_PATH}. Overwrite? [y/N] ")
        if response.lower() != 'y':
            log("Aborted.")
            return
        os.remove(DATABASE_PATH)

    # Connect to Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Check collection exists
    if not client.collection_exists(COLLECTION_NAME):
        log(f"ERROR: Collection '{COLLECTION_NAME}' does not exist in Qdrant")
        return

    # Create database
    log("Creating SQLite database...")
    db = Database(DATABASE_PATH)

    # Fetch all points from Qdrant
    points = fetch_all_qdrant_points(client, COLLECTION_NAME)

    # Process each point
    log("Processing files (computing hashes)...")
    files_batch = []
    embeddings_batch = []
    missing_count = 0
    error_count = 0

    batch_size = 500  # Commit every N files

    for point in tqdm(points, desc="Hashing files"):
        path = point.payload.get('path')
        if not path:
            continue

        # Check if file exists
        if not os.path.exists(path):
            missing_count += 1
            continue

        try:
            # Get file info
            stat = os.stat(path)
            file_size = stat.st_size
            file_mtime = stat.st_mtime
            extension = os.path.splitext(path)[1].lower()

            # Compute content hash
            file_hash = content_hash(path)

            files_batch.append((path, file_hash, file_size, file_mtime, extension))
            embeddings_batch.append((path, point.id))

            # Batch commit
            if len(files_batch) >= batch_size:
                db.add_files_batch(files_batch)
                db.add_clip_embeddings_batch(embeddings_batch)
                files_batch = []
                embeddings_batch = []

        except Exception as e:
            error_count += 1
            if error_count <= 10:
                log(f"Error processing {path}: {e}")

    # Final batch
    if files_batch:
        db.add_files_batch(files_batch)
        db.add_clip_embeddings_batch(embeddings_batch)

    log(f"Files processed: {db.file_count()}")
    log(f"Missing files (deleted?): {missing_count}")
    log(f"Errors: {error_count}")

    # Scan directories
    directories = scan_archive_directories(ARCHIVE_ROOT)
    db.update_directories_batch(directories)
    log(f"Directories indexed: {len(directories)}")

    # Final stats
    log("")
    log("=" * 60)
    log("Migration complete!")
    log("=" * 60)
    stats = db.stats()
    log(f"Files: {stats['files']}")
    log(f"CLIP embeddings: {stats['clip_embeddings']}")
    log(f"Directories: {stats['directories']}")
    log(f"Database size: {os.path.getsize(DATABASE_PATH) / 1e6:.1f} MB")

    db.close()


if __name__ == "__main__":
    main()
