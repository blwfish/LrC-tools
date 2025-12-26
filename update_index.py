#!/usr/bin/env python3
"""
Incremental update for the image search index.

Usage:
    python update_index.py              # Index new/moved files
    python update_index.py --cleanup    # Also remove orphaned entries
    python update_index.py --year 2025  # Only scan specific year
    python update_index.py --dry-run    # Show what would happen
    python update_index.py --verbose    # Detailed output
    python update_index.py --full-scan  # Ignore directory mtimes, scan everything

Workflow:
1. Scan directories that have changed since last run (using mtime)
2. For each file in changed directories:
   - Compute content hash
   - If hash exists elsewhere in DB → file was moved, update path
   - If path not in DB → new file, embed it
3. Optionally remove orphaned entries (--cleanup)
"""

import argparse
import hashlib
import os
import sys
from datetime import datetime

import torch
import open_clip
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from tqdm import tqdm

# Add parent directory to path for common module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
    ARCHIVE_ROOT,
    DATABASE_PATH,
    Database,
    content_hash,
    find_changed_directories,
    get_device,
    load_and_preprocess_image,
    scan_directory_for_images,
)
from common.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME
from common.files import get_file_info


# Globals set by args
VERBOSE = False
DRY_RUN = False


def log(msg: str, verbose_only: bool = False):
    """Log with timestamp."""
    if verbose_only and not VERBOSE:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


def get_point_id(path: str) -> int:
    """Generate a stable numeric ID from path."""
    return int(hashlib.md5(path.encode()).hexdigest()[:15], 16)


def load_clip_model():
    """Load CLIP model for embedding."""
    device = get_device()
    log(f"Loading CLIP model on {device}...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14',
        pretrained='laion2b_s32b_b82k'
    )
    model = model.to(device)
    model.eval()

    return model, preprocess, device


def process_new_files(
    new_files: list,
    db: Database,
    client: QdrantClient,
    model,
    preprocess,
    device: str
) -> tuple:
    """Embed new files and add to Qdrant.

    Returns:
        (success_count, fail_count)
    """
    if not new_files:
        return 0, 0

    if DRY_RUN:
        log(f"Would embed {len(new_files)} new files")
        return len(new_files), 0

    success = 0
    failed = 0
    batch_size = 32
    files_batch = []
    embeddings_batch = []
    points_batch = []

    for i in tqdm(range(0, len(new_files), batch_size), desc="Embedding new files"):
        batch_paths = new_files[i:i + batch_size]
        batch_images = []
        valid_paths = []
        valid_infos = []

        for path, file_hash, file_info in batch_paths:
            img, error = load_and_preprocess_image(path, preprocess)
            if img is not None:
                batch_images.append(img)
                valid_paths.append(path)
                valid_infos.append((file_hash, file_info))
            else:
                log(f"Failed to load {path}: {error}", verbose_only=True)
                failed += 1

        if not batch_images:
            continue

        # Stack and encode
        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            embeddings = model.encode_image(batch_tensor)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.cpu().numpy()

        # Prepare batches
        for path, emb, (file_hash, file_info) in zip(valid_paths, embeddings, valid_infos):
            point_id = get_point_id(path)

            files_batch.append((
                path,
                file_hash,
                file_info['file_size'],
                file_info['file_mtime'],
                file_info['extension']
            ))
            embeddings_batch.append((path, point_id))
            points_batch.append(PointStruct(
                id=point_id,
                vector=emb.tolist(),
                payload={"path": path}
            ))
            success += 1

        # Upload to Qdrant in batches
        if len(points_batch) >= 100:
            client.upsert(collection_name=COLLECTION_NAME, points=points_batch)
            db.add_files_batch(files_batch)
            db.add_clip_embeddings_batch(embeddings_batch)
            points_batch = []
            files_batch = []
            embeddings_batch = []

    # Final batch
    if points_batch:
        client.upsert(collection_name=COLLECTION_NAME, points=points_batch)
        db.add_files_batch(files_batch)
        db.add_clip_embeddings_batch(embeddings_batch)

    return success, failed


def process_moved_files(moved_files: list, db: Database, client: QdrantClient) -> int:
    """Update paths for moved files in DB and Qdrant.

    Returns:
        Number of files updated
    """
    if not moved_files:
        return 0

    if DRY_RUN:
        log(f"Would update {len(moved_files)} moved files")
        for old_path, new_path, _ in moved_files[:5]:
            log(f"  {old_path} -> {new_path}", verbose_only=True)
        return len(moved_files)

    count = 0
    for old_path, new_path, file_info in tqdm(moved_files, desc="Updating moved files"):
        # Update database
        db.update_file_path(old_path, new_path, file_info['file_mtime'])

        # Update Qdrant payload
        old_point_id = get_point_id(old_path)
        new_point_id = get_point_id(new_path)

        # If point ID changes (it will, since it's based on path), we need to:
        # 1. Read the old vector
        # 2. Delete old point
        # 3. Insert with new ID and path
        try:
            points = client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[old_point_id],
                with_vectors=True
            )
            if points:
                vector = points[0].vector
                client.delete(collection_name=COLLECTION_NAME, points_selector=[old_point_id])
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[PointStruct(
                        id=new_point_id,
                        vector=vector,
                        payload={"path": new_path}
                    )]
                )
                # Update the point_id in our DB
                db.conn.execute(
                    "UPDATE clip_embeddings SET point_id = ? WHERE path = ?",
                    (new_point_id, new_path)
                )
                db.conn.commit()
                count += 1
        except Exception as e:
            log(f"Error updating moved file {old_path}: {e}")

    return count


def cleanup_orphans(db: Database, client: QdrantClient) -> int:
    """Remove entries for files that no longer exist.

    Returns:
        Number of orphans removed
    """
    log("Checking for orphaned entries...")

    all_paths = db.get_all_paths()
    orphans = []

    for path in tqdm(all_paths, desc="Checking files"):
        if not os.path.exists(path):
            orphans.append(path)

    if not orphans:
        log("No orphaned entries found")
        return 0

    log(f"Found {len(orphans)} orphaned entries")

    if DRY_RUN:
        log("Would remove orphaned entries:")
        for path in orphans[:10]:
            log(f"  {path}")
        if len(orphans) > 10:
            log(f"  ... and {len(orphans) - 10} more")
        return len(orphans)

    # Delete from Qdrant
    point_ids = [get_point_id(path) for path in orphans]
    for i in range(0, len(point_ids), 100):
        batch = point_ids[i:i + 100]
        client.delete(collection_name=COLLECTION_NAME, points_selector=batch)

    # Delete from database
    for path in orphans:
        db.delete_file(path)

    log(f"Removed {len(orphans)} orphaned entries")
    return len(orphans)


def main():
    global VERBOSE, DRY_RUN

    parser = argparse.ArgumentParser(description="Incremental update for image search index")
    parser.add_argument('--cleanup', action='store_true', help="Remove orphaned entries")
    parser.add_argument('--year', type=str, help="Only scan specific year directory")
    parser.add_argument('--dry-run', action='store_true', help="Show what would happen")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose output")
    parser.add_argument('--full-scan', action='store_true', help="Ignore directory mtimes")
    args = parser.parse_args()

    VERBOSE = args.verbose
    DRY_RUN = args.dry_run

    log("=" * 60)
    log("Image Search Index Update")
    log("=" * 60)

    if DRY_RUN:
        log("DRY RUN - no changes will be made")

    # Check database exists
    if not os.path.exists(DATABASE_PATH):
        log(f"ERROR: Database not found at {DATABASE_PATH}")
        log("Run migrate_to_catalog.py first to create the database")
        return

    # Connect to database and Qdrant
    db = Database(DATABASE_PATH)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if not client.collection_exists(COLLECTION_NAME):
        log(f"ERROR: Collection '{COLLECTION_NAME}' does not exist in Qdrant")
        db.close()
        return

    # Determine archive root
    archive_root = ARCHIVE_ROOT
    if args.year:
        archive_root = os.path.join(ARCHIVE_ROOT, args.year)
        if not os.path.exists(archive_root):
            log(f"ERROR: Year directory not found: {archive_root}")
            db.close()
            return
        log(f"Scanning only: {archive_root}")

    # Find changed directories
    if args.full_scan:
        log("Full scan mode - checking all directories")
        stored_mtimes = {}
    else:
        stored_mtimes = db.get_all_directories()

    changed_dirs = find_changed_directories(archive_root, stored_mtimes)
    log(f"Found {len(changed_dirs)} changed directories")

    if not changed_dirs and not args.cleanup:
        log("No changes detected. Use --full-scan to force a complete scan.")
        db.close()
        return

    # Collect files from changed directories
    new_files = []  # (path, hash, file_info)
    moved_files = []  # (old_path, new_path, file_info)
    already_indexed = 0
    updated_dirs = []

    indexed_paths = db.get_all_clip_paths()

    log("Scanning changed directories...")
    for directory in tqdm(changed_dirs, desc="Scanning directories"):
        dir_mtime = os.stat(directory).st_mtime
        updated_dirs.append((directory, dir_mtime))

        for path in scan_directory_for_images(directory):
            # Skip if already indexed at this path
            if path in indexed_paths:
                already_indexed += 1
                continue

            try:
                file_info = get_file_info(path)
                file_hash = content_hash(path)

                # Check if this file exists elsewhere (moved)
                existing = db.get_file_by_hash(file_hash)
                if existing and existing['path'] != path:
                    # File was moved
                    moved_files.append((existing['path'], path, file_info))
                    log(f"Detected move: {existing['path']} -> {path}", verbose_only=True)
                else:
                    # New file
                    new_files.append((path, file_hash, file_info))

            except Exception as e:
                log(f"Error processing {path}: {e}", verbose_only=True)

    log(f"Already indexed: {already_indexed}")
    log(f"New files: {len(new_files)}")
    log(f"Moved files: {len(moved_files)}")

    # Process moved files (no embedding needed)
    if moved_files:
        moved_count = process_moved_files(moved_files, db, client)
        log(f"Updated {moved_count} moved files")

    # Process new files (need embedding)
    if new_files:
        model, preprocess, device = load_clip_model()
        success, failed = process_new_files(new_files, db, client, model, preprocess, device)
        log(f"Embedded {success} new files, {failed} failed")

    # Update directory mtimes
    if not DRY_RUN and updated_dirs:
        db.update_directories_batch(updated_dirs)
        log(f"Updated {len(updated_dirs)} directory timestamps")

    # Cleanup orphans if requested
    if args.cleanup:
        cleanup_orphans(db, client)

    # Final stats
    log("")
    log("=" * 60)
    log("Update complete!")
    log("=" * 60)
    stats = db.stats()
    log(f"Files in catalog: {stats['files']}")
    log(f"CLIP embeddings: {stats['clip_embeddings']}")

    db.close()


if __name__ == "__main__":
    main()
