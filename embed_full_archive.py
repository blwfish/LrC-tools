#!/usr/bin/env python3
"""
Embed full image archive using CLIP and store in Qdrant.
Supports checkpointing and resume after interruption.
"""

import os
import json
import subprocess
import torch
import open_clip
from PIL import Image
from io import BytesIO
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm
import hashlib
from pathlib import Path
from datetime import datetime

# Config
ARCHIVE_ROOT = "/Volumes/archive2/images/"
CHECKPOINT_FILE = "/Volumes/Additional Files/development/imageSearch/checkpoint_full.json"
LOG_FILE = "/Volumes/Additional Files/development/imageSearch/embed_full.log"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "images_full"
BATCH_SIZE = 32
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N batches

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.dng', '.raw', '.tiff', '.tif', '.nef', '.png', '.raf', '.cr2', '.cr3', '.arw', '.orf', '.rw2'}
RAW_EXTENSIONS = {'.nef', '.raw', '.dng', '.raf', '.cr2', '.cr3', '.arw', '.orf', '.rw2'}  # Use dcraw fallback for these
MAX_IMAGE_PIXELS = 200_000_000  # Skip images over 200MP to avoid memory issues

# Select best available device: CUDA (NVIDIA), MPS (Apple Silicon), or CPU
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()

def log(msg):
    """Log to both console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + "\n")

def load_checkpoint():
    """Load checkpoint if exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None

def save_checkpoint(processed_paths, successful, failed, failed_paths):
    """Save checkpoint to resume later."""
    checkpoint = {
        "processed_paths": list(processed_paths),
        "successful": successful,
        "failed": failed,
        "failed_paths": failed_paths[-100:],  # Keep last 100 failures for debugging
        "timestamp": datetime.now().isoformat(),
        "collection_name": COLLECTION_NAME
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

def collect_all_images():
    """Walk archive and collect all image paths."""
    log(f"Scanning {ARCHIVE_ROOT} for images...")
    paths = []
    for root, dirs, files in os.walk(ARCHIVE_ROOT):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                paths.append(os.path.join(root, fname))
    log(f"Found {len(paths)} images")
    return paths

def load_model():
    """Load CLIP model and preprocessing."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14',
        pretrained='laion2b_s32b_b82k'
    )
    model = model.to(DEVICE)
    model.eval()
    return model, preprocess

def get_image_id(path):
    """Generate a stable numeric ID from path."""
    return int(hashlib.md5(path.encode()).hexdigest()[:15], 16)

def load_via_dcraw(path, preprocess):
    """Load RAW file using dcraw - extract embedded preview first, demosaic as fallback."""
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
            return preprocess(img), None
    except:
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
        return preprocess(img), None
    except subprocess.TimeoutExpired:
        return None, "dcraw timeout"
    except Exception as e:
        return None, f"dcraw error: {str(e)}"

def load_via_pil(path, preprocess):
    """Load image using PIL."""
    try:
        file_size = os.path.getsize(path)
        if file_size > 500_000_000:  # 500MB file size limit
            return None, f"File too large: {file_size / 1e9:.1f}GB"

        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        img = Image.open(path)

        width, height = img.size
        if width * height > MAX_IMAGE_PIXELS:
            return None, f"Image too large: {width}x{height} ({width*height/1e6:.0f}MP)"

        if img.mode != 'RGB':
            img = img.convert('RGB')
        return preprocess(img), None
    except Exception as e:
        return None, str(e)

def load_and_preprocess_image(path, preprocess):
    """Load and preprocess image. Try PIL first, fall back to dcraw for RAW files."""
    ext = os.path.splitext(path)[1].lower()

    # Try PIL first (fast for JPEGs, works for many formats)
    result, error = load_via_pil(path, preprocess)
    if result is not None:
        return result, None

    # For RAW files, try dcraw as fallback
    if ext in RAW_EXTENSIONS:
        return load_via_dcraw(path, preprocess)

    # Not a RAW file and PIL failed
    return None, error

def main():
    log(f"Starting full archive embedding")
    log(f"Using device: {DEVICE}")

    # Collect all image paths
    all_paths = collect_all_images()

    # Check for checkpoint
    checkpoint = load_checkpoint()
    if checkpoint and checkpoint.get("collection_name") == COLLECTION_NAME:
        processed_set = set(checkpoint["processed_paths"])
        successful = checkpoint["successful"]
        failed = checkpoint["failed"]
        failed_paths = checkpoint.get("failed_paths", [])
        log(f"Resuming from checkpoint: {successful} successful, {failed} failed, {len(processed_set)} processed")
        paths_to_process = [p for p in all_paths if p not in processed_set]
        log(f"{len(paths_to_process)} images remaining")
    else:
        processed_set = set()
        successful = 0
        failed = 0
        failed_paths = []
        paths_to_process = all_paths
        log("Starting fresh (no valid checkpoint found)")

    # Load model
    log("Loading CLIP model...")
    model, preprocess = load_model()

    # Get embedding dimension
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(DEVICE)
        dim = model.encode_image(dummy).shape[1]
    log(f"Embedding dimension: {dim}")

    # Setup Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Create collection if needed (don't delete if resuming!)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        log(f"Created collection '{COLLECTION_NAME}'")
    else:
        info = client.get_collection(COLLECTION_NAME)
        log(f"Using existing collection '{COLLECTION_NAME}' with {info.points_count} points")

    # Process in batches
    batch_count = 0

    for i in tqdm(range(0, len(paths_to_process), BATCH_SIZE), desc="Embedding"):
        batch_paths = paths_to_process[i:i + BATCH_SIZE]
        batch_images = []
        valid_paths = []

        for path in batch_paths:
            img, error = load_and_preprocess_image(path, preprocess)
            if img is not None:
                batch_images.append(img)
                valid_paths.append(path)
            else:
                failed += 1
                failed_paths.append({"path": path, "error": error})
            processed_set.add(path)

        if batch_images:
            # Stack and encode
            batch_tensor = torch.stack(batch_images).to(DEVICE)
            with torch.no_grad():
                embeddings = model.encode_image(batch_tensor)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                embeddings = embeddings.cpu().numpy()

            # Upload to Qdrant
            points = [
                PointStruct(
                    id=get_image_id(path),
                    vector=emb.tolist(),
                    payload={"path": path}
                )
                for path, emb in zip(valid_paths, embeddings)
            ]

            client.upsert(collection_name=COLLECTION_NAME, points=points)
            successful += len(points)

        batch_count += 1

        # Checkpoint periodically
        if batch_count % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(processed_set, successful, failed, failed_paths)
            log(f"Checkpoint saved: {successful} successful, {failed} failed")

    # Final checkpoint
    save_checkpoint(processed_set, successful, failed, failed_paths)

    log(f"Done! Embedded {successful} images, {failed} failed")

    # Verify
    info = client.get_collection(COLLECTION_NAME)
    log(f"Collection has {info.points_count} points")

    # Write failed paths to separate file for review
    if failed_paths:
        failed_file = "/Volumes/Additional Files/development/imageSearch/failed_images.json"
        with open(failed_file, 'w') as f:
            json.dump(failed_paths, f, indent=2)
        log(f"Failed image details written to {failed_file}")

if __name__ == "__main__":
    main()
