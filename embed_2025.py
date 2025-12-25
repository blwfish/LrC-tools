#!/usr/bin/env python3
"""
Embed 2025 images using CLIP and store in Qdrant.
Uses dcraw for NEF files since PIL can't handle Z9 NEFs.
"""

import os
import json
import subprocess
import torch
import open_clip
from PIL import Image
from io import BytesIO
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from tqdm import tqdm
import hashlib
from datetime import datetime

# Config
ARCHIVE_ROOT = "/Volumes/archive2/images/2025/"
CHECKPOINT_FILE = "/Volumes/Additional Files/development/imageSearch/checkpoint_2025.json"
LOG_FILE = "/Volumes/Additional Files/development/imageSearch/embed_2025.log"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "images_full"
BATCH_SIZE = 16
CHECKPOINT_INTERVAL = 50

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.dng', '.raw', '.tiff', '.tif', '.nef', '.png', '.raf'}
RAW_EXTENSIONS = {'.nef', '.raw', '.dng', '.raf'}  # Use dcraw for these
MAX_IMAGE_PIXELS = 200_000_000

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + "\n")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None

def save_checkpoint(processed_paths, successful, failed, failed_paths):
    checkpoint = {
        "processed_paths": list(processed_paths),
        "successful": successful,
        "failed": failed,
        "failed_paths": failed_paths[-100:],
        "timestamp": datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

def collect_all_images():
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
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14',
        pretrained='laion2b_s32b_b82k'
    )
    model = model.to(DEVICE)
    model.eval()
    return model, preprocess

def get_image_id(path):
    return int(hashlib.md5(path.encode()).hexdigest()[:15], 16)

def load_raw_via_dcraw(path, preprocess):
    """Load RAW file using dcraw."""
    try:
        result = subprocess.run(
            ['dcraw', '-c', '-w', '-h', path],
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
        return None, str(e)

def load_image_pil(path, preprocess):
    """Load image using PIL."""
    try:
        file_size = os.path.getsize(path)
        if file_size > 500_000_000:
            return None, f"File too large: {file_size / 1e9:.1f}GB"

        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        img = Image.open(path)

        width, height = img.size
        if width * height > MAX_IMAGE_PIXELS:
            return None, f"Image too large: {width}x{height}"

        if img.mode != 'RGB':
            img = img.convert('RGB')
        return preprocess(img), None
    except Exception as e:
        return None, str(e)

def load_and_preprocess_image(path, preprocess):
    """Load image, using dcraw for RAW files."""
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXTENSIONS:
        return load_raw_via_dcraw(path, preprocess)
    else:
        return load_image_pil(path, preprocess)

def main():
    log("Starting 2025 images embedding")
    log(f"Using device: {DEVICE}")

    all_paths = collect_all_images()

    if not all_paths:
        log("No images found in 2025 directory")
        return

    # Check for checkpoint
    checkpoint = load_checkpoint()
    if checkpoint:
        processed_set = set(checkpoint["processed_paths"])
        successful = checkpoint["successful"]
        failed = checkpoint["failed"]
        failed_paths = checkpoint.get("failed_paths", [])
        log(f"Resuming: {successful} successful, {failed} failed")
        paths_to_process = [p for p in all_paths if p not in processed_set]
        log(f"{len(paths_to_process)} remaining")
    else:
        processed_set = set()
        successful = 0
        failed = 0
        failed_paths = []
        paths_to_process = all_paths

    # Load model
    log("Loading CLIP model...")
    model, preprocess = load_model()

    # Setup Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if not client.collection_exists(COLLECTION_NAME):
        log(f"Collection {COLLECTION_NAME} doesn't exist! Run main embed first.")
        return

    info = client.get_collection(COLLECTION_NAME)
    log(f"Adding to collection '{COLLECTION_NAME}' with {info.points_count} existing points")

    # Process in batches
    batch_count = 0

    for i in tqdm(range(0, len(paths_to_process), BATCH_SIZE), desc="2025 Images"):
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
            batch_tensor = torch.stack(batch_images).to(DEVICE)
            with torch.no_grad():
                embeddings = model.encode_image(batch_tensor)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                embeddings = embeddings.cpu().numpy()

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
        if batch_count % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(processed_set, successful, failed, failed_paths)
            log(f"Checkpoint: {successful} successful, {failed} failed")

    save_checkpoint(processed_set, successful, failed, failed_paths)
    log(f"Done! Embedded {successful} images, {failed} failed")

    info = client.get_collection(COLLECTION_NAME)
    log(f"Collection now has {info.points_count} points")

if __name__ == "__main__":
    main()
