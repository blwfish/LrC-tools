#!/usr/bin/env python3
"""
Second pass: Embed NEF files that PIL couldn't read using dcraw.
Reads failed_images.json and retries NEF files via dcraw pipeline.
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
FAILED_FILE = "/Volumes/Additional Files/development/imageSearch/failed_images.json"
CHECKPOINT_FILE = "/Volumes/Additional Files/development/imageSearch/checkpoint_nef_retry.json"
LOG_FILE = "/Volumes/Additional Files/development/imageSearch/embed_nef_retry.log"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "images_full"
BATCH_SIZE = 16  # Smaller batches since dcraw is slower
CHECKPOINT_INTERVAL = 50

# Select best available device: CUDA (NVIDIA), MPS (Apple Silicon), or CPU
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()

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

def save_checkpoint(processed_paths, successful, failed):
    checkpoint = {
        "processed_paths": list(processed_paths),
        "successful": successful,
        "failed": failed,
        "timestamp": datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

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

def load_nef_via_dcraw(path, preprocess):
    """Load NEF using dcraw, return preprocessed tensor."""
    try:
        # Use dcraw to convert to PPM, pipe to stdout
        # -c = write to stdout, -w = use camera white balance, -h = half-size (faster, still >>224px)
        result = subprocess.run(
            ['dcraw', '-c', '-w', '-h', path],
            capture_output=True,
            timeout=30
        )
        if result.returncode != 0:
            return None, f"dcraw failed: {result.stderr.decode()[:100]}"

        # Load PPM from bytes
        img = Image.open(BytesIO(result.stdout))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return preprocess(img), None
    except subprocess.TimeoutExpired:
        return None, "dcraw timeout"
    except Exception as e:
        return None, str(e)

def main():
    log("Starting NEF retry pass")
    log(f"Using device: {DEVICE}")

    # Load failed images from main run
    if not os.path.exists(FAILED_FILE):
        log(f"No failed images file found at {FAILED_FILE}")
        log("Run this after the main embedding completes.")
        return

    with open(FAILED_FILE, 'r') as f:
        failed_images = json.load(f)

    # Filter to just NEF/RAW files
    nef_extensions = {'.nef', '.raw', '.dng', '.raf'}
    nef_paths = [
        item['path'] for item in failed_images
        if os.path.splitext(item['path'])[1].lower() in nef_extensions
    ]
    log(f"Found {len(nef_paths)} NEF/RAW files to retry out of {len(failed_images)} total failures")

    if not nef_paths:
        log("No NEF files to process")
        return

    # Check for checkpoint
    checkpoint = load_checkpoint()
    if checkpoint:
        processed_set = set(checkpoint["processed_paths"])
        successful = checkpoint["successful"]
        failed = checkpoint["failed"]
        log(f"Resuming: {successful} successful, {failed} failed, {len(processed_set)} processed")
        paths_to_process = [p for p in nef_paths if p not in processed_set]
        log(f"{len(paths_to_process)} remaining")
    else:
        processed_set = set()
        successful = 0
        failed = 0
        paths_to_process = nef_paths

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

    for i in tqdm(range(0, len(paths_to_process), BATCH_SIZE), desc="NEF Retry"):
        batch_paths = paths_to_process[i:i + BATCH_SIZE]
        batch_images = []
        valid_paths = []

        for path in batch_paths:
            img, error = load_nef_via_dcraw(path, preprocess)
            if img is not None:
                batch_images.append(img)
                valid_paths.append(path)
            else:
                failed += 1
                if error:
                    log(f"Failed: {path[-60:]} - {error[:40]}")
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
            save_checkpoint(processed_set, successful, failed)
            log(f"Checkpoint: {successful} successful, {failed} failed")

    save_checkpoint(processed_set, successful, failed)
    log(f"Done! Added {successful} NEF images, {failed} still failed")

    info = client.get_collection(COLLECTION_NAME)
    log(f"Collection now has {info.points_count} points")

if __name__ == "__main__":
    main()
