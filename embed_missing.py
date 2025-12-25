#!/usr/bin/env python3
"""
Embed missing RAW images using dcraw and add to existing Qdrant collection.
"""

import os
import subprocess
import tempfile
import torch
import open_clip
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from tqdm import tqdm
import hashlib

# Config
SAMPLE_FILE = "/Volumes/Additional Files/development/imageSearch/sample_10k_paths.txt"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "images_10k"
BATCH_SIZE = 32

RAW_EXTENSIONS = {'.nef', '.raf', '.cr2', '.cr3', '.arw', '.dng', '.raw', '.orf', '.rw2'}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

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

def is_raw_file(path):
    """Check if file is a RAW format."""
    ext = os.path.splitext(path)[1].lower()
    return ext in RAW_EXTENSIONS

def load_raw_with_dcraw(path):
    """Load RAW file using dcraw, return PIL Image."""
    try:
        # dcraw -c -w -h outputs PPM to stdout
        # -c = write to stdout
        # -w = use camera white balance
        # -h = half-size (faster, still good for CLIP's 224px)
        result = subprocess.run(
            ['dcraw', '-c', '-w', '-h', path],
            capture_output=True,
            timeout=30
        )
        if result.returncode != 0:
            return None

        # Parse PPM from stdout
        from io import BytesIO
        img = Image.open(BytesIO(result.stdout))
        return img.convert('RGB')
    except Exception as e:
        return None

def load_and_preprocess_image(path, preprocess):
    """Load and preprocess a single image, using dcraw for RAW files."""
    try:
        if is_raw_file(path):
            img = load_raw_with_dcraw(path)
            if img is None:
                return None
        else:
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
        return preprocess(img)
    except Exception as e:
        return None

def get_missing_paths(client, all_paths):
    """Find paths that aren't in the collection yet."""
    indexed_ids = set()
    offset = None
    while True:
        results, offset = client.scroll(collection_name=COLLECTION_NAME, limit=1000, offset=offset)
        for p in results:
            indexed_ids.add(p.id)
        if offset is None:
            break

    missing = []
    for path in all_paths:
        if get_image_id(path) not in indexed_ids:
            missing.append(path)
    return missing

def main():
    # Load all paths
    with open(SAMPLE_FILE, 'r') as f:
        all_paths = [line.strip() for line in f if line.strip()]
    print(f"Total paths in sample: {len(all_paths)}")

    # Setup Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if not client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' doesn't exist. Run embed_images.py first.")
        return

    info = client.get_collection(COLLECTION_NAME)
    print(f"Collection currently has {info.points_count} points")

    # Find missing paths
    print("Finding missing paths...")
    missing_paths = get_missing_paths(client, all_paths)
    print(f"Missing paths to process: {len(missing_paths)}")

    if not missing_paths:
        print("Nothing to do!")
        return

    # Load model
    print("Loading CLIP model...")
    model, preprocess = load_model()

    # Process in batches
    successful = 0
    failed = 0
    failed_paths = []

    for i in tqdm(range(0, len(missing_paths), BATCH_SIZE), desc="Embedding"):
        batch_paths = missing_paths[i:i + BATCH_SIZE]
        batch_images = []
        valid_paths = []

        for path in batch_paths:
            img = load_and_preprocess_image(path, preprocess)
            if img is not None:
                batch_images.append(img)
                valid_paths.append(path)
            else:
                failed += 1
                failed_paths.append(path)

        if not batch_images:
            continue

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

    print(f"\nDone! Embedded {successful} images, {failed} still failed")

    # Verify
    info = client.get_collection(COLLECTION_NAME)
    print(f"Collection now has {info.points_count} points")

    # Report remaining failures
    if failed_paths:
        print(f"\n{len(failed_paths)} files still failed. First few:")
        for p in failed_paths[:5]:
            print(f"  {p}")

if __name__ == "__main__":
    main()
