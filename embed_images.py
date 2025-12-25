#!/usr/bin/env python3
"""
Embed images using CLIP and store in Qdrant.
"""

import os
import torch
import open_clip
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm
import hashlib

# Config
SAMPLE_FILE = "/Volumes/Additional Files/development/imageSearch/sample_10k_paths.txt"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "images_10k"
BATCH_SIZE = 32

# Select best available device: CUDA (NVIDIA), MPS (Apple Silicon), or CPU
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()
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

def load_and_preprocess_image(path, preprocess):
    """Load and preprocess a single image, handling various formats."""
    try:
        img = Image.open(path)
        # Convert to RGB (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return preprocess(img)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def main():
    # Load paths
    with open(SAMPLE_FILE, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(paths)} image paths")

    # Load model
    print("Loading CLIP model...")
    model, preprocess = load_model()

    # Get embedding dimension
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(DEVICE)
        dim = model.encode_image(dummy).shape[1]
    print(f"Embedding dimension: {dim}")

    # Setup Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Recreate collection
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )
    print(f"Created collection '{COLLECTION_NAME}'")

    # Process in batches
    successful = 0
    failed = 0

    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="Embedding"):
        batch_paths = paths[i:i + BATCH_SIZE]
        batch_images = []
        valid_paths = []

        for path in batch_paths:
            img = load_and_preprocess_image(path, preprocess)
            if img is not None:
                batch_images.append(img)
                valid_paths.append(path)
            else:
                failed += 1

        if not batch_images:
            continue

        # Stack and encode
        batch_tensor = torch.stack(batch_images).to(DEVICE)
        with torch.no_grad():
            embeddings = model.encode_image(batch_tensor)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # Normalize
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

    print(f"\nDone! Embedded {successful} images, {failed} failed")

    # Verify
    info = client.get_collection(COLLECTION_NAME)
    print(f"Collection has {info.points_count} points")

if __name__ == "__main__":
    main()
