#!/usr/bin/env python3
"""
Semantic image search using CLIP embeddings in Qdrant.
"""

import sys
import torch
import open_clip
from qdrant_client import QdrantClient

# Config
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "images_10k"

# Select best available device: CUDA (NVIDIA), MPS (Apple Silicon), or CPU
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()

def load_model():
    """Load CLIP model for text encoding."""
    model, _, _ = open_clip.create_model_and_transforms(
        'ViT-L-14',
        pretrained='laion2b_s32b_b82k'
    )
    model = model.to(DEVICE)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    return model, tokenizer

def search(query: str, model, tokenizer, client, top_k: int = 10):
    """Search for images matching the text query."""
    # Encode query
    text = tokenizer([query])
    with torch.no_grad():
        text_features = model.encode_text(text.to(DEVICE))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        query_vector = text_features.cpu().numpy()[0].tolist()

    # Search Qdrant
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )

    return results.points

def main():
    print("Loading CLIP model...")
    model, tokenizer = load_model()

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Check collection exists
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' not found. Run embed_images.py first.")
        sys.exit(1)

    info = client.get_collection(COLLECTION_NAME)
    print(f"Connected to '{COLLECTION_NAME}' with {info.points_count} images")
    print("\nEnter search queries (Ctrl+C to exit):\n")

    while True:
        try:
            query = input("Search: ").strip()
            if not query:
                continue

            results = search(query, model, tokenizer, client)

            print(f"\nTop {len(results)} results for '{query}':\n")
            for i, result in enumerate(results, 1):
                path = result.payload.get('path', 'unknown')
                score = result.score
                # Extract just the meaningful part of the path
                short_path = path.replace('/Volumes/archive2/images/', '')
                print(f"  {i:2}. [{score:.3f}] {short_path}")
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
