#!/usr/bin/env python3
"""
Semantic Image Search HTTP Server

Flask server that provides a REST API for CLIP-based image search.
Keeps the model loaded in memory for fast response times.

Endpoints:
    POST /search - Search for images matching a text query
    GET /health - Health check endpoint
"""

import os
import json
import torch
import open_clip
from flask import Flask, request, jsonify
from qdrant_client import QdrantClient
from datetime import datetime
import traceback

# Config
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "images_full")
SERVER_PORT = int(os.environ.get("SEARCH_SERVER_PORT", 5555))

# Use MPS (Metal) on Apple Silicon
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

app = Flask(__name__)

# Global model and client (loaded once at startup)
model = None
tokenizer = None
qdrant_client = None


def load_model():
    """Load CLIP model for text encoding."""
    global model, tokenizer
    print(f"Loading CLIP model on {DEVICE}...")
    model, _, _ = open_clip.create_model_and_transforms(
        'ViT-L-14',
        pretrained='laion2b_s32b_b82k'
    )
    model = model.to(DEVICE)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    print("Model loaded.")


def encode_text(query: str) -> list:
    """Encode a text query to a vector."""
    text = tokenizer([query])
    with torch.no_grad():
        text_features = model.encode_text(text.to(DEVICE))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0].tolist()


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        return jsonify({
            "status": "healthy",
            "device": DEVICE,
            "collection": COLLECTION_NAME,
            "points_count": info.points_count
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


@app.route('/search', methods=['POST'])
def search():
    """
    Search for images matching a text query.

    Request body (JSON):
        query: str - The search query text
        limit: int - Maximum results to return (default 500)
        min_score: float - Minimum similarity threshold (default 0.0)
        paths: list[str] - Optional list of paths to constrain search to
        return_all: bool - If true, ignore limit and return all above threshold
        results_file: str - Optional path to write results JSON file

    Response (JSON):
        results: list of {path: str, score: float}
        count: int
        query: str
        elapsed_ms: float
    """
    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' in request body"}), 400

        query = data['query']
        limit = data.get('limit', 500)
        min_score = data.get('min_score', 0.0)
        constrain_paths = data.get('paths', None)
        return_all = data.get('return_all', False)
        results_file = data.get('results_file', None)

        start_time = datetime.now()

        # Encode the query
        query_vector = encode_text(query)

        # Build search parameters
        search_limit = 10000 if return_all else limit

        # If constraining to specific paths, we need to filter
        if constrain_paths and len(constrain_paths) > 0:
            # For large path lists, Qdrant's filter can be slow
            # Use a should filter with path matches
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Qdrant has limits on filter size, so batch if needed
            MAX_FILTER_PATHS = 1000

            if len(constrain_paths) <= MAX_FILTER_PATHS:
                # Direct filter
                path_filter = Filter(
                    should=[
                        FieldCondition(key="path", match=MatchValue(value=p))
                        for p in constrain_paths
                    ]
                )

                results = qdrant_client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    query_filter=path_filter,
                    limit=search_limit,
                    score_threshold=min_score if min_score > 0 else None
                )
            else:
                # For large path sets, search all and filter in Python
                # This is less efficient but handles arbitrary selection sizes
                results = qdrant_client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    limit=search_limit,
                    score_threshold=min_score if min_score > 0 else None
                )
                # Filter to only paths in the constraint set
                path_set = set(constrain_paths)
                results.points = [
                    p for p in results.points
                    if p.payload.get('path') in path_set
                ]
        else:
            # Search entire collection
            results = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=search_limit,
                score_threshold=min_score if min_score > 0 else None
            )

        # Format results
        matches = [
            {"path": p.payload.get('path'), "score": p.score}
            for p in results.points
        ]

        # Apply limit if not return_all (in case we over-fetched for filtering)
        if not return_all and len(matches) > limit:
            matches = matches[:limit]

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        response = {
            "results": matches,
            "count": len(matches),
            "query": query,
            "elapsed_ms": round(elapsed_ms, 2)
        }

        # Write results file if requested
        if results_file:
            try:
                # Ensure parent directory exists
                os.makedirs(os.path.dirname(results_file), exist_ok=True)

                # Write path -> score mapping
                path_scores = {m['path']: m['score'] for m in matches}
                with open(results_file, 'w') as f:
                    json.dump({
                        "query": query,
                        "timestamp": datetime.now().isoformat(),
                        "count": len(matches),
                        "results": path_scores
                    }, f, indent=2)
                response["results_file"] = results_file
            except Exception as e:
                response["results_file_error"] = str(e)

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def main():
    global qdrant_client

    # Load model
    load_model()

    # Connect to Qdrant with longer timeout
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)

    # Verify collection exists
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        print(f"Warning: Collection '{COLLECTION_NAME}' does not exist!")
    else:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' has {info.points_count} points")

    # Run server
    print(f"Starting search server on port {SERVER_PORT}...")
    app.run(host='127.0.0.1', port=SERVER_PORT, threaded=True)


if __name__ == "__main__":
    main()
