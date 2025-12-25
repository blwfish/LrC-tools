# Semantic Image Search for Lightroom Classic

Search your photo library using natural language queries powered by CLIP embeddings and Qdrant vector database.

Instead of searching by keywords or metadata, describe what you're looking for: "red race car at sunset", "person holding coffee", "snowy mountain landscape" - and find visually matching images.

## Features

- **Natural language search** - Describe images in plain English
- **Fast** - Sub-second search across 500k+ images after initial model load
- **Selection-aware** - Search within selected photos or entire catalog
- **LrC integration** - Results appear as collections in Lightroom Classic
- **GPU accelerated** - CUDA (NVIDIA) or MPS (Apple Silicon)

## Requirements

### Hardware

| Use Case | Minimum | Recommended |
|----------|---------|-------------|
| Query only (pre-indexed) | 16GB RAM + GPU | 24GB RAM (LrC uses 8-12GB) |
| Indexing + query | 24GB RAM | 32GB RAM |
| Indexing while using LrC+PS | 32GB RAM | 48GB+ RAM |

**Supported GPUs:**
- **macOS**: Apple Silicon (M1/M2/M3/M4) - uses MPS
- **Windows/Linux**: NVIDIA GPU (RTX 3060+) - uses CUDA
- **CPU fallback**: Works but very slow, not recommended

### Software

- Python 3.10+
- [Qdrant](https://qdrant.tech/) vector database (via Docker or native)
- Adobe Lightroom Classic
- **Windows/Linux**: NVIDIA CUDA toolkit

## Architecture

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Lightroom Classic  │────▶│  Search Server   │────▶│   Qdrant    │
│  (Lua Plugin)       │     │  (Flask + CLIP)  │     │  (Vectors)  │
└─────────────────────┘     └──────────────────┘     └─────────────┘
                                    │
                                    ▼
                            ┌──────────────────┐
                            │  CLIP ViT-L-14   │
                            │  (Text Encoder)  │
                            └──────────────────┘
```

## Installation

### 1. Install Qdrant

```bash
# Using Docker (recommended)
docker pull qdrant/qdrant
docker run -p 6333:6333 -v ~/qdrant_data:/qdrant/storage qdrant/qdrant

# Or using Homebrew
brew install qdrant
qdrant
```

### 2. Clone and set up Python environment

```bash
git clone https://github.com/blwfish/LrC-tools.git
cd LrC-tools

python3 -m venv venv
source venv/bin/activate
pip install torch torchvision open-clip-torch qdrant-client flask pillow tqdm
```

### 3. Index your images

Edit `embed_full_archive.py` to set your archive path:

```python
ARCHIVE_ROOT = "/path/to/your/images/"
```

Then run the indexing:

```bash
source venv/bin/activate
python embed_full_archive.py
```

This will take several hours for large libraries (500k images ≈ 15-20 hours on M4 Max).

**For Nikon Z9 NEF files** (or other newer RAW formats PIL can't read):
```bash
# After main indexing completes
python embed_nef_retry.py
```

### 4. Install the search server as a service

```bash
# Copy launchd plist
cp com.blw.imagesearch.plist ~/Library/LaunchAgents/

# Edit paths in the plist if needed
nano ~/Library/LaunchAgents/com.blw.imagesearch.plist

# Load the service
launchctl load ~/Library/LaunchAgents/com.blw.imagesearch.plist
```

### 5. Install the Lightroom plugin

```bash
# Symlink plugin to LrC Modules folder
ln -s "$(pwd)/SemanticSearch.lrplugin" \
    ~/Library/Application\ Support/Adobe/Lightroom/Modules/SemanticSearch.lrplugin
```

Or manually add via Lightroom: **File > Plug-in Manager > Add**

### 6. Restart Lightroom Classic

## Usage

1. Open Lightroom Classic Library module
2. **Library > Plug-in Extras > Semantic Search...**
3. Enter a natural language query (e.g., "blue porsche on race track")
4. Adjust parameters if desired:
   - **Max results**: Limit number of results (default: 500)
   - **Return all matches**: Ignore limit, return everything above threshold
   - **Min similarity**: Minimum score threshold (default: 0.20)
   - **Search all photos**: Override selection and search entire catalog
5. Click **Search**
6. Results appear in a new collection under `0_Semantic_Searches`

### Tips

- CLIP similarity scores typically range 0.20-0.35 for good matches
- Lower the min similarity threshold for broader results
- Select photos first to constrain search to a subset
- Searches are fast (~50-100ms) after the model is loaded

## Managing the Service

```bash
# Check status
launchctl list | grep imagesearch

# View logs
tail -f /path/to/LrC-tools/search_server.log

# Restart
launchctl unload ~/Library/LaunchAgents/com.blw.imagesearch.plist
launchctl load ~/Library/LaunchAgents/com.blw.imagesearch.plist

# Stop
launchctl unload ~/Library/LaunchAgents/com.blw.imagesearch.plist
```

## Health Check

```bash
# Check server status
curl http://localhost:5555/health

# Test search from command line
curl -X POST http://localhost:5555/search \
  -H "Content-Type: application/json" \
  -d '{"query": "sunset over ocean", "limit": 10}'
```

## Configuration

### Search Server (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | localhost | Qdrant server host |
| `QDRANT_PORT` | 6333 | Qdrant server port |
| `COLLECTION_NAME` | images_full | Qdrant collection name |
| `SEARCH_SERVER_PORT` | 5555 | HTTP server port |

### Plugin (Config.lua)

| Setting | Default | Description |
|---------|---------|-------------|
| `DEFAULT_MAX_RESULTS` | 500 | Default result limit |
| `DEFAULT_MIN_SCORE` | 0.20 | Default similarity threshold |
| `COLLECTION_SET_NAME` | 0_Semantic_Searches | Collection set for results |

## Files

| File | Description |
|------|-------------|
| `search_server.py` | Flask HTTP server with CLIP model |
| `embed_full_archive.py` | Main indexing script |
| `embed_nef_retry.py` | Retry failed NEFs using dcraw |
| `embed_2025.py` | Index newer images with dcraw support |
| `search.py` | CLI search tool (standalone) |
| `com.blw.imagesearch.plist` | launchd service configuration |
| `SemanticSearch.lrplugin/` | Lightroom Classic plugin |

## Troubleshooting

### "Failed to connect to search server"
- Check if server is running: `curl http://localhost:5555/health`
- Start manually: `python search_server.py`
- Check logs for errors

### "Server error: timed out"
- Qdrant may be overloaded or indexing
- Restart search server
- Check Qdrant status: `curl http://localhost:6333/collections/images_full`

### "No matching images found"
- Check if images are indexed: compare Qdrant point count to catalog size
- Images may not be in the indexed paths
- Try lowering min similarity threshold

### Collection set error on first search
- This was fixed - update to latest version
- If persists, manually create `0_Semantic_Searches` collection set in LrC

## Performance

Tested on M4 Max with 500k images:

| Operation | Time |
|-----------|------|
| Model load (cold start) | ~5-6 seconds |
| Search query | 50-100ms |
| Indexing | ~3-4 images/second |

## License

MIT

## Credits

- [OpenCLIP](https://github.com/mlfoundations/open_clip) - CLIP implementation
- [Qdrant](https://qdrant.tech/) - Vector database
- [Flask](https://flask.palletsprojects.com/) - HTTP server
