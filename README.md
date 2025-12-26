# Semantic Image Search for Lightroom Classic

**Find photos by describing them in plain English.**

Instead of scrolling through thousands of images or relying on keywords you remembered to add, just type what you're looking for:

- "red Porsche at sunset"
- "person laughing at camera"
- "foggy mountain landscape"
- "crowd cheering at race track"

The system understands the *visual content* of your images, not just their filenames or metadata.

## What This Does

This tool adds a search dialog to Lightroom Classic. You type a description, and it finds matching photos from your library. Results appear as a collection you can browse, rate, or export like any other.

It works by analyzing each image and creating a mathematical "fingerprint" of its visual content. When you search, it compares your description against all those fingerprints to find the best matches. This happens in under a second, even with hundreds of thousands of images.

## Requirements

### Your Computer

- **Mac with Apple Silicon** (M1, M2, M3, or M4) - strongly recommended
- At least **24GB of RAM** if you want to use Lightroom while searching
- Enough disk space for the search index (roughly 1GB per 100,000 images)

Windows and Linux with NVIDIA graphics cards also work, but this guide focuses on Mac.

### Your Photo Library

- Images must be accessible on a mounted drive (internal, external, or NAS)
- Supports JPEG, TIFF, PNG, and most RAW formats (NEF, DNG, CR2, CR3, ARW, RAF, ORF, etc.)
- The system reads your actual image files, not Lightroom's previews

## How It Works (The Simple Version)

Three pieces work together:

1. **The Index** - A database containing the visual "fingerprint" of each image
2. **The Search Server** - A background process that handles search requests
3. **The Lightroom Plugin** - Adds the search dialog to Lightroom

You build the index once (this takes a while). After that, searching is nearly instant.

---

## Setup Guide

This requires some work in Terminal. Don't worry - you'll copy and paste most commands, and I'll explain what each one does.

### Step 1: Install the Required Tools

Open **Terminal** (find it in Applications > Utilities, or search for it in Spotlight).

First, install Homebrew if you don't have it. This is a tool that makes installing other software easier:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow any instructions it gives you. You may need to enter your password.

Now install the tools we need:

```bash
brew install python@3.11 dcraw qdrant
```

This installs:
- **Python** - A programming language that runs the search system
- **dcraw** - A tool for reading RAW camera files
- **Qdrant** - The database that stores image fingerprints

### Step 2: Download This Project

Decide where you want to keep this project. Your home folder is fine:

```bash
cd ~
git clone https://github.com/blwfish/LrC-tools.git
cd LrC-tools
```

(If you received this as a folder instead of from GitHub, just navigate to that folder in Terminal using `cd /path/to/folder`.)

### Step 3: Set Up Python

Create an isolated Python environment for this project:

```bash
python3 -m venv venv
source venv/bin/activate
```

Your terminal prompt should now show `(venv)` at the beginning.

Install the required Python packages:

```bash
pip install torch torchvision open-clip-torch qdrant-client flask pillow tqdm
```

This downloads about 2GB of files. It may take a few minutes.

### Step 4: Build the Image Index

This is the slow part. The system needs to analyze every image in your library and create its fingerprint. For 100,000 images, expect 8-10 hours. For 500,000 images, expect 2-3 days.

You only do this once. After that, you can add new images incrementally.

**Start the fingerprint database:**

Open a new Terminal window and run:

```bash
qdrant
```

Leave this window open. Qdrant needs to stay running while you build the index.

**Configure and run the indexer:**

Back in your first Terminal window, edit the indexing script to point to your images:

```bash
nano embed_full_archive.py
```

Find this line near the top:

```python
ARCHIVE_ROOT = "/Volumes/archive2/images/"
```

Change the path to wherever your images are stored. Press **Ctrl+O** to save, then **Ctrl+X** to exit.

Now start the indexing:

```bash
source venv/bin/activate
nohup python embed_full_archive.py >> embed_full.log 2>&1 &
```

This runs in the background. You can close Terminal, restart your computer, whatever - it will keep going.

**Check progress:**

```bash
tail -f embed_full.log
```

Press **Ctrl+C** to stop watching the log.

### Step 5: Set Up Automatic Startup

You want the search server and database to start automatically when you log in.

**For Qdrant**, create a startup file:

```bash
nano ~/Library/LaunchAgents/com.qdrant.plist
```

Paste this content:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.qdrant</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/qdrant</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

Save and exit (**Ctrl+O**, **Ctrl+X**).

**For the search server**, the project includes a startup file. Copy it and adjust the paths:

```bash
cp com.blw.imagesearch.plist ~/Library/LaunchAgents/
nano ~/Library/LaunchAgents/com.blw.imagesearch.plist
```

Update any paths to match where you installed the project.

**Enable both services:**

```bash
launchctl load ~/Library/LaunchAgents/com.qdrant.plist
launchctl load ~/Library/LaunchAgents/com.blw.imagesearch.plist
```

### Step 6: Install the Lightroom Plugin

Create a link from Lightroom's plugin folder to this project:

```bash
ln -s "$(pwd)/SemanticSearch.lrplugin" \
    ~/Library/Application\ Support/Adobe/Lightroom/Modules/SemanticSearch.lrplugin
```

Restart Lightroom Classic.

---

## Using the Search

1. Open Lightroom Classic and go to the **Library** module
2. From the menu: **Library > Plug-in Extras > Semantic Search...**
3. Type what you're looking for
4. Click **Search**
5. Results appear in a collection under **0_Semantic_Searches**

### Search Tips

- **Be specific**: "yellow Corvette on race track" works better than "car"
- **Describe the scene**: "crowd watching fireworks at night" finds those moments
- **Include context**: "person standing on mountain summit with clouds below"
- **Try variations**: If "sunset" doesn't find what you want, try "orange sky" or "golden hour"

### Understanding Results

Each result has a similarity score between 0 and 1. In practice:
- **0.30+** = Strong match, very likely what you're looking for
- **0.25-0.30** = Good match, worth reviewing
- **0.20-0.25** = Possible match, might be relevant
- **Below 0.20** = Weak match, probably not what you want

The default threshold is 0.20. You can adjust this in the search dialog.

---

## Keeping Your Index Updated

When you add new images to your library, you'll need to index them. The simplest approach:

1. Run the indexer again - it will skip images already in the database
2. Or create a script that indexes just your new folder

(A more sophisticated incremental update system is on the roadmap.)

---

## Troubleshooting

### "Failed to connect to search server"

The search server isn't running. Try:

```bash
# Check if it's running
curl http://localhost:5555/health

# If not, start it manually
cd ~/LrC-tools
source venv/bin/activate
python search_server.py
```

### "No matching images found"

- Your images might not be indexed yet (check if indexing is complete)
- The images might be stored in a different location than what was indexed
- Try a more general search term

### Search is slow

The first search after starting the server takes 5-10 seconds while the AI model loads into memory. Subsequent searches should be under a second.

### Lightroom doesn't show the plugin

- Make sure you restarted Lightroom after installing
- Check **File > Plug-in Manager** - the plugin should appear there
- Verify the symbolic link exists: `ls -la ~/Library/Application\ Support/Adobe/Lightroom/Modules/`

---

## Technical Details (For the Curious)

This system uses **CLIP** (Contrastive Language-Image Pre-training), an AI model developed by OpenAI that understands both images and text. It was trained on hundreds of millions of image-caption pairs from the internet.

When you index an image, CLIP converts it into a list of 768 numbers (a "vector") that captures its visual essence. When you search, CLIP converts your text query into the same kind of vector. Finding matches is just finding which image vectors are closest to your query vector.

The vector database (Qdrant) is optimized for exactly this kind of "find the nearest neighbors" search, which is why it can search hundreds of thousands of images in milliseconds.

---

## Credits

- **OpenCLIP** - Open source CLIP implementation
- **Qdrant** - Vector database
- **dcraw** - RAW file decoder by Dave Coffin
