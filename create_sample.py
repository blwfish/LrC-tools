#!/usr/bin/env python3
"""
Create a random 10k sample of image paths from the archive.
"""

import os
import random
from pathlib import Path

ARCHIVE_ROOT = "/Volumes/archive2/images/"
OUTPUT_FILE = "/Volumes/Additional Files/development/imageSearch/sample_10k_paths.txt"
SAMPLE_SIZE = 10000

# Extensions to include (case-insensitive)
EXTENSIONS = {'.jpg', '.jpeg', '.dng', '.raw', '.tiff', '.tif', '.nef', '.png', '.raf'}

def collect_image_paths():
    """Walk the archive and collect all matching image paths."""
    image_paths = []

    print(f"Scanning {ARCHIVE_ROOT}...")

    for root, dirs, files in os.walk(ARCHIVE_ROOT):
        # Skip 2025 directories (incomplete year, would skew sample)
        if '/2025' in root or root.endswith('/2025'):
            continue
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in EXTENSIONS:
                full_path = os.path.join(root, filename)
                image_paths.append(full_path)

    return image_paths

def main():
    # Collect all image paths
    all_paths = collect_image_paths()
    total_count = len(all_paths)
    print(f"Found {total_count:,} images")

    if total_count < SAMPLE_SIZE:
        print(f"Warning: Only {total_count} images found, using all of them")
        sample = all_paths
    else:
        # Random sample
        random.seed(42)  # For reproducibility
        sample = random.sample(all_paths, SAMPLE_SIZE)

    # Write to output file
    with open(OUTPUT_FILE, 'w') as f:
        for path in sample:
            f.write(path + '\n')

    print(f"Wrote {len(sample):,} paths to {OUTPUT_FILE}")

    # Quick stats
    years = {}
    for p in sample:
        # Extract year from path if possible
        parts = p.replace(ARCHIVE_ROOT, '').split('/')
        if parts:
            year_part = parts[0][:4]
            if year_part.isdigit():
                years[year_part] = years.get(year_part, 0) + 1

    print("\nSample distribution by year prefix:")
    for year in sorted(years.keys()):
        print(f"  {year}: {years[year]:,}")

if __name__ == "__main__":
    main()
