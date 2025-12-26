"""
SQLite database for tracking indexed files.
Shared by imageSearch and racing_tagger.
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import DATABASE_PATH


class Database:
    """SQLite catalog for tracking indexed files."""

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self._ensure_directory()
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _ensure_directory(self):
        """Create database directory if it doesn't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def _create_tables(self):
        """Create tables if they don't exist."""
        self.conn.executescript("""
            -- Core file tracking
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                file_size INTEGER,
                file_mtime REAL,
                extension TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_content_hash ON files(content_hash);

            -- Directory mtime tracking for fast scanning
            CREATE TABLE IF NOT EXISTS directories (
                path TEXT PRIMARY KEY,
                mtime REAL,
                last_scanned TEXT
            );

            -- CLIP embedding state
            CREATE TABLE IF NOT EXISTS clip_embeddings (
                path TEXT PRIMARY KEY,
                point_id INTEGER,
                indexed_at TEXT,
                FOREIGN KEY (path) REFERENCES files(path) ON DELETE CASCADE
            );

            -- Racing tagger state (for LrC-classification)
            CREATE TABLE IF NOT EXISTS racing_tags (
                path TEXT PRIMARY KEY,
                tagged_at TEXT,
                xmp_written_at TEXT,
                series_profile TEXT,
                keywords TEXT,  -- JSON array of keywords
                metadata TEXT,  -- JSON object from vision model
                FOREIGN KEY (path) REFERENCES files(path) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_racing_profile ON racing_tags(series_profile);
        """)
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # --- File operations ---

    def add_file(self, path: str, content_hash: str, file_size: int,
                 file_mtime: float, extension: str):
        """Add or update a file record."""
        self.conn.execute("""
            INSERT OR REPLACE INTO files (path, content_hash, file_size, file_mtime, extension)
            VALUES (?, ?, ?, ?, ?)
        """, (path, content_hash, file_size, file_mtime, extension))
        self.conn.commit()

    def get_file(self, path: str) -> Optional[sqlite3.Row]:
        """Get file record by path."""
        return self.conn.execute(
            "SELECT * FROM files WHERE path = ?", (path,)
        ).fetchone()

    def get_file_by_hash(self, content_hash: str) -> Optional[sqlite3.Row]:
        """Get file record by content hash (for move detection)."""
        return self.conn.execute(
            "SELECT * FROM files WHERE content_hash = ?", (content_hash,)
        ).fetchone()

    def get_all_paths(self) -> set:
        """Get all indexed file paths."""
        rows = self.conn.execute("SELECT path FROM files").fetchall()
        return {row['path'] for row in rows}

    def delete_file(self, path: str):
        """Delete a file record (cascades to embeddings/tags)."""
        self.conn.execute("DELETE FROM files WHERE path = ?", (path,))
        self.conn.commit()

    def update_file_path(self, old_path: str, new_path: str, new_mtime: float):
        """Update a file's path (for move detection)."""
        self.conn.execute("""
            UPDATE files SET path = ?, file_mtime = ? WHERE path = ?
        """, (new_path, new_mtime, old_path))
        # Also update in clip_embeddings
        self.conn.execute("""
            UPDATE clip_embeddings SET path = ? WHERE path = ?
        """, (new_path, old_path))
        # And racing_tags
        self.conn.execute("""
            UPDATE racing_tags SET path = ? WHERE path = ?
        """, (new_path, old_path))
        self.conn.commit()

    def file_count(self) -> int:
        """Get total number of files."""
        return self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]

    # --- Directory operations ---

    def get_directory_mtime(self, path: str) -> Optional[float]:
        """Get stored mtime for a directory."""
        row = self.conn.execute(
            "SELECT mtime FROM directories WHERE path = ?", (path,)
        ).fetchone()
        return row['mtime'] if row else None

    def update_directory(self, path: str, mtime: float):
        """Update directory mtime."""
        self.conn.execute("""
            INSERT OR REPLACE INTO directories (path, mtime, last_scanned)
            VALUES (?, ?, ?)
        """, (path, mtime, datetime.now().isoformat()))
        self.conn.commit()

    def get_all_directories(self) -> dict:
        """Get all directories with their mtimes."""
        rows = self.conn.execute("SELECT path, mtime FROM directories").fetchall()
        return {row['path']: row['mtime'] for row in rows}

    # --- CLIP embedding operations ---

    def add_clip_embedding(self, path: str, point_id: int):
        """Record that a file has been embedded in Qdrant."""
        self.conn.execute("""
            INSERT OR REPLACE INTO clip_embeddings (path, point_id, indexed_at)
            VALUES (?, ?, ?)
        """, (path, point_id, datetime.now().isoformat()))
        self.conn.commit()

    def get_clip_embedding(self, path: str) -> Optional[sqlite3.Row]:
        """Get CLIP embedding record for a file."""
        return self.conn.execute(
            "SELECT * FROM clip_embeddings WHERE path = ?", (path,)
        ).fetchone()

    def get_clip_point_id(self, path: str) -> Optional[int]:
        """Get Qdrant point ID for a file."""
        row = self.get_clip_embedding(path)
        return row['point_id'] if row else None

    def delete_clip_embedding(self, path: str):
        """Delete CLIP embedding record."""
        self.conn.execute("DELETE FROM clip_embeddings WHERE path = ?", (path,))
        self.conn.commit()

    def clip_embedding_count(self) -> int:
        """Get total number of CLIP embeddings."""
        return self.conn.execute("SELECT COUNT(*) FROM clip_embeddings").fetchone()[0]

    def get_all_clip_paths(self) -> set:
        """Get all paths that have CLIP embeddings."""
        rows = self.conn.execute("SELECT path FROM clip_embeddings").fetchall()
        return {row['path'] for row in rows}

    # --- Batch operations ---

    def add_files_batch(self, files: list):
        """Add multiple files in a single transaction.

        files: list of (path, content_hash, file_size, file_mtime, extension)
        """
        self.conn.executemany("""
            INSERT OR REPLACE INTO files (path, content_hash, file_size, file_mtime, extension)
            VALUES (?, ?, ?, ?, ?)
        """, files)
        self.conn.commit()

    def add_clip_embeddings_batch(self, embeddings: list):
        """Add multiple CLIP embedding records in a single transaction.

        embeddings: list of (path, point_id)
        """
        now = datetime.now().isoformat()
        data = [(path, point_id, now) for path, point_id in embeddings]
        self.conn.executemany("""
            INSERT OR REPLACE INTO clip_embeddings (path, point_id, indexed_at)
            VALUES (?, ?, ?)
        """, data)
        self.conn.commit()

    def update_directories_batch(self, directories: list):
        """Update multiple directories in a single transaction.

        directories: list of (path, mtime)
        """
        now = datetime.now().isoformat()
        data = [(path, mtime, now) for path, mtime in directories]
        self.conn.executemany("""
            INSERT OR REPLACE INTO directories (path, mtime, last_scanned)
            VALUES (?, ?, ?)
        """, data)
        self.conn.commit()

    # --- Racing tag operations ---

    def add_racing_tag(self, path: str, series_profile: str,
                       keywords: list = None, metadata: dict = None):
        """Record that a file has been tagged by racing_tagger.

        Args:
            path: File path
            series_profile: Profile used (e.g., 'racing-porsche')
            keywords: List of keywords extracted (stored as JSON)
            metadata: Full metadata dict from vision model (stored as JSON)
        """
        import json
        self.conn.execute("""
            INSERT OR REPLACE INTO racing_tags
            (path, tagged_at, series_profile, keywords, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            path,
            datetime.now().isoformat(),
            series_profile,
            json.dumps(keywords) if keywords else None,
            json.dumps(metadata) if metadata else None
        ))
        self.conn.commit()

    def get_racing_tag(self, path: str) -> Optional[sqlite3.Row]:
        """Get racing tag record for a file."""
        return self.conn.execute(
            "SELECT * FROM racing_tags WHERE path = ?", (path,)
        ).fetchone()

    def racing_tag_exists(self, path: str) -> bool:
        """Check if a file has been tagged."""
        return self.get_racing_tag(path) is not None

    def update_racing_xmp_written(self, path: str):
        """Record that XMP was written for a tagged file."""
        self.conn.execute("""
            UPDATE racing_tags SET xmp_written_at = ? WHERE path = ?
        """, (datetime.now().isoformat(), path))
        self.conn.commit()

    def delete_racing_tag(self, path: str):
        """Delete racing tag record."""
        self.conn.execute("DELETE FROM racing_tags WHERE path = ?", (path,))
        self.conn.commit()

    def racing_tag_count(self) -> int:
        """Get total number of racing tags."""
        return self.conn.execute("SELECT COUNT(*) FROM racing_tags").fetchone()[0]

    def get_all_racing_tag_paths(self) -> set:
        """Get all paths that have racing tags."""
        rows = self.conn.execute("SELECT path FROM racing_tags").fetchall()
        return {row['path'] for row in rows}

    def get_untagged_paths(self, limit: int = None) -> list:
        """Get paths that have CLIP embeddings but no racing tags.

        Useful for finding images to tag next.
        """
        query = """
            SELECT ce.path FROM clip_embeddings ce
            LEFT JOIN racing_tags rt ON ce.path = rt.path
            WHERE rt.path IS NULL
        """
        if limit:
            query += f" LIMIT {limit}"
        rows = self.conn.execute(query).fetchall()
        return [row['path'] for row in rows]

    # --- Stats ---

    def stats(self) -> dict:
        """Get database statistics."""
        return {
            'files': self.file_count(),
            'clip_embeddings': self.clip_embedding_count(),
            'directories': self.conn.execute(
                "SELECT COUNT(*) FROM directories"
            ).fetchone()[0],
            'racing_tags': self.racing_tag_count(),
        }
