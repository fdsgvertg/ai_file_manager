"""
utils/file_utils.py — File system helpers
"""

from __future__ import annotations
import hashlib
import mimetypes
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger("file_utils")

# ─── Extension → category mapping ────────────────────────────────────────────
EXT_CATEGORY: dict[str, str] = {
    # Images
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".bmp": "image",
    ".gif": "image", ".tiff": "image", ".tif": "image", ".webp": "image",
    ".heic": "image", ".heif": "image", ".svg": "image",
    # PDFs
    ".pdf": "pdf",
    # Audio
    ".mp3": "audio", ".wav": "audio", ".flac": "audio", ".ogg": "audio",
    ".m4a": "audio", ".aac": "audio", ".opus": "audio", ".wma": "audio",
    # Video
    ".mp4": "video", ".avi": "video", ".mkv": "video", ".mov": "video",
    ".wmv": "video", ".flv": "video", ".webm": "video", ".m4v": "video",
    # Documents
    ".txt": "document", ".md": "document", ".rst": "document",
    ".csv": "document", ".tsv": "document",
    ".json": "document", ".xml": "document", ".yaml": "document", ".yml": "document",
    ".html": "document", ".htm": "document",
    ".docx": "document", ".doc": "document",
    ".pptx": "document", ".ppt": "document",
    ".xlsx": "document", ".xls": "document",
    ".rtf": "document",
}


def get_file_category(path: Path) -> str:
    """Return file category string based on extension."""
    return EXT_CATEGORY.get(path.suffix.lower(), "unknown")


def get_file_info(path: Path) -> dict:
    """Return a dict of useful file metadata from the filesystem."""
    stat = path.stat()
    return {
        "path": str(path),
        "name": path.name,
        "stem": path.stem,
        "extension": path.suffix.lower(),
        "category": get_file_category(path),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 3),
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
    }


def safe_move(src: Path, dst: Path, overwrite: bool = False) -> Path:
    """
    Move src to dst. Creates parent dirs automatically.
    If dst already exists and overwrite=False, appends a counter suffix.
    Returns the final destination path.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and not overwrite:
        dst = _unique_path(dst)

    shutil.move(str(src), str(dst))
    log.debug(f"Moved: {src.name} → {dst}")
    return dst


def safe_copy(src: Path, dst: Path, overwrite: bool = False) -> Path:
    """Copy src to dst. Returns final destination path."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        dst = _unique_path(dst)
    shutil.copy2(str(src), str(dst))
    log.debug(f"Copied: {src.name} → {dst}")
    return dst


def _unique_path(path: Path) -> Path:
    """Append counter to stem until unique: file.txt → file_1.txt → file_2.txt"""
    counter = 1
    stem = path.stem
    while path.exists():
        path = path.parent / f"{stem}_{counter}{path.suffix}"
        counter += 1
    return path


def slugify(text: str, max_length: int = 40) -> str:
    """Convert arbitrary text to a filesystem-safe folder name."""
    text = re.sub(r"[^\w\s\-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_]+", "_", text.strip())
    text = re.sub(r"-+", "-", text)
    # Title-case for readability
    text = "_".join(w.capitalize() for w in text.split("_"))
    return text[:max_length].strip("_")


def file_hash(path: Path, algorithm: str = "md5", chunk_size: int = 65536) -> str:
    """Compute hex digest of a file (for deduplication)."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def is_readable(path: Path) -> bool:
    """Check if path is a readable file (not a symlink loop, not zero-byte, etc.)."""
    try:
        return path.is_file() and os.access(path, os.R_OK) and path.stat().st_size > 0
    except (PermissionError, OSError):
        return False
