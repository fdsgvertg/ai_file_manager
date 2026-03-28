"""
core/scanner.py — Recursively scan a folder and build a manifest of files.
"""

from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, List, Optional

from utils.config import load_config
from utils.file_utils import get_file_info, is_readable
from utils.logger import get_logger

log = get_logger("scanner")

# Names to always ignore
IGNORE_NAMES = {
    ".git", ".svn", "__pycache__", ".DS_Store",
    "Thumbs.db", ".undo_history", "desktop.ini",
}


class FileManifest:
    """Container for all scanned file metadata."""

    def __init__(self, root: Path):
        self.root = root
        self.files: List[dict] = []
        self.skipped: List[dict] = []
        self.total_size_bytes: int = 0

    def add(self, info: dict) -> None:
        self.files.append(info)
        self.total_size_bytes += info["size_bytes"]

    def skip(self, path: Path, reason: str) -> None:
        self.skipped.append({"path": str(path), "reason": reason})
        log.debug(f"Skipped {path.name}: {reason}")

    def by_category(self) -> dict[str, list]:
        result: dict[str, list] = {}
        for f in self.files:
            cat = f["category"]
            result.setdefault(cat, []).append(f)
        return result

    @property
    def count(self) -> int:
        return len(self.files)

    def summary(self) -> str:
        cats = self.by_category()
        parts = [f"{k}: {len(v)}" for k, v in sorted(cats.items())]
        size_mb = round(self.total_size_bytes / (1024 * 1024), 2)
        return (
            f"Scanned {self.count} files ({size_mb} MB) in '{self.root.name}' | "
            + " | ".join(parts)
            + f" | skipped: {len(self.skipped)}"
        )


class FolderScanner:
    """
    Scans a target directory and returns a FileManifest.
    Supports both recursive and flat (single-level) scanning.
    """

    def __init__(self, recursive: bool = True):
        cfg = load_config()
        self.recursive = recursive
        self.max_size_mb: float = 500.0  # Global cap per file
        self._cfg = cfg

    def scan(self, folder: str | Path) -> FileManifest:
        """Synchronous scan — returns FileManifest."""
        folder = Path(folder).resolve()
        if not folder.exists():
            raise FileNotFoundError(f"Target folder not found: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Target is not a directory: {folder}")

        log.info(f"Scanning: {folder}")
        manifest = FileManifest(root=folder)

        iterator = folder.rglob("*") if self.recursive else folder.iterdir()

        for path in iterator:
            if self._should_skip_dir(path):
                continue
            if not path.is_file():
                continue
            self._process_file(path, manifest)

        log.info(manifest.summary())
        return manifest

    async def scan_async(self, folder: str | Path) -> FileManifest:
        """Async wrapper — runs scan in a thread to avoid blocking."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            return await loop.run_in_executor(pool, self.scan, folder)

    # ─── internal helpers ────────────────────────────────────────────────────

    def _should_skip_dir(self, path: Path) -> bool:
        """Return True if any part of the path is in the ignore list."""
        for part in path.parts:
            if part in IGNORE_NAMES:
                return True
        return False

    def _process_file(self, path: Path, manifest: FileManifest) -> None:
        if not is_readable(path):
            manifest.skip(path, "not readable / empty")
            return

        try:
            info = get_file_info(path)
        except (PermissionError, OSError) as exc:
            manifest.skip(path, f"os error: {exc}")
            return

        if info["size_mb"] > self.max_size_mb:
            manifest.skip(path, f"too large ({info['size_mb']} MB > {self.max_size_mb} MB)")
            return

        manifest.add(info)
