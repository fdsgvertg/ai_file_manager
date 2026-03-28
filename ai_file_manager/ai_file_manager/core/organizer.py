"""
core/organizer.py — Build folder structure from enriched file metadata
and physically move files.

Supports three sorting modes:
  - topic    : LLM-assigned topic labels
  - date     : file modification month
  - relation : semantic clustering via embeddings
"""

from __future__ import annotations
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

from utils.config import load_config
from utils.file_utils import safe_move, slugify
from utils.logger import get_logger
from utils.validator import validate_metadata
from core.undo_manager import UndoManager, UndoSession

log = get_logger("organizer")

SortMode = Literal["topic", "date", "relation"]

# Maps internal category key → human-readable Level-2 folder name
CATEGORY_TO_FOLDER = {
    "image":    "Images",
    "pdf":      "PDFs",
    "audio":    "Audio",
    "video":    "Video",
    "document": "Documents",
    "unknown":  "Other",
}


class FilePlan:
    """
    Holds the proposed destination path for every file before execution.
    Enables preview-before-sort workflow.
    """

    def __init__(self):
        self.entries: list[dict] = []  # {src, dst, category, topic}
        self.conflicts: list[dict] = []

    def add(self, src: Path, dst: Path, category: str, topic: str) -> None:
        self.entries.append(
            {"src": str(src), "dst": str(dst), "category": category, "topic": topic}
        )

    def preview(self) -> str:
        lines = ["=" * 60, "📁  FILE SORT PREVIEW", "=" * 60]
        by_folder: dict[str, list] = {}
        for e in self.entries:
            folder = str(Path(e["dst"]).parent)
            by_folder.setdefault(folder, []).append(Path(e["dst"]).name)
        for folder, names in sorted(by_folder.items()):
            lines.append(f"\n  {folder}/")
            for n in names:
                lines.append(f"    └─ {n}")
        lines.append(f"\n{'='*60}")
        lines.append(f"Total: {len(self.entries)} files across {len(by_folder)} folders")
        return "\n".join(lines)


class FileOrganizer:
    """
    Orchestrates sorting: accepts enriched file records (with LLM metadata +
    embeddings), builds a FilePlan, and executes moves.
    """

    def __init__(self, root_folder: Path):
        self.root = root_folder.resolve()
        self._cfg = load_config()
        self._undo = UndoManager(self.root)
        self._confidence_threshold: float = float(
            self._cfg.clustering.confidence_threshold
        )
        self._misc_name: str = self._cfg.folders.misc_name

    # ─── Public API ──────────────────────────────────────────────────────────

    def build_plan(
        self,
        enriched_files: list[dict],
        mode: SortMode,
    ) -> FilePlan:
        """
        Build a FilePlan from enriched file records.
        enriched_files: list of dicts — each has 'file_info' + 'metadata' + optionally 'cluster_label'
        """
        plan = FilePlan()

        for record in enriched_files:
            fi = record["file_info"]
            meta = record.get("metadata") or {}
            cluster_label = record.get("cluster_label", "")

            src = Path(fi["path"])
            category = fi["category"]
            level2 = CATEGORY_TO_FOLDER.get(category, "Other")

            if mode == "topic":
                level1 = self._topic_folder(meta)
            elif mode == "date":
                level1 = self._date_folder(fi)
            elif mode == "relation":
                level1 = self._cluster_folder(cluster_label, meta)
            else:
                level1 = self._misc_name

            dst = self.root / slugify(level1) / level2 / fi["name"]
            plan.add(src, dst, category, level1)

        return plan

    def execute_plan(self, plan: FilePlan) -> dict:
        """
        Execute a FilePlan — move files and record session for undo.
        Returns a report dict.
        """
        session: UndoSession = self._undo.begin_session()
        moved = 0
        errors = []

        for entry in plan.entries:
            src = Path(entry["src"])
            dst = Path(entry["dst"])

            if not src.exists():
                errors.append(f"Source missing: {src.name}")
                continue

            try:
                # Record new folders
                parent = dst.parent
                if not parent.exists():
                    session.record_folder(parent)

                final_dst = safe_move(src, dst, overwrite=False)
                session.record_move(src, final_dst)
                moved += 1

            except Exception as exc:
                msg = f"Move failed for {src.name}: {exc}"
                errors.append(msg)
                log.error(msg)

        undo_file = self._undo.commit_session()
        log.info(f"Execution complete: {moved} moved, {len(errors)} errors.")

        return {
            "success": len(errors) == 0,
            "moved": moved,
            "errors": errors,
            "undo_session": undo_file.name if undo_file else None,
        }

    def undo_last(self) -> dict:
        return self._undo.undo_last()

    def list_undo_sessions(self) -> list[dict]:
        return self._undo.list_sessions()

    # ─── Folder name builders ─────────────────────────────────────────────────

    def _topic_folder(self, meta: dict) -> str:
        topic = meta.get("topic", "")
        confidence = float(meta.get("confidence", 0.0))
        if not topic or confidence < self._confidence_threshold:
            return self._misc_name
        return topic

    def _date_folder(self, fi: dict) -> str:
        date_str = fi.get("modified_time", "")
        fmt = self._cfg.folders.date_format
        try:
            dt = datetime.fromisoformat(date_str)
            return dt.strftime(fmt)
        except (ValueError, TypeError):
            return "Unknown_Date"

    def _cluster_folder(self, cluster_label: str, meta: dict) -> str:
        if not cluster_label:
            return self._misc_name
        # Cluster labels may come as "cluster_0" — enrich with topic if available
        topic = meta.get("topic", "")
        if topic and cluster_label.startswith("cluster_"):
            return topic
        return cluster_label or self._misc_name
