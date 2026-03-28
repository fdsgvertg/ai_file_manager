"""
core/undo_manager.py — Session-based undo for file move operations.
Each sort session is recorded as a JSON log. Undo reverses moves in order.
"""

from __future__ import annotations
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from utils.config import load_config
from utils.logger import get_logger

log = get_logger("undo_manager")


class MoveRecord:
    """Single atomic file-move record."""

    def __init__(self, src: str, dst: str, timestamp: float = None):
        self.src = src  # original absolute path
        self.dst = dst  # new absolute path
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> dict:
        return {"src": self.src, "dst": self.dst, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, d: dict) -> "MoveRecord":
        return cls(d["src"], d["dst"], d.get("timestamp", 0.0))


class UndoSession:
    """
    Represents one sorting session.
    Records all moves performed so they can be reversed.
    """

    def __init__(self, session_id: str, root_folder: str):
        self.session_id = session_id
        self.root_folder = root_folder
        self.created_at = datetime.now().isoformat()
        self.records: List[MoveRecord] = []
        self.folders_created: List[str] = []

    def record_move(self, src: Path, dst: Path) -> None:
        self.records.append(MoveRecord(str(src), str(dst)))

    def record_folder(self, folder: Path) -> None:
        self.folders_created.append(str(folder))

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "root_folder": self.root_folder,
            "created_at": self.created_at,
            "records": [r.to_dict() for r in self.records],
            "folders_created": self.folders_created,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UndoSession":
        s = cls(d["session_id"], d["root_folder"])
        s.created_at = d["created_at"]
        s.records = [MoveRecord.from_dict(r) for r in d.get("records", [])]
        s.folders_created = d.get("folders_created", [])
        return s

    @property
    def move_count(self) -> int:
        return len(self.records)


class UndoManager:
    """
    Manages undo sessions on disk under <root>/.undo_history/.
    """

    def __init__(self, root_folder: Path):
        cfg = load_config()
        self.root = root_folder
        self._history_dir = root_folder / cfg.undo.history_dir
        self._max_sessions: int = int(cfg.undo.max_sessions)
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._current: Optional[UndoSession] = None

    # ─── Session management ──────────────────────────────────────────────────

    def begin_session(self) -> UndoSession:
        """Start a new undo session. Returns the active session."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current = UndoSession(session_id, str(self.root))
        log.info(f"Undo session started: {session_id}")
        return self._current

    def commit_session(self) -> Optional[Path]:
        """Persist the current session to disk. Returns the saved file path."""
        if self._current is None or self._current.move_count == 0:
            log.info("No moves recorded — skipping undo session save.")
            self._current = None
            return None

        out_path = self._history_dir / f"{self._current.session_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self._current.to_dict(), f, indent=2)

        log.info(
            f"Undo session saved: {out_path.name} "
            f"({self._current.move_count} moves)"
        )
        self._prune_old_sessions()
        self._current = None
        return out_path

    def abort_session(self) -> None:
        """Discard the current session without saving."""
        self._current = None
        log.info("Undo session aborted.")

    @property
    def active_session(self) -> Optional[UndoSession]:
        return self._current

    # ─── Undo execution ───────────────────────────────────────────────────────

    def list_sessions(self) -> list[dict]:
        """List all saved undo sessions, newest first."""
        sessions = []
        for p in sorted(self._history_dir.glob("*.json"), reverse=True):
            try:
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
                sessions.append({
                    "session_id": d["session_id"],
                    "created_at": d["created_at"],
                    "move_count": len(d.get("records", [])),
                    "file": str(p),
                })
            except Exception:
                continue
        return sessions

    def undo_last(self) -> dict:
        """Undo the most recent session. Returns a report dict."""
        sessions = self.list_sessions()
        if not sessions:
            return {"success": False, "message": "No undo sessions found."}
        return self.undo_session(sessions[0]["session_id"])

    def undo_session(self, session_id: str) -> dict:
        """
        Reverse all file moves in the given session.
        Returns a dict with success flag, restored count, and any errors.
        """
        session_file = self._history_dir / f"{session_id}.json"
        if not session_file.exists():
            return {"success": False, "message": f"Session not found: {session_id}"}

        with open(session_file, encoding="utf-8") as f:
            data = json.load(f)

        session = UndoSession.from_dict(data)
        restored = 0
        errors = []

        # Reverse moves in LIFO order
        for record in reversed(session.records):
            src_path = Path(record.src)
            dst_path = Path(record.dst)

            if not dst_path.exists():
                errors.append(f"File no longer at destination: {dst_path}")
                continue

            try:
                src_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dst_path), str(src_path))
                restored += 1
                log.debug(f"Restored: {dst_path.name} → {src_path}")
            except Exception as exc:
                msg = f"Failed to restore {dst_path.name}: {exc}"
                errors.append(msg)
                log.error(msg)

        # Remove empty directories that were created during sorting
        self._cleanup_empty_dirs(session)

        # Remove the session file after successful undo
        if not errors:
            session_file.unlink(missing_ok=True)
            log.info(f"Undo complete: {restored} files restored, session deleted.")
        else:
            log.warning(f"Undo partial: {restored} restored, {len(errors)} errors.")

        return {
            "success": len(errors) == 0,
            "restored": restored,
            "errors": errors,
            "session_id": session_id,
        }

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _cleanup_empty_dirs(self, session: UndoSession) -> None:
        """Remove folders created during sorting if they are now empty."""
        for folder_str in reversed(session.folders_created):
            folder = Path(folder_str)
            try:
                if folder.exists() and folder.is_dir():
                    # Walk upward and remove empty dirs (but not root)
                    _remove_empty_tree(folder, stop_at=self.root)
            except Exception as exc:
                log.debug(f"Could not remove dir {folder}: {exc}")

    def _prune_old_sessions(self) -> None:
        """Keep only the N most recent undo sessions."""
        sessions = sorted(self._history_dir.glob("*.json"))
        while len(sessions) > self._max_sessions:
            oldest = sessions.pop(0)
            oldest.unlink(missing_ok=True)
            log.debug(f"Pruned old undo session: {oldest.name}")


def _remove_empty_tree(path: Path, stop_at: Path) -> None:
    """Walk up the directory tree removing empty dirs until stop_at."""
    current = path
    while current != stop_at and current != current.parent:
        try:
            if current.is_dir() and not any(current.iterdir()):
                current.rmdir()
                log.debug(f"Removed empty dir: {current}")
                current = current.parent
            else:
                break
        except Exception:
            break
