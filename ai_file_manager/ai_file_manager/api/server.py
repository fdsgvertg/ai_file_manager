"""
api/server.py — FastAPI REST API for the AI File Manager.
Exposes endpoints for scanning, processing, previewing, executing, and undoing.
"""

from __future__ import annotations
import asyncio
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from utils.config import load_config
from utils.logger import get_logger

log = get_logger("api.server")
cfg = load_config()

app = FastAPI(
    title="AI File Manager",
    version="1.0.0",
    description="Intelligent local file organizer powered by multimodal AI",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory job store ──────────────────────────────────────────────────────
# For a production system this would be Redis / Celery
_jobs: dict[str, dict] = {}


# ─── Request/Response Models ─────────────────────────────────────────────────

class ScanRequest(BaseModel):
    folder: str
    recursive: bool = True

class SortRequest(BaseModel):
    folder: str
    mode: Literal["topic", "date", "relation"] = "topic"
    recursive: bool = True
    execute: bool = False  # If False, return preview only

class UndoRequest(BaseModel):
    folder: str
    session_id: Optional[str] = None


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": cfg.project.version}


@app.post("/scan")
async def scan_folder(req: ScanRequest):
    """Scan a folder and return file manifest."""
    from core.scanner import FolderScanner

    scanner = FolderScanner(recursive=req.recursive)
    try:
        manifest = await scanner.scan_async(req.folder)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "root": str(manifest.root),
        "total_files": manifest.count,
        "total_size_mb": round(manifest.total_size_bytes / (1024**2), 2),
        "by_category": {k: len(v) for k, v in manifest.by_category().items()},
        "skipped": len(manifest.skipped),
        "files": manifest.files[:200],  # Cap for response size
    }


@app.post("/sort")
async def sort_folder(req: SortRequest, background_tasks: BackgroundTasks):
    """
    Start an async sort job.
    Returns a job_id immediately; poll /jobs/{job_id} for status.
    If execute=False, returns preview without moving files.
    """
    import uuid
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "queued", "folder": req.folder}

    background_tasks.add_task(
        _run_sort_job, job_id, req.folder, req.mode, req.recursive, req.execute
    )

    return {"job_id": job_id, "status": "queued", "message": "Sort job started"}


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    """Poll a sort job for status and results."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return _jobs[job_id]


@app.post("/undo")
async def undo(req: UndoRequest):
    """Undo the last (or specified) sort session for a folder."""
    from core.organizer import FileOrganizer

    folder = Path(req.folder)
    if not folder.exists():
        raise HTTPException(status_code=400, detail="Folder not found")

    organizer = FileOrganizer(folder)
    if req.session_id:
        from core.undo_manager import UndoManager
        um = UndoManager(folder)
        result = um.undo_session(req.session_id)
    else:
        result = organizer.undo_last()

    return result


@app.get("/undo/sessions")
def list_undo_sessions(folder: str):
    """List available undo sessions for a folder."""
    from core.organizer import FileOrganizer

    fp = Path(folder)
    if not fp.exists():
        raise HTTPException(status_code=400, detail="Folder not found")

    organizer = FileOrganizer(fp)
    return {"sessions": organizer.list_undo_sessions()}


# ─── Background sort job ─────────────────────────────────────────────────────

async def _run_sort_job(
    job_id: str,
    folder: str,
    mode: str,
    recursive: bool,
    execute: bool,
):
    """Full pipeline: scan → process → embed → cluster → plan → (execute)."""
    _jobs[job_id]["status"] = "running"

    try:
        from core.scanner import FolderScanner
        from core.router import FileRouter
        from core.organizer import FileOrganizer
        from models.llm_client import LLMClient
        from models.embedding_client import SemanticClusterer

        root = Path(folder)

        # ── 1. Scan ──────────────────────────────────────────────────────────
        _jobs[job_id]["step"] = "scanning"
        scanner = FolderScanner(recursive=recursive)
        manifest = await scanner.scan_async(folder)
        _jobs[job_id]["file_count"] = manifest.count

        # ── 2. Process each file through pipelines ───────────────────────────
        _jobs[job_id]["step"] = "processing"
        router = FileRouter()
        results = await router.process_many(manifest.files, max_concurrent=3)

        successful = [r for r in results if r.success]
        log.info(f"Processed {len(successful)}/{len(results)} files successfully")

        # ── 3. LLM metadata extraction ───────────────────────────────────────
        _jobs[job_id]["step"] = "extracting_metadata"
        llm = LLMClient.get_instance()
        enriched: list[dict] = []

        for result in results:
            fi = result.file_info
            text = result.text or f"File: {fi['name']} (type: {fi['category']})"

            metadata = llm.extract_metadata(
                content=text,
                filename=fi["name"],
                file_type=fi["category"],
            )
            enriched.append({
                "file_info": fi,
                "text": text,
                "metadata": metadata,
            })

        # ── 4. Embeddings + clustering (for 'relation' mode) ─────────────────
        if mode == "relation":
            _jobs[job_id]["step"] = "clustering"
            clusterer = SemanticClusterer()

            texts_for_embed = [
                e["text"] or e["file_info"]["name"] for e in enriched
            ]
            cluster_labels = clusterer.cluster_texts(texts_for_embed)

            # Name each cluster via LLM
            cluster_topics: dict[str, list[str]] = {}
            for label, entry in zip(cluster_labels, enriched):
                cluster_topics.setdefault(label, []).append(
                    entry["metadata"].get("topic", "")
                )

            cluster_names: dict[str, str] = {}
            for label, topics in cluster_topics.items():
                cluster_names[label] = llm.name_cluster(topics)

            for label, entry in zip(cluster_labels, enriched):
                entry["cluster_label"] = cluster_names.get(label, label)
        else:
            for entry in enriched:
                entry["cluster_label"] = ""

        # ── 5. Build folder plan ──────────────────────────────────────────────
        _jobs[job_id]["step"] = "planning"
        organizer = FileOrganizer(root)
        plan = organizer.build_plan(enriched, mode=mode)

        preview_text = plan.preview()
        _jobs[job_id]["preview"] = preview_text
        _jobs[job_id]["plan_entries"] = len(plan.entries)

        # ── 6. Execute (if requested) ─────────────────────────────────────────
        if execute:
            _jobs[job_id]["step"] = "executing"
            exec_result = organizer.execute_plan(plan)
            _jobs[job_id]["execution"] = exec_result
            _jobs[job_id]["status"] = "done"
        else:
            _jobs[job_id]["status"] = "preview_ready"

        log.info(f"Job {job_id} completed: {_jobs[job_id]['status']}")

    except Exception as exc:
        log.error(f"Job {job_id} failed: {exc}", exc_info=True)
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(exc)
