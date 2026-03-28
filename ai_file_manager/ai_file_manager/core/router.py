"""
core/router.py — Route each file to the correct processing pipeline
and return extracted text content + metadata.
"""

from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Callable, Optional

from utils.config import load_config
from utils.logger import get_logger

log = get_logger("router")


class ProcessingResult:
    """Holds the extracted text and pipeline-level metadata for one file."""

    def __init__(
        self,
        file_info: dict,
        text: str = "",
        extra: Optional[dict] = None,
        error: Optional[str] = None,
    ):
        self.file_info = file_info
        self.text = text.strip()
        self.extra = extra or {}
        self.error = error
        self.success = error is None and bool(text)

    def __repr__(self) -> str:
        status = "OK" if self.success else f"FAIL({self.error})"
        return f"ProcessingResult({self.file_info['name']!r}, {status})"


class FileRouter:
    """
    Dispatch files to the correct processing pipeline based on category.
    Pipelines are imported lazily to avoid loading heavy models until needed.
    """

    def __init__(self):
        self._cfg = load_config()
        self._pipelines: dict[str, Callable] = {}
        self._gpu_semaphore = asyncio.Semaphore(
            self._cfg.concurrency.gpu_semaphore
        )

    # ─── Public API ──────────────────────────────────────────────────────────

    async def process(self, file_info: dict) -> ProcessingResult:
        """Route a single file through the appropriate pipeline."""
        category = file_info["category"]
        path = Path(file_info["path"])

        pipeline_fn = self._get_pipeline(category)
        if pipeline_fn is None:
            # Try plain text read for unknown small files
            text = self._fallback_text_read(path)
            if text:
                return ProcessingResult(file_info, text=text)
            return ProcessingResult(
                file_info, error=f"No pipeline for category '{category}'"
            )

        try:
            # GPU-bound pipelines (image, audio, video) go through semaphore
            if category in ("image", "audio", "video", "pdf"):
                async with self._gpu_semaphore:
                    result = await pipeline_fn(file_info)
            else:
                result = await pipeline_fn(file_info)

            return result

        except Exception as exc:
            log.error(f"Pipeline error for {path.name}: {exc}", exc_info=True)
            return ProcessingResult(file_info, error=str(exc))

    async def process_many(
        self, file_infos: list[dict], max_concurrent: int = 3
    ) -> list[ProcessingResult]:
        """Process a list of files with limited concurrency."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _bounded(fi: dict) -> ProcessingResult:
            async with semaphore:
                return await self.process(fi)

        tasks = [_bounded(fi) for fi in file_infos]
        return await asyncio.gather(*tasks)

    # ─── Pipeline dispatch ────────────────────────────────────────────────────

    def _get_pipeline(self, category: str) -> Optional[Callable]:
        if category in self._pipelines:
            return self._pipelines[category]

        fn = None
        if category == "image":
            from pipelines.image_pipeline import process_image
            fn = process_image
        elif category == "pdf":
            from pipelines.pdf_pipeline import process_pdf
            fn = process_pdf
        elif category == "audio":
            from pipelines.audio_pipeline import process_audio
            fn = process_audio
        elif category == "video":
            from pipelines.video_pipeline import process_video
            fn = process_video
        elif category == "document":
            from pipelines.document_pipeline import process_document
            fn = process_document

        if fn:
            self._pipelines[category] = fn
        return fn

    # ─── Fallback ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_text_read(path: Path, max_chars: int = 4000) -> str:
        """Try to read unknown files as plain UTF-8 text."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            return text[:max_chars]
        except Exception:
            return ""
