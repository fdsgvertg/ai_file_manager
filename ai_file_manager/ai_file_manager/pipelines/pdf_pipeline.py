"""
pipelines/pdf_pipeline.py — Extract text from PDFs using PyMuPDF,
with Tesseract OCR fallback for scanned/image-only PDFs.
"""

from __future__ import annotations
import asyncio
import io
from pathlib import Path
from typing import Optional

from core.router import ProcessingResult
from utils.config import load_config
from utils.logger import get_logger

log = get_logger("pdf_pipeline")


async def process_pdf(file_info: dict) -> ProcessingResult:
    """Pipeline entry point for PDF files."""
    path = Path(file_info["path"])
    loop = asyncio.get_event_loop()

    try:
        text, extra = await loop.run_in_executor(None, _extract_pdf, path)
    except Exception as exc:
        log.error(f"PDF pipeline error for {path.name}: {exc}")
        return ProcessingResult(file_info, error=str(exc))

    if not text:
        return ProcessingResult(file_info, error="No text extracted from PDF")

    return ProcessingResult(file_info, text=text, extra=extra)


def _extract_pdf(path: Path) -> tuple[str, dict]:
    """
    Extract text from a PDF:
    1. Try PyMuPDF direct text extraction
    2. If result is empty/too short, try OCR with Tesseract via pytesseract
    """
    cfg = load_config().processing.pdf
    max_pages = int(cfg.max_pages)

    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError("PyMuPDF not installed: pip install pymupdf")

    text_parts: list[str] = []
    total_pages = 0
    ocr_used = False

    with fitz.open(str(path)) as doc:
        total_pages = len(doc)
        pages_to_process = min(total_pages, max_pages)

        for page_num in range(pages_to_process):
            page = doc[page_num]
            page_text = page.get_text("text").strip()

            if page_text and len(page_text) > 50:
                text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
            else:
                # Page is image-only — attempt OCR
                ocr_text = _ocr_page(page, cfg)
                if ocr_text:
                    text_parts.append(f"[Page {page_num + 1} - OCR]\n{ocr_text}")
                    ocr_used = True

    combined = "\n\n".join(text_parts)
    full_text = f"[PDF: {path.name} | {total_pages} pages]\n\n{combined}"

    extra = {
        "total_pages": total_pages,
        "pages_processed": pages_to_process,
        "ocr_used": ocr_used,
    }

    return full_text[:12000], extra  # Cap total characters sent to LLM


def _ocr_page(page, cfg) -> str:
    """Render a single PDF page to image and OCR it with Tesseract."""
    try:
        import pytesseract
        from PIL import Image

        dpi = int(cfg.ocr_dpi)
        lang = str(cfg.ocr_lang)
        zoom = dpi / 72  # 72 dpi = 1:1 scale in fitz
        mat = __import__("fitz").Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img, lang=lang)
        return text.strip()

    except ImportError:
        log.warning("pytesseract or Pillow not installed — skipping OCR")
        return ""
    except Exception as exc:
        log.warning(f"OCR failed on page: {exc}")
        return ""
