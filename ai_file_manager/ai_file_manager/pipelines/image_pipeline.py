"""
pipelines/image_pipeline.py — Extract text description from images via Qwen2-VL.
"""

from __future__ import annotations
import asyncio
from pathlib import Path

from core.router import ProcessingResult
from models.vision_client import VisionClient
from utils.config import load_config
from utils.logger import get_logger

log = get_logger("image_pipeline")


async def process_image(file_info: dict) -> ProcessingResult:
    """
    Pipeline entry point for image files.
    Runs vision model inference in a thread to keep the event loop free.
    """
    path = Path(file_info["path"])
    cfg = load_config().processing.image

    # Size guard
    if file_info["size_mb"] > float(cfg.max_size_mb):
        return ProcessingResult(
            file_info,
            error=f"Image too large: {file_info['size_mb']} MB > {cfg.max_size_mb} MB",
        )

    max_px = int(cfg.resize_max_px)

    loop = asyncio.get_event_loop()
    client = VisionClient.get_instance()

    try:
        description = await loop.run_in_executor(
            None, client.describe_image, path, max_px
        )
    except Exception as exc:
        log.error(f"Image pipeline error for {path.name}: {exc}")
        description = None

    if not description:
        # Fallback: use filename and basic EXIF if vision fails
        description = _fallback_image_text(path)
        if not description:
            return ProcessingResult(file_info, error="Vision model returned no output")

    full_text = f"[Image File: {path.name}]\n{description}"
    return ProcessingResult(
        file_info,
        text=full_text,
        extra={"vision_description": description},
    )


def _fallback_image_text(path: Path) -> str:
    """Extract EXIF metadata as text fallback if vision model fails."""
    parts = [f"Image file: {path.name}"]
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        with Image.open(path) as img:
            parts.append(f"Size: {img.width}x{img.height}")
            parts.append(f"Mode: {img.mode}")

            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag in ("DateTime", "DateTimeOriginal", "Make", "Model",
                               "GPSInfo", "ImageDescription", "UserComment"):
                        parts.append(f"{tag}: {value}")
    except Exception as exc:
        log.debug(f"EXIF extraction failed for {path.name}: {exc}")

    return " | ".join(parts) if len(parts) > 1 else ""
