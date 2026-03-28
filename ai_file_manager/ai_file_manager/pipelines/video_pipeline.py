"""
pipelines/video_pipeline.py — Extract representative frames from video using
FFmpeg, then describe them with the vision model.
"""

from __future__ import annotations
import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from core.router import ProcessingResult
from utils.config import load_config
from utils.logger import get_logger

log = get_logger("video_pipeline")


async def process_video(file_info: dict) -> ProcessingResult:
    """Pipeline entry point for video files."""
    path = Path(file_info["path"])
    cfg = load_config().processing.video
    max_dur = int(cfg.max_duration_sec)
    n_frames = int(cfg.frames_to_extract)

    loop = asyncio.get_event_loop()

    # Get video duration
    duration = await loop.run_in_executor(None, _get_video_duration, path)

    if duration is None:
        return ProcessingResult(file_info, error="Could not read video metadata")

    if duration > max_dur:
        log.warning(f"Video capped at {max_dur}s (actual: {duration:.0f}s)")
        duration = float(max_dur)

    # Extract frames
    try:
        frame_paths = await loop.run_in_executor(
            None, _extract_frames, path, duration, n_frames
        )
    except Exception as exc:
        return ProcessingResult(file_info, error=f"Frame extraction failed: {exc}")

    if not frame_paths:
        return ProcessingResult(file_info, error="No frames extracted from video")

    # Describe each frame with vision model
    from models.vision_client import VisionClient
    client = VisionClient.get_instance()

    descriptions = []
    for i, fp in enumerate(frame_paths, 1):
        desc = await loop.run_in_executor(None, client.describe_image, fp, 512)
        if desc:
            descriptions.append(f"[Frame {i}]: {desc}")
        # Clean up temp frame
        try:
            fp.unlink(missing_ok=True)
        except Exception:
            pass

    if not descriptions:
        return ProcessingResult(file_info, error="Vision model returned no descriptions")

    combined = "\n\n".join(descriptions)
    full_text = (
        f"[Video File: {path.name} | Duration: {duration:.0f}s]\n\n"
        f"{combined}"
    )

    return ProcessingResult(
        file_info,
        text=full_text,
        extra={"duration_sec": duration, "frames_analyzed": len(descriptions)},
    )


def _get_video_duration(path: Path) -> Optional[float]:
    """Get video duration in seconds via ffprobe."""
    try:
        import json
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(path),
            ],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception as exc:
        log.warning(f"Could not get duration for {path.name}: {exc}")
        return None


def _extract_frames(
    path: Path, duration: float, n_frames: int
) -> list[Path]:
    """
    Extract n_frames evenly spaced frames from the video using FFmpeg.
    Returns list of paths to temporary PNG files.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="ai_fm_frames_"))
    frame_paths = []

    # Calculate evenly-spaced timestamps (avoid 0 and end)
    interval = duration / (n_frames + 1)
    timestamps = [interval * (i + 1) for i in range(n_frames)]

    for i, ts in enumerate(timestamps):
        out_path = tmp_dir / f"frame_{i:03d}.png"
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(ts),
            "-i", str(path),
            "-vframes", "1",
            "-q:v", "2",
            "-vf", "scale=512:-1",      # Resize to 512px width
            str(out_path),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=30
            )
            if result.returncode == 0 and out_path.exists():
                frame_paths.append(out_path)
            else:
                log.warning(f"FFmpeg failed for frame at {ts:.1f}s: {result.stderr[:100]}")
        except subprocess.TimeoutExpired:
            log.warning(f"FFmpeg timeout for frame at {ts:.1f}s")
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg not found. Install from: https://ffmpeg.org/download.html"
            )

    return frame_paths
