"""
pipelines/audio_pipeline.py — Transcribe audio files to text using
faster-whisper (CTranslate2-based, optimized for low-VRAM devices).
"""

from __future__ import annotations
import asyncio
from pathlib import Path

from core.router import ProcessingResult
from utils.config import load_config
from utils.logger import get_logger

log = get_logger("audio_pipeline")

# Singleton holder for the faster-whisper model
_whisper_model = None
_whisper_lock = asyncio.Lock()


async def process_audio(file_info: dict) -> ProcessingResult:
    """Pipeline entry point for audio files."""
    path = Path(file_info["path"])
    cfg = load_config().processing.audio

    max_dur = int(cfg.max_duration_sec)

    loop = asyncio.get_event_loop()

    # Check duration before committing to transcription
    duration = await loop.run_in_executor(None, _get_audio_duration, path)
    if duration and duration > max_dur:
        log.warning(
            f"Audio too long: {path.name} ({duration:.0f}s > {max_dur}s) — truncating"
        )

    try:
        transcript, extra = await loop.run_in_executor(
            None, _transcribe, path, duration
        )
    except Exception as exc:
        log.error(f"Audio pipeline error for {path.name}: {exc}")
        return ProcessingResult(file_info, error=str(exc))

    if not transcript:
        return ProcessingResult(file_info, error="No speech detected in audio")

    full_text = f"[Audio Transcript: {path.name}]\n\n{transcript}"
    return ProcessingResult(file_info, text=full_text, extra=extra)


def _get_whisper_model():
    """Lazy-load faster-whisper model (singleton)."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(
            "faster-whisper not installed: pip install faster-whisper"
        )

    cfg = load_config().models.audio
    model_size = str(cfg.model_size)
    device = str(cfg.device)
    compute_type = str(cfg.compute_type)

    # Fallback: if CUDA unavailable, use CPU with int8
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
            compute_type = "int8"
            log.info("CUDA not available — using CPU for Whisper")
    except ImportError:
        device = "cpu"
        compute_type = "int8"

    log.info(f"Loading Whisper {model_size} on {device} ({compute_type})")
    _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
    log.info("Whisper model loaded ✓")
    return _whisper_model


def _transcribe(path: Path, duration_hint: float | None) -> tuple[str, dict]:
    """Transcribe audio. Returns (transcript_text, metadata_dict)."""
    cfg_audio = load_config().models.audio
    lang = cfg_audio.language  # None = auto-detect

    model = _get_whisper_model()

    segments, info = model.transcribe(
        str(path),
        language=lang if lang else None,
        beam_size=5,
        vad_filter=True,       # Voice activity detection — skip silence
        vad_parameters={"min_silence_duration_ms": 500},
    )

    text_parts = []
    total_duration = 0.0

    for segment in segments:
        # Only process up to max duration
        if segment.end > load_config().processing.audio.max_duration_sec:
            break
        text_parts.append(segment.text.strip())
        total_duration = segment.end

    transcript = " ".join(text_parts)

    extra = {
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration_sec": round(total_duration, 1),
        "segment_count": len(text_parts),
    }

    return transcript, extra


def _get_audio_duration(path: Path) -> float | None:
    """Get audio duration in seconds without loading the whole file."""
    try:
        import subprocess, json
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(path),
            ],
            capture_output=True, text=True, timeout=10
        )
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception:
        return None
