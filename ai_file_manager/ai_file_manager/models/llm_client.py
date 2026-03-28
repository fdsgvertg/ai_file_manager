"""
models/llm_client.py — Phi-3 Mini inference via llama-cpp-python (GGUF).
Provides structured metadata extraction from file text.
"""

from __future__ import annotations
import json
import threading
from pathlib import Path
from typing import Optional

from utils.config import load_config
from utils.logger import get_logger
from utils.validator import extract_json, validate_metadata

log = get_logger("llm_client")

# ─── Prompt Templates ────────────────────────────────────────────────────────

METADATA_EXTRACTION_PROMPT = """\
You are a file classification assistant. Analyze the following file content and return ONLY a valid JSON object with these exact fields:

{{
  "topic": "<2-4 word descriptive category e.g. 'Medical Records', 'Travel Photos', 'Python Code', 'Financial Reports'>",
  "summary": "<one sentence summary of the content>",
  "keywords": ["<keyword1>", "<keyword2>", "<keyword3>"],
  "confidence": <float between 0.0 and 1.0 indicating classification confidence>,
  "file_date": "<YYYY-MM-DD if a date is mentioned in the content, else null>"
}}

Rules:
- Output ONLY the JSON object, no prose, no markdown fences
- topic must be concise and suitable as a folder name
- keywords must be an array of 3-5 strings
- confidence: 1.0 = very clear content, 0.5 = ambiguous, 0.0 = unreadable

File name: {filename}
File type: {file_type}
Content:
{content}
"""

TOPIC_NAMING_PROMPT = """\
You are given a list of file topics from a semantic cluster. Generate a single concise folder name (2-4 words, Title Case) that best represents all of them.

Topics: {topics}

Return ONLY the folder name as a plain string — no JSON, no explanation.
"""


class LLMClient:
    """
    Singleton-style wrapper around llama-cpp-python Llama model.
    Thread-safe via a lock (single GPU slot).
    """

    _instance: Optional["LLMClient"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._cfg = load_config().models.llm
        self._model = None
        self._model_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "LLMClient":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ─── Model loading ────────────────────────────────────────────────────────

    def _load_model(self):
        """Lazy-load the GGUF model on first use."""
        if self._model is not None:
            return

        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python not installed. Run: pip install llama-cpp-python"
            )

        model_path = Path(self._cfg.path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"LLM model not found at: {model_path}\n"
                "Download from: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf"
            )

        log.info(f"Loading LLM: {model_path.name}")
        self._model = Llama(
            model_path=str(model_path),
            n_ctx=int(self._cfg.n_ctx),
            n_gpu_layers=int(self._cfg.n_gpu_layers),
            n_threads=int(self._cfg.n_threads),
            verbose=False,
        )
        log.info("LLM loaded ✓")

    # ─── Core inference ───────────────────────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        """Run inference. Thread-safe."""
        with self._model_lock:
            self._load_model()
            output = self._model(
                prompt,
                max_tokens=int(self._cfg.max_tokens),
                temperature=float(self._cfg.temperature),
                repeat_penalty=float(self._cfg.repeat_penalty),
                stop=["</s>", "<|end|>", "<|endoftext|>"],
                echo=False,
            )
        return output["choices"][0]["text"].strip()

    # ─── Public methods ───────────────────────────────────────────────────────

    def extract_metadata(
        self,
        content: str,
        filename: str,
        file_type: str,
        max_content_chars: int = 4000,
    ) -> Optional[dict]:
        """
        Send file content to LLM and return validated metadata dict.
        Returns None on failure.
        """
        if not content or not content.strip():
            return self._empty_metadata()

        truncated = content[:max_content_chars]
        prompt = METADATA_EXTRACTION_PROMPT.format(
            filename=filename,
            file_type=file_type,
            content=truncated,
        )

        try:
            raw = self._generate(prompt)
            log.debug(f"LLM raw output for {filename!r}: {raw[:200]}")
            parsed = extract_json(raw)
            if parsed is None:
                log.warning(f"Failed to parse JSON from LLM for {filename!r}")
                return self._empty_metadata()
            validated = validate_metadata(parsed)
            return validated or self._empty_metadata()

        except Exception as exc:
            log.error(f"LLM inference error for {filename!r}: {exc}")
            return self._empty_metadata()

    def name_cluster(self, topics: list[str]) -> str:
        """Ask the LLM to generate a human folder name from a list of topics."""
        if not topics:
            return "Miscellaneous"

        unique_topics = list(dict.fromkeys(t for t in topics if t))[:12]
        if not unique_topics:
            return "Miscellaneous"

        prompt = TOPIC_NAMING_PROMPT.format(topics=", ".join(unique_topics))

        try:
            result = self._generate(prompt).strip()
            # Keep only the first line, cap length
            name = result.split("\n")[0].strip()[:50]
            return name or "Miscellaneous"
        except Exception as exc:
            log.error(f"Cluster naming error: {exc}")
            return "Miscellaneous"

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_metadata() -> dict:
        return {
            "topic": "Uncategorized",
            "summary": "",
            "keywords": [],
            "confidence": 0.0,
            "file_date": None,
        }
