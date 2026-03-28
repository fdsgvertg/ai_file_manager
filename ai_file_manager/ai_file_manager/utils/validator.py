"""
utils/validator.py — Strict JSON extraction and schema validation for LLM outputs
"""

from __future__ import annotations
import json
import re
from typing import Any, Optional
from utils.logger import get_logger

log = get_logger("validator")


def extract_json(text: str) -> Optional[dict | list]:
    """
    Robustly extract the first valid JSON object or array from an LLM response.
    Handles:
      - Raw JSON
      - JSON inside ```json ... ``` fences
      - JSON embedded in surrounding prose
    Returns None if no valid JSON found.
    """
    if not text or not isinstance(text, str):
        return None

    # 1. Try the whole string first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2. Try code fence extraction
    fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
    for match in fence_pattern.finditer(text):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue

    # 3. Try to find first { ... } or [ ... ] block
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # 4. Attempt to fix common LLM JSON mistakes (trailing commas, single quotes)
    try:
        fixed = _fix_json(text)
        return json.loads(fixed)
    except Exception:
        pass

    log.warning(f"Could not extract JSON from text: {text[:200]!r}")
    return None


def _fix_json(text: str) -> str:
    """Apply heuristic fixes to malformed JSON."""
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Replace single-quoted strings with double-quoted (simple cases)
    text = re.sub(r"'([^'\\]*)'", r'"\1"', text)
    return text


def validate_metadata(data: Any) -> Optional[dict]:
    """
    Validate that LLM-returned metadata has required fields.
    Expected schema:
    {
        "topic": str,
        "summary": str,
        "keywords": [str, ...],
        "confidence": float (0-1),
        "file_date": str | null
    }
    Returns cleaned dict or None if invalid.
    """
    if not isinstance(data, dict):
        return None

    required_fields = {"topic", "summary", "keywords", "confidence"}
    missing = required_fields - data.keys()
    if missing:
        log.warning(f"Metadata missing fields: {missing}")
        # Fill with defaults rather than failing hard
        data.setdefault("topic", "Uncategorized")
        data.setdefault("summary", "")
        data.setdefault("keywords", [])
        data.setdefault("confidence", 0.0)

    # Type coercion and clamping
    data["topic"] = str(data["topic"]).strip()[:80]
    data["summary"] = str(data.get("summary", "")).strip()[:500]
    data["confidence"] = float(_clamp(data.get("confidence", 0.0), 0.0, 1.0))

    if not isinstance(data.get("keywords"), list):
        data["keywords"] = []
    else:
        data["keywords"] = [str(k).strip() for k in data["keywords"] if k][:10]

    # Optional field
    data.setdefault("file_date", None)

    return data


def _clamp(value: Any, lo: float, hi: float) -> float:
    try:
        v = float(value)
        return max(lo, min(hi, v))
    except (TypeError, ValueError):
        return lo
