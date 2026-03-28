"""
utils/logger.py — Centralised logging for AI File Manager
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str, log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """
    Return a named logger with file + console handlers.
    Handlers are only attached once (idempotent).
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    if logger.handlers:
        _loggers[name] = logger
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    try:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        fh = logging.FileHandler(log_path / f"ai_file_manager_{date_str}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception as exc:
        logger.warning(f"Could not create file logger: {exc}")

    _loggers[name] = logger
    return logger


# Module-level default logger
log = get_logger("ai_file_manager")
