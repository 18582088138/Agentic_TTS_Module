"""日志 / Logging —— 一个 handler，够用就行。"""

from __future__ import annotations

import logging
import os
import sys

_ROOT = "agentic_tts"
_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return
    logger = logging.getLogger(_ROOT)
    logger.setLevel(os.environ.get("TTS_LOG_LEVEL", "INFO").upper())
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s",
                                               datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """取一个子 logger / Get a child logger. `TTS_LOG_LEVEL` 控制级别。"""
    _configure()
    return logging.getLogger(f"{_ROOT}.{name}")


__all__ = ["get_logger"]
