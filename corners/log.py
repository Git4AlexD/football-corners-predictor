"""Logging setup shared by every command."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from corners.config import LOG_DIR

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: int = logging.INFO, log_file: str | None = "pipeline.log") -> None:
    """Log to the console and, optionally, a rotating file under ``logs/``."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(LOG_DIR / log_file, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
        )
    logging.basicConfig(level=level, format=_FORMAT, datefmt="%H:%M:%S", handlers=handlers, force=True)
    # Third-party libraries are noisy at INFO.
    for name in ("urllib3", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)
