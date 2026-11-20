"""Logging helpers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure the root ``sanvia`` logger (console and optional file)."""
    root = logging.getLogger("sanvia")
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_FMT))
    root.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(_FMT))
        root.addHandler(fh)
    return root


def attach_file_handler(path: Path) -> logging.Handler:
    """Attach a per-run file handler to the ``sanvia`` logger and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter(_FMT))
    logging.getLogger("sanvia").addHandler(handler)
    return handler


def detach_handler(handler: logging.Handler) -> None:
    """Remove and close a handler created by :func:`attach_file_handler`."""
    logging.getLogger("sanvia").removeHandler(handler)
    handler.close()
