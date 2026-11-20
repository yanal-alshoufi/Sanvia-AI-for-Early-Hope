"""Small JSON/CSV helpers with NaN-safe serialisation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    """Convert numpy/pandas/Path objects to JSON types. NaN and inf become ``None``."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    try:
        import pandas as pd

        if obj is pd.NA or obj is pd.NaT:
            return None
    except ImportError:  # pragma: no cover
        pass
    return str(obj)


def save_json(obj: Any, path: Path) -> Path:
    """Atomically write ``obj`` as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(obj), fh, indent=2, sort_keys=True, ensure_ascii=False)
    os.replace(tmp, path)
    return path


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file using streaming reads."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
