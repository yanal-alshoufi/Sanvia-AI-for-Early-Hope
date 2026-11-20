"""Atomic checkpoint save/load and resume compatibility checks."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> int:
    """Write atomically (temp file then rename) and return the file size in bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)
    return path.stat().st_size


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def verify_resume_compatibility(ckpt: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Refuse to resume when the dataset, split, configuration or backbone differ from the checkpoint."""
    mismatches = {k: (ckpt.get(k), v) for k, v in expected.items() if ckpt.get(k) != v}
    if mismatches:
        raise RuntimeError(
            "Checkpoint is incompatible with the current experiment; start a new run instead. "
            f"Mismatches (checkpoint, current): {mismatches}"
        )
