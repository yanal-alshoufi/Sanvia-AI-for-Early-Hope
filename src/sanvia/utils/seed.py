"""Seeding and determinism helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

NONDETERMINISM_NOTES = [
    "CUDA atomics in some backward kernels can vary between runs even with deterministic flags.",
    "Mixed precision (AMP) changes numerical rounding and can differ across GPU models and driver versions.",
    "DataLoader worker scheduling is seeded per worker, but augmentation draws depend on worker assignment.",
    "Resuming from a checkpoint restores RNG states, but the DataLoader shuffle order of the interrupted epoch is not replayed.",
    "Different torch, torchvision, CUDA or cuDNN versions may change results.",
]


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch and CUDA; optionally request deterministic algorithms."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` that derives per-worker seeds from the torch base seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
