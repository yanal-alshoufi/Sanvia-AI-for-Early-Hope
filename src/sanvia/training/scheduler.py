"""Linear warm-up followed by cosine annealing, stepped per optimizer update."""

from __future__ import annotations

import math

import torch


def build_warmup_cosine(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int, min_lr_ratio: float = 0.0
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps, warmup_steps = max(total_steps, 1), min(max(warmup_steps, 0), total_steps)

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)
