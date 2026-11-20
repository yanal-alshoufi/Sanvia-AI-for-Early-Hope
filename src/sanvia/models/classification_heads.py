"""Independent classification heads returning logits."""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
