"""Feature fusion and optional age branch (identical for all backbones)."""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionBlock(nn.Module):
    """Concatenated ``[CC, MLO]`` features -> fully connected fusion block."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)
        )
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AgeBranch(nn.Module):
    """Small fixed MLP over ``[normalised_age, missing_flag]``."""

    def __init__(self, hidden_dim: int, embed_dim: int, in_features: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, embed_dim), nn.ReLU(inplace=True)
        )
        self.out_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
