"""Optimizers for the two training phases."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from sanvia.models.multitask_model import MultiViewMultiTaskModel


def build_phase1_optimizer(model: MultiViewMultiTaskModel, cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    """Frozen backbone: AdamW over fusion, age branch and heads only."""
    t = cfg["training"]
    return torch.optim.AdamW(model.head_parameters(), lr=t["phase1"]["lr"], weight_decay=t["optimizer"]["weight_decay"])


def build_phase2_optimizer(model: MultiViewMultiTaskModel, cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    """Fine-tuning: lower learning rate for the backbone. All backbone tensors are registered from the start;
    frozen tensors have no gradient and are skipped until gradual unfreezing enables them."""
    t = cfg["training"]
    groups = [
        {"params": model.backbone_parameters(), "lr": t["phase2"]["lr_backbone"], "name": "backbone"},
        {"params": model.head_parameters(), "lr": t["phase2"]["lr_head"], "name": "head"},
    ]
    return torch.optim.AdamW(groups, weight_decay=t["optimizer"]["weight_decay"])
