"""Masked, class-weighted multi-task cross-entropy."""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_class_weights(labels: np.ndarray, num_classes: int, scheme: str = "balanced") -> np.ndarray:
    """Inverse-frequency weights from Train labels only, normalised to mean 1 over present classes.

    Classes absent from Train receive weight 0 (they can never be a valid target in Train).
    """
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = np.zeros(num_classes)
    present = counts > 0
    if scheme != "balanced":
        raise ValueError(f"Unknown class-weight scheme: {scheme}")
    weights[present] = counts.sum() / (present.sum() * counts[present])
    weights[present] /= weights[present].mean()
    if (~present).any():
        logger.warning("Classes absent from Train get zero weight: %s", np.flatnonzero(~present).tolist())
    return weights


class MaskedWeightedCrossEntropy(nn.Module):
    """Cross-entropy on logits; samples whose ``mask`` is False contribute nothing."""

    def __init__(self, weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("weight", None if weight is None else weight.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        if not mask.any():
            return logits.sum() * 0.0
        return F.cross_entropy(logits[mask].float(), targets[mask], weight=self.weight)


class MultiTaskLoss(nn.Module):
    """``L_total = w_birads * L_birads + w_density * L_density`` (equal weights by default)."""

    def __init__(
        self,
        birads_weights: torch.Tensor | None,
        density_weights: torch.Tensor | None,
        task_weights: Mapping[str, float] | None = None,
    ):
        super().__init__()
        self.birads = MaskedWeightedCrossEntropy(birads_weights)
        self.density = MaskedWeightedCrossEntropy(density_weights)
        tw = task_weights or {}
        self.w_birads, self.w_density = float(tw.get("birads", 1.0)), float(tw.get("density", 1.0))

    def forward(self, outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]):
        lb = self.birads(outputs["birads"], batch["birads"], batch["birads_mask"])
        ld = self.density(outputs["density"], batch["density"], batch["density_mask"])
        return self.w_birads * lb + self.w_density * ld, {"birads": lb.detach(), "density": ld.detach()}
