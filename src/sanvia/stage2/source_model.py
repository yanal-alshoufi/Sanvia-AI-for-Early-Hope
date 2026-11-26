"""Source-only view pooling, scope-aware heads, and masked supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sanvia.stage2.stage1_artifacts import require


class SourceModel(nn.Module):
    def __init__(self, backbone: nn.Module, registry: dict):
        super().__init__()
        self.backbone = backbone
        self.registry = registry
        self.heads = nn.ModuleDict(
            {
                s: nn.ModuleDict({k: nn.Linear(backbone.feature_dim, len(t["classes"])) for k, t in tasks.items()})
                for s, tasks in registry.items()
            }
        )

    def forward(self, images: torch.Tensor, valid: torch.Tensor, source: str) -> dict:
        require(
            images.ndim == 5 and valid.shape == images.shape[:2] and valid.dtype == torch.bool and valid.any(1).all(),
            "Every sample needs at least one valid image",
        )
        features = self.backbone(images[valid])
        slots = features.new_zeros((*valid.shape, features.shape[-1]))
        slots[valid] = features
        pooled = slots.sum(dim=1) / valid.sum(1, keepdim=True)
        return {k: self.heads[source][k](pooled if t["scope"] == "breast" else slots) for k, t in self.registry[source].items()}

    def set_phase(self, frozen: bool) -> None:
        self.train()
        self.backbone.requires_grad_(not frozen)
        self.backbone.train(not frozen)


def label_mass(batches: list[dict], tasks: dict, weights: dict) -> dict[str, float]:
    """Compute each task's effective denominator across a gradient-accumulation window."""
    mass = dict.fromkeys(tasks, 0.0)
    for batch in batches:
        for name in tasks:
            mask = batch[name + "_mask"]
            if name in weights:
                labels = batch[name][mask].long()
                mass[name] += float(weights[name][labels].sum())
            else:
                mass[name] += int(mask.sum())
    return mass


def masked_loss(
    outputs: dict, batch: dict, tasks: dict, weights: dict, window_mass: dict | None = None
) -> tuple[torch.Tensor | None, dict]:
    """Contribution of this microbatch to the weighted mean over all valid labels in the window."""
    mass = window_mass if window_mass is not None else label_mass([batch], tasks, weights)
    task_weight = sum(tasks[k]["weight"] for k, n in mass.items() if n > 0)
    parts = {}
    terms = []
    for name, logits in outputs.items():
        mask = batch[name + "_mask"]
        if mass[name] <= 0 or not mask.any():
            continue
        target = batch[name][mask]
        values = logits[mask].float()
        task = tasks[name]
        if task["loss"] == "bce":
            numerator = F.binary_cross_entropy_with_logits(values, target.float(), reduction="sum")
        else:
            numerator = F.cross_entropy(values, target.long(), weight=weights.get(name), reduction="sum")
        loss = numerator / mass[name]
        parts[name] = float(loss.detach())
        terms.append(loss * task["weight"] / task_weight)
    return (sum(terms) if terms else None), parts
