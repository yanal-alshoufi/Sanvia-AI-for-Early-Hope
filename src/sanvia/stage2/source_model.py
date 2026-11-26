"""Source-only view pooling, scope-aware heads, and masked supervision."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sanvia.stage2.stage1_artifacts import require


class SourceModel(nn.Module):
    def __init__(self, backbone: nn.Module, registry: dict):
        super().__init__()
        self.backbone=backbone
        self.registry=registry
        self.heads=nn.ModuleDict({s:nn.ModuleDict({k:nn.Linear(backbone.feature_dim,len(t["classes"])) for k,t in tasks.items()}) for s,tasks in registry.items()})

    def forward(self, images: torch.Tensor, valid: torch.Tensor, source: str) -> dict:
        require(images.ndim==5 and valid.shape==images.shape[:2] and valid.dtype==torch.bool and valid.any(1).all(),"Every sample needs at least one valid image")
        features=self.backbone(images[valid])
        slots=features.new_zeros((*valid.shape,features.shape[-1]))
        slots[valid]=features
        pooled=slots.sum(dim=1)/valid.sum(1,keepdim=True)
        return {k: self.heads[source][k](pooled if t["scope"]=="breast" else slots) for k,t in self.registry[source].items()}

    def set_phase(self, frozen: bool) -> None:
        self.train()
        self.backbone.requires_grad_(not frozen)
        self.backbone.train(not frozen)


def masked_loss(outputs: dict, batch: dict, tasks: dict, weights: dict) -> tuple[torch.Tensor | None,dict]:
    """Normalize over active tasks; an invalid head remains outside the optimizer graph."""
    parts={}
    weighted=[]
    scale=0.
    for name,logits in outputs.items():
        t=tasks[name]
        mask=batch[name+"_mask"]
        if not mask.any():
            continue
        target=batch[name][mask]
        values=logits[mask].float()
        loss=(F.binary_cross_entropy_with_logits(values,target.float()) if t["loss"]=="bce" else
              F.cross_entropy(values,target.long(),weight=weights.get(name)))
        parts[name]=float(loss.detach())
        weighted.append(loss*t["weight"])
        scale+=t["weight"]
    return (sum(weighted)/scale if weighted else None),parts

