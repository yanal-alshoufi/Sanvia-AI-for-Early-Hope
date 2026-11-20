"""Backbone factory: one interface for six torchvision ImageNet backbones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torchvision


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    builder: str
    weights_enum: str
    weights_name: str
    input_size: int
    feature_dim: int
    approx_params_m: float | None = None


def get_backbone_spec(cfg: Mapping[str, Any], name: str) -> BackboneSpec:
    for b in cfg["backbones"]:
        if b["name"] == name:
            return BackboneSpec(**{k: b.get(k) for k in BackboneSpec.__dataclass_fields__})
    raise KeyError(f"Unknown backbone {name!r}. Available: {[b['name'] for b in cfg['backbones']]}")


def resolve_weights(spec: BackboneSpec):
    """Resolve the torchvision weights enum member (no download)."""
    enum = getattr(torchvision.models, spec.weights_enum)
    return enum[spec.weights_name]


def weights_metadata(spec: BackboneSpec) -> dict[str, Any]:
    w = resolve_weights(spec)
    t = w.transforms()
    return {
        "weights_identifier": f"{spec.weights_enum}.{spec.weights_name}",
        "url": w.url,
        "torchvision_version": torchvision.__version__,
        "mean": list(t.mean),
        "std": list(t.std),
        "torchvision_resize_size": getattr(t, "resize_size", None),
        "torchvision_crop_size": getattr(t, "crop_size", None),
    }


class FeatureBackbone(nn.Module):
    """Wraps a torchvision classifier whose final classification layer was replaced by ``Identity``."""

    def __init__(self, net: nn.Module, spec: BackboneSpec):
        super().__init__()
        self.net = net
        self.spec = spec
        self.feature_dim = spec.feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if not isinstance(out, torch.Tensor):  # defensive: Inception aux outputs
            out = out.logits
        return out


def build_backbone(spec: BackboneSpec, pretrained: bool = True) -> FeatureBackbone:
    """Build the backbone, load ImageNet weights, strip the classifier and return pooled features ``[N, D]``."""
    builder = getattr(torchvision.models, spec.builder)
    weights = resolve_weights(spec) if pretrained else None
    if spec.builder == "inception_v3":
        net = (
            builder(weights=weights)
            if pretrained
            else builder(weights=None, aux_logits=True, transform_input=True, init_weights=False)
        )
        net.AuxLogits = None
        net.aux_logits = False
        net.dropout = nn.Identity()
        net.fc = nn.Identity()
    else:
        net = builder(weights=weights)
        if spec.builder == "resnet50":
            net.fc = nn.Identity()
        elif spec.builder in {"densenet121", "mobilenet_v3_large", "efficientnet_b4", "efficientnet_v2_s"}:
            net.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone builder: {spec.builder}")
    return FeatureBackbone(net, spec)


@torch.no_grad()
def infer_feature_dim(backbone: FeatureBackbone, size: int | None = None) -> int:
    was_training = backbone.training
    backbone.eval()
    out = backbone(torch.zeros(2, 3, size or backbone.spec.input_size, size or backbone.spec.input_size))
    backbone.train(was_training)
    return int(out.shape[-1])
