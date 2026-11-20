"""Multi-view, multi-task model with a shared backbone and two classification heads."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from sanvia.models.backbone_factory import BackboneSpec, FeatureBackbone, build_backbone
from sanvia.models.classification_heads import ClassificationHead
from sanvia.models.fusion import AgeBranch, FusionBlock


class MultiViewMultiTaskModel(nn.Module):
    """Input ``[B, 2, 3, H, W]`` (CC, MLO) -> shared backbone -> concat -> fusion -> two heads (logits)."""

    def __init__(
        self, backbone: FeatureBackbone, training_cfg: Mapping[str, Any], use_age: bool, num_birads: int = 5, num_density: int = 4
    ):
        super().__init__()
        self.backbone = backbone
        self.use_age = use_age
        self.fusion = FusionBlock(
            2 * backbone.feature_dim, training_cfg["fusion"]["hidden_dim"], training_cfg["fusion"]["dropout"]
        )
        head_in = self.fusion.out_dim
        self.age_branch = None
        if use_age:
            ab = training_cfg["age_branch"]
            self.age_branch = AgeBranch(ab["hidden_dim"], ab["embed_dim"])
            head_in += self.age_branch.out_dim
        self.birads_head = ClassificationHead(head_in, num_birads, training_cfg["head"]["dropout"])
        self.density_head = ClassificationHead(head_in, num_density, training_cfg["head"]["dropout"])

    # ------------------------------------------------------------------ forward
    def forward(self, images: torch.Tensor, age: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if images.ndim != 5 or images.shape[1] != 2:
            raise ValueError(f"Expected images [B, 2, 3, H, W], got {tuple(images.shape)}")
        b, v = images.shape[:2]
        feats = self.backbone(images.flatten(0, 1))  # [B*2, D], same weights for both views
        fused_in = feats.view(b, v, -1).flatten(1)  # [B, 2D] = concat(CC, MLO)
        z = self.fusion(fused_in)
        if self.use_age:
            if age is None:
                raise ValueError("Model was built with the age branch but no age input was given.")
            z = torch.cat([z, self.age_branch(age)], dim=1)
        return {"birads": self.birads_head(z), "density": self.density_head(z)}

    # -------------------------------------------------------- freezing helpers
    def backbone_parameters(self) -> list[nn.Parameter]:
        return list(self.backbone.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        ids = {id(p) for p in self.backbone.parameters()}
        return [p for p in self.parameters() if id(p) not in ids]

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable
        self._apply_bn_policy()

    def unfreeze_top_fraction(self, fraction: float) -> int:
        """Unfreeze the last ``fraction`` of backbone parameter tensors (deepest layers first)."""
        params = self.backbone_parameters()
        n = int(round(len(params) * min(max(fraction, 0.0), 1.0)))
        for i, p in enumerate(params):
            p.requires_grad = i >= len(params) - n
        self._apply_bn_policy()
        return n

    def _apply_bn_policy(self) -> None:
        """BatchNorm layers whose parameters are frozen stay in eval mode so running stats are not updated."""
        if not self.training:
            return
        for m in self.backbone.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                weight = getattr(m, "weight", None)
                if weight is not None and not weight.requires_grad:
                    m.eval()

    def train(self, mode: bool = True):  # noqa: D102
        super().train(mode)
        if mode:
            self._apply_bn_policy()
        return self

    def parameter_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "backbone": sum(p.numel() for p in self.backbone.parameters())}


def build_model(
    spec: BackboneSpec, training_cfg: Mapping[str, Any], use_age: bool, pretrained: bool = True
) -> MultiViewMultiTaskModel:
    return MultiViewMultiTaskModel(build_backbone(spec, pretrained=pretrained), training_cfg, use_age)


def model_summary_text(model: nn.Module, spec: BackboneSpec) -> str:
    counts = model.parameter_counts()
    return (
        f"Backbone: {spec.name} ({spec.weights_enum}.{spec.weights_name}), input {spec.input_size}x{spec.input_size}, "
        f"feature_dim={spec.feature_dim}\nParameters: total={counts['total']:,} trainable={counts['trainable']:,} "
        f"backbone={counts['backbone']:,}\n\n{model}"
    )
