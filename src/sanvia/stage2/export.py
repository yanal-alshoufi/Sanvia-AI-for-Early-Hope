"""Portable, strict feature-backbone exports. Source heads never enter transfer files."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from sanvia.models.backbone_factory import BackboneSpec, build_backbone
from sanvia.stage2.stage1_artifacts import require
from sanvia.training.checkpointing import save_checkpoint
from sanvia.utils.io import load_json, save_json, sha256_file


def export_backbone(backbone, spec: BackboneSpec, path: Path, metadata: dict) -> dict:
    state = {k: v.detach().cpu() for k, v in backbone.state_dict().items()}
    save_checkpoint(path, state)
    record = dict(
        metadata,
        spec=asdict(spec),
        sha256=sha256_file(path),
        file=path.name,
        tensors={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in state.items()},
    )
    save_json(record, path.with_name(path.stem + "_metadata.json"))
    return record


def load_export(path: Path, spec: BackboneSpec | None = None, backbone=None):
    meta = load_json(path.with_name(path.stem + "_metadata.json"))
    require(sha256_file(path) == meta["sha256"], f"Backbone checksum mismatch: {path}")
    actual = BackboneSpec(**meta["spec"])
    require(spec is None or spec == actual, "Backbone architecture mismatch")
    model = backbone if backbone is not None else build_backbone(actual, pretrained=False)
    require(getattr(model, "spec", actual) == actual, "Destination backbone mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    for k, v in model.state_dict().items():
        require(k in state and state[k].shape == v.shape and state[k].dtype == v.dtype, f"Incompatible backbone tensor {k}")
    model.load_state_dict(state, strict=True)
    return model
