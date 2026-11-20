"""Register the two selected research candidates in the MLflow Model Registry (never as Production)."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Mapping

import mlflow
import mlflow.pytorch
import torch
from mlflow.tracking import MlflowClient

from sanvia.models.backbone_factory import get_backbone_spec
from sanvia.models.multitask_model import build_model
from sanvia.training.checkpointing import load_checkpoint

logger = logging.getLogger(__name__)


def registered_model_name(prefix: str, backbone: str) -> str:
    return f"{prefix}_{backbone.replace('-', '')}"


def _log_model_compat(model: torch.nn.Module, registered_name: str) -> None:
    params = inspect.signature(mlflow.pytorch.log_model).parameters
    key = "name" if "name" in params else "artifact_path"
    extra = {}
    if "serialization_format" in params:
        # Full-model pickle keeps the multi-input (images, age) forward signature; loading requires `sanvia` to be installed.
        extra["serialization_format"] = "pickle"
    mlflow.pytorch.log_model(model, **{key: "model"}, registered_model_name=registered_name, **extra)


def register_selected_models(
    selected: list[Mapping[str, Any]], cfg: Mapping[str, Any], lineage_by_backbone: Mapping[str, Mapping[str, Any]], use_age: bool
) -> list[dict[str, Any]]:
    """Log each selected checkpoint's model inside its original MLflow run and register it as a research candidate."""
    reg_cfg = cfg["mlops"]["registry"]
    client = MlflowClient()
    results = []
    for summary in selected:
        name = registered_model_name(reg_cfg["model_name_prefix"], summary["backbone"])
        spec = get_backbone_spec(cfg, summary["backbone"])
        model = build_model(spec, cfg["training"], use_age=use_age, pretrained=False)
        ckpt = load_checkpoint(Path(summary["best_checkpoint"]))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        with mlflow.start_run(run_id=summary["mlflow_run_id"]):
            _log_model_compat(model, name)
        versions = client.search_model_versions(f"name='{name}'")
        version = max(versions, key=lambda v: int(v.version))
        lineage = lineage_by_backbone[summary["backbone"]]
        client.set_registered_model_alias(name, reg_cfg["alias"], version.version)
        client.update_model_version(
            name,
            version.version,
            description=(
                f"Research candidate (not production). Validation selection score "
                f"{summary['validation']['selection_score']:.4f}. Official Test was not used."
            ),
        )
        tags = {
            "stage": "research_candidate",
            "production": "false",
            "run_id": summary["run_id"],
            "mlflow_run_id": summary["mlflow_run_id"],
            "dataset_id": lineage["dataset_version"],
            "git_commit": lineage["git_commit"],
            "split_id": lineage["split_id"],
            "checkpoint": summary["best_checkpoint"],
            "val_selection_score": f"{summary['validation']['selection_score']:.4f}",
            "val_birads_macro_f1": f"{summary['validation']['birads']['macro_f1']:.4f}",
            "val_density_macro_f1": f"{summary['validation']['density']['macro_f1']:.4f}",
        }
        for k, v in tags.items():
            client.set_model_version_tag(name, version.version, k, str(v))
        results.append({"registered_name": name, "version": version.version, "backbone": summary["backbone"]})
        logger.info("Registered %s version %s as research candidate", name, version.version)
    return results
