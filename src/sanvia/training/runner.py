"""Run one backbone experiment end to end: tracking, training, Validation evaluation and artifacts."""

from __future__ import annotations

import logging
import platform
import traceback
from typing import Any, Mapping

import pandas as pd
import torch

from sanvia.data.pipeline import TrainingData, make_dataloaders
from sanvia.evaluation.evaluator import evaluate, measure_inference_time
from sanvia.evaluation.visualizations import plot_confusion_matrix, plot_learning_curves, plot_per_class_metrics
from sanvia.mlops.mlflow_tracking import MLflowTracker, NullTracker
from sanvia.mlops.reproducibility import find_resumable_run, new_run_id, run_dir_for, update_run_meta, utc_now, write_run_meta
from sanvia.models.backbone_factory import get_backbone_spec, weights_metadata
from sanvia.models.multitask_model import build_model, model_summary_text
from sanvia.training.losses import MultiTaskLoss
from sanvia.training.trainer import Trainer
from sanvia.utils.config import config_hash, run_config_subset, save_resolved_config
from sanvia.utils.io import save_json, to_jsonable
from sanvia.utils.logging import attach_file_handler, detach_handler
from sanvia.utils.paths import ProjectPaths
from sanvia.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


def _mlflow_params(
    cfg: Mapping[str, Any],
    spec,
    wmeta,
    td: TrainingData,
    git: Mapping[str, Any],
    env: Mapping[str, Any],
    cfg_hash: str,
    accum: int,
    pretrained: bool,
) -> dict[str, Any]:
    t = cfg["training"]
    return {
        "backbone": spec.name,
        "weights_identifier": wmeta["weights_identifier"],
        "pretrained": pretrained,
        "input_resolution": spec.input_size,
        "feature_dim": spec.feature_dim,
        "batch_size": t["batch_size"],
        "effective_batch_size": t["effective_batch_size"],
        "grad_accumulation_steps": accum,
        "frozen_epochs": t["phase1"]["epochs"],
        "finetune_epochs": t["phase2"]["epochs"],
        "lr_phase1_head": t["phase1"]["lr"],
        "lr_backbone": t["phase2"]["lr_backbone"],
        "lr_head": t["phase2"]["lr_head"],
        "optimizer": t["optimizer"]["name"],
        "weight_decay": t["optimizer"]["weight_decay"],
        "scheduler": t["scheduler"],
        "loss": t["loss"]["name"],
        "class_weights_birads": [round(float(x), 4) for x in td.class_weights["birads"]],
        "class_weights_density": [round(float(x), 4) for x in td.class_weights["density"]],
        "augmentation": cfg["data"]["augmentation"],
        "age_branch_enabled": td.age_enabled,
        "seed": t["seed"],
        "split_id": td.split_id,
        "dataset_id": td.dataset_id,
        "dataset_manifest_hash": td.dataset_manifest_hash,
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
        "config_hash": cfg_hash,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": wmeta["torchvision_version"],
        "gpu": env["gpu"].get("gpu_name"),
        "normalization": td.normalizer.to_dict(),
        "amp": t["amp"],
    }


def train_backbone(
    name: str,
    cfg: Mapping[str, Any],
    paths: ProjectPaths,
    td: TrainingData,
    git: Mapping[str, Any],
    env: Mapping[str, Any],
    resume: bool = False,
    use_mlflow: bool = True,
    pretrained: bool = True,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train and evaluate one backbone. Previous runs are never overwritten; interrupted runs can be resumed."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = get_backbone_spec(cfg, name)
    wmeta = weights_metadata(spec)
    cfg_hash = config_hash(run_config_subset(cfg, name))
    existing = find_resumable_run(paths.checkpoints_dir, name, cfg_hash, td.dataset_id, td.split_id) if resume else None
    if existing is not None:
        run_dir, run_id = existing, existing.name
        meta = update_run_meta(run_dir, status="running", resumed_utc=utc_now())
    else:
        run_id = new_run_id(name)
        run_dir = run_dir_for(paths.checkpoints_dir, name, run_id)
        run_dir.mkdir(parents=True, exist_ok=False)
        meta = {
            "run_id": run_id,
            "backbone": name,
            "dataset_id": td.dataset_id,
            "split_id": td.split_id,
            "config_hash": cfg_hash,
            "git_commit": git.get("commit"),
            "git_dirty": git.get("dirty"),
            "start_utc": utc_now(),
            "end_utc": None,
            "status": "running",
            "mlflow_run_id": None,
        }
        write_run_meta(run_dir, meta)
    handler = attach_file_handler(run_dir / "training.log")
    tracker: NullTracker = NullTracker()
    try:
        set_global_seed(cfg["training"]["seed"], cfg["training"]["deterministic"])
        if use_mlflow:
            tracker = MLflowTracker(
                run_id,
                {
                    "backbone": name,
                    "dataset_id": td.dataset_id,
                    "split_id": td.split_id,
                    "git_commit": git.get("commit"),
                    "stage": "stage1_backbone_benchmark",
                },
                existing_run_id=meta.get("mlflow_run_id"),
            )
            update_run_meta(run_dir, mlflow_run_id=tracker.run_id)
        accum = cfg["training"]["effective_batch_size"] // cfg["training"]["batch_size"]
        tracker.log_params(_mlflow_params(cfg, spec, wmeta, td, git, env, cfg_hash, accum, pretrained))
        resolved = save_resolved_config(cfg, run_dir / "resolved_config.yaml")
        tracker.log_artifact(resolved, "config")

        model = build_model(spec, cfg["training"], td.age_enabled, pretrained=pretrained)
        (run_dir / "model_summary.txt").write_text(model_summary_text(model, spec), encoding="utf-8")
        train_loader, val_loader = make_dataloaders(td, spec, cfg)
        w = {k: torch.tensor(v, dtype=torch.float32) for k, v in td.class_weights.items()}
        criterion = MultiTaskLoss(w["birads"], w["density"], cfg["training"]["loss"]["task_weights"])
        identity = {
            "config_hash": cfg_hash,
            "dataset_id": td.dataset_id,
            "split_id": td.split_id,
            "dataset_manifest_hash": td.dataset_manifest_hash,
            "backbone_name": name,
            "weights_identifier": wmeta["weights_identifier"],
            "run_id": run_id,
            "git_commit": git.get("commit"),
        }
        extras = {
            "model_config": to_jsonable(
                {
                    **spec.__dict__,
                    "use_age": td.age_enabled,
                    "fusion": cfg["training"]["fusion"],
                    "head": cfg["training"]["head"],
                    "age_branch": cfg["training"]["age_branch"],
                }
            ),
            "preprocessing_config": to_jsonable(
                {"normalizer": td.normalizer.to_dict(), "input_size": spec.input_size, **cfg["data"]["preprocessing"]}
            ),
            "age_processor": None if td.age_processor is None else td.age_processor.to_dict(),
            "class_weights": {k: v.tolist() for k, v in td.class_weights.items()},
        }
        trainer = Trainer(model, cfg, train_loader, val_loader, criterion, device, run_dir, identity, extras, tracker)
        fit = trainer.fit(resume=existing is not None)

        amp = bool(cfg["training"]["amp"] and device.type == "cuda")
        val = evaluate(model, val_loader, criterion.to(device), device, amp)
        it = cfg["training"]["inference_timing"]
        infer_ms = measure_inference_time(
            model, spec.input_size, device, td.age_enabled, amp, it["batch_size"], it["warmup_iters"], it["iters"]
        )
        counts = model.parameter_counts()
        pred_df = val.pop("predictions")
        pred_df.to_csv(run_dir / "validation_predictions.csv", index=False)
        pd.DataFrame(fit["history"]).to_csv(run_dir / "history.csv", index=False)
        arts = [run_dir / "history.csv", run_dir / "validation_predictions.csv", run_dir / "model_summary.txt"]
        for task, space in (("birads", td.birads_space), ("density", td.density_space)):
            m = val[task]
            arts.append(
                plot_confusion_matrix(
                    m["confusion_matrix"], space.class_names, f"{name} {task}", run_dir / f"confusion_{task}.png"
                )
            )
            arts.append(
                plot_per_class_metrics(
                    m["per_class"], space.class_names, f"{name} {task} per-class", run_dir / f"per_class_{task}.png"
                )
            )
            pd.DataFrame({**{k: v for k, v in m["per_class"].items()}}, index=space.class_names).to_csv(
                run_dir / f"per_class_{task}.csv"
            )
            arts.append(run_dir / f"per_class_{task}.csv")
        arts.append(plot_learning_curves(fit["history"], cfg["training"]["phase1"]["epochs"], run_dir / "learning_curves.png"))
        summary = {
            "backbone": name,
            "run_id": run_id,
            "mlflow_run_id": tracker.run_id,
            "input_size": spec.input_size,
            "weights_identifier": wmeta["weights_identifier"],
            "validation": val,
            "best_epoch": fit["best_epoch"],
            "train_val_f1_gap": fit["train_val_f1_gap"],
            "training_time_s": fit["training_time_s"],
            "avg_epoch_time_s": fit["avg_epoch_time_s"],
            "peak_gpu_mb": fit["peak_gpu_mb"],
            "epochs_run": fit["epochs_run"],
            "params_total": counts["total"],
            "params_trainable": counts["trainable"],
            "checkpoint_mb": fit["checkpoint_mb"],
            "inference_ms_per_batch": infer_ms,
            "best_checkpoint": str(run_dir / "best.pt"),
            "run_dir": str(run_dir),
            "config_hash": cfg_hash,
            "dataset_id": td.dataset_id,
            "split_id": td.split_id,
            "git_commit": git.get("commit"),
            "preprocessing_hash": config_hash(extras["preprocessing_config"]),
        }
        save_json(summary, run_dir / "run_summary.json")
        arts.append(run_dir / "run_summary.json")
        for a in arts + [run_dir / "training.log"]:
            tracker.log_artifact(a, f"{name}")
        tracker.log_artifact(run_dir / "best.pt", f"{name}/checkpoint")
        tracker.log_metrics(
            {
                "final_val_selection_score": val["selection_score"],
                "training_time_s": fit["training_time_s"],
                "inference_ms_per_batch": infer_ms,
                "params_total": counts["total"],
                "params_trainable": counts["trainable"],
                "checkpoint_mb": fit["checkpoint_mb"],
            },
            step=fit["epochs_run"],
        )
        update_run_meta(run_dir, status="completed", end_utc=utc_now())
        tracker.end("FINISHED")
        return summary
    except BaseException as exc:
        update_run_meta(
            run_dir, status="failed", end_utc=utc_now(), error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
        )
        tracker.end("FAILED")
        raise
    finally:
        detach_handler(handler)
