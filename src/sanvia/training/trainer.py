"""Two-phase trainer: frozen backbone, then gradual fine-tuning with AMP, accumulation, resume and early stopping."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from sanvia.evaluation.evaluator import NUM_CLASSES, autocast_context, batch_to_device, evaluate
from sanvia.evaluation.metrics import compute_task_metrics
from sanvia.mlops.mlflow_tracking import NullTracker
from sanvia.models.multitask_model import MultiViewMultiTaskModel
from sanvia.training.checkpointing import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    verify_resume_compatibility,
)
from sanvia.training.early_stopping import EarlyStopping
from sanvia.training.optimizer import build_phase1_optimizer, build_phase2_optimizer
from sanvia.training.scheduler import build_warmup_cosine

logger = logging.getLogger(__name__)


class Trainer:
    """Trains one backbone under the shared Stage 1 protocol and selects the best epoch on Validation only."""

    def __init__(
        self,
        model: MultiViewMultiTaskModel,
        cfg: Mapping[str, Any],
        train_loader,
        val_loader,
        criterion,
        device: torch.device,
        run_dir: Path,
        identity: Mapping[str, Any],
        checkpoint_extras: Mapping[str, Any] | None = None,
        tracker: NullTracker | None = None,
    ):
        self.model, self.cfg, self.t = model, cfg, cfg["training"]
        self.train_loader, self.val_loader, self.criterion = train_loader, val_loader, criterion.to(device)
        self.device, self.run_dir = device, Path(run_dir)
        self.identity = dict(identity)  # config_hash, dataset_id, split_id, backbone_name, ... (checked on resume)
        self.extras = dict(checkpoint_extras or {})
        self.tracker = tracker or NullTracker()
        self.amp = bool(self.t["amp"] and device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.accum = self.t["effective_batch_size"] // self.t["batch_size"]
        self.f_epochs, self.t_epochs = self.t["phase1"]["epochs"], self.t["phase2"]["epochs"]
        self.early = EarlyStopping(self.t["phase2"]["early_stopping_patience"], self.t["phase2"]["early_stopping_min_delta"])
        self.optimizer = self.scheduler = None
        self.phase: int | None = None
        self.history: list[dict[str, float]] = []
        self.best_score, self.best_epoch = -math.inf, 0
        self._pending: dict[str, Any] | None = None
        self.best_path, self.last_path = self.run_dir / "best.pt", self.run_dir / "last.pt"

    # ------------------------------------------------------------------ phases
    def _enter_phase(self, phase: int) -> None:
        if phase == 1:
            self.model.set_backbone_trainable(False)
            self.optimizer, self.scheduler = build_phase1_optimizer(self.model, self.cfg), None
        else:
            p2 = self.t["phase2"]
            self.model.set_backbone_trainable(False)
            self.optimizer = build_phase2_optimizer(self.model, self.cfg)
            steps_per_epoch = math.ceil(len(self.train_loader) / self.accum)
            self.scheduler = build_warmup_cosine(
                self.optimizer, p2["epochs"] * steps_per_epoch, p2["warmup_epochs"] * steps_per_epoch, p2["min_lr_ratio"]
            )
            self.early.reset()
        self.phase = phase
        pend = self._pending
        if pend is not None and pend["phase"] == phase:
            self.optimizer.load_state_dict(pend["optimizer_state_dict"])
            if self.scheduler is not None and pend.get("scheduler_state_dict"):
                self.scheduler.load_state_dict(pend["scheduler_state_dict"])
            if phase == 2:
                self.early.load_state_dict(pend["early_stopping"])
            self._pending = None
        logger.info("Entered phase %d", phase)

    # --------------------------------------------------------------- one epoch
    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        n_batches = len(self.train_loader)
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum, n_items = 0.0, 0
        store = {t: {"true": [], "pred": []} for t in NUM_CLASSES}
        for i, batch in enumerate(self.train_loader):
            batch = batch_to_device(batch, self.device)
            with autocast_context(self.device, self.amp):
                out = self.model(batch["images"], batch.get("age"))
                loss, _ = self.criterion(out, batch)
            self.scaler.scale(loss / self.accum).backward()
            if (i + 1) % self.accum == 0 or (i + 1) == n_batches:
                clip = self.t.get("gradient_clip_norm")
                if clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
            bs = batch["images"].shape[0]
            loss_sum += float(loss.detach()) * bs
            n_items += bs
            for t in NUM_CLASSES:
                mask = batch[f"{t}_mask"].bool()
                store[t]["true"].append(batch[t][mask].cpu().numpy())
                store[t]["pred"].append(out[t].detach().argmax(1)[mask].cpu().numpy())
        res = {"train_loss": loss_sum / max(n_items, 1)}
        for t, k in NUM_CLASSES.items():
            y, p = np.concatenate(store[t]["true"]), np.concatenate(store[t]["pred"])
            res[f"train_macro_f1_{t}"] = compute_task_metrics(y, p, np.eye(k)[p] if len(p) else np.zeros((0, k)), k)["macro_f1"]
        valid = [res["train_macro_f1_birads"], res["train_macro_f1_density"]]
        res["train_macro_f1_mean"] = float(np.mean(valid)) if not any(math.isnan(v) for v in valid) else math.nan
        return res

    # --------------------------------------------------------------- checkpoint
    def _payload(self, epoch: int) -> dict[str, Any]:
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": None if self.scheduler is None else self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "epoch": epoch,
            "phase": self.phase,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "early_stopping": self.early.state_dict(),
            "rng_state": capture_rng_state(),
            "seed": self.t["seed"],
            **self.identity,
            **self.extras,
        }

    def _restore(self, ckpt: Mapping[str, Any]) -> int:
        verify_resume_compatibility(
            ckpt,
            {
                k: v
                for k, v in self.identity.items()
                if k in ("config_hash", "dataset_id", "split_id", "backbone_name", "dataset_manifest_hash")
            },
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.history, self.best_score, self.best_epoch = list(ckpt["history"]), ckpt["best_score"], ckpt["best_epoch"]
        self._pending = dict(ckpt)
        restore_rng_state(ckpt["rng_state"])
        logger.info("Resumed from epoch %d (phase %d)", ckpt["epoch"], ckpt["phase"])
        return ckpt["epoch"] + 1

    # ---------------------------------------------------------------------- fit
    def fit(self, resume: bool = False) -> dict[str, Any]:
        self.model.to(self.device)
        start = 1
        if resume and self.last_path.exists():
            start = self._restore(load_checkpoint(self.last_path, map_location=self.device))
        total = self.f_epochs + self.t_epochs
        for epoch in range(start, total + 1):
            phase = 1 if epoch <= self.f_epochs else 2
            if self.phase != phase:
                self._enter_phase(phase)
            if phase == 2:
                self.model.unfreeze_top_fraction(min(1.0, (epoch - self.f_epochs) / max(1, self.t["phase2"]["unfreeze_epochs"])))
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            train_res = self._train_epoch()
            val = evaluate(self.model, self.val_loader, self.criterion, self.device, self.amp)
            score = val["selection_score"]
            rec = {
                "epoch": epoch,
                "phase": phase,
                **train_res,
                "val_loss": val["loss"],
                "val_macro_f1_birads": val["birads"]["macro_f1"],
                "val_macro_f1_density": val["density"]["macro_f1"],
                "val_macro_f1_mean": score,
                "val_qwk_birads": val["birads"]["qwk"],
                "val_qwk_density": val["density"]["qwk"],
                "val_auc_birads": val["birads"]["macro_auc"],
                "val_auc_density": val["density"]["macro_auc"],
                "lr_backbone": self.optimizer.param_groups[0]["lr"],
                "lr_head": self.optimizer.param_groups[-1]["lr"],
                "epoch_time_s": time.perf_counter() - t0,
                "peak_gpu_mb": torch.cuda.max_memory_allocated() / 1024**2 if self.device.type == "cuda" else 0.0,
            }
            self.history.append(rec)
            self.tracker.log_metrics({**rec, "selection_score": score}, step=epoch)
            improved = (not math.isnan(score) and score > self.best_score) or not self.best_path.exists()
            if improved:
                self.best_score, self.best_epoch = (score if not math.isnan(score) else self.best_score), epoch
            if phase == 2:
                self.early.step(score)
            payload = self._payload(epoch)
            if improved:
                save_checkpoint(self.best_path, payload)
            save_checkpoint(self.last_path, payload)
            logger.info(
                "epoch %d/%d phase %d | train_loss %.4f val_loss %.4f | val F1 birads %.4f density %.4f | score %.4f%s",
                epoch,
                total,
                phase,
                rec["train_loss"],
                rec["val_loss"],
                rec["val_macro_f1_birads"],
                rec["val_macro_f1_density"],
                score,
                " *" if improved else "",
            )
            if phase == 2 and self.early.should_stop:
                logger.info("Early stopping at epoch %d", epoch)
                break
        best = load_checkpoint(self.best_path, map_location=self.device)
        self.model.load_state_dict(best["model_state_dict"])
        best_rec = next(r for r in self.history if r["epoch"] == best["epoch"])
        gap = best_rec["train_macro_f1_mean"] - best_rec["val_macro_f1_mean"]
        times = [r["epoch_time_s"] for r in self.history]
        return {
            "best_epoch": best["epoch"],
            "best_score": best["best_score"],
            "history": self.history,
            "train_val_f1_gap": gap,
            "training_time_s": float(sum(times)),
            "avg_epoch_time_s": float(np.mean(times)),
            "peak_gpu_mb": float(max(r["peak_gpu_mb"] for r in self.history)),
            "checkpoint_mb": self.best_path.stat().st_size / 1024**2,
            "epochs_run": len(self.history),
        }
