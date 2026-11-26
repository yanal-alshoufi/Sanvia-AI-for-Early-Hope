"""Validation evaluation and consistent inference-time measurement."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from sanvia.evaluation.metrics import compute_task_metrics, selection_score

NUM_CLASSES = {"birads": 5, "density": 4}


def batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def autocast_context(device: torch.device, amp: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(amp and device.type == "cuda"))


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, criterion, device: torch.device, amp: bool, *, include_probabilities: bool = False) -> dict[str, Any]:
    """Run the model on a loader (Validation only in Stage 1) and compute all task metrics."""
    model.eval()
    total, n_items = 0.0, 0
    comp = {"birads": 0.0, "density": 0.0}
    store = {t: {"true": [], "pred": [], "prob": []} for t in NUM_CLASSES}
    ids: list[str] = []
    masks = {t: [] for t in NUM_CLASSES}
    for batch in loader:
        batch = batch_to_device(batch, device)
        with autocast_context(device, amp):
            out = model(batch["images"], batch.get("age"))
            loss, parts = criterion(out, batch)
        bs = batch["images"].shape[0]
        total += float(loss) * bs
        n_items += bs
        for t in NUM_CLASSES:
            comp[t] += float(parts[t]) * bs
            prob = torch.softmax(out[t].float(), dim=1)
            store[t]["true"].append(batch[t].cpu().numpy())
            store[t]["pred"].append(prob.argmax(1).cpu().numpy())
            store[t]["prob"].append(prob.cpu().numpy())
            masks[t].append(batch[f"{t}_mask"].cpu().numpy())
        ids.extend(batch["sample_id"])
    result: dict[str, Any] = {
        "loss": total / max(n_items, 1),
        "loss_birads": comp["birads"] / max(n_items, 1),
        "loss_density": comp["density"] / max(n_items, 1),
        "n_samples": n_items,
    }
    pred_df = pd.DataFrame({"sample_id": ids})
    for t, k in NUM_CLASSES.items():
        mask = np.concatenate(masks[t]).astype(bool)
        y_true, y_pred = np.concatenate(store[t]["true"])[mask], np.concatenate(store[t]["pred"])[mask]
        y_prob = np.concatenate(store[t]["prob"])[mask]
        result[t] = compute_task_metrics(y_true, y_pred, y_prob, k)
        pred_df[f"{t}_true"] = np.concatenate(store[t]["true"])
        pred_df[f"{t}_pred"] = np.concatenate(store[t]["pred"])
        pred_df[f"{t}_valid"] = mask
        if include_probabilities:
            probabilities = np.concatenate(store[t]["prob"])
            for class_index in range(k):
                pred_df[f"{t}_prob_{class_index}"] = probabilities[:, class_index]
    result["selection_score"] = selection_score(result["birads"]["macro_f1"], result["density"]["macro_f1"])
    result["predictions"] = pred_df
    return result


@torch.no_grad()
def measure_inference_time(
    model: torch.nn.Module,
    input_size: int,
    device: torch.device,
    use_age: bool,
    amp: bool,
    batch_size: int = 8,
    warmup: int = 3,
    iters: int = 10,
) -> float:
    """Mean forward time per batch (ms) on synthetic input; identical procedure for every backbone."""
    model.eval()
    images = torch.randn(batch_size, 2, 3, input_size, input_size, device=device)
    age = torch.zeros(batch_size, 2, device=device) if use_age else None
    for _ in range(warmup):
        with autocast_context(device, amp):
            model(images, age)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        with autocast_context(device, amp):
            model(images, age)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0
