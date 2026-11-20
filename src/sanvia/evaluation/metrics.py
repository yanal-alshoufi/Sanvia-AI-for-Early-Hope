"""Task metrics with explicit NaN for undefined values (never silently zero)."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score


def _nanmean(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    return float(valid.mean()) if valid.size else math.nan


def compute_task_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, num_classes: int) -> dict[str, Any]:
    """Accuracy, per-class/macro precision-recall-F1, OvR macro AUC, QWK and confusion matrix.

    Macro averages use classes present in the ground truth (support > 0). Per-class precision is NaN when a class
    is never predicted; recall and F1 are NaN when the class is absent from the ground truth.
    """
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    labels = list(range(num_classes))
    n = len(y_true)
    nan = math.nan
    result: dict[str, Any] = {"n_samples": int(n), "notes": []}
    if n == 0:
        result.update(
            accuracy=nan,
            macro_precision=nan,
            macro_recall=nan,
            macro_f1=nan,
            macro_auc=nan,
            qwk=nan,
            per_class={
                "precision": [nan] * num_classes,
                "recall": [nan] * num_classes,
                "f1": [nan] * num_classes,
                "support": [0] * num_classes,
            },
            confusion_matrix=np.zeros((num_classes, num_classes), dtype=int).tolist(),
            classes_missing_in_truth=labels,
        )
        result["notes"].append("No valid samples: all metrics undefined.")
        return result
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(axis=1).astype(float), cm.sum(axis=0).astype(float)
    precision = np.where(predicted > 0, tp / np.maximum(predicted, 1), nan)
    recall = np.where(support > 0, tp / np.maximum(support, 1), nan)
    denom = support + predicted
    f1 = np.where(support > 0, 2 * tp / np.maximum(denom, 1), nan)
    precision_eval = np.where(support > 0, precision, nan)
    missing = [i for i in labels if support[i] == 0]
    if missing:
        result["notes"].append(
            f"Classes {missing} absent from ground truth: their per-class metrics are NaN and excluded from macro averages."
        )
    aucs = np.full(num_classes, nan)
    for c in labels:
        pos = y_true == c
        if pos.any() and (~pos).any():
            aucs[c] = roc_auc_score(pos.astype(int), y_prob[:, c])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qwk = float(cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic"))
    result.update(
        accuracy=float((y_true == y_pred).mean()),
        macro_precision=_nanmean(precision_eval),
        macro_recall=_nanmean(recall),
        macro_f1=_nanmean(f1),
        macro_auc=_nanmean(aucs),
        qwk=qwk if not math.isnan(qwk) else nan,
        per_class={
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.astype(int).tolist(),
            "auc": aucs.tolist(),
        },
        confusion_matrix=cm.tolist(),
        classes_missing_in_truth=missing,
    )
    return result


def selection_score(birads_f1: float, density_f1: float) -> float:
    """Mean Macro F1 over both tasks; NaN if either is undefined."""
    if math.isnan(birads_f1) or math.isnan(density_f1):
        return math.nan
    return (birads_f1 + density_f1) / 2.0
