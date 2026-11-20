"""Plots for evaluation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrix(cm: Sequence[Sequence[int]], class_names: Sequence[str], title: str, path: Path) -> Path:
    cm = np.asarray(cm)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.full(cm.shape, np.nan), where=row_sums > 0)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"{title} (counts)")
    sns.heatmap(
        norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, xticklabels=class_names, yticklabels=class_names, ax=axes[1]
    )
    axes[1].set_title(f"{title} (row-normalised)")
    for ax in axes:
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    return _save(fig, path)


def plot_per_class_metrics(per_class: Mapping[str, Sequence[float]], class_names: Sequence[str], title: str, path: Path) -> Path:
    df = pd.DataFrame({m: per_class[m] for m in ("precision", "recall", "f1")}, index=class_names).reset_index(names="class")
    fig, ax = plt.subplots(figsize=(8, 4))
    df.melt(id_vars="class", var_name="metric", value_name="value").pipe(
        lambda d: sns.barplot(data=d, x="class", y="value", hue="metric", ax=ax)
    )
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    return _save(fig, path)


def plot_learning_curves(history: Sequence[Mapping[str, float]], phase1_epochs: int, path: Path) -> Path:
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(h["epoch"], h["train_loss"], label="train")
    axes[0].plot(h["epoch"], h["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[1].plot(h["epoch"], h["train_macro_f1_mean"], label="train")
    axes[1].plot(h["epoch"], h["val_macro_f1_mean"], label="validation")
    axes[1].set_title("Mean Macro F1 (BI-RADS, density)")
    axes[1].set_xlabel("epoch")
    for ax in axes:
        ax.axvline(phase1_epochs + 0.5, color="grey", linestyle="--", linewidth=0.8)
        ax.legend()
    return _save(fig, path)


def plot_backbone_comparison(table: pd.DataFrame, path: Path) -> Path:
    cols = ["birads_macro_f1", "density_macro_f1", "selection_score"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    table.set_index("backbone")[cols].plot.bar(ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_title("Validation comparison across backbones")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, path)
