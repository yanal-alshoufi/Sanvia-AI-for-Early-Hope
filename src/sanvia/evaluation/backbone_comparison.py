"""Backbone comparison table and Validation-only selection of the best two backbones."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _min_recall(per_class: Mapping[str, Sequence[float]]) -> float:
    recall = np.array(per_class["recall"], dtype=float)
    return float(np.nanmin(recall)) if (~np.isnan(recall)).any() else math.nan


def build_comparison_table(summaries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """One row per completed run; all values originate from Validation results."""
    rows = []
    for s in summaries:
        v = s["validation"]
        rows.append(
            {
                "backbone": s["backbone"],
                "input_size": s["input_size"],
                "birads_macro_f1": v["birads"]["macro_f1"],
                "density_macro_f1": v["density"]["macro_f1"],
                "birads_qwk": v["birads"]["qwk"],
                "density_qwk": v["density"]["qwk"],
                "selection_score": v["selection_score"],
                "birads_macro_auc": v["birads"]["macro_auc"],
                "density_macro_auc": v["density"]["macro_auc"],
                "birads_min_class_recall": _min_recall(v["birads"]["per_class"]),
                "density_min_class_recall": _min_recall(v["density"]["per_class"]),
                "train_val_f1_gap": s["train_val_f1_gap"],
                "best_epoch": s["best_epoch"],
                "training_time_min": s["training_time_s"] / 60.0,
                "avg_epoch_time_s": s["avg_epoch_time_s"],
                "peak_gpu_mb": s["peak_gpu_mb"],
                "params_total": s["params_total"],
                "params_trainable": s["params_trainable"],
                "checkpoint_mb": s["checkpoint_mb"],
                "inference_ms_per_batch": s["inference_ms_per_batch"],
                "run_id": s["run_id"],
                "mlflow_run_id": s.get("mlflow_run_id"),
            }
        )
    return pd.DataFrame(rows).sort_values("selection_score", ascending=False, na_position="last").reset_index(drop=True)


def select_top_backbones(table: pd.DataFrame, n_select: int, min_birads_fraction: float, max_gap: float) -> dict[str, Any]:
    """Rank by mean Macro F1; guard against weak BI-RADS or obvious overfitting. Test data is never involved."""
    best_birads = table["birads_macro_f1"].max()
    flags = []
    for _, r in table.iterrows():
        issues = []
        if not math.isnan(best_birads) and r["birads_macro_f1"] < min_birads_fraction * best_birads:
            issues.append(f"BI-RADS Macro F1 below {min_birads_fraction:.0%} of the best")
        if not math.isnan(r["train_val_f1_gap"]) and r["train_val_f1_gap"] > max_gap:
            issues.append(f"train-validation F1 gap {r['train_val_f1_gap']:.2f} exceeds {max_gap}")
        if math.isnan(r["selection_score"]):
            issues.append("selection score undefined")
        flags.append("; ".join(issues))
    table = table.assign(guard_issues=flags)
    eligible = table[table["guard_issues"] == ""]
    warning = None
    if len(eligible) < n_select:
        warning = (
            f"Only {len(eligible)} backbones passed the guards; remaining slots filled by score order "
            "among the others and flagged in the rationale."
        )
        rest = table[table["guard_issues"] != ""].dropna(subset=["selection_score"])
        chosen = pd.concat([eligible, rest]).head(n_select)
    else:
        chosen = eligible.head(n_select)
    lines = []
    for rank, (_, r) in enumerate(chosen.iterrows(), start=1):
        lines.append(
            f"{rank}. {r['backbone']}: selection score {r['selection_score']:.4f} "
            f"(BI-RADS F1 {r['birads_macro_f1']:.4f}, density F1 {r['density_macro_f1']:.4f}, "
            f"BI-RADS QWK {r['birads_qwk']:.4f}, density QWK {r['density_qwk']:.4f}, "
            f"train-validation gap {r['train_val_f1_gap']:.3f}, training time {r['training_time_min']:.1f} min, "
            f"peak GPU {r['peak_gpu_mb']:.0f} MB, {int(r['params_total']):,} parameters)."
            + (f" Guard issues: {r['guard_issues']}." if r["guard_issues"] else "")
        )
    return {
        "table": table,
        "selected": chosen["backbone"].tolist(),
        "warning": warning,
        "rationale": "Selection uses Validation results only (mean of BI-RADS and density Macro F1).\n" + "\n".join(lines),
    }
