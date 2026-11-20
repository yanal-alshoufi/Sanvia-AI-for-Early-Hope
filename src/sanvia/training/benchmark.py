"""Benchmark-level orchestration: run all backbones, compare, select (Validation only), register, report."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import pandas as pd
import torch

from sanvia.data.pipeline import TrainingData
from sanvia.evaluation.backbone_comparison import build_comparison_table, select_top_backbones
from sanvia.evaluation.visualizations import plot_backbone_comparison
from sanvia.mlops.model_registry import register_selected_models
from sanvia.mlops.reproducibility import build_lineage, build_reproducibility_report, write_model_card
from sanvia.training.runner import train_backbone
from sanvia.utils.config import config_hash
from sanvia.utils.io import load_json, save_json
from sanvia.utils.paths import ProjectPaths

logger = logging.getLogger(__name__)


def run_all_backbones(
    cfg: Mapping[str, Any],
    paths: ProjectPaths,
    td: TrainingData,
    git: Mapping[str, Any],
    env: Mapping[str, Any],
    names: list[str] | None = None,
    resume: bool = True,
    use_mlflow: bool = True,
    pretrained: bool = True,
    device: torch.device | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Train the backbones sequentially with the same protocol. Failures are recorded and reported at the end."""
    summaries, failures = [], {}
    for name in names or [b["name"] for b in cfg["backbones"]]:
        logger.info("=== Backbone %s ===", name)
        try:
            summaries.append(
                train_backbone(
                    name, cfg, paths, td, git, env, resume=resume, use_mlflow=use_mlflow, pretrained=pretrained, device=device
                )
            )
        except Exception as exc:  # noqa: BLE001 - run is marked failed; continue with the next backbone
            failures[name] = f"{type(exc).__name__}: {exc}"
            logger.error("Backbone %s failed: %s", name, failures[name])
    return summaries, failures


def finalize_benchmark(
    cfg: Mapping[str, Any],
    paths: ProjectPaths,
    td: TrainingData,
    summaries: list[Mapping[str, Any]],
    git: Mapping[str, Any],
    env: Mapping[str, Any],
    register: bool = True,
) -> dict[str, Any]:
    """Compare backbones, select the best two on Validation, write reports and register the research candidates."""
    expected = [b["name"] for b in cfg["backbones"]]
    missing = sorted(set(expected) - {s["backbone"] for s in summaries})
    if missing:
        raise RuntimeError(f"Cannot select backbones: runs missing or failed for {missing}.")
    table = build_comparison_table(summaries)
    sel_cfg = cfg["selection"]
    selection = select_top_backbones(
        table, sel_cfg["n_select"], sel_cfg["min_birads_f1_fraction_of_best"], sel_cfg["max_overfit_gap"]
    )
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    selection["table"].to_csv(paths.reports_dir / "backbone_comparison.csv", index=False)
    plot_backbone_comparison(table, paths.plots_dir / "backbone_comparison.png")
    decision = {
        "selected": selection["selected"],
        "rationale": selection["rationale"],
        "warning": selection["warning"],
        "basis": "Validation only; official Test not used",
        "comparison": table.to_dict(orient="records"),
    }
    save_json(decision, paths.reports_dir / "backbone_comparison.json")
    save_json(decision, paths.reports_dir / "selection_decision.json")

    dataset = load_json(paths.manifests_dir / "dataset_manifest.json")
    split_summary = load_json(paths.splits_dir / "split_summary.json")
    cfg_hash = config_hash(cfg)
    selected = [s for s in summaries if s["backbone"] in selection["selected"]]
    selected.sort(key=lambda s: selection["selected"].index(s["backbone"]))
    lineage = {s["backbone"]: build_lineage(s, dataset, split_summary, git, cfg_hash) for s in selected}
    for s in selected:
        write_model_card(paths.reports_dir / f"model_card_{s['backbone']}.md", s, lineage[s["backbone"]], selection["rationale"])
    save_json(lineage, paths.reports_dir / "lineage_report.json")

    registered: list[dict[str, Any]] = []
    if register and cfg["mlops"]["registry"]["register"]:
        registered = register_selected_models(selected, cfg, lineage, td.age_enabled)
        save_json(registered, paths.reports_dir / "registered_models.json")

    report = build_reproducibility_report(
        dataset=dataset,
        split=split_summary,
        git=git,
        config_hash_value=cfg_hash,
        environment=env,
        training_cfg=cfg["training"],
        summaries=list(summaries),
        selection=decision,
        age_status=td.age_reason,
    )
    report["registered_models"] = registered
    save_json(report, paths.reports_dir / "reproducibility_report.json")
    return {"table": pd.DataFrame(table), "selection": selection, "lineage": lineage, "registered": registered, "report": report}
