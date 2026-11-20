"""Run identity, resume discovery, reproducibility and lineage reports, model cards."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from sanvia.mlops.git_metadata import reproducibility_statement
from sanvia.utils.io import load_json, save_json
from sanvia.utils.seed import NONDETERMINISM_NOTES


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id(backbone: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{backbone}-{stamp}-{uuid.uuid4().hex[:6]}"


def run_dir_for(checkpoints_dir: Path, backbone: str, run_id: str) -> Path:
    return Path(checkpoints_dir) / backbone / run_id


def write_run_meta(run_dir: Path, meta: Mapping[str, Any]) -> None:
    save_json(meta, Path(run_dir) / "run_meta.json")


def update_run_meta(run_dir: Path, **updates: Any) -> dict[str, Any]:
    meta = load_json(Path(run_dir) / "run_meta.json")
    meta.update(updates)
    write_run_meta(run_dir, meta)
    return meta


def find_resumable_run(checkpoints_dir: Path, backbone: str, config_hash: str, dataset_id: str, split_id: str) -> Path | None:
    """Latest unfinished run of this backbone with identical configuration, dataset and split."""
    base = Path(checkpoints_dir) / backbone
    if not base.exists():
        return None
    candidates = []
    for d in sorted(base.iterdir()):
        meta_path = d / "run_meta.json"
        if not meta_path.exists() or not (d / "last.pt").exists():
            continue
        m = load_json(meta_path)
        if m.get("status") in {"running", "failed", "interrupted"} and (
            m.get("config_hash"),
            m.get("dataset_id"),
            m.get("split_id"),
        ) == (config_hash, dataset_id, split_id):
            candidates.append(d)
    return candidates[-1] if candidates else None


def build_reproducibility_report(
    *,
    dataset: Mapping[str, Any],
    split: Mapping[str, Any],
    git: Mapping[str, Any],
    config_hash_value: str,
    environment: Mapping[str, Any],
    training_cfg: Mapping[str, Any],
    summaries: list[Mapping[str, Any]],
    selection: Mapping[str, Any],
    age_status: str,
) -> dict[str, Any]:
    return {
        "generated_utc": utc_now(),
        "dataset": dict(dataset),
        "split": dict(split),
        "git": dict(git),
        "git_statement": reproducibility_statement(dict(git)),
        "config_hash": config_hash_value,
        "environment": dict(environment),
        "training_parameters": dict(training_cfg),
        "age_integration": age_status,
        "experiments": [
            {
                "backbone": s["backbone"],
                "run_id": s["run_id"],
                "mlflow_run_id": s.get("mlflow_run_id"),
                "checkpoint": s["best_checkpoint"],
                "validation_selection_score": s["validation"]["selection_score"],
            }
            for s in summaries
        ],
        "final_selection": dict(selection),
        "known_nondeterminism": NONDETERMINISM_NOTES,
        "official_test_used": False,
    }


def build_lineage(
    summary: Mapping[str, Any],
    dataset: Mapping[str, Any],
    split: Mapping[str, Any],
    git: Mapping[str, Any],
    config_hash_value: str,
) -> dict[str, Any]:
    """Dataset -> manifests -> split -> preprocessing -> backbone -> run -> validation -> checkpoint -> registry."""
    return {
        "dataset_version": dataset.get("dataset_id"),
        "dataset_release": dataset.get("dataset_release_id"),
        "image_manifest_checksum": dataset.get("png_manifest_checksum"),
        "sample_manifest_checksum": dataset.get("sample_manifest_checksum"),
        "split_id": split.get("split_id"),
        "split_manifest_checksum": dataset.get("split_manifest_checksum"),
        "preprocessing_config_hash": summary.get("preprocessing_hash"),
        "backbone": summary["backbone"],
        "weights_identifier": summary.get("weights_identifier"),
        "run_id": summary["run_id"],
        "mlflow_run_id": summary.get("mlflow_run_id"),
        "config_hash": config_hash_value,
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
        "validation_metrics": {
            "selection_score": summary["validation"]["selection_score"],
            "birads_macro_f1": summary["validation"]["birads"]["macro_f1"],
            "density_macro_f1": summary["validation"]["density"]["macro_f1"],
        },
        "checkpoint": summary["best_checkpoint"],
    }


def write_model_card(path: Path, summary: Mapping[str, Any], lineage: Mapping[str, Any], rationale: str) -> Path:
    v = summary["validation"]
    text = f"""# Model card: SANVIA Stage 1 - {summary["backbone"]} (research candidate)

Status: research candidate. Not validated for clinical use. Not production. Not deployed.

## Task
Breast-level multi-task classification from paired CC and MLO mammograms (VinDr-Mammo, PNG):
BI-RADS (5 classes) and breast density (4 classes).

## Architecture
Shared ImageNet-pretrained {summary["backbone"]} ({summary.get("weights_identifier")}), input {summary["input_size"]}x{summary["input_size"]},
concatenation of CC/MLO features, fully connected fusion block, optional age branch, two independent classification heads.

## Validation results (official Test was not used)
- BI-RADS Macro F1: {v["birads"]["macro_f1"]:.4f}, QWK: {v["birads"]["qwk"]:.4f}, Macro AUC: {v["birads"]["macro_auc"]:.4f}
- Density Macro F1: {v["density"]["macro_f1"]:.4f}, QWK: {v["density"]["qwk"]:.4f}, Macro AUC: {v["density"]["macro_auc"]:.4f}
- Selection score (mean Macro F1): {v["selection_score"]:.4f}

## Selection rationale
{rationale}

## Lineage
{chr(10).join(f"- {k}: {val}" for k, val in lineage.items())}

## Limitations
Single dataset and single seed; Validation-based selection only; no external or Test-set evaluation; images have
different input resolutions across backbones; labels originate from one population and acquisition setting.
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return Path(path)
