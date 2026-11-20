"""Data preparation steps shared by the notebook and the command-line script."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sanvia.data import age_processing as ap
from sanvia.data.audit import run_data_audit
from sanvia.data.dataset import BreastPairDataset, PairedAugmenter, age_features_for
from sanvia.data.image_indexing import find_unexpected_files, index_png_files, match_annotations_to_images
from sanvia.data.labels import LabelSpace, build_label_spaces
from sanvia.data.loading import build_schema_report, load_tables, validate_required_columns
from sanvia.data.preprocessing import (
    IntensityNormalizer,
    PNGPreprocessor,
    fit_global_window,
    validate_normalization_against_audit,
)
from sanvia.data.sample_builder import build_breast_samples, sample_distribution
from sanvia.data.splitting import build_split, split_summary, study_label_distribution
from sanvia.mlops.data_versioning import build_png_manifest, checksum_files, manifest_checksum, write_dataset_manifest
from sanvia.models.backbone_factory import BackboneSpec
from sanvia.training.losses import compute_class_weights
from sanvia.utils.io import save_json, sha256_file
from sanvia.utils.paths import ProjectPaths
from sanvia.utils.seed import make_generator, seed_worker

logger = logging.getLogger(__name__)


@dataclass
class TrainingData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    birads_space: LabelSpace
    density_space: LabelSpace
    normalizer: IntensityNormalizer
    class_weights: dict[str, np.ndarray]
    age_enabled: bool
    age_processor: ap.AgeProcessor | None
    age_reason: str
    dataset_id: str
    dataset_manifest_hash: str
    split_id: str
    extras: dict[str, Any] = field(default_factory=dict)


def csv_files(paths: ProjectPaths) -> dict[str, Path]:
    return {"metadata": paths.metadata_csv, "breast": paths.breast_annotations_csv, "finding": paths.finding_annotations_csv}


def index_and_manifest(cfg: Mapping[str, Any], paths: ProjectPaths) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """Index PNGs, build/save the SHA-256 PNG manifest and the dataset manifest with the dataset id."""
    icfg = cfg["data"]["indexing"]
    index = index_png_files(paths.images_dir, icfg["extension"])
    manifest_path = paths.manifests_dir / "png_image_manifest.csv"
    previous = pd.read_csv(manifest_path) if (manifest_path.exists() and icfg["reuse_previous_manifest_hashes"]) else None
    manifest = build_png_manifest(index, icfg["hash_workers"], previous)
    manifest.to_csv(manifest_path, index=False)
    record = write_dataset_manifest(
        paths.manifests_dir / "dataset_manifest.json",
        cfg["mlops"]["dataset_release_id"],
        checksum_files(csv_files(paths)),
        manifest_checksum(manifest),
        int(len(manifest)),
    )
    return index, manifest, record


def audit_stage(cfg: Mapping[str, Any], paths: ProjectPaths, index, manifest: pd.DataFrame) -> dict[str, Any]:
    """Load CSVs, validate columns, match images, run the audit and evaluate the integrity gates."""
    cols = cfg["data"]["columns"]
    tables = load_tables(paths)
    validate_required_columns(tables["breast"], cols.values(), "breast-level_annotations.csv")
    schema = build_schema_report(tables, csv_files(paths))
    birads_space, density_space = build_label_spaces(cfg)
    match_df = match_annotations_to_images(tables["breast"][cols["image_id"]], index, cfg["data"]["indexing"]["extension"])
    unexpected = find_unexpected_files(index, set(match_df["image_id"]))
    audit = run_data_audit(
        tables, manifest, paths.images_dir, paths.audit_dir, schema, cfg, match_df, unexpected, birads_space, density_space
    )
    audit.update(tables=tables, match_df=match_df, birads_space=birads_space, density_space=density_space)
    issues = evaluate_gates(audit["summary"], cfg)
    save_json({"critical_issues": issues, "passed": not issues}, paths.audit_dir / "integrity_gate.json")
    audit["critical_issues"] = issues
    return audit


def evaluate_gates(summary: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[str]:
    """Critical conditions that must be resolved before any training."""
    issues = []
    ann = summary["annotations"]
    if ann["unexpected_laterality_values"] or ann["unexpected_view_values"]:
        issues.append(f"Unexpected laterality/view values: {ann['unexpected_laterality_values']} {ann['unexpected_view_values']}")
    if ann["n_unknown_label_records"] and not cfg["data"]["sample_construction"]["allow_unknown_labels"]:
        issues.append(f"Unknown label strings found: {ann['unknown_labels']}")
    if summary["images"]["n_images"] == 0:
        issues.append("No PNG images were indexed.")
    return issues


def split_stage(cfg: Mapping[str, Any], paths: ProjectPaths, audit: Mapping[str, Any]) -> dict[str, Any]:
    """Create Train/Validation from official Training only; save split artifacts."""
    cols = cfg["data"]["columns"]
    split = build_split(audit["tables"]["breast"], cols, cfg["data"]["split"], audit["birads_space"], audit["density_space"])
    s = split["studies"]
    d = paths.splits_dir
    d.mkdir(parents=True, exist_ok=True)
    keep = ["study_id", "official_split", "final_split", "sig_full", "sig_birads"]
    s[s["final_split"] == "train"][keep].to_csv(d / "train_studies.csv", index=False)
    s[s["final_split"] == "validation"][keep].to_csv(d / "validation_studies.csv", index=False)
    s[s["final_split"] == "official_test"][keep].to_csv(d / "official_test_studies.csv", index=False)
    study_label_distribution(s).to_csv(d / "study_stratification_check.csv", index=False)
    split["summary"] = split_summary(split, cfg["data"]["split"])
    save_json(split["summary"], d / "split_summary.json")
    return split


def sample_stage(
    cfg: Mapping[str, Any], paths: ProjectPaths, audit: Mapping[str, Any], split: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify age, build breast-level samples, write the manifest and update dataset/split reports."""
    cols, dcfg = cfg["data"]["columns"], cfg["data"]
    tables = audit["tables"]
    age_report, image_age = ap.verify_age_join(
        tables["metadata"], tables["breast"], dcfg["age"], cols["image_id"], cols["study_id"]
    )
    enabled, reason = ap.decide_age_enabled(dcfg["age"]["mode"], age_report)
    study_age, age_summary = (
        ap.derive_study_age(tables["breast"], image_age, cols["image_id"], cols["study_id"]) if enabled else (None, {})
    )
    if study_age is not None:
        study_age["study_id"] = study_age["study_id"].astype(str)
    save_json(
        {"enabled": enabled, "reason": reason, "join": age_report, "study_age": age_summary}, paths.splits_dir / "age_report.json"
    )
    studies = split["studies"]
    corrupted = set(audit["corrupted"]["image_id"].astype(str))
    sc = dcfg["sample_construction"]
    samples, excluded = build_breast_samples(
        tables["breast"],
        audit["match_df"],
        corrupted,
        cols,
        audit["birads_space"],
        audit["density_space"],
        sc["laterality_values"],
        sc["view_values"],
        dict(zip(studies["study_id"], studies["final_split"])),
        dict(zip(studies["study_id"], studies["official_split"])),
        study_age,
        sc["allow_unknown_labels"],
    )
    samples.to_csv(paths.splits_dir / "final_sample_manifest.csv", index=False)
    excluded.to_csv(paths.splits_dir / "excluded_samples.csv", index=False)
    dist = sample_distribution(samples, audit["birads_space"], audit["density_space"])
    dist.to_csv(paths.splits_dir / "split_distribution.csv", index=False)
    n_expected = len(samples) + len(excluded)
    frac = len(excluded) / max(n_expected, 1)
    issues = []
    if frac > dcfg["audit"]["max_excluded_sample_fraction"]:
        issues.append(
            f"{frac:.1%} of breasts were excluded (limit {dcfg['audit']['max_excluded_sample_fraction']:.0%}); see excluded_samples.csv"
        )
    for name in ("train", "validation"):
        if (samples["final_split"] == name).sum() == 0:
            issues.append(f"No valid samples in {name}")
    summary = dict(split["summary"])
    summary["n_samples"] = samples["final_split"].value_counts().to_dict()
    summary["n_excluded_samples"] = len(excluded)
    summary["exclusion_reasons"] = excluded["reasons"].str.split(";").explode().str.split(":").str[0].value_counts().to_dict()
    save_json(summary, paths.splits_dir / "split_summary.json")
    split_files = [paths.splits_dir / f for f in ("train_studies.csv", "validation_studies.csv", "official_test_studies.csv")]
    split_checksum = manifest_checksum(
        pd.DataFrame({"f": [f.name for f in split_files], "h": [sha256_file(f) for f in split_files]})
    )
    record = write_dataset_manifest(
        paths.manifests_dir / "dataset_manifest.json",
        cfg["mlops"]["dataset_release_id"],
        checksum_files(csv_files(paths)),
        manifest_checksum(pd.read_csv(paths.manifests_dir / "png_image_manifest.csv")),
        int(pd.read_csv(paths.manifests_dir / "png_image_manifest.csv").shape[0]),
        {
            "n_valid_breast_samples": int(len(samples)),
            "sample_manifest_checksum": sha256_file(paths.splits_dir / "final_sample_manifest.csv"),
            "split_manifest_checksum": split_checksum,
            "split_id": split["split_id"],
        },
    )
    return {
        "samples": samples,
        "excluded": excluded,
        "age_enabled": enabled,
        "age_reason": reason,
        "age_report": age_report,
        "issues": issues,
        "dataset_record": record,
        "distribution": dist,
    }


def make_training_data(
    cfg: Mapping[str, Any],
    paths: ProjectPaths,
    samples: pd.DataFrame,
    audit: Mapping[str, Any],
    age_enabled: bool,
    age_reason: str,
    dataset_record: Mapping[str, Any],
    split_id: str,
) -> TrainingData:
    """Fit all data-derived preprocessing on Train only and assemble the training bundle."""
    seed = cfg["project"]["seed"]
    train_df = samples[samples["final_split"] == "train"].reset_index(drop=True)
    val_df = samples[samples["final_split"] == "validation"].reset_index(drop=True)
    norm_cfg = cfg["data"]["normalization"]
    norm_report = validate_normalization_against_audit(audit["image_audit"], norm_cfg)
    if norm_cfg["strategy"] == "global_window":
        normalizer = fit_global_window(
            pd.concat([train_df["cc_image_path"], train_df["mlo_image_path"]]).tolist(), norm_cfg, seed
        )
    else:
        normalizer = IntensityNormalizer("dtype_range")
    weights = {
        "birads": compute_class_weights(
            train_df.loc[train_df["birads_valid"], "birads_encoded"].astype(int).to_numpy(),
            5,
            cfg["training"]["class_weights"]["scheme"],
        ),
        "density": compute_class_weights(
            train_df.loc[train_df["density_valid"], "density_encoded"].astype(int).to_numpy(),
            4,
            cfg["training"]["class_weights"]["scheme"],
        ),
    }
    processor = None
    if age_enabled:
        processor = ap.AgeProcessor().fit(train_df.drop_duplicates("study_id")["age"].to_numpy(dtype=float))
        processor.save(paths.preprocessing_dir / "age_processor.json")
    save_json(
        {
            "normalizer": normalizer.to_dict(),
            "validation": norm_report,
            "class_weights": {k: v.tolist() for k, v in weights.items()},
            "age_processor": None if processor is None else processor.to_dict(),
            "age_reason": age_reason,
        },
        paths.preprocessing_dir / "fitted_preprocessing.json",
    )
    return TrainingData(
        train_df,
        val_df,
        audit["birads_space"],
        audit["density_space"],
        normalizer,
        weights,
        age_enabled,
        processor,
        age_reason,
        dataset_record["dataset_id"],
        dataset_record["dataset_manifest_hash"],
        split_id,
        {"normalization_report": norm_report},
    )


def make_dataloaders(td: TrainingData, spec: BackboneSpec, cfg: Mapping[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Train loader with augmentation, deterministic Validation loader. No Test loader exists in Stage 1."""
    pre = PNGPreprocessor(spec.input_size, td.normalizer, cfg["data"]["preprocessing"])
    aug_cfg = cfg["data"]["augmentation"]
    augmenter = PairedAugmenter(aug_cfg) if aug_cfg["enabled"] else None
    age_tr = age_features_for(td.train_df, td.age_processor) if td.age_enabled else None
    age_va = age_features_for(td.val_df, td.age_processor) if td.age_enabled else None
    t = cfg["training"]
    common = dict(
        batch_size=t["batch_size"],
        num_workers=t["num_workers"],
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        persistent_workers=t["num_workers"] > 0,
    )
    train = DataLoader(
        BreastPairDataset(td.train_df, pre, augmenter, age_tr), shuffle=True, generator=make_generator(t["seed"]), **common
    )
    val = DataLoader(BreastPairDataset(td.val_df, pre, None, age_va), shuffle=False, **common)
    return train, val
