"""Read a private Stage 1 output snapshot without running its indexing or audit pipeline."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sanvia.data.age_processing import AgeProcessor
from sanvia.data.labels import build_label_spaces
from sanvia.data.pipeline import TrainingData
from sanvia.data.preprocessing import IntensityNormalizer
from sanvia.data.splitting import compute_split_id
from sanvia.mlops.data_versioning import compute_dataset_id, manifest_checksum
from sanvia.models.backbone_factory import get_backbone_spec, weights_metadata
from sanvia.training.losses import compute_class_weights
from sanvia.utils.config import config_hash, run_config_subset, validate_config
from sanvia.utils.io import load_json, sha256_file


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def local_path(root: Path, relative: str) -> Path:
    """Reject stale absolute paths and symlinks escaping the configured dataset root."""
    path = (root / relative).resolve()
    require(not Path(relative).is_absolute() and path.is_relative_to(root.resolve()), f"Unsafe relative path {relative!r}")
    return path


@dataclass
class Snapshot:
    config: dict
    selected: list[str]
    data: TrainingData
    baselines: dict
    report: dict
    png_hashes: set[str]


def read_target(root: Path, images: Path, cfg: dict, dataset: dict) -> tuple[TrainingData, set[str]]:
    """Verify study files and target bytes. Use test identifiers only for integrity."""
    splits = root / "splits"
    development = {}
    groups = []
    files = {}
    settings = cfg["data"]["split"]
    for partition in ("train", "validation", "official_test"):
        path = splits / f"{partition}_studies.csv"
        files[path.name] = sha256_file(path)
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=["study_id","official_split","final_split"])
        official = settings["official_test_value" if partition == "official_test" else "official_train_value"]
        ids = frame.study_id.tolist()
        require(ids and all(ids) and len(set(ids)) == len(ids) and not set(ids).intersection(development), "Overlapping or missing study IDs")
        require(set(frame.official_split) == {official} and set(frame.final_split) == {partition}, f"Incorrect study partition in {path}")
        development.update(dict.fromkeys(ids, partition))
        groups.append(sorted(ids))
    digest = manifest_checksum(pd.DataFrame({"f": list(files), "h": list(files.values())}))
    summary = load_json(splits / "split_summary.json")
    sid = compute_split_id(settings["seed"],settings["val_fraction"],summary["stratification_level"],*groups)
    require(digest == dataset["split_manifest_checksum"] and sid == dataset["split_id"] == summary["split_id"], "Stage 1 split fingerprint mismatch")
    path = splits / "final_sample_manifest.csv"
    require(sha256_file(path) == dataset["sample_manifest_checksum"], "Stage 1 sample manifest checksum mismatch")
    png_path = root / "manifests/png_image_manifest.csv"
    require(manifest_checksum(pd.read_csv(png_path)) == dataset["png_manifest_checksum"], "Stage 1 PNG manifest checksum mismatch")
    require(compute_dataset_id(dataset["csv_checksums"], dataset["png_manifest_checksum"]) == dataset["dataset_id"], "Stage 1 dataset ID mismatch")
    index = pd.read_csv(png_path,dtype=str,keep_default_na=False).set_index("relative_path",verify_integrity=True)
    rows = []
    image_splits = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            part = development.get(row["study_id"])
            require(part in {"train","validation","official_test"} and row["final_split"] == part, "Sample/study split mismatch")
            official = settings["official_test_value" if part == "official_test" else "official_train_value"]
            require(row["official_split"] == official, "Sample official assignment mismatch")
            for view in ("cc","mlo"):
                rel = row[f"{view}_image_relpath"]
                entry = index.loc[rel]
                require(entry.image_id == row[f"{view}_image_id"], "Image identifier disagrees with PNG manifest")
                require(image_splits.setdefault(entry.sha256,part) == part,"Image fingerprint crosses target splits")
            if part != "official_test":
                rows.append(row)
    frame = pd.DataFrame(rows)
    require(not frame.sample_id.duplicated().any() and set(frame.final_split)=={"train","validation"},"Invalid eligible target sample identities")
    for view in ("cc","mlo"):
        paths = []
        for rel in frame[f"{view}_image_relpath"]:
            actual = local_path(images,rel)
            require(sha256_file(actual) == index.at[rel,"sha256"], f"Changed eligible VinDr image {rel}")
            paths.append(str(actual))
        frame[f"{view}_image_path"] = paths
    spaces = build_label_spaces(cfg)
    for space in spaces:
        task = space.name
        mask = frame[f"{task}_valid"].map({"True":True,"False":False})
        require(mask.notna().all(),f"Invalid {task} mask")
        frame[f"{task}_valid"] = mask
        frame[f"{task}_encoded"] = pd.to_numeric(frame[f"{task}_encoded"])
        raw = frame.loc[mask,f"{task}_original"].str.strip().str.upper().map(space.mapping)
        require(raw.notna().all() and (raw == frame.loc[mask,f"{task}_encoded"]).all(),f"Invalid {task} encoding")
    frame["age"] = pd.to_numeric(frame.age)
    train, val = [frame[frame.final_split == p].reset_index(drop=True) for p in ("train","validation")]
    fitted = load_json(root / "preprocessing/fitted_preprocessing.json")
    age = AgeProcessor.from_dict(fitted["age_processor"]) if fitted["age_processor"] else None
    age_report = load_json(splits/"age_report.json")
    require(bool(age) == age_report["enabled"],"Inconsistent age artifacts")
    if age:
        expected = AgeProcessor().fit(train.drop_duplicates("study_id").age.to_numpy()).to_dict()
        require(all(np.isclose(expected[k],v) for k,v in age.to_dict().items()),"Age processor differs from training-only fit")
    weights = {}
    for space in spaces:
        labels = train.loc[train[f"{space.name}_valid"], f"{space.name}_encoded"].astype(int)
        weights[space.name] = compute_class_weights(labels.to_numpy(),space.num_classes,cfg["training"]["class_weights"]["scheme"])
        require(np.allclose(weights[space.name],fitted["class_weights"][space.name]),"Target class weights differ from training-only fit")
    return TrainingData(train,val,*spaces,IntensityNormalizer.from_dict(fitted["normalizer"]),weights,bool(age),age,
                        fitted["age_reason"],dataset["dataset_id"],dataset["dataset_manifest_hash"],sid),set(index.sha256)


def read_snapshot(root: Path, images: Path, pair: list[str] | None = None) -> Snapshot:
    """Read actual Stage 1 output files. A missing baseline can be reproduced on the saved split."""
    reports = root / "reports"
    report = load_json(reports / "reproducibility_report.json")
    decision = load_json(reports / "selection_decision.json")
    cfg = yaml.safe_load((reports / "resolved_config.yaml").read_text())
    validate_config(cfg)
    require(report["official_test_used"] is False and decision == report["final_selection"], "Unverified Stage 1 selection")
    require(config_hash(cfg) == report["config_hash"],"Stage 1 resolved configuration mismatch")
    selected = pair if pair is not None else decision["selected"]
    require(len(selected)==2 and len(set(selected))==2,"Exactly two distinct backbones required")
    dataset = load_json(root/"manifests/dataset_manifest.json")
    require(dataset == report["dataset"],"Dataset and reproducibility report disagree")
    td, hashes = read_target(root,images,cfg,dataset)
    baselines = {}
    for name in selected:
        spec = get_backbone_spec(cfg,name)
        runs = [e for e in report["experiments"] if e["backbone"]==name]
        require(len(runs)==1, f"Expected one Stage 1 run for {name}")
        run = local_path(root/"checkpoints",f"{name}/{runs[0]['run_id']}")
        summary = load_json(run/"run_summary.json")
        expected = dict(backbone=name, run_id=runs[0]["run_id"], dataset_id=td.dataset_id,split_id=td.split_id,input_size=spec.input_size,
                        weights_identifier=weights_metadata(spec)["weights_identifier"],config_hash=config_hash(run_config_subset(cfg,name)),
                        git_commit=report["git"].get("commit"))
        issues = [k for k,v in expected.items() if summary.get(k)!=v]
        run_cfg = yaml.safe_load((run/"resolved_config.yaml").read_text())
        if run_config_subset(run_cfg,name)!=run_config_subset(cfg,name):
            issues.append("resolved target protocol")
        if summary["validation"]["selection_score"]!=runs[0]["validation_selection_score"]:
            issues.append("validation metrics")
        predictions = run/"validation_predictions.csv"
        if predictions.exists():
            ids=pd.read_csv(predictions,usecols=["sample_id"],dtype=str).sample_id
            if sorted(ids)!=sorted(td.val_df.sample_id):
                issues.append("eligible validation samples")
        # Old summaries record the enum even when pretrained=False. Matched
        # reproduction supplies positive provenance for each seed and package version.
        issues.append("historical summary does not prove ImageNet initialization")
        baselines[name] = dict(summary=summary,checkpoint=run/"best.pt",predictions=predictions, incompatibilities=issues)
    return Snapshot(cfg,list(selected),td,baselines,report,hashes)

