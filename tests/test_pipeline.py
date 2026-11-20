import json

import numpy as np
import pandas as pd

from sanvia.data import age_processing as ap
from sanvia.data.audit import audit_findings
from sanvia.data.image_indexing import index_png_files
from sanvia.evaluation.backbone_comparison import build_comparison_table, select_top_backbones
from sanvia.mlops.data_versioning import compute_dataset_id, manifest_checksum, verify_png_manifest


def test_pipeline_outputs_and_counts(prepared):
    p, s = prepared["paths"], prepared["samples"]["samples"]
    for f in (
        "data_audit_summary.json",
        "csv_schema_report.json",
        "image_audit.csv",
        "annotation_audit.csv",
        "label_distribution.csv",
        "duplicate_records.csv",
        "missing_images.csv",
        "corrupted_images.csv",
        "ambiguous_image_mappings.csv",
    ):
        assert (p.audit_dir / f).exists(), f
    for f in (
        "train_studies.csv",
        "validation_studies.csv",
        "official_test_studies.csv",
        "final_sample_manifest.csv",
        "split_summary.json",
        "split_distribution.csv",
    ):
        assert (p.splits_dir / f).exists(), f
    assert len(s) == 80 and s["final_split"].value_counts().to_dict() == {"train": 48, "official_test": 20, "validation": 12}
    assert not prepared["audit"]["critical_issues"] and not prepared["samples"]["issues"]
    for col in (
        "sample_id",
        "cc_image_id",
        "mlo_image_id",
        "birads_encoded",
        "density_encoded",
        "age",
        "age_missing_flag",
        "official_split",
        "final_split",
    ):
        assert col in s.columns
    assert (s.loc[s.final_split == "validation", "official_split"] == "training").all()
    td = prepared["td"]
    assert set(td.train_df.final_split) == {"train"} and set(td.val_df.final_split) == {"validation"}


def test_age_verified_and_fitted_on_train_only(prepared):
    td = prepared["td"]
    assert td.age_enabled and td.age_processor is not None
    train_ages = td.train_df.drop_duplicates("study_id")["age"].to_numpy(float)
    assert abs(td.age_processor.median - np.median(train_ages)) < 1e-9
    feats = td.age_processor.transform(td.val_df["age"].to_numpy(float))
    assert feats.shape[1] == 2 and set(np.unique(feats[:, 1])) <= {0.0, 1.0}


def test_age_disabled_when_join_not_verifiable(synth):
    cfg, paths = synth
    meta = pd.read_csv(paths.metadata_csv)
    meta["SOP Instance UID"] = ["zzz" + str(i) for i in range(len(meta))]
    breast = pd.read_csv(paths.breast_annotations_csv)
    report, _ = ap.verify_age_join(meta, breast, cfg["data"]["age"], "image_id", "study_id")
    enabled, reason = ap.decide_age_enabled("auto", report)
    assert not report["verified"] and not enabled and "match rate" in reason


def test_age_disabled_without_column(synth):
    cfg, paths = synth
    meta = pd.read_csv(paths.metadata_csv).drop(columns=["Patient's Age"])
    report, _ = ap.verify_age_join(meta, pd.read_csv(paths.breast_annotations_csv), cfg["data"]["age"], "image_id", "study_id")
    assert not ap.decide_age_enabled("auto", report)[0]


def test_age_parsing_and_conflicts():
    s = ap.parse_age(pd.Series(["057Y", "63", "abc", "12 years", None]))
    assert s.iloc[0] == 57 and s.iloc[1] == 63 and s.iloc[2:].isna().all()
    breast = pd.DataFrame({"study_id": ["s", "s", "t"], "image_id": ["a", "b", "c"]})
    ages = pd.DataFrame({"image_id": ["a", "b", "c"], "age": [50.0, 51.0, 40.0]})
    study_age, summary = ap.derive_study_age(breast, ages, "image_id", "study_id")
    assert summary["n_conflicting_studies"] == 1 and np.isnan(study_age.set_index("study_id").loc["s", "age"])


def test_png_manifest_checksums_and_verification(prepared):
    m, idx, p = prepared["manifest"], prepared["index"], prepared["paths"]
    assert {"image_id", "relative_path", "file_size", "sha256", "width", "height", "bit_depth"} <= set(m.columns)
    assert m["sha256"].str.len().eq(64).all() and len(m) == 160
    assert verify_png_manifest(m, idx)["identical"]
    victim = p.images_dir / m.loc[0, "relative_path"]
    victim.write_bytes(victim.read_bytes()[:-1] + b"\x00")
    rep = verify_png_manifest(m, index_png_files(p.images_dir))
    assert rep["changed"] == [m.loc[0, "relative_path"]] and not rep["identical"]


def test_dataset_id_stable_and_manifest_written(prepared):
    rec = json.loads((prepared["paths"].manifests_dir / "dataset_manifest.json").read_text())
    assert rec["dataset_id"] == compute_dataset_id(rec["csv_checksums"], rec["png_manifest_checksum"])
    assert rec["n_valid_breast_samples"] == 80 and rec["sample_manifest_checksum"] and rec["split_manifest_checksum"]
    assert manifest_checksum(prepared["manifest"]) == rec["png_manifest_checksum"]


def test_finding_audit_detects_bad_boxes(prepared):
    f = pd.read_csv(prepared["paths"].finding_annotations_csv)
    f.loc[0, "xmax"] = 999
    f.loc[1, "xmin"] = 50.0
    s = audit_findings(f, prepared["manifest"])
    assert s["n_outside_png_bounds"] >= 1 and s["n_invalid_ordering"] >= 1 and s["category_counts"]["Mass"] == len(f)


def _summary(name, b, d, gap=0.05):
    pc = {"recall": [0.5] * 5, "precision": [0.5] * 5, "f1": [0.5] * 5}
    return {
        "backbone": name,
        "run_id": name,
        "input_size": 224,
        "validation": {
            "birads": {"macro_f1": b, "qwk": 0.5, "macro_auc": 0.7, "per_class": pc},
            "density": {"macro_f1": d, "qwk": 0.5, "macro_auc": 0.7, "per_class": pc},
            "selection_score": (b + d) / 2,
        },
        "train_val_f1_gap": gap,
        "best_epoch": 3,
        "training_time_s": 60,
        "avg_epoch_time_s": 5,
        "peak_gpu_mb": 100,
        "params_total": 10,
        "params_trainable": 10,
        "checkpoint_mb": 1,
        "inference_ms_per_batch": 1,
    }


def test_selection_uses_validation_score_and_guards():
    table = build_comparison_table(
        [_summary("A", 0.30, 0.95), _summary("B", 0.60, 0.60), _summary("C", 0.55, 0.62), _summary("D", 0.58, 0.58, gap=0.6)]
    )
    res = select_top_backbones(table, 2, 0.85, 0.30)
    assert res["selected"] == ["B", "C"]  # A has the top density F1 but weak BI-RADS; D overfits
    assert "Validation results only" in res["rationale"]
