"""Generate notebooks/Sanvia_Stage1_Backbone_Benchmark.ipynb (run once; the notebook is committed)."""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md("""
# SANVIA Stage 1: Reproducible Multi-Task Mammography Backbone Benchmark

Compares six ImageNet-pretrained CNN backbones (ResNet50, DenseNet121, InceptionV3, MobileNetV3-Large,
EfficientNet-B4, EfficientNetV2-S) on VinDr-Mammo PNG images. One sample is one breast (CC + MLO).
Tasks: BI-RADS (5 classes) and breast density (4 classes). The two best backbones are selected on **Validation only**;
the official Test split is never used for training, tuning, checkpoint selection or backbone selection.

This notebook only orchestrates the modular code in `src/sanvia`. Set `REPO_DIR` below to the repository location
(clone it into Google Drive or `/content`). Runs are resumable: re-running Section 13 continues interrupted runs.
""")

md("## Section 1 - Google Drive and environment setup")
code("""
import os
import sys
from pathlib import Path

# Must be set before CUDA is initialised: reduces memory fragmentation for the large EfficientNet inputs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ON_COLAB = "google.colab" in sys.modules
REPO_DIR = Path("/content/drive/MyDrive/SANVIA")   # location of the cloned repository
OVERRIDES = {}        # optional configuration overrides, e.g. {"training": {"num_workers": 2}}
PRETRAINED = True     # ImageNet weights; keep True for the real benchmark
RESUME = True         # continue interrupted runs of identical configuration/dataset/split

if ON_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
""")
code("""
if ON_COLAB:
    %pip install -q -r {REPO_DIR}/requirements.txt
    %pip install -q -e {REPO_DIR} --no-deps
""")
code("""
import platform
import torch
import torchvision
import pandas as pd
from IPython.display import Image, display

from sanvia.utils.config import config_hash, load_config, save_resolved_config
from sanvia.utils.environment import collect_environment
from sanvia.utils.logging import setup_logging
from sanvia.utils.paths import build_paths, ensure_output_dirs, verify_input_paths
from sanvia.utils.seed import NONDETERMINISM_NOTES, set_global_seed

cfg = load_config(REPO_DIR / "configs", OVERRIDES)
setup_logging(cfg["logging"]["level"])
paths = build_paths(cfg)
verify_input_paths(paths)        # raises a clear error naming every missing path
ensure_output_dirs(paths)

print("Python     :", platform.python_version())
print("PyTorch    :", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA       :", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU        : {props.name}, {props.total_memory / 1024**3:.1f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True      # faster matmuls on Ampere+, negligible effect on accuracy
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
else:
    print("WARNING: no GPU detected. Training six backbones on CPU is impractical; select a GPU runtime.")
""")

md("## Section 2 - Project configuration")
code("""
# load_config already validated: seeds consistent, no full-image DVC tracking, no Test evaluation, six backbones.
set_global_seed(cfg["project"]["seed"], cfg["training"]["deterministic"])
save_resolved_config(cfg, paths.reports_dir / "resolved_config.yaml")
CONFIG_HASH = config_hash(cfg)
print("Configuration hash:", CONFIG_HASH)
print("Output directory  :", paths.output_dir)
print("Known sources of nondeterminism:")
for note in NONDETERMINISM_NOTES:
    print(" -", note)
""")

md("## Section 3 - Git initialization and metadata")
code("""
from sanvia.mlops.git_metadata import get_git_metadata, reproducibility_statement
from sanvia.utils.io import save_json

git = get_git_metadata(REPO_DIR)
env = collect_environment()
save_json({"git": git, "environment": env}, paths.reports_dir / "environment_and_git.json")
print("Commit :", git["commit"], "| branch:", git["branch"], "| dirty:", git["dirty"])
print(reproducibility_statement(git))
display(pd.Series(env["packages"], name="version").to_frame())
""")

md("""
## Section 4 - DVC initialization and dataset versioning

DVC tracks only small files (CSVs, manifests, splits). The PNG images are **not** copied or tracked by DVC;
they are verified through a SHA-256 manifest. The manifest detects changes but cannot restore old image versions.
""")
code("""
from sanvia.data import pipeline as pl
from sanvia.data.image_indexing import index_png_files
from sanvia.mlops.data_versioning import DVCManager, verify_png_manifest

dvc_cfg = cfg["mlops"]["dvc"]
assert dvc_cfg["track_full_images"] is False

manifest_path = paths.manifests_dir / "png_image_manifest.csv"
if dvc_cfg["verify_image_checksums"] and manifest_path.exists():
    previous = pd.read_csv(manifest_path)
    report = verify_png_manifest(previous, index_png_files(paths.images_dir, cfg["data"]["indexing"]["extension"]),
                                 cfg["data"]["indexing"]["hash_workers"])
    print("Previous manifest verification:", {k: (v if not isinstance(v, list) else len(v)) for k, v in report.items()})
    if not report["identical"]:
        print("WARNING: images differ from the saved manifest; a new dataset id will be created.")

index, png_manifest, dataset_record = pl.index_and_manifest(cfg, paths)
print("Indexed PNG files :", len(png_manifest))
print("Dataset id        :", dataset_record["dataset_id"])
print("PNG manifest hash :", dataset_record["png_manifest_checksum"][:16])
display(png_manifest.head())
""")
code("""
DVC_ENABLED = dvc_cfg["enabled"]
dvc_status = "disabled"
if DVC_ENABLED:
    try:
        dvc = DVCManager(REPO_DIR, cfg, paths.dvc_remote_dir)   # persistent remote on Google Drive (small files only)
        dvc.ensure_initialized()
        tracked = {}
        if dvc_cfg["track_csv_files"]:
            tracked.update({p.name: p for p in (paths.metadata_csv, paths.breast_annotations_csv, paths.finding_annotations_csv)})
        if dvc_cfg["track_manifests"]:
            tracked.update({"png_image_manifest.csv": manifest_path, "dataset_manifest.json": paths.manifests_dir / "dataset_manifest.json"})
        dvc.stage_files(tracked)
        dvc.push()
        dvc_status = "tracked and pushed; remote in sync" if dvc.remote_in_sync() else "pushed; remote NOT in sync"
    except Exception as exc:       # DVC problems must not corrupt the experiment; they are reported, not hidden
        dvc_status = f"failed: {type(exc).__name__}: {exc}"
print("DVC status:", dvc_status)
print("Remember to `git add data/tracked/*.dvc .dvc dvc.yaml` and commit before training.")
""")

md("## Section 5 - Data loading and audit")
code("""
audit = pl.audit_stage(cfg, paths, index, png_manifest)
s = audit["summary"]
print("Images:", s["images"]["n_images"], "| unreadable:", s["images"]["n_unreadable"], "| bit depths:", s["images"]["bit_depth_counts"])
print("Studies:", s["annotations"]["n_unique_studies"], "| unique images:", s["annotations"]["n_unique_images"],
      "| studies without 4 images:", s["annotations"]["studies_without_four_images"])
print("Missing BI-RADS:", s["annotations"]["birads_missing"], "| missing density:", s["annotations"]["density_missing"],
      "| unknown labels:", s["annotations"]["unknown_labels"])
print("Image mapping:", s["mapping"])
print("Duplicate records:", s["annotations"]["n_duplicate_records"],
      "| inconsistent breasts (BI-RADS/density):", s["annotations"]["n_breasts_birads_inconsistent"], s["annotations"]["n_breasts_density_inconsistent"])
print("Findings:", {k: v for k, v in s["findings"].items() if k != "columns"})
for plot in audit["plots"]:
    display(Image(filename=str(plot)))
if audit["critical_issues"]:
    raise RuntimeError(f"Critical data issues, training blocked: {audit['critical_issues']}")
print("Integrity gate passed.")
""")

md("## Section 6 - Official split verification and internal Train/Validation split")
code("""
split = pl.split_stage(cfg, paths, audit)
print("Official split (studies):", split["verification"]["studies_per_official_split"])
print("Split id:", split["split_id"], "| stratification:", split["stratification_level"])
print("Studies  train / validation / official test:", len(split["train_ids"]), len(split["val_ids"]), len(split["test_ids"]))
""")

md("## Section 7 - Breast-level sample construction")
code("""
out = pl.sample_stage(cfg, paths, audit, split)
samples, excluded = out["samples"], out["excluded"]
print("Valid breast samples:", len(samples), "| excluded:", len(excluded))
print(samples["final_split"].value_counts().to_string())
print("Age integration:", out["age_reason"])
if len(excluded):
    print("Exclusion reasons:")
    print(excluded["reasons"].str.split(";").explode().str.split(":").str[0].value_counts().to_string())
display(out["distribution"].pivot_table(index=["task", "class"], columns="final_split", values="count", aggfunc="sum"))
display(samples.head())
if out["issues"]:
    raise RuntimeError(f"Critical sample-construction issues, training blocked: {out['issues']}")
if DVC_ENABLED and dvc_status.startswith("tracked"):
    dvc.stage_files({p.name: p for p in [paths.splits_dir / "final_sample_manifest.csv", paths.splits_dir / "train_studies.csv",
                                         paths.splits_dir / "validation_studies.csv", paths.splits_dir / "official_test_studies.csv",
                                         paths.manifests_dir / "dataset_manifest.json"]})
    dvc.push()
""")

md("""
## Section 8 - Preprocessing

Normalisation is derived from the PNG audit (no blind division by 255). Images are letterboxed (aspect ratio preserved,
no center crop), replicated to three channels and ImageNet-normalised. Fitted quantities (class weights, age
imputation/scaling, optional global window) use Train only.
""")
code("""
td = pl.make_training_data(cfg, paths, samples, audit, out["age_enabled"], out["age_reason"], out["dataset_record"], split["split_id"])
print("Normalisation:", td.normalizer.to_dict())
print("Audit-based validation:", td.extras["normalization_report"])
print("Class weights BI-RADS:", td.class_weights["birads"].round(3).tolist())
print("Class weights density:", td.class_weights["density"].round(3).tolist())
print("Age branch enabled:", td.age_enabled, "|", td.age_reason)

from sanvia.data.preprocessing import PNGPreprocessor, plot_preprocessing_examples
for size in sorted({b["input_size"] for b in cfg["backbones"]}):
    pre = PNGPreprocessor(size, td.normalizer, cfg["data"]["preprocessing"])
    info = plot_preprocessing_examples(td.train_df, pre, paths.preprocessing_dir / f"preprocessing_{size}.png", n=2)
    display(Image(filename=str(paths.preprocessing_dir / f"preprocessing_{size}.png")))
save_json(info, paths.preprocessing_dir / "last_preprocessing_example_info.json")
""")

md("## Section 9 - Dataset and DataLoader creation")
code("""
from sanvia.models.backbone_factory import get_backbone_spec

spec0 = get_backbone_spec(cfg, cfg["backbones"][0]["name"])
train_loader, val_loader = pl.make_dataloaders(td, spec0, cfg)    # Train: augmentation; Validation: deterministic; no Test loader
batch = next(iter(train_loader))
print("Backbone for this check:", spec0.name)
print("images :", tuple(batch["images"].shape), batch["images"].dtype, "(batch, views[CC, MLO], channels, H, W)")
print("birads :", tuple(batch["birads"].shape), "mask valid:", int(batch["birads_mask"].sum()))
print("density:", tuple(batch["density"].shape), "mask valid:", int(batch["density_mask"].sum()))
print("age    :", tuple(batch["age"].shape) if "age" in batch else "disabled")
print("Train samples:", len(td.train_df), "| Validation samples:", len(td.val_df))
del batch
""")

md("## Section 10 - Model initialization (all six backbones)")
code("""
from sanvia.models.multitask_model import build_model
from sanvia.models.backbone_factory import infer_feature_dim, weights_metadata

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rows = []
for b in cfg["backbones"]:
    spec = get_backbone_spec(cfg, b["name"])
    model = build_model(spec, cfg["training"], td.age_enabled, pretrained=PRETRAINED).to(device).eval()
    assert infer_feature_dim(model.backbone.cpu(), 96 if spec.input_size < 300 else 128) == spec.feature_dim
    model.to(device)
    x = torch.zeros(2, 2, 3, spec.input_size, spec.input_size, device=device)
    age = torch.zeros(2, 2, device=device) if td.age_enabled else None
    with torch.no_grad():
        o = model(x, age)
    c = model.parameter_counts()
    rows.append({"backbone": spec.name, "weights": weights_metadata(spec)["weights_identifier"], "input": spec.input_size,
                 "feature_dim": spec.feature_dim, "birads_out": tuple(o["birads"].shape), "density_out": tuple(o["density"].shape),
                 "backbone_params_M": round(c["backbone"] / 1e6, 2), "total_params_M": round(c["total"] / 1e6, 2)})
    del model, x, o
    if device.type == "cuda":
        torch.cuda.empty_cache()
display(pd.DataFrame(rows))
""")

md("""
## Section 11 - MLflow initialization

SQLite backend and artifact store on Google Drive. Valid only for single-user, sequential runs; do not run concurrent writers.
""")
code("""
from sanvia.mlops.mlflow_tracking import setup_mlflow, verify_mlflow_persistence

experiment_id = setup_mlflow(cfg, paths)
check = verify_mlflow_persistence(paths, cfg["mlops"]["mlflow"]["experiment_name"])
print("Experiment id:", experiment_id)
print(check)
assert check["db_exists"] and check["artifact_dir_exists"] and check["readback_ok"]
""")

md("## Section 12 - Automated preflight tests")
code("""
import subprocess

from sanvia.data.splitting import assert_no_leakage

# Runtime checks on the real prepared data
assert_no_leakage(split)
assert set(td.train_df["final_split"]) == {"train"} and set(td.val_df["final_split"]) == {"validation"}
assert not (set(td.train_df["study_id"]) & set(td.val_df["study_id"]))
assert not ((set(td.train_df["study_id"]) | set(td.val_df["study_id"])) & set(split["test_ids"]))
assert (td.train_df["official_split"] == cfg["data"]["split"]["official_train_value"]).all()
assert (td.val_df["official_split"] == cfg["data"]["split"]["official_train_value"]).all()
assert (samples.groupby("study_id")["final_split"].nunique() == 1).all()
assert samples["sample_id"].is_unique and samples["cc_image_id"].ne(samples["mlo_image_id"]).all()
assert all(td.class_weights[k].sum() > 0 for k in td.class_weights)
print("Runtime data checks passed.")

# Unit tests on synthetic data (data handling, splitting, preprocessing, models, losses, metrics, checkpoints)
if (REPO_DIR / "tests").exists():
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", str(REPO_DIR / "tests")], cwd=REPO_DIR, capture_output=True, text=True)
    print(result.stdout[-1500:])
    if result.returncode != 0:
        raise RuntimeError("Preflight unit tests failed; training blocked.")
""")

md("""
## Section 13 - Backbone training

The six backbones are trained sequentially with the same protocol (frozen phase, then gradual fine-tuning).
Each backbone gets its own MLflow run and run directory; previous runs and checkpoints are never overwritten.
If the session is interrupted, re-run this cell: an unfinished run with identical configuration, dataset id and split id is resumed.
""")
code("""
from sanvia.training.benchmark import run_all_backbones

summaries, failures = run_all_backbones(cfg, paths, td, git, env, resume=RESUME, use_mlflow=True, pretrained=PRETRAINED, device=device)
print("Completed:", [s["backbone"] for s in summaries])
print("Failed   :", failures or "none")
""")

md("## Section 14 - Validation evaluation")
code("""
from sanvia.evaluation.backbone_comparison import build_comparison_table

table = build_comparison_table(summaries)
display(table[["backbone", "input_size", "birads_macro_f1", "density_macro_f1", "birads_qwk", "density_qwk", "selection_score", "training_time_min"]])
display(table[["backbone", "birads_macro_auc", "density_macro_auc", "birads_min_class_recall", "density_min_class_recall",
               "train_val_f1_gap", "best_epoch", "avg_epoch_time_s", "peak_gpu_mb", "params_total", "params_trainable",
               "checkpoint_mb", "inference_ms_per_batch"]])
for s in summaries:
    for name in ("learning_curves.png", "confusion_birads.png", "confusion_density.png"):
        display(Image(filename=str(Path(s["run_dir"]) / name)))
""")

md("## Section 15 - Select the best two backbones (Validation only)")
code("""
from sanvia.evaluation.backbone_comparison import select_top_backbones

sel_cfg = cfg["selection"]
preview = select_top_backbones(table, sel_cfg["n_select"], sel_cfg["min_birads_f1_fraction_of_best"], sel_cfg["max_overfit_gap"])
print("Selected:", preview["selected"])
print(preview["rationale"])
if preview["warning"]:
    print("WARNING:", preview["warning"])
display(preview["table"][["backbone", "selection_score", "birads_macro_f1", "density_macro_f1", "guard_issues"]])
""")

md("""
## Section 16 - Model registry

Registers only the two selected candidates as research candidates (alias `research_candidate`, never Production)
and writes lineage, model cards, the selection decision and the reproducibility report.
""")
code("""
from sanvia.training.benchmark import finalize_benchmark

final = finalize_benchmark(cfg, paths, td, summaries, git, env, register=True)
print("Selected :", final["selection"]["selected"])
print("Registered:", final["registered"])
""")

md("## Section 17 - Reproducibility report")
code("""
report = final["report"]
print("Dataset id          :", report["dataset"]["dataset_id"])
print("Dataset manifest hash:", report["dataset"]["dataset_manifest_hash"])
print("Split id            :", report["split"]["split_id"])
print("Git commit          :", report["git"]["commit"], "| dirty:", report["git"]["dirty"])
print(report["git_statement"])
print("Configuration hash  :", report["config_hash"])
display(pd.DataFrame(report["experiments"]))
print("Report saved to:", paths.reports_dir / "reproducibility_report.json")
print("Lineage saved to:", paths.reports_dir / "lineage_report.json")
""")

md("## Section 18 - Final project summary")
code("""
ds = report["dataset"]
print("DATA")
print("  Indexed PNG files      :", ds["n_indexed_pngs"])
print("  Valid breast samples   :", ds["n_valid_breast_samples"])
print("  Studies (train/val/test):", report["split"]["n_studies"])
print("  Samples per split      :", report["split"].get("n_samples"))
print("  Excluded breasts       :", report["split"].get("n_excluded_samples"))
print("  Age integration        :", td.age_reason)
print("MODELS")
display(final["table"][["backbone", "birads_macro_f1", "density_macro_f1", "birads_qwk", "density_qwk", "selection_score", "training_time_min"]])
print("  Selected (Validation only):", final["selection"]["selected"])
for s in summaries:
    if s["backbone"] in final["selection"]["selected"]:
        print("  Checkpoint", s["backbone"], ":", s["best_checkpoint"])
print("MLOPS")
print("  MLflow experiment      :", cfg["mlops"]["mlflow"]["experiment_name"], "at", paths.mlflow_db)
print("  DVC                    :", dvc_status, "| full image tracking: disabled (SHA-256 manifest only)")
print("  Reproducibility        :", report["git_statement"])
print("  Official Test used     :", report["official_test_used"])
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
}
out = Path(__file__).resolve().parents[1] / "notebooks" / "Sanvia_Stage1_Backbone_Benchmark.ipynb"
nbf.write(nb, out)
print("wrote", out)
