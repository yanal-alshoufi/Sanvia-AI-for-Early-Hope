# SANVIA Stage 2: supervised transfer benchmark

The Stage 2 feature branch adds sequential and joint supervised source training, matched VinDr baselines and validation-only initialization selection. Start with the [Stage 2 protocol and commands](docs/stage2.md), [Colab notebook](notebooks/Sanvia_Stage2_Domain_Adaptation.ipynb) and [verification ledger](docs/stage2_implementation_status.md).

## Quick start (Google Colab)

1. Put this repository on Google Drive (for example `MyDrive/SANVIA`) or clone it, then `git init`/commit if needed
   (DVC and run metadata need a Git repository).
2. Open `notebooks/Sanvia_Stage1_Backbone_Benchmark.ipynb`, select a GPU runtime, set `REPO_DIR` in the first cell, run all.
3. Data is read from `/content/drive/MyDrive/VnDir_Mammo`; all persistent outputs go to
   `/content/drive/MyDrive/VnDir_Mammo/sanvia_outputs`.

Command line (data stages only): `python scripts/prepare_data.py --data-dir /path/to/VnDir_Mammo`.
Synthetic end-to-end check (no real data, no downloads): `python scripts/smoke_test_synthetic.py --backbones MobileNetV3-Large DenseNet121 ResNet50`.

## Layout

| Path | Role |
|---|---|
| `configs/*.yaml` | base, data, training, mlops, backbones configuration (the resolved config is logged per run) |
| `src/sanvia/data` | CSV loading, PNG indexing/IO, audit, labels, splitting, sample builder, age, preprocessing, dataset, pipeline steps |
| `src/sanvia/models` | backbone factory (one interface, six backbones), fusion, heads, multi-task model |
| `src/sanvia/training` | losses, optimizer, scheduler, early stopping, checkpointing, trainer, per-backbone runner, benchmark orchestration |
| `src/sanvia/evaluation` | metrics (NaN for undefined values), evaluator, plots, comparison and Validation-only selection |
| `src/sanvia/mlops` | Git metadata, SHA-256 manifests and dataset id, DVC wrapper, MLflow tracking and registry, reproducibility/lineage |
| `tests/` | pytest suite on synthetic grayscale PNGs (no real data) |
| `.github/workflows/ci.yml` | CI: compile, Ruff, pytest on CPU |

## Key methodological decisions

- **Split**: official Training studies are split 80/20 into Train/Validation by `study_id` (stratified on a study-level
  BI-RADS/density signature, with documented fallback). Official Test studies are kept unchanged and isolated. The old logic that
  built Validation from official Test studies was not reused.
- **Samples**: only complete CC+MLO pairs with readable files and consistent labels. Missing views, duplicate records, conflicting
  labels, unknown label strings and unreadable images are reported in `excluded_samples.csv`; nothing is substituted or resolved silently.
  A gate stops the pipeline if more than 5% of breasts are excluded.
- **Labels**: exact string matching from `data.labels` (no regex). Missing labels are masked per task (never class 0).
- **PNG handling**: bit depth is preserved (8/16-bit). Intensity scaling uses the container range verified by the audit; a 16-bit set whose
  observed maximum is far below 65535 stops the pipeline instead of guessing. Letterboxing preserves aspect ratio; no center crop.
- **Age**: used only if the metadata join is verified (exact match rate, duplicates, conflicts); imputation and scaling are fitted on Train only.
  Otherwise the age branch is disabled for all six backbones and the reason is recorded.
- **Model**: shared backbone for both views, concatenation `[CC, MLO]`, fusion MLP, optional age MLP, two heads returning logits.
  Identical for every backbone. Reported backbone parameter counts exclude the removed ImageNet classifier, so they are lower than the
  "approximate parameters" in the protocol table.
- **Training**: frozen-backbone phase (AdamW, lr 1e-3), then gradual unfreezing with lr 2e-5 (backbone) / 1e-4 (heads), warm-up + cosine,
  AMP, gradient accumulation to effective batch 16, early stopping on Validation mean Macro F1. Weighted cross-entropy with Train-only class weights.
  Frozen BatchNorm layers stay in eval mode.
- **Selection**: mean of BI-RADS and density Macro F1 on Validation, with guards for weak BI-RADS (below 85% of the best) and a
  train-validation gap above 0.30. Thresholds are in `configs/training.yaml`.

## MLOps scope (what is and is not versioned)

- **Git** versions code, configs, notebooks, tests and CI. Each run records the commit hash and dirty state; a dirty tree is flagged as not fully reproducible.
- **DVC** versions only small files (source CSVs, manifests, splits) via `data/tracked/` and a Drive-based remote configured with `--local`
  (machine-specific, not committed). `track_full_images` must stay `false`; PNGs are **not** copied or tracked.
- **PNG integrity** is verified through `png_image_manifest.csv` (size, SHA-256, dimensions, bit depth). It detects changed, missing and added
  images but **cannot restore** previous image versions; that needs a real backup.
- **MLflow** uses SQLite plus an artifact directory on Google Drive. This is suitable for single-user, sequential runs only; do not run concurrent writers.
  Registered models are research candidates (alias `research_candidate`), never Production. Logged models use pickle serialization and need `sanvia` installed to load.
- A stable `dataset_id` (CSV + PNG manifest checksums) and `dataset_manifest_hash` (adds sample and split manifests) are logged with every run.

## Development

```bash
pip install -r requirements.txt && pip install -e . --no-deps
ruff check src tests scripts && ruff format --check src tests scripts
pytest -q
python scripts/build_notebook.py   # regenerate the notebook
```

## Known limitations

- Single seed and a single dataset; results are Validation-only research indicators, not clinical evidence.
- Backbones use different input resolutions (standard size per weights); a common-resolution comparison of the final two is recommended afterwards.
- Left/right orientation is not standardized; images are used as stored.
- Full determinism is not guaranteed (see `sanvia.utils.seed.NONDETERMINISM_NOTES`).
- Pretrained weights are downloaded from torchvision at run time (internet required in Colab).
