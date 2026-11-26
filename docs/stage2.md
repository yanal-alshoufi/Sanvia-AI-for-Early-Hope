# Supervised cross-dataset transfer for SANVIA

Stage 2 asks whether supervision from external mammography collections improves two backbones on the held-out VinDr **validation** studies. Each selected backbone gets three target initializations for each seed. All three arms use the same controlled Stage 1 VinDr training protocol.

| Arm | Initialization used for target training |
|---|---|
| Matched baseline | The exact torchvision ImageNet weight enum recorded in Stage 1 |
| Sequential | The selected checkpoint after CBIS-DDSM, CMMD, then INbreast by default |
| Multi-dataset | The selected joint checkpoint from homogeneous, quota-balanced source batches |

Starting external training from ImageNet isolates external supervision. Starting from a VinDr-fine-tuned checkpoint would introduce target supervision before the intervention. Stage 1 checkpoints and validation summaries provide baseline evidence. Stage 1 logs its weight enum even for a run with pretraining disabled, and its prediction CSV omits probabilities. Stage 2 reproduces a fresh baseline at each evaluated seed. The split seed and original age and normalization fits stay fixed across target training seeds.

## Installation and first run

Run these commands from a checkout of the feature branch, using Python 3.10 or newer and compatible PyTorch/torchvision wheels. CPU verification used Python 3.12 and PyTorch 2.7.1 CPU.

```bash
git clone --branch stage2-domain-adaptation https://github.com/yanal-alshoufi/Sanvia-AI-for-Early-Hope.git
cd Sanvia-AI-for-Early-Hope
python -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[stage2,dev]'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 sanvia-stage2 smoke-test --output /tmp/sanvia-stage2
```

For GPU execution, install the PyTorch and torchvision build appropriate for your CUDA runtime. The smoke command generates small synthetic DICOM and PNG collections in its output directory. Its designated fixture backbones, random initialization, small resolution and recorded scores are for software verification only. The notebook and CLI share the same research runner.

In Colab, open `notebooks/Sanvia_Stage2_Domain_Adaptation.ipynb` from the feature branch, enable a GPU for real data, run the Drive setup and confirm `REPO_DIR`. In section 3 set `MODE`, `OUTPUT_DIR`, `STAGE1_OUTPUT_DIR`, `VINDR_IMAGES`, `DATASET_ROOTS`, `ENABLED_SOURCES`, `SELECTED_STRATEGIES`, `PRETRAINED`, `SEEDS`, `RESUME` and a private YAML `OVERRIDES` mapping. Run the remaining cells in order. Full mode reads the Stage 1 selection and uses pretrained weights. Pilot mode uses authorized real inputs and a separately configured small source budget, and never produces a research handoff. Keep notebook outputs containing medical images or validation records private. The notebook makes no Git commits or pushes.

## Private inputs

The original Stage 1 output tree must contain the following files. Copy the original artifacts intact. Updating a path in the saved YAML changes its hash. Use the separate Stage 2 `vindr_images` setting to relocate the authorized PNG directory.

```text
reports/{resolved_config.yaml,selection_decision.json,reproducibility_report.json}
manifests/{png_image_manifest.csv,dataset_manifest.json}
splits/{train_studies.csv,validation_studies.csv,official_test_studies.csv,final_sample_manifest.csv,split_summary.json,age_report.json}
preprocessing/fitted_preprocessing.json
checkpoints/<backbone>/<original-run-id>/{resolved_config.yaml,run_summary.json}
```

When available, keep the selected runs' `best.pt` and `validation_predictions.csv`. Use only trusted checkpoint files from your own Stage 1 run. Stage 2 checks study assignments and sample checksums, uses original image identifiers and relative paths, and verifies the bytes of eligible train/validation images. Official-test identifiers and existing manifest hashes are used only for integrity checks. No official-test loader is constructed. Acquire VinDr through the authorized PhysioNet access process. [VinDr-Mammo v1.0.0](https://physionet.org/content/vindr-mammo/1.0.0/)

### External sources

- **CBIS-DDSM.** Obtain the full mammogram DICOM series and the four mass/calcification train/test case-description CSVs through TCIA, following the collection's download and attribution terms. Supply a common root with the original paths, or DICOMs with series UIDs matching the parent directory in the official full-image reference. The adapter distinguishes the full-image, crop and ROI-mask columns. Both density column spellings occur in the official metadata. The underlying case-description patient ID groups mass, calcification, views and both sides across official partitions. Multiple lesion rows for one image produce a single image, with lesion and raw annotation tables. Mixed pathology is masked by default. Optional `any_malignant` defines an **image-level** target for the annotated lesions on that projection. Positive-only finding supervision can learn a trivial all-positive predictor and has no validated specificity. Its recall is descriptive and excluded from checkpoint selection. [TCIA CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)
- **CMMD.** Obtain the released DICOM collection and revised clinical workbook from TCIA. The clinical join uses `ID1`, `LeftRight` and `classification`. Only a matching side receives diagnosis. Two examinations for the same diagnosed side require additional verified linkage. Unmatched DICOM sides retain missing pathology. A clinical row without a matching side blocks training. [TCIA CMMD](https://www.cancerimagingarchive.net/collection/cmmd/)
- **INbreast.** Request licensed access from the dataset custodians. Point the adapter at the distributed annotation table and DICOM root. Verify `File Name`, `Laterality`, `View`, `Bi-Rads` and `ACR` on your authorized copy. DICOM patient/study UIDs group projections; the numeric filename prefix links image annotations. Native 0, 6 and category 4 subdivisions have distinct logits. Unknown categories require an explicit task mapping. The original publication identifies the dataset and custodian context. [INbreast publication](https://pubmed.ncbi.nlm.nih.gov/22078258/)
- **BCDR-FM and BCDR-DM.** Apply to the official repository and obtain permission for each subset and intended use. Both sources are disabled by default. Supply a separate version, `schema_reference`, task definitions and column bindings for each authorized release. The configurable tabular adapter requires verified patient, study, image, laterality, view, file path and image-role columns. Use `annotations[].columns` to map canonical field names to native headers, and `vocabulary` for explicit side, role, view and split translations. The only accepted primary role is `full`; `mask` and `crop` are excluded. Bind each task's raw label field explicitly. No native BCDR release schema was available for verification here. [BCDR access portal](https://bcdr.ceta-ciemat.es/)

Preserve original identifiers and authorized files under separate source roots. Do not copy images or restricted metadata into Git. The unit of grouping is the verified examination and side. If a release lacks a usable patient UID, study UID or view, resolve the release-specific identity using authorized documentation before using it. The adapters stop on ambiguous joins or multiple distinct projections in a single view slot. De-identification prevents a claim of patient disjointness across collections.

## Configuring a real experiment

`configs/stage2/default.yaml` contains the editable settings and `configs/stage2/tasks.yaml` defines the primary tasks. A private YAML override merges recursively. The following is a template. Replace each angle-bracket value with an authorized path or verified release identifier.

```yaml
mode: full
output: /private/SANVIA_Stage2_Outputs
stage1_output: /private/VinDr/sanvia_outputs
vindr_images: /private/VinDr/images/images_png
seeds: [42, 123, 2026]
cache: /private/fast-local-cache
sources:
  CBIS-DDSM:
    root: /private/CBIS
    version: '<verified CBIS release>'
    annotations:
      - {path: mass_case_description_train_set.csv, official_split: train}
      - {path: calc_case_description_train_set.csv, official_split: train}
      - {path: mass_case_description_test_set.csv, official_split: test}
      - {path: calc_case_description_test_set.csv, official_split: test}
  CMMD:
    root: /private/CMMD
    version: '<verified CMMD release>'
    annotations: [{path: CMMD_clinicaldata_revision.xlsx}]
  INbreast:
    root: /private/INbreast
    version: '<verified INbreast release>'
    annotations: [{path: '<released annotation CSV>', separator: ';'}]
```

A single-seed full benchmark may set `seeds: [42]`. Set `name` and `reduced_sources: true` for an explicitly reduced-source experiment. A development backbone override requires a separate experiment name. BCDR-FM and BCDR-DM can be inserted in the ordered `enabled_sources` list after their individual releases have passed audit. Both strategies use precisely that list and the same accepted manifests.

Each task declares scope, loss, original-to-canonical mapping, ordered classes and weight. The class count is the length of the ordered list. Eligibility is per task and per slot. Missing values are masked, unknown raw labels trigger review and explicit exclusions are recorded. CE uses training-only balanced weights. Each accumulated source update normalizes by the valid label mass across the entire effective batch, including uneven partial-label masks. BCE uses a separate mask for each verified positive finding. Source selection uses the equal-weight mean of evaluable CE task macro F1s within each dataset and then across datasets for the joint strategy. Configure a release-specific task registry under `sources.<dataset>.tasks` to replace that source's defaults. Changing mappings creates a new experiment identity.

## Audit and preprocessing

Every accepted source has normalized patient, study, breast, image, lesion, task and partition tables, original annotations, SHA-256 manifests, plots, missing-label reasons and a critical-issue report. Official source test patients stay out of training and validation. Remaining eligible patients undergo grouped splitting, stratified on their observed class signatures when group counts permit. An accepted snapshot persists the assignments. Resume verifies its fingerprint, source annotation hashes and source file bytes.

Exact source image hashes are checked within and across collections and against the existing VinDr manifest. Optional per-source `decoded_hashes: true` records hashes of normalized decoded arrays. `near_duplicates: true` flags **equal 64-bit difference hashes** as review candidates among enabled sources. These checks cannot rule out all transformed duplicates. Unresolved candidates block training. Keep a reviewed release record if you quarantine or relink cases and assign the curated inputs a new version. Failed audits record the blocking error and may have partial tables. An unresolved linkage error must be corrected before final counts are available.

External DICOM processing uses pydicom modality and VOI transforms and excludes stored padding from grayscale scaling. The first VOI alternative is used. MONOCHROME1 has a single inversion; consistent explicit presentation shapes are honored. Complex presentation LUT sequences, contradictory polarity, multiframe images, processing-intent DICOMs, constant DICOM ranges and unsupported color images need an explicit verified presentation policy. Install a compatible pydicom codec for the transfer syntax in your release. PNG, TIFF and JPEG accept unsigned grayscale or identical RGB channels. Source intensities are letterboxed with the Stage 1 geometry and normalized with the Stage 1 ImageNet settings. Validation has deterministic transforms. The optional cache binds source bytes, preprocessing settings, version and resolution and checks the saved array. For authorized data previews set `preview_images: true` and use private output storage.

## Execute and resume

Run from the repository root with a private override file.

```bash
sanvia-stage2 plan --config /private/stage2.yaml
sanvia-stage2 audit --config /private/stage2.yaml
sanvia-stage2 prepare --config /private/stage2.yaml
sanvia-stage2 train-sequential --config /private/stage2.yaml
sanvia-stage2 train-multi-dataset --config /private/stage2.yaml
sanvia-stage2 finetune-vindr --config /private/stage2.yaml
sanvia-stage2 evaluate --config /private/stage2.yaml
sanvia-stage2 compare --config /private/stage2.yaml
sanvia-stage2 export --config /private/stage2.yaml
```

`run-all` runs the primary pipeline in order. `evaluate` and `compare` use saved, checksum-verified validation predictions and the recorded target metrics. The optional `linear-probe` action trains separate linear BI-RADS and density heads on fixed concatenated view features and optional fitted age inputs. Run it after source training. Its files live in `linear_probe/` and are excluded from selection.

`source_training` controls source-only effective batch size (microbatch size × accumulation), fixed head-only and trainable updates, per-source fine-tuning quotas, cosine decay, separate learning rates, AMP, clipping, prefetching and workers. Per-backbone batch overrides use `batch_by_backbone`. Planned and observed optimizer steps, projected image counts, actual images, repeated samples and valid labels are exported. A positive `replay_fraction` schedules extra training-source updates in a separately configured sequential variant. Positive `l2sp` regularizes the encoder toward its initial weights. Neither is enabled in the primary benchmark. Sequential transitions restore the selected validation checkpoint and start a fresh optimizer. Joint training selects on the equal-dataset validation score. Matched intended quotas do not guarantee equal elapsed time or valid image counts.

Rerun the same command after interruption. Source recovery resumes at a committed optimizer update and restores its best-state reference, source draw positions, optimizer, scheduler, scaler and RNG. Target recovery resumes at an epoch boundary, including the shuffle-generator state. Stage 1 persistent-worker augmentation can yield different draws after restarting a multiworker target job. Exact multiworker target numerical replay is unverified. Set a new configuration and record its identity if resource limits require a change. Full and pilot training run the repository's regression suite once per verified test/code fingerprint. Diagnostic logs and private partial checkpoints remain on disk after errors.

## Reporting and research handoff

```text
<output>/<mode>/<name>/<benchmark-key>/
  audit/<source>/                 normalized tables, split assignments, plots
  <backbone>/<seed>/<strategy>/
    source/<stage>/               selected and last source checkpoints
    source/                       budgets, trajectories, forgetting, pre-target export
    target/                       complete target model, post-target encoder, last state,
                                  held-out predictions, metrics, lineage and plots
    linear_probe/                 optional frozen-feature evaluation
  comparison/                     seed results, class changes, intervals, decisions
  tracking/                       MLflow SQLite, artifacts, optional local DVC storage
  stage3_handoff/                 full-mode research candidates
  reproducibility_report.json
```

The primary score is the mean of BI-RADS and density macro F1 with Stage 1 class ordering. The existing evaluator excludes unsupported ground-truth classes from macro F1 and records undefined per-class metrics and AUC. Source/target stage selection only uses validation. Optional paired bootstrap samples whole studies with replacement; the same study multiplicities apply to each pair. Per-seed deltas, mean, standard deviation and class-level changes accompany the comparison. Relative gain is undefined for a zero baseline score.

The policy evaluates each backbone independently. A strategy needs positive mean improvement above the configured threshold, adequate BI-RADS performance relative to the best, acceptable per-seed task/QWK/minority recall changes and a bounded overfit gap. Missing guard metrics disqualify that adaptation. With bootstrap enabled, every compared seed must have a positive lower confidence bound. Eligible strategies rank by mean score, then lower seed dispersion and configured default arm priority (sequential first). Otherwise select ImageNet. The representative seed is closest to the selected arm's mean score, with smaller seed breaking a tie.

The handoff directory contains portable verified **pre-VinDr** feature backbones, specifications, research model cards, checksums and selection lineage. This preserves a source-only initialization for future target experiments. Fine-tuned target encoders are separate alternative artifacts. Load a handoff independently:

```python
from pathlib import Path
from sanvia.stage2.export import load_export

encoder = load_export(Path("/private/benchmark/stage3_handoff/<backbone>/backbone.pt"))
encoder.eval()
```

## MLOps and troubleshooting

Stage 2 uses the local `SANVIA_Stage2_Domain_Adaptation` MLflow experiment with per-arm parents and source/target child runs. A filesystem lock serializes writers to an output root. Use one Colab runtime per Google Drive output root because distributed Drive lock semantics depend on the mounted filesystem. Local files remain available if tracking fails; inspect `mlflow_failure.json`. Full-mode registration uses the `research_candidate` alias. These are research artifacts without clinical validation.

Optional DVC uses its own private Git/DVC workspace and local remote under the benchmark output. It stages only the compact accepted source hash references. Image hashes verify bytes. Restoring a reference file cannot restore a medical image. Keep all authorized clinical material, validation records and per-sample reports in private storage with appropriate access controls.

For a missing Stage 1 artifact, restore the original run directory with its reports and per-run YAML. For a changed sample or split hash, retrieve the corresponding snapshot. For unknown labels or view conflicts, inspect the original release and use a documented mapping. For compressed DICOM failures install a compatible decoder and rerun the audit. The runner records available bytes for the output and cache filesystems and checks the configurable minimum_free_gib bound. For missing space select a suitable private volume. To debug a failed preflight inspect `preflight.log`. In a read-only environment set `DVC_SITE_CACHE_DIR` to a writable directory. Details of executed checks and remaining real-data validation are in [the implementation ledger](stage2_implementation_status.md).
