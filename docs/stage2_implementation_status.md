# SANVIA Stage 2 implementation status

Base commit `050eae24f2d4ee5be9d9f9f3f929e3fa672f6a3a` on `origin/master`. Development branch `stage2-domain-adaptation`. Update this ledger after each acceptance run.

| Milestone | Acceptance criteria | State |
|---|---|---|
| 1. Stage 1 contract and reproducibility | Run the previously excluded checksum test, explain the cause, retain historical fingerprint interpretation, load the actual selection, run configuration and original study assignments. Reject mixed snapshots and official-test access. | Implemented and fixture-tested. Private snapshot pending. |
| 2. Sources and audit | Read documented source schemas, distinguish image types and grouping levels, retain raw labels, gate ambiguous records, persist deterministic patient splits and detect duplicates across sources and target. Verify missing/partial labels, DICOM polarity, resizing and corrupt files on fixtures. | Implemented and fixture-tested. Authorized release audits pending. |
| 3. Supervised source experiments | Encode valid views through the Stage 1 factory, route task heads, verify masked losses and frozen buffers, match per-source update budgets, verify both strategies and interrupted-run recovery. | Implemented. Both strategies passed CPU interruption recovery. |
| 4. Controlled target benchmark | Restore exact target architecture, transforms, losses and split. Transfer only verified backbone weights. Run and compare paired baselines for each seed, test validation-only metrics, selection and handoff reload. | Implemented. Six-arm smoke and portable reload passed. |
| 5. Operations and delivery | Exercise CLI and notebook, separate modes and tracking, validate reports, run all existing and new tests, scan changes for secrets, document real-data acceptance. | CLI, tracking and DVC verified. Final notebook regression in progress. |

## Inspection and decisions

The Stage 1 notebook is under `notebooks/`. The repository contains six torchvision specifications, a feature-backbone wrapper, paired-view fusion with an optional age branch, train-only fitted preprocessing, a two-phase Trainer and validation-only guarded selection. The runner writes per-run YAML, checkpoint dictionaries, summaries and predictions. The benchmark writes selection, lineage and reproducibility reports. Dataset, split and fitted preprocessing artifacts normally live in the configured private output directory. The Stage 1 preparation path audits the complete image collection. Stage 2 must consume saved artifacts without invoking that path on real target data. Historical run paths may point to another machine.

MLflow uses SQLite and a local artifact store. DVC tracks explicitly staged small files. Keep Stage 2 outputs in a distinct directory and use one writer at a time. Smoke and pilot results do not qualify for research selection.

## Outstanding inputs and issues

No private Stage 1 output directory, licensed source collections, or VinDr PNG root accompanied the specification. Real counts, generalization results and a research handoff require those inputs. The specification flags a possible credential in historical notebook copies. The checked-out notebook scan found no recognizable GitHub token. Revoke any credential that was ever embedded, remove it from shared copies and use Colab Secrets or environment-based authentication. Do not copy credential-bearing cells into new notebooks.

## Verification log

The results below distinguish software verification from the outstanding real-data acceptance run.

### Milestone 1 verification

Original master reproduced the failing checksum assertion. CSV round trips convert empty header errors from None to NaN. Canonicalizing missing cells to the historical disk spelling preserves persisted Stage 1 fingerprints under this pandas version. Rehash previously unreadable entries when reusing the image index. Removed the notebook test exclusion. Historical records stay read-only. The reader verifies the saved split and sample hashes, training-only fits and eligible image bytes. Stage 1 summaries log the selected weight enum even on random-initialization runs, so use fresh matched ImageNet baselines for comparisons.

Ran `pytest -q tests/test_pipeline.py tests/stage2/test_stage1_artifacts.py` with CPU PyTorch. 10 passed. The official-test protection test removed every synthetic test image before reading the snapshot. Private snapshot integration remains pending.

### Milestone 2 verification

Inspected the official TCIA mass and calcification case-description headers and revised CMMD workbook header. CBIS uses two spellings for density. CMMD exposes patient and side without an exam identifier in the clinical workbook. Read exam identity from DICOM and block ambiguous cross-examination matches. INbreast binds CSV file numbers to authorized DICOMs. FM and DM require independently supplied release-specific bindings. These last three bindings still need validation on the owner's distributions. BCDR's official endpoint returned HTTP 403 in this environment.

Ran `pytest -q tests/stage2/test_source_audit.py`. 3 passed. Native-format fixtures exercise primary joins, mixed pathology, non-exhaustive findings, native BI-RADS, duplicate gating, grouped splits, DICOM presentation, high-bit-depth input and cache invalidation.

### Milestone 3 verification

The shared source engine plans fixed warm-up and trainable updates per dataset. The joint schedule shuffles a quota-constrained sequence. Sequential transitions restore their best validation checkpoint and create a new optimizer. Persisted draw offsets, RNG state and named best snapshots support completed-update recovery. Invalid views bypass the encoder and inactive heads receive no gradient. Positive-only finding labels contribute masked BCE and positive recall. They do not enter the validation selection score.

Ran `pytest -q tests/stage2/test_source_training.py`. 3 passed. Both interrupted strategies produced bit-identical final tensors to their respective uninterrupted CPU runs. Backbone factory, strict export and full downstream path still require the integration checks below.

### Milestone 4 component verification

Stage 2 reuses the Stage 1 target model, augmentations, DataLoaders, two-phase Trainer and metric code. The evaluator can optionally include per-class probability columns. Its default output is unchanged. Source and baseline arms initialize target heads with the same seed and load only verified backbone tensors. Predefine per-task guardrails, validate sample alignment and resample whole studies. Export the pre-target artifact from the representative seed closest to its strategy mean.

Ran `pytest -q tests/stage2/test_target.py`. 3 passed. Verified Stage 1 layout import, incompatible historical baseline provenance, factory reload, checksum rejection and paired study resampling. End-to-end target training and selection are checked in the final milestone.


### Milestone 5 verification

The packaged CLI smoke run used the actual torchvision MobileNetV3-Large and DenseNet121 factories at fixture resolution. It completed two synthetic baseline arms and four adaptations across three native-format synthetic sources. Each adapted arm exported and independently reloaded its pre-target feature weights. The six target runs produced probabilities, confusion matrices, paired comparison rows and mode-ineligible decisions. Official VinDr test usage was false.

Ran `pytest -q tests/stage2/test_integration.py`. Three integration tests passed, including full workflow reuse, isolated linear probes, independent selection with ImageNet fallback, handoff loading, SQLite tracking readback and DVC hash-reference backup. The selection-policy fixture uses controlled scores exclusively for testing guards.

The first Stage 1 suite run had 85 passes and one DVC failure. The environment mounted DVC's default /var/tmp cache read-only. Setting DVC_SITE_CACHE_DIR to writable private storage restored the DVC round-trip test. A follow-up run of the then-current Stage 2 suite plus tests/test_mlops.py passed all 14 checks. No tests were excluded from the final preflight.

## Implementation inventory

The isolated `src/sanvia/stage2/` package contains the Stage 1 reader, source adapters, audit, preprocessing, datasets, source model and training engine, controlled target training, comparison, export, tracking, preflight, synthetic fixture generator, orchestration and CLI. `configs/stage2/` holds the default protocol and task registries. The new notebook and tests use those same entry points. Packaging adds a CLI and optional source-format dependencies. Stage 1 changes are limited to restoring the manifest regression check, its null serialization and cache handling, and opt-in probability output in its evaluator.

## Next real-data acceptance steps

1. Mount the original private Stage 1 report, split, manifest, fitted preprocessing and checkpoint tree and the corresponding eligible VinDr PNGs. Run `sanvia-stage2 plan --config /private/stage2.yaml` to verify the selected pair, fingerprints and historical run records.
2. Set source roots, authorized annotation paths and observed release versions. Run `sanvia-stage2 audit --config /private/stage2.yaml`. Review raw vocabularies, excluded cases, multi-lesion aggregation, grouping, conflicting split assignments, linked derivatives and duplicate candidates. Verify source schemas with the local releases. Configure optional BCDR subsets individually only after this review.
3. Run a named pilot on the intended GPU. Verify compressed DICOM decoders, storage, memory, convergence, throughput, preemption and single-writer Drive tracking. Confirm permissions for any retained patient-level tables and plots.
4. Freeze the protocol and run `sanvia-stage2 run-all --config /private/stage2.yaml` in full mode. Review all matched seeds, validation safeguards, source/target lineage, research registry readback and independent handoff reload. The published branch contains no actual clinical results or selected research weights.

See [protocol, configurations, commands and handoff loading](stage2.md) for the Colab cells, complete private YAML template and output layout.

The notebook's first synthetic execution completed all 26 code cells with zero cell errors, including its 99-test preflight. A local registry test exposed that MLflow 3.18 uses an exported PyTorch graph by default and requires an input example. Registration now provides a synthetic array at the recorded resolution. A second registration and alias readback succeeded with a real torchvision encoder. No private images entered this test. The final acceptance run also covers newly added release-binding and weighted gradient-accumulation checks.
