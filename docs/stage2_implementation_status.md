# SANVIA Stage 2 implementation status

Base commit `050eae24f2d4ee5be9d9f9f3f929e3fa672f6a3a` on `origin/master`. Development branch `stage2-domain-adaptation`. Update this ledger after each acceptance run.

| Milestone | Acceptance criteria | State |
|---|---|---|
| 1. Stage 1 contract and reproducibility | Run the previously excluded checksum test, explain the cause, retain historical fingerprint interpretation, load the actual selection, run configuration and original study assignments. Reject mixed snapshots and official-test access. | In progress |
| 2. Sources and audit | Read documented source schemas, distinguish image types and grouping levels, retain raw labels, gate ambiguous records, persist deterministic patient splits and detect duplicates across sources and target. Verify missing/partial labels, DICOM polarity, resizing and corrupt files on fixtures. | Pending |
| 3. Supervised source experiments | Encode valid views through the Stage 1 factory, route task heads, verify masked losses and frozen buffers, match per-source update budgets, verify both strategies and interrupted-run recovery. | Pending |
| 4. Controlled target benchmark | Restore exact target architecture, transforms, losses and split. Transfer only verified backbone weights. Run and compare paired baselines for each seed, test validation-only metrics, selection and handoff reload. | Pending |
| 5. Operations and delivery | Exercise CLI and notebook, separate modes and tracking, validate reports, run all existing and new tests, scan changes for secrets, document real-data acceptance. | Pending |

## Inspection and decisions

The Stage 1 notebook is under `notebooks/`. The repository contains six torchvision specifications, a feature-backbone wrapper, paired-view fusion with an optional age branch, train-only fitted preprocessing, a two-phase Trainer and validation-only guarded selection. The runner writes per-run YAML, checkpoint dictionaries, summaries and predictions. The benchmark writes selection, lineage and reproducibility reports. Dataset, split and fitted preprocessing artifacts normally live in the configured private output directory. The Stage 1 preparation path audits the complete image collection. Stage 2 must consume saved artifacts without invoking that path on real target data. Historical run paths may point to another machine.

MLflow uses SQLite and a local artifact store. DVC tracks explicitly staged small files. Keep Stage 2 outputs in a distinct directory and use one writer at a time. Smoke and pilot results do not qualify for research selection.

## Outstanding inputs and issues

No private Stage 1 output directory, licensed source collections, or VinDr PNG root accompanied the specification. Real counts, generalization results and a research handoff require those inputs. The historical notebook's Git authentication cells warrant owner review. Revoke any credential that was ever embedded, remove it from shared copies and use Colab Secrets or environment-based authentication. Do not copy credential-bearing cells into new notebooks.

## Verification log

Branch, base and source inventory verified. Pending checks are listed above.

### Milestone 1 verification

Original master reproduced the failing checksum assertion. CSV round trips convert empty header errors from None to NaN. Canonicalizing missing cells to the historical disk spelling preserves persisted Stage 1 fingerprints under this pandas version. Rehash previously unreadable entries when reusing the image index. Removed the notebook test exclusion. Historical records stay read-only. The reader verifies the saved split and sample hashes, training-only fits and eligible image bytes. Stage 1 summaries log the selected weight enum even on random-initialization runs, so use fresh matched ImageNet baselines for comparisons.

Ran `pytest -q tests/test_pipeline.py tests/stage2/test_stage1_artifacts.py` with CPU PyTorch. 10 passed. The official-test protection test removed every synthetic test image before reading the snapshot. Private snapshot integration remains pending.

### Milestone 2 verification

Inspected the official TCIA mass and calcification case-description headers and revised CMMD workbook header. CBIS uses two spellings for density. CMMD exposes patient and side without an exam identifier in the clinical workbook. Read exam identity from DICOM and block ambiguous cross-examination matches. INbreast binds CSV file numbers to authorized DICOMs. FM and DM require independently supplied release-specific bindings. These last three bindings still need validation on the owner's distributions. BCDR's official endpoint returned HTTP 403 in this environment.

Ran `pytest -q tests/stage2/test_source_audit.py`. 3 passed. Native-format fixtures exercise primary joins, mixed pathology, non-exhaustive findings, native BI-RADS, duplicate gating, grouped splits, DICOM presentation, high-bit-depth input and cache invalidation.

### Milestone 3 verification

The shared source engine plans fixed warm-up and trainable updates per dataset. The joint schedule shuffles a quota-constrained sequence. Sequential transitions restore their best validation checkpoint and create a new optimizer. Persisted draw offsets, RNG state and named best snapshots support completed-update recovery. Invalid views bypass the encoder and inactive heads receive no gradient. Positive-only finding labels contribute masked BCE and positive recall. They do not enter the validation selection score.

Ran `pytest -q tests/stage2/test_source_training.py`. 3 passed. Both interrupted strategies produced bit-identical final tensors to their respective uninterrupted CPU runs. Backbone factory, strict export and full downstream path still require the integration checks below.
